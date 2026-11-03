#!/usr/bin/env python3
import time
import threading
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import deque
from ft300python.ft300_stream import FT300Stream
from matplotlib.animation import FuncAnimation
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from scipy.spatial.transform import Rotation as R

# --- ROS Node for Roll Calculation ---
JOINTS = [
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "shoulder_pan_joint",
]

class RollNode(Node):
    def __init__(self):
        super().__init__('roll_node')
        self.joint_positions = {}
        self.roll_data = deque(maxlen=3000)
        self.time_data = deque(maxlen=3000)
        self.start_time = time.time()
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for FK service...")

    def joint_state_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in JOINTS:
                self.joint_positions[name] = pos
        if all(name in self.joint_positions for name in JOINTS):
            self.get_current_pose()

    def get_current_pose(self):
        req = GetPositionFK.Request()
        req.header.frame_id = 'base'
        req.fk_link_names = ['tool0']
        req.robot_state = RobotState()
        req.robot_state.joint_state.name = JOINTS
        req.robot_state.joint_state.position = [self.joint_positions[n] for n in JOINTS]
        future = self.fk_client.call_async(req)
        future.add_done_callback(self.fk_response)

    def fk_response(self, future):
        try:
            resp = future.result()
            if resp.error_code.val != 1:
                return
            pose = resp.pose_stamped[0].pose
            r = R.from_quat([
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            ])
            euler = r.as_euler('xyz', degrees=True)
            roll = euler[0]
            current_time = (self.get_clock().now().nanoseconds / 1e9) - self.start_time
            self.roll_data.append(roll)
            self.time_data.append(current_time)
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def get_roll_data(self):
        return list(self.time_data), list(self.roll_data)

# --- Configuration ---
PORT = "/dev/ttyUSB0"
TIMEOUT = 1
G = 9.81
MOUNT_MASS = 0.100
GRIPPER_MASS = 1.0 + 0.25

# --- Shared resources & State ---
latest_readings = (0.0, 0.0)
sample_count = 0
data_lock = threading.Lock()
running = True

# Buffers
MAX_SAMPLES = 3000
z_history = deque(maxlen=MAX_SAMPLES)
t_history = deque(maxlen=MAX_SAMPLES)

# Calibration Data State
collected_data = []
current_step_idx = 0

# --- Sensor Worker ---
def sensor_worker():
    global running, latest_readings, sample_count
    try:
        sensor = FT300Stream(PORT, timeout=TIMEOUT, zero_reset=False)
        for _ in range(10):
            sensor.get_force_torque()
            time.sleep(0.005)
        
        start_time = time.time()
        while running:
            readings = sensor.get_force_torque()
            with data_lock:
                latest_readings = (readings[2], readings[3])
                sample_count += 1
                z_history.append(readings[2])
                t_history.append(time.time() - start_time)
            time.sleep(0.005)
    except Exception as e:
        print(f"Sensor Error: {e}")
        running = False

def calc_expected_force(added_mass):
    return (GRIPPER_MASS + added_mass) * G

steps = [
    ("Pull_Up_No_Load", 0.0, "Orient gripper to PULL UP - NO extra weights"),
    ("Mount_Only", MOUNT_MASS, f"Attach {MOUNT_MASS*1000:.0f}g mount")
]
weight_points = []
for r in [range(10, 460, 10), [500], range(510, 650, 10)]:
    weight_points.extend(list(r))
for w in weight_points:
    steps.append((f"Mount+{w}g", MOUNT_MASS + (w / 1000.0), f"Add {w}g weight"))

# --- GUI / Plotting Class ---
class CalibrationGUI:
    def __init__(self, roll_node):
        self.roll_node = roll_node
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("FT300 Calibration Visualizer")

        # 1. Live Roll Plot (Top - Full Width)
        self.ax_roll = plt.subplot(2, 1, 1)
        self.line_roll, = self.ax_roll.plot([], [], 'red', lw=1.5)
        self.ax_roll.set_title("Live Roll Orientation")
        self.ax_roll.set_ylabel("Roll (Degrees)")
        self.ax_roll.set_ylim(-180, 180)
        self.ax_roll.grid(True, alpha=0.3)

        # 2. Scatter Plot (Bottom - Full Width)
        self.ax_scatter = plt.subplot(2, 1, 2)
        self.scatter_pts = self.ax_scatter.scatter([], [], color='blue', label='Samples')
        self.ax_scatter.set_title("Calibration Curve (Samples Taken)")
        self.ax_scatter.set_ylabel("Raw Fz")
        self.ax_scatter.set_xlabel("Expected N")
        self.ax_scatter.grid(True, alpha=0.3)

        plt.subplots_adjust(bottom=0.2, hspace=0.4)

        self.txt_instruction = self.fig.text(0.5, 0.12, "", ha='center', fontsize=12, fontweight='bold', color='blue')
        self.txt_status = self.fig.text(0.5, 0.08, "Ready to start", ha='center', fontsize=10)

        # Buttons
        self.ax_btn_sample = plt.axes([0.35, 0.02, 0.14, 0.05])
        self.btn_sample = Button(self.ax_btn_sample, 'Take Sample', color='limegreen', hovercolor='green')
        self.btn_sample.on_clicked(self.on_sample)

        self.ax_btn_undo = plt.axes([0.51, 0.02, 0.14, 0.05])
        self.btn_undo = Button(self.ax_btn_undo, 'Delete/Undo', color='tomato', hovercolor='red')
        self.btn_undo.on_clicked(self.on_undo)

        self.update_instruction()

    def update_instruction(self):
        if current_step_idx < len(steps):
            label, _, instr = steps[current_step_idx]
            self.txt_instruction.set_text(f"STEP {current_step_idx+1}/{len(steps)}: {instr}")
        else:
            self.txt_instruction.set_text("COLLECTION COMPLETE! Close window to save.")
            self.btn_sample.set_active(False)

    def on_sample(self, event):
        global current_step_idx
        if current_step_idx >= len(steps): return

        self.txt_status.set_text("Sampling 100 points... please hold still.")
        self.fig.canvas.draw_idle()

        label, extra_m, _ = steps[current_step_idx]
        samples = []
        with data_lock: start_cnt = sample_count

        while len(samples) < 100:
            with data_lock:
                if sample_count > start_cnt:
                    samples.append(latest_readings[0])
                    start_cnt = sample_count
            self.fig.canvas.flush_events()
            time.sleep(0.001)

        avg_fz = sum(samples) / len(samples)
        expected_n = calc_expected_force(extra_m)

        collected_data.append({"label": label, "fz_raw": avg_fz, "fz_n": expected_n})
        current_step_idx += 1

        self.update_instruction()
        self.update_scatter()
        self.txt_status.set_text(f"Recorded: {label} (Fz: {avg_fz:.2f})")

    def on_undo(self, event):
        global current_step_idx
        if len(collected_data) > 0:
            item = collected_data.pop()
            current_step_idx -= 1
            self.update_instruction()
            self.update_scatter()
            self.txt_status.set_text(f"Removed: {item['label']}")

    def update_scatter(self):
        if not collected_data:
            self.scatter_pts.set_offsets(np.empty((0, 2)))
        else:
            x = [p["fz_raw"] for p in collected_data]
            y = [p["fz_n"] for p in collected_data]
            self.scatter_pts.set_offsets(np.c_[y, x])
            self.ax_scatter.set_xlim(min(y) - 0.5, max(y) + 0.5)
            self.ax_scatter.set_ylim(min(x) - 1, max(x) + 1)
        self.fig.canvas.draw_idle()

    def update_plot(self):
        roll_time, roll_data = self.roll_node.get_roll_data()
        if not roll_time:
            return

        decimation = max(1, len(roll_time) // 500)
        roll_time_dec = roll_time[::decimation]
        roll_data_dec = roll_data[::decimation]

        self.line_roll.set_data(roll_time_dec, roll_data_dec)
        if roll_time_dec:
            self.ax_roll.set_xlim(roll_time_dec[0], roll_time_dec[-1] + 0.5)

    def animate(self, i):
        self.update_plot()
        return self.line_roll,

# --- Main Execution ---
def main():
    global running
    rclpy.init()
    roll_node = RollNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(roll_node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    sensor_thread = threading.Thread(target=sensor_worker, daemon=True)
    sensor_thread.start()

    print("Initializing Sensor and GUI...")
    time.sleep(1)

    gui = CalibrationGUI(roll_node)
    ani = FuncAnimation(gui.fig, gui.animate, interval=50, blit=True)

    plt.show()

    running = False
    if collected_data:
        output_file = "ft300_fz_data.json"
        with open(output_file, "w") as f:
            json.dump(collected_data, f, indent=4)
        print(f"Saved {len(collected_data)} points to {output_file}")

    roll_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()