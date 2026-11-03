import time
import threading
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from collections import deque
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK

# Note: Keeping your specific import
try:
    from ft300python.ft300_stream import FT300Stream
except ImportError:
    # Dummy class for testing if lib is missing
    class FT300Stream:
        def __init__(self, *args, **kwargs): pass
        def get_force_torque(self):
            # Simulated: [Fx, Fy, Fz, Tx, Ty, Tz]
            return [0, 0, -9.8, np.random.normal(0, 0.1), np.random.normal(0, 0.05), 0]

# --- ROS Node for Roll Calculation ---
JOINTS = [
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "shoulder_pan_joint",
]

global running

class RollNode(Node):
    def __init__(self):
        super().__init__('roll_node')
        self.joint_positions = {}
        self.roll_data = deque(maxlen=3000)
        self.time_data = deque(maxlen=3000)
        self.start_time = self.get_clock().now().nanoseconds / 1e9
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
D_GRIPPER = 0.080
D_LOAD = 0.278

# --- Shared State ---
latest_readings = (0.0, 0.0)
sample_count = 0
data_lock = threading.Lock()
running = True

MAX_SAMPLES = 3000  # ~15 seconds at 200 Hz
tx_history = deque(maxlen=MAX_SAMPLES)
time_history = deque(maxlen=MAX_SAMPLES)

collected_data = []
current_step_idx = 0

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
                tx_history.append(readings[3])
                time_history.append(time.time() - start_time)
            time.sleep(0.005)
    except Exception as e:
        print(f"[Background] Sensor Error: {e}")
        running = False

def calc_expected_torque(load_mass):
    gripper_moment = GRIPPER_MASS * G * D_GRIPPER
    load_moment = load_mass * G * D_LOAD
    return (gripper_moment + load_moment)

# --- Steps Generation ---
steps = [
    ("Base_Gripper", 0.0, "Ensure gripper is at 90 deg (Horizontal) NO load"),
    ("Base_Gripper+Mount", MOUNT_MASS, "Add Mount without weight")
]
weight_points = []
for r in [range(10, 460, 10), [500], range(510, 650, 10)]:
    weight_points.extend(list(r))
for w in weight_points:
    total_load = MOUNT_MASS + (w / 1000.0)
    steps.append((f"Mount+{w}g", total_load, f"Add {w}g weight"))

class CalibrationGUI:
    def __init__(self, roll_node):
        self.roll_node = roll_node
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("FT300 Torque & Roll Calibration")

        # 1. Live Roll Plot (Top)
        self.ax_roll = plt.subplot(2, 1, 1)
        self.line_roll, = self.ax_roll.plot([], [], 'red', lw=1.5)
        self.ax_roll.set_title("Live Roll Orientation")
        self.ax_roll.set_ylabel("Degrees")
        self.ax_roll.set_ylim(-180, 180)
        self.ax_roll.grid(True, alpha=0.3)

        # 2. Scatter Plot (Bottom)
        self.ax_scatter = plt.subplot(2, 1, 2)
        self.scatter_pts = self.ax_scatter.scatter([], [], color='blue', s=30, label='Samples')
        self.ax_scatter.set_title("Calibration Curve (Raw Tx vs Expected Nm)")
        self.ax_scatter.set_ylabel("Raw Tx")
        self.ax_scatter.set_xlabel("Expected Torque (Nm)")
        self.ax_scatter.grid(True, alpha=0.3)

        plt.subplots_adjust(bottom=0.2, hspace=0.4)

        self.txt_instruction = self.fig.text(0.5, 0.12, "", ha='center', fontsize=11, fontweight='bold', color='darkred')
        self.txt_status = self.fig.text(0.5, 0.08, "Waiting for sensor...", ha='center', fontsize=10)

        # Buttons
        self.ax_btn_sample = plt.axes([0.35, 0.02, 0.14, 0.05])
        self.btn_sample = Button(self.ax_btn_sample, 'Take Sample', color='palegreen', hovercolor='limegreen')
        self.btn_sample.on_clicked(self.on_sample)

        self.ax_btn_undo = plt.axes([0.51, 0.02, 0.14, 0.05])
        self.btn_undo = Button(self.ax_btn_undo, 'Undo Last', color='mistyrose', hovercolor='tomato')
        self.btn_undo.on_clicked(self.on_undo)

        self.update_instruction()

    def update_instruction(self):
        if current_step_idx < len(steps):
            label, _, instr = steps[current_step_idx]
            self.txt_instruction.set_text(f"STEP {current_step_idx+1}/{len(steps)}: {instr}")
        else:
            self.txt_instruction.set_text("COLLECTION COMPLETE! Close window to save JSON.")
            self.btn_sample.set_active(False)

    def on_sample(self, event):
        global current_step_idx
        if current_step_idx >= len(steps): return
        self.txt_status.set_text("Sampling 100 points...")
        self.fig.canvas.draw_idle()

        label, mass, _ = steps[current_step_idx]
        samples = []
        with data_lock: start_cnt = sample_count

        while len(samples) < 100:
            with data_lock:
                if sample_count > start_cnt:
                    samples.append(latest_readings[1])
                    start_cnt = sample_count
            self.fig.canvas.flush_events()
            time.sleep(0.001)

        avg_tx_raw = sum(samples) / len(samples)
        expected_nm = calc_expected_torque(mass)
        collected_data.append({"label": label, "tx_raw": round(avg_tx_raw, 6), "tx_nm": round(expected_nm, 6)})

        current_step_idx += 1
        self.update_instruction()
        self.update_scatter()
        self.txt_status.set_text(f"Recorded: {label} | Tx: {avg_tx_raw:.4f}")

    def on_undo(self, event):
        global current_step_idx
        if len(collected_data) > 0:
            removed = collected_data.pop()
            current_step_idx -= 1
            self.update_instruction()
            self.update_scatter()
            self.txt_status.set_text(f"Removed: {removed['label']}")

    def update_scatter(self):
        if not collected_data:
            self.scatter_pts.set_offsets(np.empty((0, 2)))
        else:
            x_vals = [p["tx_nm"] for p in collected_data]
            y_vals = [p["tx_raw"] for p in collected_data]
            offsets = np.column_stack((x_vals, y_vals))
            self.scatter_pts.set_offsets(offsets)
            self.ax_scatter.set_xlim(min(x_vals) - 0.5, max(x_vals) + 0.5)
            self.ax_scatter.set_ylim(min(y_vals) - 1, max(y_vals) + 1)
        self.fig.canvas.draw_idle()

    def animate(self, i):
        # Get roll data from ROS node
        roll_time, roll_data = self.roll_node.get_roll_data()

        if roll_time:
            # Decimate data for smoother rendering
            decimation = max(1, len(roll_time) // 500)
            roll_time_dec = roll_time[::decimation]
            roll_data_dec = roll_data[::decimation]

            # Update Roll Plot
            self.line_roll.set_data(roll_time_dec, roll_data_dec)
            if roll_time_dec:
                self.ax_roll.set_xlim(roll_time_dec[0], roll_time_dec[-1] + 0.5)

        return self.line_roll, self.scatter_pts

if __name__ == "__main__":
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
        output_file = "ft300_tx_data.json"
        with open(output_file, "w") as f:
            json.dump(collected_data, f, indent=4)
        print(f"\nSaved {len(collected_data)} points to {output_file}")

    roll_node.destroy_node()
    rclpy.shutdown()