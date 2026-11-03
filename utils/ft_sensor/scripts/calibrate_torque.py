#!/usr/bin/env python3
import time
import threading
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import deque
from matplotlib.animation import FuncAnimation

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from scipy.spatial.transform import Rotation as R

# Import your custom message
from ft_sensor_msgs.msg import SensorData



JOINTS = [
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "shoulder_pan_joint",
]


TIMEOUT = 1
G = 9.81
MOUNT_MASS = 0.100
GRIPPER_MASS = 1.387  # Total gripper assembly
D_GRIPPER = 0.080          # CoM of gripper (m)
D_LOAD = 0.278             # CoM of added weights (m)

def calc_expected_torque(load_mass):
    # Torque = (m_gripper * g * d_gripper) + (m_load * g * d_load)
    return (GRIPPER_MASS * G * D_GRIPPER) + (load_mass * G * D_LOAD)

# --- Shared resources & State ---
data_lock = threading.Lock()
running = True

# Buffers (Switching focus to tx - Index 4 in FT300Stream)
MAX_SAMPLES = 3000
tx_history = deque(maxlen=MAX_SAMPLES)

# Calibration Data State
collected_data = []
current_step_idx = 0



class UnifiedSensorNode(Node):
    def __init__(self):
        super().__init__('unified_sensor_node')
        self.joint_positions = {}
        
        # FK Client
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for FK service...")

        # Subscriptions
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(SensorData, '/sensor_data', self.sensor_data_cb, 10)

    def sensor_data_cb(self, msg: SensorData):
        """Callback for the FT sensor data via ROS topic"""
        global tx_history
        with data_lock:
            # Append the Tx value to history
            tx_history.append(msg.tx)

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
        global latest_roll_value
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
            latest_roll_value = euler[0]
        except Exception as e:
            self.get_logger().error(f"FK Service call failed: {e}")



# --- Steps Generation ---
steps = [
    ("Base_Gripper", 0.0, "Ensure gripper horizontal - NO load"),
    ("Mount_Only", MOUNT_MASS, f"Attach {MOUNT_MASS*1000:.0f}g mount")
]

weight_points = list(range(10, 460, 10)) + [500] + list(range(510, 650, 10))
for w in weight_points:
    steps.append((f"Mount+{w}g", MOUNT_MASS + (w / 1000.0), f"Add {w}g weight"))

# --- GUI / Plotting Class ---
class CalibrationGUI:
    def __init__(self, roll_node):
        self.roll_node = roll_node
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("FT300 Torque Calibration Visualizer")

        self.ax_scatter = plt.subplot(1, 1, 1)
        self.scatter_pts = self.ax_scatter.scatter([], [], color='purple', label='Samples')
        self.ax_scatter.set_title("Calibration Curve (Raw tx vs Expected Torque)")
        self.ax_scatter.set_ylabel("Raw tx (Internal Units)")
        self.ax_scatter.set_xlabel("Expected Torque (Nm)")
        self.ax_scatter.grid(True, alpha=0.3)

        plt.subplots_adjust(bottom=0.2)

        self.txt_instruction = self.fig.text(0.5, 0.12, "", ha='center', fontsize=12, fontweight='bold', color='blue')
        self.txt_status = self.fig.text(0.5, 0.08, "Ready to start", ha='center', fontsize=10)
        self.txt_roll = self.fig.text(0.98, 0.95, "Roll: 0.0°", ha='right', va='top', fontsize=14, fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Buttons
        self.ax_btn_sample = plt.axes([0.25, 0.02, 0.14, 0.05])
        self.btn_sample = Button(self.ax_btn_sample, 'Take Sample', color='limegreen', hovercolor='palegreen')
        self.btn_sample.on_clicked(self.on_sample)

        self.ax_btn_undo = plt.axes([0.42, 0.02, 0.14, 0.05])
        self.btn_undo = Button(self.ax_btn_undo, 'Delete/Undo', color='tomato', hovercolor='salmon')
        self.btn_undo.on_clicked(self.on_undo)

        self.ax_btn_save = plt.axes([0.59, 0.02, 0.14, 0.05])
        self.btn_save = Button(self.ax_btn_save, 'Save Data', color='gold', hovercolor='khaki')
        self.btn_save.on_clicked(self.on_save)

        self.update_instruction()

    def on_save(self, event):
        if not collected_data:
            self.txt_status.set_text("No data to save!")
            return
        os.makedirs("calibration", exist_ok=True)
        output_file = "calibration/tx.json"
        with open(output_file, "w") as f:
            json.dump(collected_data, f, indent=4)
        self.txt_status.set_text(f"Saved {len(collected_data)} points to {output_file}")

    def update_instruction(self):
        if current_step_idx < len(steps):
            label, _, instr = steps[current_step_idx]
            self.txt_instruction.set_text(f"STEP {current_step_idx+1}/{len(steps)}: {instr}")
        else:
            self.txt_instruction.set_text("COLLECTION COMPLETE! Close window to save.")
            self.btn_sample.set_active(False)

    def on_sample(self, event):
            global current_step_idx
            if current_step_idx >= len(steps): 
                return

            # 1. Provide immediate UI feedback
            self.txt_status.set_text("Collecting 100 NEW samples...")
            self.btn_sample.set_active(False) # Disable to prevent double-clicks
            self.fig.canvas.draw_idle()

            # 2. Clear or mark the start of collection
            with data_lock:
                tx_history.clear() 

            # 3. Wait for exactly 100 samples (Non-blocking enough for 0.5s)
            # At 200Hz (0.005s sleep in worker), 100 samples take ~0.5 seconds
            start_wait = time.time()
            while len(tx_history) < 100:
                plt.pause(0.01) # This keeps the GUI alive/responsive while waiting
                if time.time() - start_wait > 2.0: # Timeout safety
                    self.txt_status.set_text("Error: Sensor timeout!")
                    self.btn_sample.set_active(True)
                    return

            # 4. Process the data now that we know it's fresh
            with data_lock:
                samples = list(tx_history)
                avg_tx = sum(samples) / len(samples)

            # 5. Save and Advance
            label, load_m, _ = steps[current_step_idx]
            expected_torque = calc_expected_torque(load_m)

            collected_data.append({
                "label": label, 
                "tx_raw": avg_tx, 
                "tx_expected": expected_torque,
                "roll_deg": latest_roll_value,
                "timestamp": time.time()
            })
            
            current_step_idx += 1
            self.update_instruction()
            self.update_scatter()
            self.btn_sample.set_active(True) # Re-enable
            self.txt_status.set_text(f"Recorded {label}")

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
            x = [p["tx_expected"] for p in collected_data]
            y = [p["tx_raw"] for p in collected_data]
            self.scatter_pts.set_offsets(np.c_[x, y])
            self.ax_scatter.set_xlim(min(x) - 0.5, max(x) + 0.5)
            self.ax_scatter.set_ylim(min(y) - 2, max(y) + 2)
        self.fig.canvas.draw_idle()

    def animate(self, i):
        self.txt_roll.set_text(f"Roll: {latest_roll_value:.2f}°")
        return self.scatter_pts, self.txt_roll
    

# --- Main Execution ---
def main():
    global running
    
    rclpy.init()
    roll_node = UnifiedSensorNode()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(roll_node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()


    print("Initializing Sensor and GUI for Torque Calibration...")
    time.sleep(1)

    gui = CalibrationGUI(roll_node)
    ani = FuncAnimation(gui.fig, gui.animate, interval=100, blit=False, cache_frame_data=False)

    plt.show()

    running = False


    roll_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()