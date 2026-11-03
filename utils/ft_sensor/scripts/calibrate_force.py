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

# --- Global Store ---
latest_roll_value = 0.0
data_lock = threading.Lock()
MAX_SAMPLES = 3000
z_history = deque(maxlen=MAX_SAMPLES)
running = True

# --- Configuration ---
G = 9.81
MOUNT_MASS = 0.100  # 100g
GRIPPER_MASS = 1.387  # Combined mass of gripper components

JOINTS = [
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "shoulder_pan_joint",
]

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
        global z_history
        with data_lock:
            # Append the Fz value to history
            z_history.append(msg.fz)

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

# --- ROS Spinner Worker ---
def ros_worker():
    rclpy.init()
    node = UnifiedSensorNode()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"ROS Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

def calc_expected_force(added_mass):
    return (GRIPPER_MASS + added_mass) * G

# --- Build Calibration Steps ---
steps = [
    ("Base_Gripper", 0.0, "Ensure gripper is vertical (down) - NO load"),
    ("Mount_Only", MOUNT_MASS, f"Attach {MOUNT_MASS*1000:.0f}g mount")
]
# Create granular weight increments
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

        # Scatter Plot
        self.ax_scatter = plt.subplot(1, 1, 1)
        self.scatter_pts = self.ax_scatter.scatter([], [], color='blue', label='Samples')
        self.ax_scatter.set_title("Calibration Curve (Raw Fz vs Expected Force)")
        self.ax_scatter.set_ylabel("Raw Fz (Internal Units)")
        self.ax_scatter.set_xlabel("Expected Force (N)")
        self.ax_scatter.grid(True, alpha=0.3)

        plt.subplots_adjust(bottom=0.2)

        # Labels
        self.txt_instruction = self.fig.text(0.5, 0.12, "", ha='center', fontsize=12, fontweight='bold', color='blue')
        self.txt_status = self.fig.text(0.5, 0.08, "Ready to start", ha='center', fontsize=10)
        
        # Live Roll Label
        self.txt_roll = self.fig.text(0.98, 0.95, "Roll: 0.0°", ha='right', va='top', 
                                      fontsize=14, fontweight='bold', color='black',
                                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

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
        output_file = "calibration/fz.json"
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

            # 1. Immediate UI Feedback & Lock Button
            self.txt_status.set_text("Collecting 100 NEW samples... Hold steady!")
            self.btn_sample.set_active(False) 
            self.fig.canvas.draw_idle()

            # 2. Clear history to ensure only post-click data is used
            with data_lock:
                z_history.clear() 

            # 3. Wait until the sensor_worker has pushed 100 new samples
            # Sensor worker runs at ~200Hz, so this should take ~0.5s
            start_wait = time.time()
            timeout_duration = 2.0 # 2-second safety buffer
            
            while len(z_history) < 100:
                plt.pause(0.01) # Keeps the GUI alive while we wait
                if time.time() - start_wait > timeout_duration:
                    self.txt_status.set_text("Error: Sensor timeout!")
                    self.btn_sample.set_active(True)
                    return

            # 4. Process the fresh samples
            with data_lock:
                # We take exactly the first 100 collected after the click
                samples = list(z_history)[:100]
                avg_fz = sum(samples) / len(samples)

            # 5. Save Data and Advance Step
            label, extra_m, _ = steps[current_step_idx]
            expected_n = calc_expected_force(extra_m)

            collected_data.append({
                "label": label, 
                "fz_raw": avg_fz, 
                "fz_expected": expected_n,
                "roll_deg": latest_roll_value,
                "timestamp": time.time()
            })
            
            current_step_idx += 1
            self.update_instruction()
            self.update_scatter()
            
            # 6. Re-enable UI
            self.btn_sample.set_active(True)
            self.txt_status.set_text(f"Recorded {label}")
            self.fig.canvas.draw_idle()

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
            # Display Expected N on X-axis and Raw Fz on Y-axis
            x = [p["fz_expected"] for p in collected_data]
            y = [p["fz_raw"] for p in collected_data]
            self.scatter_pts.set_offsets(np.c_[x, y])
            self.ax_scatter.set_xlim(min(x) - 1, max(x) + 1)
            self.ax_scatter.set_ylim(min(y) - 5, max(y) + 5)
        self.fig.canvas.draw_idle()

    def animate(self, i):
        self.txt_roll.set_text(f"Roll: {latest_roll_value:.2f}°")
        return self.scatter_pts, self.txt_roll

# --- Main Execution ---
def main():
    global running
    
    # Init ROS 2
    rclpy.init()
    roll_node = UnifiedSensorNode()
    
    # Spin ROS 2 in a background thread
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(roll_node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()


    print("Initializing Sensor and GUI...")
    time.sleep(1)

    # Setup GUI
    gui = CalibrationGUI(roll_node)
    # cache_frame_data=False avoids the warning seen in your logs
    ani = FuncAnimation(gui.fig, gui.animate, interval=100, blit=False, cache_frame_data=False)

    plt.show() # Execution blocks here until window is closed

    # Cleanup
    running = False


    roll_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()