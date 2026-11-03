#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import json
import numpy as np
import math
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from scipy.spatial.transform import Rotation as R

# FT300 specific import
from ft300python.ft300_stream import FT300Stream

# --- Physical Constants ---
G = 9.81  
MOUNT_MASS = 0.100  
GRIPPER_MASS = 1.25  
D_GRIPPER = 0.080  
D_LOAD = 0.278      
CALIBRATION_FILE = "ft300_tx_data.json"

# Updated Joint List order from your working example
JOINTS = [
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "shoulder_pan_joint",
]

class MassEstimatorNode(Node):
    def __init__(self):
        super().__init__('mass_estimator_node')
        
        self.callback_group = ReentrantCallbackGroup()

        # 1. Load Calibration
        self.calibration_data = self.load_calibration(CALIBRATION_FILE)
        
        # 2. Initialize FT300 Sensor
        self.get_logger().info("Connecting to FT300 sensor...")
        try:
            self.sensor = FT300Stream("/dev/ttyUSB0", timeout=1, zero_reset=False)
            self.samples = []
            self.batch_size = 10 
        except Exception as e:
            self.get_logger().error(f"Sensor Connection Failed: {e}")
            raise e

        # 3. ROS2 Setup
        self.joint_positions = {}
        self.current_roll_deg = 0.0 
        
        self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_cb, 
            10,
            callback_group=self.callback_group)
            
        self.fk_client = self.create_client(
            GetPositionFK, 
            '/compute_fk',
            callback_group=self.callback_group)

        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('FK service not available, waiting...')
        
        self.create_timer(0.02, self.process_sensor_data, callback_group=self.callback_group)

    def load_calibration(self, filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                return sorted(data, key=lambda x: x['tx_raw'])
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {e}")
            return []

    def joint_state_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in JOINTS:
                self.joint_positions[name] = pos
        
        if all(n in self.joint_positions for n in JOINTS):
            self.update_orientation_from_fk()

    def update_orientation_from_fk(self):
        req = GetPositionFK.Request()
        # Updated to 'base' from your working example
        req.header.frame_id = 'base' 
        req.fk_link_names = ['tool0']
        req.robot_state = RobotState()
        req.robot_state.joint_state.name = JOINTS
        req.robot_state.joint_state.position = [self.joint_positions[n] for n in JOINTS]

        future = self.fk_client.call_async(req)
        future.add_done_callback(self.fk_response_cb)

    def fk_response_cb(self, future):
        try:
            resp = future.result()
            if resp.error_code.val == 1:
                pose = resp.pose_stamped[0].pose
                r = R.from_quat([pose.orientation.x, pose.orientation.y, 
                                 pose.orientation.z, pose.orientation.w])
                
                # Extract Euler angles - Using ROLL (index 0) as per your second example
                euler = r.as_euler('xyz', degrees=True)
                self.current_roll_deg = euler[0] 
        except Exception as e:
            self.get_logger().error(f"FK update failed: {e}")

    def get_calibrated_torque(self, current_raw):
        if not self.calibration_data:
            return 0.0
            
        x_coords = np.array([p['tx_raw'] for p in self.calibration_data])
        y_coords = np.array([p['tx_nm'] for p in self.calibration_data])

        if current_raw < x_coords[0] or current_raw > x_coords[-1]:
            dist_sorted = sorted(self.calibration_data, key=lambda x: abs(x['tx_raw'] - current_raw))
            closest_4 = dist_sorted[:4]
            x_reg = np.array([p['tx_raw'] for p in closest_4])
            y_reg = np.array([p['tx_nm'] for p in closest_4])
            slope, intercept = np.polyfit(x_reg, y_reg, 1)
            return slope * current_raw + intercept
        return np.interp(current_raw, x_coords, y_coords)

    def estimate_mass(self, torque_nm, angle_deg):
        # Using the absolute sine of the roll angle for gravity projection
        angle_rad = math.radians(angle_deg)
        gravity_projection = abs(math.sin(angle_rad))
        
        if gravity_projection < 0.05: 
            return 0.0

        m_grip_moment = GRIPPER_MASS * D_GRIPPER
        total_moment_req = torque_nm / (G * gravity_projection)
        
        load_mass = (total_moment_req - m_grip_moment) / D_LOAD
        return load_mass - MOUNT_MASS

    def process_sensor_data(self):
        try:
            readings = self.sensor.get_force_torque()
            if readings:
                self.samples.append(readings[3]) # Tx axis
            
            if len(self.samples) >= self.batch_size:
                avg_raw = sum(self.samples) / len(self.samples)
                torque_nm = self.get_calibrated_torque(avg_raw)
                mass_kg = self.estimate_mass(torque_nm, self.current_roll_deg)
                
                print(f"\rRoll: {self.current_roll_deg:6.1f}° | Raw: {avg_raw:8.1f} | Mass: {mass_kg*1000:8.2f}g", end="", flush=True)
                self.samples = []
        except Exception as e:
            self.get_logger().warn(f"Data processing error: {e}")

def main():
    rclpy.init()
    node = MassEstimatorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()