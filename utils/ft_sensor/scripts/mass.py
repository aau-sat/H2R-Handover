import threading
import json
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from scipy.spatial.transform import Rotation as R
import math
from std_msgs.msg import Float32, Bool

# Import your custom message type
from force_estimation.msg import SensorData

# Configuration
TORQUE_CALIB_FILE = "calibration/tx.json"
FORCE_CALIB_FILE = "calibration/fz.json"
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# Physical Constants
G = 9.81
GRIPPER_MASS = 1.387
D_GRIPPER = 0.080
D_LOAD = 0.278

# Global drift offsets (set by zero calibration)
DRIFT_FZ = 0.0
DRIFT_TX = 0.0

record = False
history = []

class CombinedMassEstimator(Node):
    def __init__(self):
        super().__init__('combined_mass_estimator')

        # 1. Load Calibration Data
        self.torque_calib_data = self.load_calibration(TORQUE_CALIB_FILE, "tx_raw")
        self.force_calib_data = self.load_calibration(FORCE_CALIB_FILE, "fz_raw")

        # 2. State Setup
        self.joint_positions = {}
        self.current_roll_deg = 90.0
        self.latest_mass_torque_kg = 0.0
        self.calibration_done = False
        self.raw_tx = 0.0
        self.raw_fz = 0.0
        self.weight_estimation_active = False

        # 3. ROS Communications
        self.create_subscription(SensorData, '/sensor_data', self.sensor_data_cb, 10)
        self.create_subscription(Bool, '/weight_estimation_status', self.weight_estimation_status_cb, 10)
        # self.mass_pub = self.create_publisher(Float32, '/estimated_mass', 10)

        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')

        # Wait for the service to be available before proceeding
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_fk service...')

        self.get_logger().info(
            "Mass Estimator Node Started. Listening to /sensor_data and /weight_estimation_status. "
            "Running zero calibration automatically on startup..."
        )

    def joint_state_cb(self, msg):
        # Map incoming joint names to positions
        for name, pos in zip(msg.name, msg.position):
            if name in JOINTS:
                self.joint_positions[name] = pos

        # Only update orientation if we have data for all joints
        if all(n in self.joint_positions for n in JOINTS):
            self.update_orientation()

    def update_orientation(self):
        if not self.fk_client.service_is_ready(): return
        
        req = GetPositionFK.Request()
        req.header.frame_id = 'base' # Calculate relative to the robot base
        req.fk_link_names = ['tool0'] # The link the sensor is attached to
        
        # Fill the robot state with current joint positions
        req.robot_state = RobotState()
        req.robot_state.joint_state.name = JOINTS
        req.robot_state.joint_state.position = [self.joint_positions[n] for n in JOINTS]

        # Call the service asynchronously so the node doesn't freeze
        future = self.fk_client.call_async(req)
        future.add_done_callback(self.fk_response_cb)

    def fk_response_cb(self, future):
        try:
            resp = future.result()
            if resp.error_code.val == 1: # 1 means Success in MoveIt
                pose = resp.pose_stamped[0].pose
                
                # Convert Quaternions to a Scipy Rotation object
                rot = R.from_quat([
                    pose.orientation.x, 
                    pose.orientation.y,
                    pose.orientation.z, 
                    pose.orientation.w
                ])
                
                # --- TILT CALCULATION ---
                # Apply rotation to a unit Z vector to see how much it points "down"
                local_z_in_base = rot.apply([0, 0, 1])
                
                # --- ROLL CALCULATION ---
                # Convert to Euler angles (X-Y-Z sequence)
                # Euler[0] corresponds to 'x' rotation, which is typically 'Roll'
                euler = rot.as_euler('xyz', degrees=True)
                self.current_roll_deg = euler[0]
                
        except Exception as e:
            self.get_logger().error(f"FK Error: {e}")

    def run_startup_calibration(self):
        global DRIFT_TX
        self.get_logger().info('Running startup zero calibration for Tx...')

        # Capture current snapshots (actual raw sensor readings)
        actual_raw_tx = getattr(self, 'raw_tx', 0.0)
        self.get_logger().debug(f"Actual Raw Readings -> raw_tx: {actual_raw_tx}")

        # 1. PHYSICS PREDICTION (Expected Units)
        # Calculate what torque should be due to gripper alone at current orientation
        roll_rad = math.radians(self.current_roll_deg)
        self.current_tilt_cos = math.cos(roll_rad)
        expected_tx_from_gravity = (GRIPPER_MASS * G * D_GRIPPER) * math.sin(roll_rad)

        self.get_logger().info(f"Physics Predictions (Gripper Only) -> Expected Tx: {expected_tx_from_gravity:.4f} Nm "
                            f"(Tilt Cos: {self.current_tilt_cos:.3f}, Roll: {self.current_roll_deg:.2f} deg)")

        # 2. CONVERT EXPECTED VALUES TO RAW SENSOR UNITS
        # Use the calibration table in reverse to find what raw values should correspond to the expected values
        expected_raw_tx = self.process_reading(expected_tx_from_gravity, self.torque_calib_data, "tx_expected", "tx_raw")

        self.get_logger().info(f"Expected Raw Values (from table) -> raw_tx: {expected_raw_tx:.4f}")

        # 3. DRIFT CALCULATION IN RAW SENSOR SPACE
        # Drift = actual_raw - expected_raw (offset to subtract from future readings)
        DRIFT_TX = actual_raw_tx - expected_raw_tx

        self.get_logger().info(f"Calibration Complete - Raw Sensor Offsets -> Tx Drift: {DRIFT_TX:.4f}")

        self.calibration_done = True

    def finish_weight_estimation(self):
        global history
        global record
        self.get_logger().info("STOP Weight Estimation")
        self.weight_estimation_active = False
        record = False

        if len(history) > 0:
            self.get_logger().info(f"ESTIMATED MASS: {np.median(history)}")
            history = []
        else:
            self.get_logger().error(f"NO DATA RECORDED")

    def weight_estimation_status_cb(self, msg: Bool):
        global record

        if msg.data and not self.weight_estimation_active:
            self.weight_estimation_active = True
            record = True
            self.get_logger().info("START Weight Estimation")
        elif (not msg.data) and self.weight_estimation_active:
            self.finish_weight_estimation()
    
    def load_calibration(self, filename, sort_key):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                valid_data = [d for d in data if sort_key in d]
                return sorted(valid_data, key=lambda x: x[sort_key])
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration {filename}: {e}")
            return []

    def get_gripper_weight(self):
        gripper_entry = next((item for item in self.force_calib_data if item.get("label") == "Base_Gripper"), None)
        if gripper_entry:
            return gripper_entry.get("fz_expected", 0.0)
        return 0.0

    def process_reading(self, current_raw, calib_data, raw_key, unit_key, k=3):
        if not calib_data:
            return 0.0

        try:
            raw_vals = np.array([p[raw_key] for p in calib_data])
            unit_vals = np.array([p[unit_key] for p in calib_data])
        except KeyError as e:
            self.get_logger().error(f"Mapping Error: Key {e} not found in calibration file.")
            return 0.0

        if current_raw < raw_vals[0] or current_raw > raw_vals[-1]:
            distances = np.abs(raw_vals - current_raw)
            idx = np.argsort(distances)[:k]
            coeffs = np.polyfit(raw_vals[idx], unit_vals[idx], 1)
            return coeffs[0] * current_raw + coeffs[1]
        else:
            return np.interp(current_raw, raw_vals, unit_vals)

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in JOINTS:
                self.joint_positions[name] = pos

    def sensor_data_cb(self, msg: SensorData):
        global history
        self.raw_tx = msg.tx
        self.raw_fz = msg.fz

        if not self.calibration_done:
            self.run_startup_calibration()

        angle_rad = math.radians(self.current_roll_deg)
        gravity_proj_horiz = math.sin(angle_rad)
        sin_roll_abs = abs(gravity_proj_horiz)

        compensated_raw_tx = self.raw_tx - DRIFT_TX
        current_tx_nm = self.process_reading(compensated_raw_tx, self.torque_calib_data, "tx_raw", "tx_expected")
        net_tx = current_tx_nm

        gripper_torque_contribution = (GRIPPER_MASS * G * D_GRIPPER) * gravity_proj_horiz
        load_torque_only = net_tx - gripper_torque_contribution

        mass_torque = (abs(load_torque_only) / (G * D_LOAD * sin_roll_abs))

        self.latest_mass_torque_kg = mass_torque if mass_torque is not None else 0.0

        # EXPORT MASS HERE FOR REPORTING
        # Publish the estimated mass
        # mass_msg = Float32()
        # mass_msg.data = float(self.latest_mass_torque_kg)
        # self.mass_pub.publish(mass_msg)

        if record:
            history.append(self.latest_mass_torque_kg)

def main():
    rclpy.init()
    node = CombinedMassEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()