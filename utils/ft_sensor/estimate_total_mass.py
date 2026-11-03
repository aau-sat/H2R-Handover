import threading
import time
import json
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from scipy.spatial.transform import Rotation as R
import math

# Configuration
PORT = "/dev/ttyUSB0"
TORQUE_CALIB_FILE = "ft300_tx_data.json"
FORCE_CALIB_FILE = "ft300_fz_data.json"
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# Physical Constants
G = 9.81
MOUNT_MASS = 0.100  # kg
GRIPPER_MASS = 1.25  # kg
D_GRIPPER = 0.080  # m
D_LOAD = 0.278  # m

class CombinedMassEstimator(Node):
    def __init__(self):
        super().__init__('combined_mass_estimator')

        # 1. Load Calibration Data
        self.torque_calib_data = self.load_calibration(TORQUE_CALIB_FILE)
        self.force_calib_data = self.load_calibration(FORCE_CALIB_FILE)
        self.gripper_weight_n = self.get_gripper_weight()

        # 2. Sensor & State Setup
        try:
            from ft300python.ft300_stream import FT300Stream
            self.ft_stream = FT300Stream(PORT, timeout=1, zero_reset=False)
        except Exception as e:
            self.get_logger().error(f"Sensor Connection/Import Failed: {e}")
            exit()

        self.joint_positions = {}
        self.current_tilt_cos = 1.0
        self.current_roll_deg = 0.0
        self.running = True

        # 3. ROS Communications
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')

        self.thread = threading.Thread(target=self._run_sensor_loop, daemon=True)
        self.thread.start()

    def load_calibration(self, filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                return sorted(data, key=lambda x: x.get("tx_raw", x.get("fz_raw", 0)))
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {e}")
            return []

    def get_gripper_weight(self):
        gripper_entry = next((item for item in self.force_calib_data if item.get("label") == "Pull_Up_No_Load"), None)
        return gripper_entry["fz_n"] if gripper_entry else 0.0

    def process_torque_reading(self, current_raw_tx, k=3):
        raw_vals = np.array([p["tx_raw"] for p in self.torque_calib_data])
        n_vals = np.array([p["tx_nm"] for p in self.torque_calib_data])

        min_raw = raw_vals[0]
        max_raw = raw_vals[-1]

        if current_raw_tx < min_raw or current_raw_tx > max_raw:
            distances = np.abs(raw_vals - current_raw_tx)
            idx = np.argsort(distances)[:k]
            X_local = raw_vals[idx]
            y_local = n_vals[idx]
            coeffs = np.polyfit(X_local, y_local, 1)
            return coeffs[0] * current_raw_tx + coeffs[1]
        else:
            return np.interp(current_raw_tx, raw_vals, n_vals)

    def process_force_reading(self, current_raw_fz, k=3):
        raw_vals = np.array([p["fz_raw"] for p in self.force_calib_data])
        n_vals = np.array([p["fz_n"] for p in self.force_calib_data])

        min_raw = raw_vals[0]
        max_raw = raw_vals[-1]

        if current_raw_fz < min_raw or current_raw_fz > max_raw:
            distances = np.abs(raw_vals - current_raw_fz)
            idx = np.argsort(distances)[:k]
            X_local = raw_vals[idx]
            y_local = n_vals[idx]
            coeffs = np.polyfit(X_local, y_local, 1)
            return coeffs[0] * current_raw_fz + coeffs[1]
        else:
            return np.interp(current_raw_fz, raw_vals, n_vals)

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in JOINTS:
                self.joint_positions[name] = pos

        if all(n in self.joint_positions for n in JOINTS):
            self.update_orientation()

    def update_orientation(self):
        if not self.fk_client.service_is_ready(): return
        req = GetPositionFK.Request()
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
                rot = R.from_quat([pose.orientation.x, pose.orientation.y,
                                   pose.orientation.z, pose.orientation.w])

                # Z-axis alignment with gravity
                local_z_in_base = rot.apply([0, 0, 1])
                self.current_tilt_cos = abs(local_z_in_base[2])

                # Roll angle for torque method
                euler = rot.as_euler('xyz', degrees=True)
                self.current_roll_deg = euler[0]
        except Exception as e:
            self.get_logger().error(f"FK Error: {e}")

    def estimate_mass_from_torque(self, torque_nm):
        angle_rad = math.radians(self.current_roll_deg)
        gravity_projection = abs(math.sin(angle_rad))

        if gravity_projection < 0.05:
            return None

        m_grip_moment = GRIPPER_MASS * D_GRIPPER
        total_moment_req = torque_nm / (G * gravity_projection)
        load_mass = (total_moment_req - m_grip_moment) / D_LOAD
        return load_mass + MOUNT_MASS

    def estimate_mass_from_force(self, force_n):
        expected_gripper_fz = self.gripper_weight_n * self.current_tilt_cos
        applied_load_fz = force_n - expected_gripper_fz

        if self.current_tilt_cos > 0.1:
            true_added_mass_n = applied_load_fz / self.current_tilt_cos
            return (true_added_mass_n / G) + MOUNT_MASS
        else:
            return None

    def _run_sensor_loop(self):
        print("\nCombined Mass Estimation (Torque + Force)...")
        while self.running:
            readings = self.ft_stream.get_force_torque()
            if readings:
                raw_tx, raw_fz = readings[3], readings[2]

                # 1. Torque-based mass (horizontal)
                torque_nm = self.process_torque_reading(raw_tx)
                mass_torque = self.estimate_mass_from_torque(torque_nm)

                # 2. Force-based mass (vertical)
                force_n = self.process_force_reading(raw_fz)
                mass_force = self.estimate_mass_from_force(force_n)

                # 3. Select the best estimate based on orientation
                if abs(self.current_tilt_cos) > 0.7:  # Mostly vertical
                    mass_kg = mass_force if mass_force is not None else 0.0
                    method = "Force"
                elif abs(math.sin(math.radians(self.current_roll_deg))) > 0.7:  # Mostly horizontal
                    mass_kg = mass_torque if mass_torque is not None else 0.0
                    method = "Torque"
                else:  # Transition region, average
                    mass_kg = (mass_torque or 0.0 + mass_force or 0.0) / 2
                    method = "Combined"

                print(f"Raw Tx: {raw_tx:8.2f} | Raw Fz: {raw_fz:8.2f} | Mass: {mass_kg*1000:8.2f}g | Method: {method:8} | Tilt: {self.current_tilt_cos:.2f} | Roll: {self.current_roll_deg:.1f}°", end='\r')

            time.sleep(0.01)

def main():
    rclpy.init()
    node = CombinedMassEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        rclpy.shutdown()

if __name__ == "__main__":
    main()