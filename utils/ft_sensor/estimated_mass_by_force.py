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

# Configuration
PORT = "/dev/ttyUSB0"
CALIB_FILE = "ft300_fz_data.json"
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

class LocalRegressionWeightMonitor(Node):
    def __init__(self):
        super().__init__('local_regression_weight_monitor')
        
        # 1. Load and Sort Calibration Data
        self.calib_data = []
        self.gripper_weight_n = 0.0 
        
        try:
            with open(CALIB_FILE, "r") as f:
                self.calib_data = json.load(f)
            
            # Sort data by raw value to make interpolation easier
            self.calib_data.sort(key=lambda x: x["fz_raw"])
            
            # Find the gripper weight from the specific label provided
            gripper_entry = next((item for item in self.calib_data if item.get("label") == "Pull_Up_No_Load"), None)
            if gripper_entry:
                self.gripper_weight_n = gripper_entry["fz_n"]
                self.get_logger().info(f"Gripper weight identified: {self.gripper_weight_n} N")
            else:
                self.get_logger().warn("Label 'Pull_Up_No_Load' not found. Gripper subtraction may be inaccurate.")
                
            self.get_logger().info(f"Loaded {len(self.calib_data)} calibration points.")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {e}")
            exit()

        # 2. Sensor & State Setup
        try:
            from ft300python.ft300_stream import FT300Stream
            self.ft_stream = FT300Stream(PORT, timeout=1, zero_reset=False)
        except Exception as e:
            self.get_logger().error(f"Sensor Connection/Import Failed: {e}")
            exit()

        self.joint_positions = {}
        self.current_tilt_cos = 1.0 
        self.running = True

        # 3. ROS Communications
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        
        self.thread = threading.Thread(target=self._run_sensor_loop, daemon=True)
        self.thread.start()

    def process_fz_reading(self, current_raw_fz, k=3):
        """
        Logic: Linear Interpolation for values inside range.
        Linear Regression (extrapolation) for values outside range.
        """
        raw_vals = np.array([p["fz_raw"] for p in self.calib_data])
        n_vals = np.array([p["fz_n"] for p in self.calib_data])

        min_raw = raw_vals[0]
        max_raw = raw_vals[-1]

        # --- CASE 1: OUTSIDE BOUNDS (REGRESSION) ---
        if current_raw_fz < min_raw or current_raw_fz > max_raw:
            # Find k nearest points for the regression
            distances = np.abs(raw_vals - current_raw_fz)
            idx = np.argsort(distances)[:k]
            
            X_local = raw_vals[idx]
            y_local = n_vals[idx]

            coeffs = np.polyfit(X_local, y_local, 1)
            return coeffs[0] * current_raw_fz + coeffs[1]

        # --- CASE 2: INSIDE BOUNDS (INTERPOLATION) ---
        else:
            # Use numpy's built-in 1D interpolation
            # Since we sorted self.calib_data, raw_vals is monotonically increasing
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
        except Exception as e:
            self.get_logger().error(f"FK Error: {e}")

    def _run_sensor_loop(self):
        print("\nMonitoring Applied Load (Adaptive Interpolation/Regression)...")
        while self.running:
            readings = self.ft_stream.get_force_torque()
            if readings:
                raw_fz = readings[2]
                
                # 1. Get the calibrated total force (Hybrid approach)
                total_calibrated_fz = self.process_fz_reading(raw_fz)
                
                # 2. Subtract gravity component of the gripper
                expected_gripper_fz = self.gripper_weight_n * self.current_tilt_cos
                applied_load_fz = total_calibrated_fz - expected_gripper_fz
                
                # 3. Scale back to true vertical mass if sensor is tilted
                if self.current_tilt_cos > 0.1:
                    true_added_mass_n = applied_load_fz / self.current_tilt_cos
                else:
                    true_added_mass_n = 0.0

                print(f"Raw Fz: {raw_fz:8.2f} | Net Load: {true_added_mass_n:8.4f} N | Tilt: {self.current_tilt_cos:.2f}", end='\r')

            time.sleep(0.01)

def main():
    rclpy.init()
    node = LocalRegressionWeightMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        rclpy.shutdown()

if __name__ == "__main__":
    main()