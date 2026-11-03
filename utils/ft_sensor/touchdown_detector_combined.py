#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import time
import json
import os
import statistics
import threading
import numpy as np
from ft300python.ft300_stream import FT300Stream
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from touchdown_detection.action import DetectTouchdown
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import JointState

# python combined.py --ros-args -p config_id:='7'

# Physical Constants for Mass Estimation
G = 9.81
GRIPPER_MASS = 1.387
D_GRIPPER = 0.080
D_LOAD = 0.278
TORQUE_CALIB_FILE = "scripts/calibration/tx.json"
FORCE_CALIB_FILE = "scripts/calibration/fz.json"
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

class IntegratedTouchdownNode(Node):
    def __init__(self):
        super().__init__('integrated_touchdown_node')

        # --- Parameters & Setup ---
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('force_threshold', 10.0) # 10.8
        self.declare_parameter('config_id', 0)
        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.threshold = self.get_parameter('force_threshold').get_parameter_value().double_value
        config_id_value = self.get_parameter('config_id').get_parameter_value().integer_value
        self.config_id = str(config_id_value)
        self.submission_json_file = f"scripts/output/mass_estimate_{self.config_id}.json"
        
        self.mass_est_start_time = None
        self.mass_est_end_time = None
        
        self.data_lock = threading.Lock()
        self.current_sensor_values = [0.0] * 6 
        self.current_magnitude = 0.0
        self.touch_down_detected = False
        self.cal_limit = 100
        self.window_size = 5
        
        # --- Mass Estimation State ---
        self.torque_calib_data = self.load_calibration(TORQUE_CALIB_FILE, "tx_raw")
        self.joint_positions = {n: 0.0 for n in JOINTS}
        self.current_roll_deg = 90.0
        self.drift_tx = 0.0
        self.calibration_done = False
        self.weight_estimation_active = False
        self.weight_estimation_logged = False
        self.mass_history = []
        
        self.min_weight_g = 2.0
        self.max_weight_g = 500.0
        self.finger_bias_g = 40.0;

        # --- ROS Communications ---
        self.callback_group = ReentrantCallbackGroup()
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(Bool, '/weight_estimation_status', self.weight_estimation_status_cb, 10)
        self.mass_pub = self.create_publisher(Float32, '/estimated_mass', 10)
        
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        
        self.action_server = ActionServer(
            self, DetectTouchdown, 'detect_touchdown',
            execute_callback=self.execute_callback,
            callback_group=self.callback_group
        )

        # --- Sensor Hardware ---
        try:
            self.ft_stream = FT300Stream(self.port, timeout=1, zero_reset=True)
            self.get_logger().info(f"FT300 connected on {self.port}.")
        except Exception as error:
            self.get_logger().error(f"Failed to connect FT300: {error}")

        self.read_thread = threading.Thread(target=self.sensor_loop, daemon=True)
        self.read_thread.start()

    # --- Mass Estimation Logic ---
    def load_calibration(self, filename, sort_key):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                return sorted([d for d in data if sort_key in d], key=lambda x: x[sort_key])
        except Exception:
            return []

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in JOINTS:
                self.joint_positions[name] = pos
        # Call this without waiting/spinning
        self.update_orientation()

    def update_orientation(self):
        # 1. Check readiness
        if not self.fk_client.service_is_ready():
            return

        # 2. Build Request
        req = GetPositionFK.Request()
        req.header.frame_id = 'base'
        req.fk_link_names = ['tool0']
        req.robot_state = RobotState()
        req.robot_state.joint_state.name = JOINTS
        req.robot_state.joint_state.position = [self.joint_positions[n] for n in JOINTS]
        
        # 3. Call Async and ATTACH a callback
        # This returns immediately; no spinning required!
        future = self.fk_client.call_async(req)
        future.add_done_callback(self.fk_response_cb)

    def fk_response_cb(self, future):
        try:
            resp = future.result()
            if resp is not None and resp.error_code.val == 1:
                pose = resp.pose_stamped[0].pose
                rot = R.from_quat([
                    pose.orientation.x, 
                    pose.orientation.y, 
                    pose.orientation.z, 
                    pose.orientation.w
                ])
                self.current_roll_deg = rot.as_euler('xyz', degrees=True)[0]
                # Now this log will actually appear!
                # self.get_logger().info(f"Updated Roll: {self.current_roll_deg:.2f} degrees")
        except Exception as e:
            self.get_logger().error(f"FK Error: {e}")

    def run_mass_zero_calibration(self, raw_tx):
        roll_rad = math.radians(self.current_roll_deg)
        expected_tx_gravity = (GRIPPER_MASS * G * D_GRIPPER) * math.sin(roll_rad)
        
        # Reverse map physics to raw units
        raw_vals = np.array([p["tx_raw"] for p in self.torque_calib_data])
        unit_vals = np.array([p["tx_expected"] for p in self.torque_calib_data])
        expected_raw_tx = np.interp(expected_tx_gravity, unit_vals, raw_vals)
        
        self.drift_tx = raw_tx - expected_raw_tx
        self.calibration_done = True
        self.get_logger().info(f"Mass Calibration Complete. Drift: {self.drift_tx:.4f}")

    def process_reading(self, current_raw, calib_data):
        raw_vals = np.array([p["tx_raw"] for p in calib_data])
        unit_vals = np.array([p["tx_expected"] for p in calib_data])
        return np.interp(current_raw, raw_vals, unit_vals)

    def append_submission_json(self, payload):
        output_path = self.submission_json_file
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        records = []
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as infile:
                    existing = json.load(infile)
                    if isinstance(existing, list):
                        records = existing
                    elif isinstance(existing, dict):
                        records = [existing]
            except Exception as exc:
                self.get_logger().warn(f"Could not parse existing JSON file '{output_path}': {exc}. Overwriting with a fresh array.")

        records.append(payload)
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(records, outfile, indent=2)

        self.get_logger().info(f"Submission payload saved to {output_path}")

    # --- Unified Sensor Loop ---
    def sensor_loop(self):
        while rclpy.ok():
            if self.ft_stream is None:
                time.sleep(0.5)
                continue

            raw_vals = self.ft_stream.get_force_torque()
            if not raw_vals: continue

            with self.data_lock:
                self.current_sensor_values = list(raw_vals)
                # For Touchdown (Magnitude)
                fx, fy, fz = raw_vals[0], raw_vals[1], raw_vals[2]
                self.current_magnitude = math.sqrt(fx**2 + fy**2 + fz**2)

                # For Mass Estimation
                if not self.calibration_done:
                    self.run_mass_zero_calibration(raw_vals[3]) # raw_vals[3] is Tx
                
                if self.weight_estimation_active:
                    angle_rad = math.radians(self.current_roll_deg)
                    sin_roll = math.sin(angle_rad)
                    
                    if abs(sin_roll) > 0.05: # Avoid division by zero at vertical
                        comp_tx = raw_vals[3] - self.drift_tx
                        current_tx_nm = self.process_reading(comp_tx, self.torque_calib_data)
                        
                        gripper_contrib = (GRIPPER_MASS * G * D_GRIPPER) * sin_roll
                        load_torque = current_tx_nm - gripper_contrib
                        mass_kg = abs(load_torque) / (G * D_LOAD * abs(sin_roll))
                        
                        self.mass_history.append(mass_kg)
            
            time.sleep(0.002)

    # --- Callbacks & Actions ---
    def weight_estimation_status_cb(self, msg: Bool):
        if msg.data and not self.touch_down_detected and not self.weight_estimation_active:
            self.mass_history = []
            self.weight_estimation_active = True
            self.get_logger().info("Mass Recording Started")
            self.weight_estimation_logged = False
            self.mass_est_start_time = int(time.time_ns() // 1_000_000)

        if not msg.data or self.touch_down_detected:
            self.mass_est_end_time = int(time.time_ns() // 1_000_000)
            self.weight_estimation_active = False
            if self.mass_history:
                final_mass = statistics.median(self.mass_history)
                self.get_logger().info(f"FINAL ESTIMATED MASS: {final_mass:.4f} kg")
                # self.mass_pub.publish(Float32(data=float(final_mass)))
                final_mass_g = float(final_mass) * 1000.0
                final_mass_g -= self.finger_bias_g
                final_mass_g = max(final_mass_g, self.min_weight_g)
                final_mass_g = min(final_mass_g, self.max_weight_g)

                if not self.weight_estimation_logged:
                    delivery_location_data = {
                        "robot_mass_est_available": 1,
                        "robot_mass_est_g": final_mass_g,
                        "robot_mass_est_timepoint_t0": self.mass_est_start_time,
                        "robot_mass_est_timepoint_t1": self.mass_est_end_time,
                    }
                    self.get_logger().get_child("SUBMISSION").info(
                        f"{json.dumps(delivery_location_data)}"
                    )
                    self.append_submission_json(delivery_location_data)
                    self.weight_estimation_logged = True

    def execute_callback(self, goal_handle):
        self.get_logger().info("Touchdown detection active...")
        force_buffer = []
        result = DetectTouchdown.Result()

        try:
            bias_fx, bias_fy, bias_fz = 0.0, 0.0, 0.0
            for _ in range(self.cal_limit):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.status = False
                    self.get_logger().info("Goal canceled during calibration phase.")
                    return result

                ft_values = self.current_sensor_values
                bias_fx += ft_values[0]
                bias_fy += ft_values[1]
                bias_fz += ft_values[2]
                # print(f"Calibration Reading: fx={ft_values[0]:.2f}, fy={ft_values[1]:.2f}, fz={ft_values[2]:.2f}")
                time.sleep(0.01)

            bias_fx /= self.cal_limit
            bias_fy /= self.cal_limit
            bias_fz /= self.cal_limit
            
            self.get_logger().info(f"Bias: {bias_fx}, {bias_fy}, {bias_fz}")

            self.get_logger().info(
                f"Bias calculated. Monitoring for > {self.threshold}N (Median of {self.window_size})."
            )

            force_buffer = []
            median_force = 0.0

            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.status = False
                    self.get_logger().info("Goal canceled during detection phase.")
                    return result

                ft_values = self.current_sensor_values
                fx = ft_values[0] - bias_fx
                fy = ft_values[1] - bias_fy
                fz = ft_values[2] - bias_fz

                current_magnitude = math.sqrt(fx**2 + fy**2 + fz**2)
                force_buffer.append(current_magnitude)

                if len(force_buffer) > self.window_size:
                    force_buffer.pop(0)

                if len(force_buffer) == self.window_size:
                    median_force = statistics.median(force_buffer)
                    if median_force > self.threshold:
                        self.get_logger().warn(
                            f"TOUCHDOWN DETECTED: Median {median_force:.2f}N"
                        )
                        goal_handle.succeed()
                        result.status = True
                        
                        self.weight_estimation_status_cb(Bool(data=False))
                        
                        return result

                time.sleep(0.005)

            goal_handle.abort()
            result.status = False
            return result

        except Exception as error:
            self.get_logger().error(f"Error during touchdown detection: {error}")
            goal_handle.abort()
            result.status = False
            return result
        
        

        # while rclpy.ok():
        #     if goal_handle.is_cancel_requested:
        #         goal_handle.canceled()
        #         return result

        #     with self.data_lock:
        #         mag = self.current_magnitude

        #     force_buffer.append(mag)
        #     if len(force_buffer) > 5: force_buffer.pop(0)

        #     if len(force_buffer) == 5 and statistics.median(force_buffer) > self.threshold:
        #         self.get_logger().warn(f"TOUCHDOWN Median Force: {statistics.median(force_buffer):.2f}N")
        #         self.touch_down_detected = True
        #         goal_handle.succeed()
        #         result.status = True
                
        #         self.weight_estimation_status_cb(Bool(data=False))
                
        #         return result
            
        #     time.sleep(0.01)
        
        # goal_handle.abort()
        # return result

def main():
    rclpy.init()
    node = IntegratedTouchdownNode()
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