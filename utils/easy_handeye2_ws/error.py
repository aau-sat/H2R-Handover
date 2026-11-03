#!/usr/bin/env python3
import sys
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK

JOINTS = [
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "shoulder_pan_joint",
]

class ErrorEvaluationClient(Node):
    """ROS 2 Node to evaluate FK positioning error against a goal pose.

    Attributes:
        joint_positions (dict): Latest joint positions.
        latest_goal_pos (np.ndarray): Latest position from /goal_pose.
        latest_tcp_pos (np.ndarray): Latest computed TCP position.
        errors (list): Recorded Euclidean errors.
    """

    def __init__(self):
        """Initializes the node, subscribers, clients, and keyboard listener."""
        super().__init__('error_evaluation_client')

        self.joint_positions = {}
        self.latest_goal_pos = None
        self.latest_tcp_pos = None
        self.errors = []

        # Subscribers
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_cb, 10)

        # Publisher for FK-derived end-effector pose
        self.end_pose_pub = self.create_publisher(PoseStamped, '/end_pose', 10)

        # FK Service Client
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for FK service...")

        # Keyboard listener thread (Ubuntu non-blocking IO)
        self.listener = keyboard.Listener(on_release=self.on_key_release)
        self.listener.start()
        self.get_logger().info("Press 'Enter' to sample error. Press 'Esc' to compute stats and exit.")

    def goal_pose_cb(self, msg: PoseStamped):
        """Callback for the goal pose subscriber.

        Args:
            msg (PoseStamped): The received goal pose.
        """
        self.latest_goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

    def joint_state_cb(self, msg: JointState):
        """Callback for joint states to trigger FK computation.

        Args:
            msg (JointState): The received joint states.
        """
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos
        self.request_fk()

    def request_fk(self):
        """Requests Forward Kinematics from MoveIt."""
        for name in JOINTS:
            if name not in self.joint_positions:
                return

        req = GetPositionFK.Request()
        req.header.frame_id = 'base'
        req.fk_link_names = ['tool0']
        req.robot_state = RobotState()
        req.robot_state.joint_state.name = JOINTS
        req.robot_state.joint_state.position = [self.joint_positions[n] for n in JOINTS]

        future = self.fk_client.call_async(req)
        future.add_done_callback(self.fk_response)

    def fk_response(self, future):
        """Handles the FK service response and applies the TCP offset.

        Args:
            future (rclpy.task.Future): The future object containing the FK response.
        """
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

            # Local tool offset: 4.5 cm in Z
            local_offset = [0.0, 0.0, 0.085]
            global_offset = r.apply(local_offset)

            self.latest_tcp_pos = np.array([
                pose.position.x + global_offset[0],
                pose.position.y + global_offset[1],
                pose.position.z + global_offset[2]
            ])

            end_pose_msg = PoseStamped()
            end_pose_msg.header.stamp = self.get_clock().now().to_msg()
            end_pose_msg.header.frame_id = 'base'
            end_pose_msg.pose.position.x = float(self.latest_tcp_pos[0])
            end_pose_msg.pose.position.y = float(self.latest_tcp_pos[1])
            end_pose_msg.pose.position.z = float(self.latest_tcp_pos[2])
            end_pose_msg.pose.orientation = pose.orientation
            self.end_pose_pub.publish(end_pose_msg)

        except Exception as e:
            self.get_logger().error(f"FK request failed: {e}")

    def on_key_release(self, key):
        """Handles keyboard input events as a background thread.

        Args:
            key (pynput.keyboard.Key): The key released.
        """
        if key == keyboard.Key.enter:
            self.sample_error()
        elif key == keyboard.Key.esc:
            self.compute_statistics()
            self.listener.stop()
            rclpy.shutdown()

    def sample_error(self):
            """Calculates and stores the L1 error (absolute difference) for x, y, z and the total magnitude.

            Returns:
                None
            """
            if self.latest_goal_pos is None or self.latest_tcp_pos is None:
                self.get_logger().warn("Waiting for both /goal_pose and FK position...")
                return

            # Calculate L1 norm (absolute error) for each axis
            diff = np.abs(self.latest_goal_pos - self.latest_tcp_pos) * 100.0
            err_x, err_y, err_z = float(diff[0]), float(diff[1]), float(diff[2])
            
            # Total L1 norm magnitude
            err_total = float(np.sum(diff))
            
            self.errors.append([err_x, err_y, err_z, err_total])
            self.get_logger().info(
                f"Sample [{len(self.errors)}] L1 Errors -> X: {err_x:.4f}, Y: {err_y:.4f}, Z: {err_z:.4f}, Total: {err_total:.4f} cm"
            )

    def compute_statistics(self):
        """Computes and prints mean, variance, min, and max of the L1 errors.

        Returns:
            None
        """
        if not self.errors:
            self.get_logger().info("No samples recorded.")
            return

        # Convert to numpy array for column-wise operations; Shape: (N, 4)
        err_arr = np.array(self.errors)
        
        mean_errs = np.mean(err_arr, axis=0)
        var_errs = np.var(err_arr, axis=0)
        min_errs = np.min(err_arr, axis=0)
        max_errs = np.max(err_arr, axis=0)

        labels = ['X', 'Y', 'Z', 'Total L1']
        
        print("\n--- L1 Error Statistics ---")
        print(f"Samples: {len(err_arr)}\n")
        for i, label in enumerate(labels):
            print(f"--- {label} ---")
            print(f"Mean (MAE): {mean_errs[i]:.6f} cm")
            print(f"Std\t:   {np.sqrt(var_errs[i]):.6f} cm")
            print(f"Min Err:    {min_errs[i]:.6f} cm")
            print(f"Max Err:    {max_errs[i]:.6f} m\n")
        print("---------------------------\n")


def main():
    rclpy.init()
    node = ErrorEvaluationClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()