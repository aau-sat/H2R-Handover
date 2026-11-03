#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, RobotState, PositionConstraint, OrientationConstraint, BoundingVolume
from moveit_msgs.srv import GetPositionFK

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped


from scipy.spatial.transform import Rotation as R
import time

JOINTS = [
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "shoulder_pan_joint",
]

class Client(Node):
    def __init__(self):
        super().__init__('client')

        self.joint_positions = {}

        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')

        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for FK service...")

    def joint_state_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos
            self.get_current_pose()

    def get_current_pose(self):
        for name in JOINTS:
            if name not in self.joint_positions:
                self.get_logger().warn(f"Waiting for joint states: {name}")
                rclpy.spin_once(self, timeout_sec=0.1)
                return None



        req = GetPositionFK.Request()
        req.header.frame_id = 'base'
        req.fk_link_names = ['tool0']
        req.robot_state = RobotState()
        req.robot_state.joint_state.name = JOINTS
        req.robot_state.joint_state.position = [self.joint_positions[n] for n in JOINTS]

        future = self.fk_client.call_async(req)
        future.add_done_callback(self.fk_response)



        
    # def fk_response(self, future):
    #     try:
    #         resp = future.result()
    #         if resp.error_code.val != 1:
    #             self.get_logger().error("FK service failed")
    #         else:
    #             #print(f"x: {resp.pose_stamped[0].pose.position.x:0.2f}, y: {resp.pose_stamped[0].pose.position.y:0.2f}, z: {resp.pose_stamped[0].pose.position.z:0.2f}")
    #             print(f"{{position: {{x: {resp.pose_stamped[0].pose.position.x:0.2f}, y: {resp.pose_stamped[0].pose.position.y:0.2f}, z: {resp.pose_stamped[0].pose.position.z:0.2f}}}, orientation: {{x: {resp.pose_stamped[0].pose.orientation.x:0.2f}, y: {resp.pose_stamped[0].pose.orientation.y:0.2f}, z: {resp.pose_stamped[0].pose.orientation.z:0.2f}, w: {resp.pose_stamped[0].pose.orientation.w:0.2f}}}}}")
    #     except Exception as e:
    #         self.get_logger().error(f"Service call failed: {e}")

    def fk_response(self, future):
            """
            Handles the response from the Forward Kinematics (FK) service and applies a TCP offset.

            Args:
                future (rclpy.task.Future): The future object containing the FK service response.

            Returns:
                None
            """
            try:
                resp = future.result()
                if resp.error_code.val != 1:
                    self.get_logger().error("FK service failed")
                    return

                pose = resp.pose_stamped[0].pose

                # Extract orientation as a rotation object
                r = R.from_quat([
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w
                ])

                # Local tool offset: 4.5 cm in Z (0.045 meters)
                local_offset = [0.0, 0.0, 0.045]

                # Rotate the local offset into the global frame
                global_offset = r.apply(local_offset)

                # Add the global offset to the current position
                tcp_x = pose.position.x + global_offset[0]
                tcp_y = pose.position.y + global_offset[1]
                tcp_z = pose.position.z + global_offset[2]

                print(f"{{position: {{x: {tcp_x:0.2f}, y: {tcp_y:0.2f}, z: {tcp_z:0.2f}}}, "
                    f"orientation: {{x: {pose.orientation.x:0.2f}, y: {pose.orientation.y:0.2f}, "
                    f"z: {pose.orientation.z:0.2f}, w: {pose.orientation.w:0.2f}}}}}")

            except Exception as e:
                self.get_logger().error(f"Service call failed: {e}")

def main():
    rclpy.init()
    node = Client()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

