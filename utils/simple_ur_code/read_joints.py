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

        print([f"{self.joint_positions[n]:+5.2f}" for n in JOINTS])


        return 0

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

