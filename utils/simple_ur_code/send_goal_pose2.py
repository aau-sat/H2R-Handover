#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, RobotState, PositionConstraint, OrientationConstraint, BoundingVolume
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped


from scipy.spatial.transform import Rotation as R

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


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

    def get_current_pose(self):
        for name in JOINTS:
            if name not in self.joint_positions:
                self.get_logger().warn(f"Waiting for joint states: {name}")
                rclpy.spin_once(self, timeout_sec=0.1)
                return None



        req = GetPositionFK.Request()
        req.header.frame_id = 'world'
        req.fk_link_names = ['tool0']
        req.robot_state = RobotState()
        req.robot_state.joint_state.name = JOINTS
        req.robot_state.joint_state.position = [self.joint_positions[n] for n in JOINTS]

        future = self.fk_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        resp = future.result()
        if resp.error_code.val != 1:
            self.get_logger().error("FK service failed")
        else:
            return resp.pose_stamped[0].pose

    def send_cartesian_pose_goal(self, target_pose: Pose):
        self.move_group_client.wait_for_server()
        goal = MoveGroup.Goal()
        #goal.request.planner_id = "PTP"
        req = goal.request

        req.group_name = "ur_manipulator"
        req.allowed_planning_time = 5.0
        req.num_planning_attempts = 10
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2

        # -------------------------
        # Position constraint
        # -------------------------
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "world"
        pos_constraint.link_name = "tool0"
        pos_constraint.weight = 1.0

        solid = SolidPrimitive()
        solid.type = SolidPrimitive.BOX
        solid.dimensions = [0.05, 0.05, 0.05]  # 1 cm tolerance box

        bounding_volume = BoundingVolume()
        bounding_volume.primitives.append(solid)
        bounding_volume.primitive_poses.append(target_pose)

        pos_constraint.constraint_region = bounding_volume

        # -------------------------
        # Orientation constraint
        # -------------------------
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = "world"
        ori_constraint.link_name = "tool0"
        ori_constraint.orientation = target_pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.01
        ori_constraint.absolute_y_axis_tolerance = 0.01
        ori_constraint.absolute_z_axis_tolerance = 0.01
        ori_constraint.weight = 1.0

        # -------------------------
        # Combine constraints
        # -------------------------
        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)

        req.goal_constraints.append(constraints)

        # -------------------------
        # Send goal
        # -------------------------
        print("sending goal ...")
        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Cartesian goal rejected")
            return

        self.get_logger().info("Cartesian goal accepted")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info("Cartesian motion finished")



def main():
    rclpy.init()
    node = Client()

    rclpy.spin_once(node)

    while node.joint_positions == {}:
        rclpy.spin_once(node)


    pose = node.get_current_pose()

    if pose:
        print(pose.position)

        new_pose = Pose()
        new_pose.position.x = pose.position.x
        new_pose.position.y = pose.position.y
        new_pose.position.z = pose.position.z
        new_pose.orientation = pose.orientation

        node.send_cartesian_pose_goal(new_pose)
    
    pose = node.get_current_pose()

    if pose:
        print(pose.position)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

