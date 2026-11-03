#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    MoveItErrorCodes
)
from shape_msgs.msg import SolidPrimitive


class PilzPTPClient(Node):

    def __init__(self):
        super().__init__("pilz_ptp_client")

        self.group_name = "ur_manipulator"

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            "/move_action"
        )

        self.get_logger().info("Waiting for MoveGroup action server...")
        self.move_group_client.wait_for_server()
        self.get_logger().info("Connected to MoveGroup.")

    def send_ptp(self):

        goal = MoveGroup.Goal()
        req = goal.request

        # ========= CRITICAL SETTINGS =========
        req.group_name = self.group_name
        req.pipeline_id = "pilz_industrial_motion_planner"
        req.planner_id = "PTP"
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2
        # =====================================

        # ===== TARGET POSE =====
        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"

        target_pose.pose.position.x = 0.3
        target_pose.pose.position.y = 0.2
        target_pose.pose.position.z = 0.4

        target_pose.pose.orientation.x = 0.0
        target_pose.pose.orientation.y = 1.0
        target_pose.pose.orientation.z = 0.0
        target_pose.pose.orientation.w = 0.0
        # ========================

        # ===== POSITION CONSTRAINT (Sphere) =====
        pos_con = PositionConstraint()
        pos_con.header.frame_id = "base_link"
        pos_con.link_name = "tool0"

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]   # 1 cm tolerance

        pos_con.constraint_region.primitives.append(sphere)
        pos_con.constraint_region.primitive_poses.append(target_pose.pose)
        pos_con.weight = 1.0
        # ========================================

        # ===== ORIENTATION CONSTRAINT =====
        ori_con = OrientationConstraint()
        ori_con.header.frame_id = "base_link"
        ori_con.link_name = "tool0"
        ori_con.orientation = target_pose.pose.orientation
        ori_con.absolute_x_axis_tolerance = 0.2
        ori_con.absolute_y_axis_tolerance = 0.2
        ori_con.absolute_z_axis_tolerance = 0.2
        ori_con.weight = 1.0
        # ===================================

        constraints = Constraints()
        constraints.position_constraints.append(pos_con)
        constraints.orientation_constraints.append(ori_con)

        req.goal_constraints.append(constraints)

        self.get_logger().info("Sending Pilz PTP goal...")

        send_future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return

        self.get_logger().info("Goal accepted, waiting...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result.result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("Motion SUCCESS ✅")
        else:
            self.get_logger().error(
                f"Motion failed. Error code: {result.result.error_code.val}"
            )


def main():
    rclpy.init()
    node = PilzPTPClient()
    node.send_ptp()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
