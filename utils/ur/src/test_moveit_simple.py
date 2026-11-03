#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# Standard ROS 2 Imports
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints, 
    PositionConstraint, 
    OrientationConstraint, 
    BoundingVolume
)

class UR3eMover(Node):
    def __init__(self):
        super().__init__('ur3e_mover')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')

    def send_cartesian_goal(self, target_pose_stamped):
        """
        Accepts a PoseStamped object and sends a MoveGroup action goal.
        """
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "ur_custom_manipulator"
        
        # --- 1. Position Constraint (The Box) ---
        pos_con = PositionConstraint()
        pos_con.header = target_pose_stamped.header # Use frame_id from input
        pos_con.link_name = "grasp_center"          # The tool frame to move
        pos_con.weight = 1.0

        # Define the 1mm target tolerance box
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.001, 0.001, 0.001]
        
        # Place the box at the target position
        bv = BoundingVolume()
        bv.primitives.append(box)
        bv.primitive_poses.append(target_pose_stamped.pose)
        
        pos_con.constraint_region = bv

        # --- 2. Orientation Constraint (The Rotation) ---
        ori_con = OrientationConstraint()
        ori_con.header = target_pose_stamped.header
        ori_con.link_name = "grasp_center"
        
        # IMPORTANT: Use the orientation from your PoseStamped
        ori_con.orientation = target_pose_stamped.pose.orientation
        
        ori_con.weight = 1.0
        # Tolerances (in radians) - Adjust if planning fails
        ori_con.absolute_x_axis_tolerance = 0.05
        ori_con.absolute_y_axis_tolerance = 0.05
        ori_con.absolute_z_axis_tolerance = 0.05

        # --- 3. Bundle Constraints ---
        constraints = Constraints()
        constraints.position_constraints.append(pos_con)
        constraints.orientation_constraints.append(ori_con)
        goal_msg.request.goal_constraints.append(constraints)
        
        # Planning settings
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1

        print(f"Waiting for action server...")
        self._action_client.wait_for_server()
        
        print(f"Sending goal...")
        print(f"Pos: {target_pose_stamped.pose.position}")
        print(f"Ori: {target_pose_stamped.pose.orientation}")
        
        return self._action_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    client = UR3eMover()

    # Define your specific pose in base_link
    target = PoseStamped()
    target.header.frame_id = "base_link"
    
    # Your coordinates
    target.pose.position.x = -0.016
    target.pose.position.y = 0.336
    target.pose.position.z = 0.503
    
    # Your orientation
    target.pose.orientation.x = 0.891
    target.pose.orientation.y = -0.068
    target.pose.orientation.z = 0.043
    target.pose.orientation.w = -0.446

    # Send goal
    future = client.send_cartesian_goal(target)
    
    # Wait for result
    rclpy.spin_until_future_complete(client, future)
    
    result = future.result()
    if result:
        print("Goal accepted by MoveIt!")
    else:
        print("Goal rejected or failed.")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()