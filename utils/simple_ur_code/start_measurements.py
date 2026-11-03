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
import copy
import time

from scipy.spatial.transform import Rotation as R

#from ft300python.ft300_stream import FT300Stream

# Joint names for each robot
ROBOT_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

def pose_to_xyz_rpy(pose: Pose):
    x = pose.position.x
    y = pose.position.y
    z = pose.position.z
    q = pose.orientation
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    r = R.from_quat(q)
    roll, pitch, yaw = r.as_euler('xyz')
    return x, y, z, roll, pitch, yaw

class ArmClient(Node):
    def __init__(self):
        super().__init__('dual_arm_client')

        # Store the latest joint states
        self.joint_positions = {}

        # Subscribe to combined joint_states
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

        # Action clients for MoveIt move_action
        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')

        # FK service client
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for FK service...")

    def joint_state_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def get_current_pose(self, ee_link: str):
        """Call MoveIt FK service to get the current EE pose from joint states."""

        joint_names = ROBOT_JOINTS

        # Wait until we have all joint positions
        for name in joint_names:
            if name not in self.joint_positions:
                self.get_logger().warn(f"Waiting for joint states: {name}")
                rclpy.spin_once(self, timeout_sec=0.1)
                return None

        #print(joint_names)
        #print([self.joint_positions[n] for n in joint_names])

        # Prepare FK request
        req = GetPositionFK.Request()
        req.header.frame_id = 'base'
        req.fk_link_names = [ee_link]
        req.robot_state = RobotState()
        req.robot_state.joint_state.name = joint_names
        req.robot_state.joint_state.position = [self.joint_positions[n] for n in joint_names]

        future = self.fk_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp.error_code.val != 1:
            self.get_logger().error("FK service failed")
            return None

        return resp.pose_stamped[0].pose

    def send_cartesian_pose_goal(self, client, group_name: str, ee_link: str, target_pose: Pose):
        client.wait_for_server()
        print("action server ready ...")
        goal = MoveGroup.Goal()
        req = goal.request

        req.group_name = group_name
        req.allowed_planning_time = 5.0
        req.num_planning_attempts = 5
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2

        # -------------------------
        # Position constraint
        # -------------------------
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base"
        pos_constraint.link_name = ee_link
        pos_constraint.weight = 1.0

        solid = SolidPrimitive()
        solid.type = SolidPrimitive.BOX
        solid.dimensions = [0.0001, 0.0001, 0.0001]  # 0.1 mm tolerance box

        bounding_volume = BoundingVolume()
        bounding_volume.primitives.append(solid)
        bounding_volume.primitive_poses.append(target_pose)

        pos_constraint.constraint_region = bounding_volume

        # -------------------------
        # Orientation constraint
        # -------------------------
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = "base"
        ori_constraint.link_name = ee_link
        ori_constraint.orientation = target_pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.0001
        ori_constraint.absolute_y_axis_tolerance = 0.0001
        ori_constraint.absolute_z_axis_tolerance = 0.0001
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
        future = client.send_goal_async(goal)
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
    node = ArmClient()

    # --- Get current poses ---
    rclpy.spin_once(node, timeout_sec=0.5)

    while node.joint_positions == {}:
        rclpy.spin_once(node, timeout_sec=0.5)
    
    print(f"our joints are: {node.joint_positions}")

    #initialize the ft300 connection
    #port = "/dev/ttyUSB0"  # "/dev/ttyUSB0" for example in Ubuntu
    #ft_stream = FT300Stream(port, timeout=1, zero_reset=True)

    num_points  = 100
    #TODO: take the number of points (num_points) in regards to the overall area (8.5cm x 18cm) and compute a 2D array
    #      -> e.g. take the start point as (0,0,0) and just increase x and y
    #TODO: save one pose as a start pose and e.g. one as a home pose
    #TODO: make a loop with optional waits in between that changes x/y/z in a given format (one example is given below with one z loop)


    curr_pose = node.get_current_pose('wrist_3_link')
    if curr_pose:
        print("EE pose:", pose_to_xyz_rpy(curr_pose))

    new_pose = copy.deepcopy(curr_pose)

    #reset the force sensor
    #ft_stream.reset_zero_force_torque()

    # --- Example: move EE 0.5 mm down along Z in 10 steps ---
    for i in range(0,5,1):
        if curr_pose:
            #decrease z by 0.5mmm per step
            new_pose.position.x = new_pose.position.x
            new_pose.position.y = new_pose.position.y
            new_pose.position.z = new_pose.position.z - 0.01
            new_pose.orientation = new_pose.orientation

            #send the move command
            node.send_cartesian_pose_goal(node.move_group_client, group_name="ur_manipulator", ee_link="wrist_3_link", target_pose=new_pose)
            #wait for 100ms
            time.sleep(0.2)
            #get force values
            #ft_values = ft_stream.get_force_torque()
            #print(f"Recorded force values: {ft_values}")

    curr_pose2 = node.get_current_pose('wrist_3_link')
    if curr_pose2:
        print("EE pose:", pose_to_xyz_rpy(curr_pose2))
        print("starting pose:", pose_to_xyz_rpy(curr_pose))

    #TODO: here you could move back to the starting pose (5mm up)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

