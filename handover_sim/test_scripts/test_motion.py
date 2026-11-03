import threading

import rclpy
from sensor_msgs.msg import JointState
import math
import time

# ROS2 node for controlling the robot's joint positions and velocities

rclpy.init()
node = rclpy.create_node('position_velocity_publisher')
pub = node.create_publisher(JointState, 'joint_command', 10)

# Spin in a separate thread
thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
thread.start()

joint_state = JointState()

# current position:
# - 0.0
# - -1.5477
# - 1.5837
# - 0.9789
# - 1.54
# - 3.1816
# - 0.02
# - 0.021
# - -0.0015
# - -0.0015
# - -0.0296
# - 0.0308
# - -0.0292
# - -0.0304

joint_state.name = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                             "wrist_2_joint", "wrist_3_joint", "finger_joint",
                             "right_outer_knuckle_joint", "left_outer_finger_joint", "right_outer_finger_joint",
                             "left_inner_finger_joint", "right_inner_finger_joint",
                             "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"]
joint_state.position = [0.0, -1.5477, 1.5837, 0.9789, 1.54, 3.1816, 0.5, 0.5, -0.0015, -0.0015, -0.0296,
                                 0.0308, -0.0292, -0.0304]
joint_state.velocity = [0.0] * len(joint_state.name)

rate = node.create_rate(10)
try:
    while rclpy.ok():
        t = time.time()
        joint_state.position[0] = (-1.57 + 1.0 * math.sin(1.0 * t))
        pub.publish(joint_state)
        rate.sleep()
except KeyboardInterrupt:
    pass
rclpy.shutdown()
thread.join()
