#!/usr/bin/env python3
import math
from typing import Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_msgs.msg import TFMessage

from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy


# ----------------------------
# Math helpers
# ----------------------------

def transform_to_matrix(t: TransformStamped) -> np.ndarray:
    tx = t.transform.translation.x
    ty = t.transform.translation.y
    tz = t.transform.translation.z

    qx = t.transform.rotation.x
    qy = t.transform.rotation.y
    qz = t.transform.rotation.z
    qw = t.transform.rotation.w

    R = quat_to_rot(qx, qy, qz, qw)

    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = R
    T[0:3, 3] = [tx, ty, tz]
    return T


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0.0:
        return np.eye(3)

    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),   1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx), 1 - 2*(xx + yy)],
    ], dtype=float)


def rot_to_quat(R: np.ndarray):
    tr = float(np.trace(R))
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            S = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    return (qx/n, qy/n, qz/n, qw/n) if n > 0 else (0, 0, 0, 1)


def pose_to_matrix(p: PoseStamped) -> np.ndarray:
    tx = p.pose.position.x
    ty = p.pose.position.y
    tz = p.pose.position.z

    qx = p.pose.orientation.x
    qy = p.pose.orientation.y
    qz = p.pose.orientation.z
    qw = p.pose.orientation.w

    R = quat_to_rot(qx, qy, qz, qw)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [tx, ty, tz]
    return T


def matrix_to_pose_stamped(T, frame_id, stamp):
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    qx, qy, qz, qw = rot_to_quat(T[0:3, 0:3])
    p = T[0:3, 3]

    msg.pose.position.x = float(p[0])
    msg.pose.position.y = float(p[1])
    msg.pose.position.z = float(p[2])

    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


# ----------------------------
# ROS node
# ----------------------------

class ComputeGoalPose(Node):
    def __init__(self):
        super().__init__("compute_goal_pose")

        self.declare_parameter("base_frame", "base")
        self.declare_parameter("camera_frame", "left_color_optical_frame")


        #self.declare_parameter("base_frame", "base")
        #self.declare_parameter("camera_frame", "camera_color_optical_frame")
        # self.declare_parameter("input_topic", "/aruco_marker_pose")
        self.declare_parameter("input_topic", "/estimated_pose")
        self.declare_parameter("output_topic", "/goal_pose")
        self.declare_parameter("output_frame", "")

        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.output_frame = self.get_parameter("output_frame").value
        self.input_topic = str(self.get_parameter("input_topic").value or "/estimated_pose")
        self.output_topic = str(self.get_parameter("output_topic").value or "/goal_pose")

        self.T_base_to_cam = None

        # tf_static subscription
        qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE
        )
        qos_sensor = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
             depth=10
        )

        self.sub_tf_static = self.create_subscription(
            TFMessage,
            "/tf_static",
            self.on_tf_static,
            qos
        )

        self.sub_pose = self.create_subscription(
            PoseStamped,
            self.input_topic,
            self.on_pose,
            qos_sensor
        )

        self.pub_pose = self.create_publisher(
            PoseStamped,
            self.output_topic,
            10
        )

        self.get_logger().info(f"Camera frame: {self.camera_frame}")
        self.get_logger().info(f"Base frame: {self.base_frame}")
        self.get_logger().info("Waiting for static TF...")

    def on_tf_static(self, msg: TFMessage):
        if self.T_base_to_cam is not None:
            return

        for t in msg.transforms:
                       
            if (
                t.header.frame_id == self.base_frame and
                t.child_frame_id == self.camera_frame
            ):
                self.T_base_to_cam = transform_to_matrix(t)
                self.get_logger().info(
                    f"Received static TF {self.base_frame} → {self.camera_frame}"                
                )
                
                self.get_logger().info(f"Received static TF with transform {t.transform.translation.x}, ")
                break

    def on_pose(self, msg: PoseStamped):
        if self.T_base_to_cam is None:
            self.get_logger().warn("Static TF not received yet")
            return

        # Use this to bring the goal pose into baselink frame!!
        T_baselink_base = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)


        T_in = pose_to_matrix(msg)
        T_out = self.T_base_to_cam @ T_in        
        # T_out_baselink = T_baselink_base @ self.T_base_to_cam @ T_in
        
        T_gripper_offset = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.25], # 0.25 meters = 25 cm
            [0, 0, 0, 1]
        ], dtype=float)
        
        # T_out = self.T_base_to_cam @ T_in @ T_gripper_offset  

        out_frame = self.output_frame if self.output_frame else self.base_frame
        out_msg = matrix_to_pose_stamped(T_out, out_frame, msg.header.stamp)
        self.pub_pose.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ComputeGoalPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
