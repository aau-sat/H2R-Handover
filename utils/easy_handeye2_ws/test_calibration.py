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


def matrix_inverse(T: np.ndarray) -> np.ndarray:
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[0:3, 0:3] = R.T
    T_inv[0:3, 3] = -R.T @ p
    return T_inv


def rot_to_rpy_deg(Rm: np.ndarray):
    sy = math.sqrt(Rm[0, 0] * Rm[0, 0] + Rm[1, 0] * Rm[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(Rm[2, 1], Rm[2, 2])
        pitch = math.atan2(-Rm[2, 0], sy)
        yaw = math.atan2(Rm[1, 0], Rm[0, 0])
    else:
        roll = math.atan2(-Rm[1, 2], Rm[1, 1])
        pitch = math.atan2(-Rm[2, 0], sy)
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])


# ----------------------------
# ROS node
# ----------------------------

class ComputeGoalPose(Node):
    def __init__(self):
        super().__init__("compute_goal_pose")

        self.declare_parameter("base_frame", "base")
        self.declare_parameter("camera_frame", "right_color_optical_frame") # needs to be named after calibration target frame in tf_static


        #self.declare_parameter("base_frame", "base")
        #self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("input_topic", "/aruco_marker_pose")
        #self.declare_parameter("input_topic", "/estimated_pose_segmentation")
        self.declare_parameter("output_topic", "/goal_pose")
        self.declare_parameter("output_frame", "")
        self.declare_parameter("invert_static_tf", False)
        self.declare_parameter("debug_every_n", 20)
        self.declare_parameter("publish_input_debug_topic", True)
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.output_frame = self.get_parameter("output_frame").value
        self.invert_static_tf = bool(self.get_parameter("invert_static_tf").value)
        self.debug_every_n = max(1, int(self.get_parameter("debug_every_n").value))
        self.publish_input_debug_topic = bool(self.get_parameter("publish_input_debug_topic").value)

        self.pose_count = 0
        self.logged_startup_compute_dump = False

        self.T_base_to_cam = None
        self.selected_tf_stamp_ns = -1
        self.pose_count = 0

        # tf_static subscription
        qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE
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
            50
        )

        self.pub_pose = self.create_publisher(
            PoseStamped,
            self.output_topic,
            50
        )

        self.pub_input_debug = self.create_publisher(
            PoseStamped,
            "/goal_pose_debug/input_pose",
            50
        )

        self.get_logger().info("Waiting for static TF...")
        self.get_logger().info(
            "Config: "
            f"base_frame='{self.base_frame}', "
            f"camera_frame='{self.camera_frame}', "
            f"input_topic='{self.get_parameter('input_topic').value}', "
            f"output_topic='{self.get_parameter('output_topic').value}', "
            f"output_frame='{self.output_frame if self.output_frame else self.base_frame}', "
            f"invert_static_tf={self.invert_static_tf}, "
            f"debug_every_n={self.debug_every_n}, "
            f"publish_input_debug_topic={self.publish_input_debug_topic}"
        )

    def on_tf_static(self, msg: TFMessage):
        matches = [
            t for t in msg.transforms
            if t.header.frame_id == self.base_frame and t.child_frame_id == self.camera_frame
        ]

        if not matches:
            return

        for t in msg.transforms:
                       
            if (
                t.header.frame_id == self.base_frame and
                t.child_frame_id == self.camera_frame
            ):
                self.T_base_to_cam = transform_to_matrix(t)
                if self.invert_static_tf:
                    self.T_base_to_cam = matrix_inverse(self.T_base_to_cam)
                    self.get_logger().warn(
                        "invert_static_tf=True: using inverse of matched static TF"
                    )

                tr = self.T_base_to_cam[0:3, 3]
                rpy = rot_to_rpy_deg(self.T_base_to_cam[0:3, 0:3])
                self.get_logger().info(
                    f"Received static TF {self.base_frame} → {self.camera_frame}"                
                )
                self.get_logger().info(
                    "Using T_base_to_cam "
                    f"xyz=({tr[0]:.4f}, {tr[1]:.4f}, {tr[2]:.4f}) m "
                    f"rpy=({rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}) deg"
                )
                break

            if (
                t.header.frame_id == self.camera_frame and
                t.child_frame_id == self.base_frame
            ):
                # Some setups publish camera->base; invert to build base->camera.
                self.T_base_to_cam = matrix_inverse(transform_to_matrix(t))
                if self.invert_static_tf:
                    self.T_base_to_cam = matrix_inverse(self.T_base_to_cam)
                    self.get_logger().warn(
                        "invert_static_tf=True: using inverse of inverted static TF"
                    )

                tr = self.T_base_to_cam[0:3, 3]
                rpy = rot_to_rpy_deg(self.T_base_to_cam[0:3, 0:3])
                self.get_logger().warn(
                    f"Found static TF {self.camera_frame} → {self.base_frame}; auto-inverted to build {self.base_frame} → {self.camera_frame}"
                )
                self.get_logger().info(
                    "Using T_base_to_cam "
                    f"xyz=({tr[0]:.4f}, {tr[1]:.4f}, {tr[2]:.4f}) m "
                    f"rpy=({rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}) deg"
                )
                break

    def on_pose(self, msg: PoseStamped):
        self.pose_count += 1
        if self.T_base_to_cam is None:
            self.get_logger().warn("Static TF not received yet")
            return

        self.pose_count += 1

        if self.publish_input_debug_topic:
            self.pub_input_debug.publish(msg)

        if msg.header.frame_id and msg.header.frame_id != self.camera_frame:
            self.get_logger().warn(
                f"Incoming pose frame is '{msg.header.frame_id}', expected '{self.camera_frame}'. "
                "This often causes large pose errors."
            )

        # Use this to bring the goal pose into baselink frame!!
        # T_baselink_base = np.array([
        #     [-1, 0, 0, 0],
        #     [0, -1, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ], dtype=float)


        T_in = pose_to_matrix(msg)
        T_out = self.T_base_to_cam @ T_in
        T_out_if_inverse = np.linalg.inv(self.T_base_to_cam) @ T_in
        # T_out_baselink = T_baselink_base @ self.T_base_to_cam @ T_in
        
        # T_gripper_offset = np.array([
        #     [1, 0, 0, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 1, 0.25], # 0.25 meters = 25 cm
        #     [0, 0, 0, 1]
        # ], dtype=float)
        
        # T_out = self.T_base_to_cam @ T_in @ T_gripper_offset  

        out_frame = self.output_frame if self.output_frame else self.base_frame
        out_msg = matrix_to_pose_stamped(T_out, out_frame, msg.header.stamp)
        self.pub_pose.publish(out_msg)

        if not self.logged_startup_compute_dump:
            self.logged_startup_compute_dump = True
            self.get_logger().info("=== STARTUP COMPUTE DATA DUMP ===")
            self.get_logger().info(
                f"Input pose header: frame_id='{msg.header.frame_id}', "
                f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
            )
            self.get_logger().info(
                "Input pose xyz="
                f"({msg.pose.position.x:.6f}, {msg.pose.position.y:.6f}, {msg.pose.position.z:.6f}) m "
                "quat="
                f"({msg.pose.orientation.x:.6f}, {msg.pose.orientation.y:.6f}, "
                f"{msg.pose.orientation.z:.6f}, {msg.pose.orientation.w:.6f})"
            )
            self.get_logger().info(
                "T_base_to_cam matrix:\n"
                f"{np.array2string(self.T_base_to_cam, precision=6, suppress_small=False)}"
            )
            self.get_logger().info(
                "T_in (from input pose) matrix:\n"
                f"{np.array2string(T_in, precision=6, suppress_small=False)}"
            )
            self.get_logger().info(
                "T_out = T_base_to_cam @ T_in matrix:\n"
                f"{np.array2string(T_out, precision=6, suppress_small=False)}"
            )
            self.get_logger().info(
                "Output pose xyz="
                f"({out_msg.pose.position.x:.6f}, {out_msg.pose.position.y:.6f}, {out_msg.pose.position.z:.6f}) m "
                "quat="
                f"({out_msg.pose.orientation.x:.6f}, {out_msg.pose.orientation.y:.6f}, "
                f"{out_msg.pose.orientation.z:.6f}, {out_msg.pose.orientation.w:.6f}) "
                f"frame='{out_msg.header.frame_id}'"
            )
            self.get_logger().info("=== END STARTUP COMPUTE DATA DUMP ===")

        if self.pose_count % self.debug_every_n == 0:
            p_in = T_in[0:3, 3]
            p_out = T_out[0:3, 3]
            rpy_in = rot_to_rpy_deg(T_in[0:3, 0:3])
            rpy_out = rot_to_rpy_deg(T_out[0:3, 0:3])
            self.get_logger().info(
                f"Pose debug #{self.pose_count}: in[{self.camera_frame}] "
                f"xyz=({p_in[0]:.4f}, {p_in[1]:.4f}, {p_in[2]:.4f}) m "
                f"rpy=({rpy_in[0]:.2f}, {rpy_in[1]:.2f}, {rpy_in[2]:.2f}) deg -> "
                f"out[{out_frame}] xyz=({p_out[0]:.4f}, {p_out[1]:.4f}, {p_out[2]:.4f}) m "
                f"rpy=({rpy_out[0]:.2f}, {rpy_out[1]:.2f}, {rpy_out[2]:.2f}) deg"
            )


def main():
    rclpy.init()
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