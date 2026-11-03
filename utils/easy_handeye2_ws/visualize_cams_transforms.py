# Visualizes the active cam-to-cam transform based on the unified cam->cam transform yaml and the static TFs from the cameras.
# run set_cams_transform.py first to compute the unified cam-to-cam transform and then use this script to visualize it.

import rclpy
from rclpy.node import Node
import sys
from geometry_msgs.msg import TransformStamped, PoseStamped
import argparse

import os
import math
import yaml
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_msgs.msg import TFMessage

from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
import numpy as np

def load_transform_yaml(calib_file: str):
    try:
        with open(calib_file, 'r') as handle:
            calib = yaml.safe_load(handle)
    except IOError:
        print ('Failed reading calibration parameters from file %s' % calib_file)
        sys.exit(-1)

    if not isinstance(calib, dict):
        print ('Invalid calibration file format in %s' % calib_file)
        sys.exit(-1)

    if 'parameters' in calib and 'transform' in calib:
        return {
            'frame_id': calib['parameters']['robot_base_frame'],
            'child_frame_id': calib['parameters']['tracking_base_frame'],
            'translation': calib['transform']['translation'],
            'rotation': calib['transform']['rotation'],
        }

    if 'frame_id' in calib and 'child_frame_id' in calib and 'translation' in calib and 'rotation' in calib:
        return {
            'frame_id': calib['frame_id'],
            'child_frame_id': calib['child_frame_id'],
            'translation': calib['translation'],
            'rotation': calib['rotation'],
        }

    print ('Unsupported calibration file format in %s' % calib_file)
    sys.exit(-1)

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

def matrix_to_tf_stamped(T, frame_id, child_frame_id, stamp):
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id

    qx, qy, qz, qw = rot_to_quat(T[0:3, 0:3])
    p = T[0:3, 3]

    msg.transform.translation.x = float(p[0])
    msg.transform.translation.y = float(p[1])
    msg.transform.translation.z = float(p[2])

    msg.transform.rotation.x = float(qx)
    msg.transform.rotation.y = float(qy)
    msg.transform.rotation.z = float(qz)
    msg.transform.rotation.w = float(qw)
    return msg

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

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * sr - sy * sp * cr  # x
    q[1] = sy * cp * sr + cy * sp * cr  # y
    q[2] = sy * cp * cr - cy * sp * sr  # z
    q[3] = cy * cp * cr + sy * sp * sr  # w

    return q

class CamToCamTransformConfig:
    def __init__(self, from_camera_name, to_camera_name, calib_file):
        self.from_camera_name = from_camera_name
        self.to_camera_name = to_camera_name
        calib = load_transform_yaml(calib_file)
        self.tf = TransformStamped()
        self.tf.header.frame_id = calib['frame_id']
        self.tf.child_frame_id = calib['child_frame_id']
        self.tf.transform.translation.x = calib['translation']['x']
        self.tf.transform.translation.y = calib['translation']['y']
        self.tf.transform.translation.z = calib['translation']['z']
        self.tf.transform.rotation.x = calib['rotation']['x']
        self.tf.transform.rotation.y = calib['rotation']['y']
        self.tf.transform.rotation.z = calib['rotation']['z']
        self.tf.transform.rotation.w = calib['rotation']['w']

    def get_frame_id(self):
        return self.tf.header.frame_id

    def transform_to_matrix(self) -> np.ndarray:
        tx = self.tf.transform.translation.x
        ty = self.tf.transform.translation.y
        tz = self.tf.transform.translation.z

        qx = self.tf.transform.rotation.x
        qy = self.tf.transform.rotation.y
        qz = self.tf.transform.rotation.z
        qw = self.tf.transform.rotation.w
        n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if n == 0.0:
            R = np.eye(3)
        else:
            qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
            xx, yy, zz = qx*qx, qy*qy, qz*qz
            xy, xz, yz = qx*qy, qx*qz, qy*qz
            wx, wy, wz = qw*qx, qw*qy, qw*qz

            R = np.array([
                [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
                [2*(xy + wz),   1 - 2*(xx + zz),     2*(yz - wx)],
                [2*(xz - wy),       2*(yz + wx), 1 - 2*(xx + yy)],
            ], dtype=float)

        T = np.eye(4, dtype=float)
        T[0:3, 0:3] = R
        T[0:3, 3] = [tx, ty, tz]
        return T
    
    def get_translation(self):
        return self.tf.transform.translation.x, self.tf.transform.translation.y, self.tf.transform.translation.z

    def get_euler(self):
        return euler_from_quaternion(self.tf.transform.rotation)
    
    def get_rotation(self):
        return self.tf.transform.rotation.x, self.tf.transform.rotation.y, self.tf.transform.rotation.z, self.tf.transform.rotation.w
    
    def set_translation(self, x, y, z):
        self.tf.transform.translation.x = x
        self.tf.transform.translation.y = y
        self.tf.transform.translation.z = z
    
    def set_rotation_from_euler(self, roll, pitch, yaw):
        quat = quaternion_from_euler(roll, pitch, yaw)
        self.tf.transform.rotation.x = quat[0]
        self.tf.transform.rotation.y = quat[1]
        self.tf.transform.rotation.z = quat[2]
        self.tf.transform.rotation.w = quat[3]

class CamToBaseTransformConfig:
    def __init__(self, camera_name: str, calib_file: str):
        self.camera_name = camera_name
        calib = load_transform_yaml(calib_file)
        self.tf = TransformStamped()
        self.tf.header.frame_id = calib['frame_id']
        self.tf.child_frame_id = calib['child_frame_id']
        self.tf.transform.translation.x = calib['translation']['x']
        self.tf.transform.translation.y = calib['translation']['y']
        self.tf.transform.translation.z = calib['translation']['z']
        self.tf.transform.rotation.x = calib['rotation']['x']
        self.tf.transform.rotation.y = calib['rotation']['y']
        self.tf.transform.rotation.z = calib['rotation']['z']
        self.tf.transform.rotation.w = calib['rotation']['w']
    
    def get_frame_id(self):
        return self.tf.header.frame_id

    def transform_to_matrix(self) -> np.ndarray:
        tx = self.tf.transform.translation.x
        ty = self.tf.transform.translation.y
        tz = self.tf.transform.translation.z

        qx = self.tf.transform.rotation.x
        qy = self.tf.transform.rotation.y
        qz = self.tf.transform.rotation.z
        qw = self.tf.transform.rotation.w
        n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if n == 0.0:
            R = np.eye(3)
        else:
            qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
            xx, yy, zz = qx*qx, qy*qy, qz*qz
            xy, xz, yz = qx*qy, qx*qz, qy*qz
            wx, wy, wz = qw*qx, qw*qy, qw*qz

            R = np.array([
                [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
                [2*(xy + wz),   1 - 2*(xx + zz),     2*(yz - wx)],
                [2*(xz - wy),       2*(yz + wx), 1 - 2*(xx + yy)],
            ], dtype=float)

        T = np.eye(4, dtype=float)
        T[0:3, 0:3] = R
        T[0:3, 3] = [tx, ty, tz]
        return T

class CamToCamTransformBroadcaster(Node):
    def __init__(self, cam_to_cam_cfg: CamToCamTransformConfig):
        super().__init__('cam_to_cam_transform_broadcaster')
        self.broadcaster = StaticTransformBroadcaster(self)
        timer_period = 0.1  # seconds
        self.cam_to_cam_cfg = cam_to_cam_cfg
        self.from_cam_name = self.cam_to_cam_cfg.from_camera_name
        self.to_cam_name = self.cam_to_cam_cfg.to_camera_name
        self.tf = None
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
        self.timer = self.create_timer(timer_period, self.publish_transform)
    
    def on_tf_static(self, msg: TFMessage):
        if self.tf is not None:
            return

        # Store all received transforms in a lookup dictionary
        if not hasattr(self, 'tf_lookup'):
            self.tf_lookup = {}

        for t in msg.transforms:
            key = (t.header.frame_id, t.child_frame_id)
            self.tf_lookup[key] = transform_to_matrix(t)
            # Log for debugging
            self.get_logger().info(f"Buffered static TF: {key[0]} -> {key[1]}")

    def _get_path_transform(self, frame_id, child_frame_id):
        """Helper to get a transform even if it's inverted in the lookup."""
        if (frame_id, child_frame_id) in self.tf_lookup:
            return self.tf_lookup[(frame_id, child_frame_id)]
        elif (child_frame_id, frame_id) in self.tf_lookup:
            return np.linalg.inv(self.tf_lookup[(child_frame_id, frame_id)])
        return None

    def _calculate_cam_to_cam_transform(self):
        r_depth_opt = f"{self.from_cam_name}_depth_optical_frame"
        r_depth_phys = f"{self.from_cam_name}_depth_frame"
        r_link = f"{self.from_cam_name}_link"
        r_color_phys = f"{self.from_cam_name}_color_frame"
        r_color_opt = f"{self.from_cam_name}_color_optical_frame"

        l_color_opt = f"{self.to_cam_name}_color_optical_frame"
        l_color_phys = f"{self.to_cam_name}_color_frame"
        l_link = f"{self.to_cam_name}_link"
        l_depth_phys = f"{self.to_cam_name}_depth_frame"
        l_depth_opt = f"{self.to_cam_name}_depth_optical_frame"

        try:
            # 1. Right Depth Optical -> Right Color Optical
            # Path: opt -> phys -> link -> phys -> opt
            T_r_dopt_to_dphys = self._get_path_transform(child_frame_id=r_depth_opt, frame_id=r_depth_phys)
            T_r_dphys_to_link = self._get_path_transform(child_frame_id=r_depth_phys, frame_id=r_link)
            T_r_link_to_cphys = self._get_path_transform(child_frame_id=r_link, frame_id=r_color_phys)
            T_r_cphys_to_copt  = self._get_path_transform(child_frame_id=r_color_phys, frame_id=r_color_opt)

            # 2. Left Color Optical -> Left Depth Optical
            T_l_copt_to_cphys = self._get_path_transform(child_frame_id=l_color_opt, frame_id=l_color_phys)
            T_l_cphys_to_link = self._get_path_transform(child_frame_id=l_color_phys, frame_id=l_link)
            T_l_link_to_dphys = self._get_path_transform(child_frame_id=l_link, frame_id=l_depth_phys)
            T_l_dphys_to_dopt = self._get_path_transform(child_frame_id=l_depth_phys, frame_id=l_depth_opt)

            # Check if all internal bits were found
            internals = [T_r_dopt_to_dphys, T_r_dphys_to_link, T_r_link_to_cphys, T_r_cphys_to_copt,
                         T_l_copt_to_cphys, T_l_cphys_to_link, T_l_link_to_dphys, T_l_dphys_to_dopt]
            
            if any(m is None for m in internals):
                self.get_logger().warn("Not all RealSense static TFs available yet")
                return

            # 3. The Calibrated Bridge
            T_cam_to_cam = self.cam_to_cam_cfg.transform_to_matrix()

            # 4. Combine everything
            # Chain: R_Depth_Opt -> R_Color_Opt -> L_Color_Opt -> L_Depth_Opt
            T_right_internal = T_r_cphys_to_copt @ T_r_link_to_cphys @ T_r_dphys_to_link @ T_r_dopt_to_dphys
            T_left_internal  = T_l_dphys_to_dopt @ T_l_link_to_dphys @ T_l_cphys_to_link @ T_l_copt_to_cphys
            
            full_matrix = T_left_internal @ T_cam_to_cam @ T_right_internal
            cam_to_cam_matrix = T_cam_to_cam

            stamp = self.get_clock().now().to_msg()
            self.tf = matrix_to_tf_stamped(full_matrix, l_depth_opt, r_depth_opt, stamp)
            print ('Successfully computed full depth-to-depth transform:')
            print (full_matrix)
            self.get_logger().info("Successfully computed full cam-to-cam transform.")
            print (cam_to_cam_matrix)

        except Exception as e:
            self.get_logger().warn(f"Waiting for more TFs... {e}")
                
    def publish_transform(self):
        if self.tf is None:
            self._calculate_cam_to_cam_transform()
            return
        self.tf.header.stamp = self.get_clock().now().to_msg()
        self.broadcaster.sendTransform(self.tf)


class CamToBaseTransformBroadcaster(Node):
    def __init__(self, first_cam_cfg: CamToBaseTransformConfig, second_cam_cfg: CamToBaseTransformConfig):
        super().__init__('cam_to_base_transform_broadcaster')
        self.broadcaster = StaticTransformBroadcaster(self)
        timer_period = 0.1  # seconds
        self.first_cam_cfg = first_cam_cfg
        self.second_cam_cfg = second_cam_cfg
        self.first_cam_name = self.first_cam_cfg.camera_name
        self.second_cam_name = self.second_cam_cfg.camera_name
        self.tf_first = None
        self.tf_second = None
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
        self.timer = self.create_timer(timer_period, self.publish_transform)
    
    def on_tf_static(self, msg: TFMessage):
        if self.tf_first is not None and self.tf_second is not None:
            return

        # Store all received transforms in a lookup dictionary
        if not hasattr(self, 'tf_lookup'):
            self.tf_lookup = {}

        for t in msg.transforms:
            key = (t.header.frame_id, t.child_frame_id)
            self.tf_lookup[key] = transform_to_matrix(t)
            # Log for debugging
            self.get_logger().info(f"Buffered static TF: {key[0]} -> {key[1]}")

    def _get_path_transform(self, frame_id, child_frame_id):
        """Helper to get a transform even if it's inverted in the lookup."""
        if (frame_id, child_frame_id) in self.tf_lookup:
            return self.tf_lookup[(frame_id, child_frame_id)]
        elif (child_frame_id, frame_id) in self.tf_lookup:
            return np.linalg.inv(self.tf_lookup[(child_frame_id, frame_id)])
        return None

    def _calculate_cam_to_base_transform(self):
        first_depth_opt = f"{self.first_cam_name}_depth_optical_frame"
        first_depth_phys = f"{self.first_cam_name}_depth_frame"
        first_link = f"{self.first_cam_name}_link"
        first_color_phys = f"{self.first_cam_name}_color_frame"
        first_color_opt = f"{self.first_cam_name}_color_optical_frame"

        second_depth_opt = f"{self.second_cam_name}_depth_optical_frame"
        second_depth_phys = f"{self.second_cam_name}_depth_frame"
        second_link = f"{self.second_cam_name}_link"
        second_color_phys = f"{self.second_cam_name}_color_frame"
        second_color_opt = f"{self.second_cam_name}_color_optical_frame"

        try:
            # 1. Right Depth Optical -> Right Color Optical
            # Path: opt -> phys -> link -> phys -> opt
            T_first_dopt_to_dphys = self._get_path_transform(child_frame_id=first_depth_opt, frame_id=first_depth_phys)
            T_first_dphys_to_link = self._get_path_transform(child_frame_id=first_depth_phys, frame_id=first_link)
            T_first_link_to_cphys = self._get_path_transform(child_frame_id=first_link, frame_id=first_color_phys)
            T_first_cphys_to_copt  = self._get_path_transform(child_frame_id=first_color_phys, frame_id=first_color_opt)

            # 2. Left Depth Optical -> Left Color Optical
            T_second_dopt_to_dphys = self._get_path_transform(child_frame_id=second_depth_opt, frame_id=second_depth_phys)
            T_second_dphys_to_link = self._get_path_transform(child_frame_id=second_depth_phys, frame_id=second_link)
            T_second_link_to_cphys = self._get_path_transform(child_frame_id=second_link, frame_id=second_color_phys)
            T_second_cphys_to_copt  = self._get_path_transform(child_frame_id=second_color_phys, frame_id=second_color_opt)

            # Check if all internal bits were found
            internals = [T_first_dopt_to_dphys, T_first_dphys_to_link, T_first_link_to_cphys, T_first_cphys_to_copt,
                         T_second_dopt_to_dphys, T_second_dphys_to_link, T_second_link_to_cphys, T_second_cphys_to_copt]
            
            if any(m is None for m in internals):
                self.get_logger().warn("Not all RealSense static TFs available yet")
                return

            # 3. The Calibrated Bridge
            T_first_to_base = self.first_cam_cfg.transform_to_matrix()
            T_second_to_base = self.second_cam_cfg.transform_to_matrix()

            # 4. Combine everything
            # Chain: R_Depth_Opt -> R_Color_Opt -> L_Color_Opt -> L_Depth_Opt
            T_first_internal = T_first_cphys_to_copt @ T_first_link_to_cphys @ T_first_dphys_to_link @ T_first_dopt_to_dphys
            T_second_internal  = T_second_cphys_to_copt @ T_second_link_to_cphys @ T_second_dphys_to_link @ T_second_dopt_to_dphys
            
            first_full_matrix = T_first_to_base @ T_first_internal
            second_full_matrix = T_second_to_base @ T_second_internal

            stamp = self.get_clock().now().to_msg()
            self.tf_first = matrix_to_tf_stamped(first_full_matrix, self.first_cam_cfg.get_frame_id(), first_depth_opt, stamp)
            self.tf_second = matrix_to_tf_stamped(second_full_matrix, self.second_cam_cfg.get_frame_id(), second_depth_opt, stamp)
            print ('Successfully computed full depth-to-depth transform:')
            print (first_full_matrix)
            print (second_full_matrix)

        except Exception as e:
            self.get_logger().warn(f"Waiting for more TFs... {e}")
                
    def publish_transform(self):
        if self.tf_first is None or self.tf_second is None:
            self._calculate_cam_to_base_transform()
            return
        self.tf_first.header.stamp = self.get_clock().now().to_msg()
        self.tf_second.header.stamp = self.get_clock().now().to_msg()
        self.broadcaster.sendTransform([self.tf_first, self.tf_second])

parser = argparse.ArgumentParser(description='Visualizes the computed transform between two cameras.')
parser.add_argument('--tf_type', type=str, default='cam-to-base', help='Type of TF to publish: cam-to-cam or cam-to-base. Default: cam-to-base')
parser.add_argument('--from_cam_name', type=str, default='right', help='Name of the first camera (child frame)')
parser.add_argument('--to_cam_name', type=str, default='left', help='Name of the second camera (parent frame)')
parser.add_argument('--cam_to_cam_calibration', type=str, default='cam_to_cam.yaml', help='Calibration file for the first camera')
parser.add_argument('--first_to_base_calibration', type=str, default='/home/student-team/.ros2/easy_handeye2/calibrations/calibration_right_camera.calib', 
                    help='Calibration file for the first camera (for cam-to-base TF)')
parser.add_argument('--refined_first_to_base', action='store_true', default=False,
                    help='Use refined calibration for the first camera (for cam-to-base TF)')
parser.add_argument('--refined_first_to_base_calibration', type=str, default='right_refined_cam_to_base.yaml',
                    help='Refined calibration file for the first camera (for cam-to-base TF, optional)')
parser.add_argument('--second_to_base_calibration', type=str, default='/home/student-team/.ros2/easy_handeye2/calibrations/calibration_left_camera.calib', 
                    help='Calibration file for the second camera (for cam-to-base TF)')
parser.add_argument('--refined_second_to_base', action='store_true', default=False, 
                    help='Use refined calibration for the second camera (for cam-to-base TF)')
parser.add_argument('--refined_second_to_base_calibration', type=str, default='left_refined_cam_to_base.yaml', 
                    help='Refined calibration file for the second camera (for cam-to-base TF, optional)')

if __name__ == '__main__':
    args = parser.parse_args()
    
    from_cam, to_cam = args.from_cam_name, args.to_cam_name
    
    rclpy.init()
    if args.tf_type == 'cam-to-base':
        print('Mode: cam-to-base.')
        if args.refined_first_to_base:
            first_to_base_calib = args.refined_first_to_base_calibration
        else:
            first_to_base_calib = args.first_to_base_calibration
        if args.refined_second_to_base:
            second_to_base_calib = args.refined_second_to_base_calibration
        else:
            second_to_base_calib = args.second_to_base_calibration
        
        first_cam_cfg = CamToBaseTransformConfig(from_cam, first_to_base_calib)
        second_cam_cfg = CamToBaseTransformConfig(to_cam, second_to_base_calib)
        broadcaster = CamToBaseTransformBroadcaster(first_cam_cfg, second_cam_cfg)
    else:
        print('Mode: cam-to-cam.')
        cam_to_cam_calib = args.cam_to_cam_calibration
    
        cam_to_cam_tf = CamToCamTransformConfig(from_cam, to_cam, cam_to_cam_calib)
        broadcaster = CamToCamTransformBroadcaster(cam_to_cam_tf)
    rclpy.spin(broadcaster)
