# Allows user to refine the active cam-to-cam transform based on the unified cam->cam transform yaml and the static TFs from the cameras. 
# Run set_cams_transform.py first to compute the unified cam-to-cam transform and then use this script to visualize and refine it.
# launch rviz2 and set the fixed frame to child_frame_id of the cam-to-cam transform to see the effect of changes in real time.

import rclpy
from rclpy.node import Node
import sys
from geometry_msgs.msg import TransformStamped
import argparse

import termios
import tty
import os
import math
import yaml
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_msgs.msg import TFMessage

from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

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

def load_transform_yaml(calib_file: str):
    try:
        with open(calib_file, 'r') as handle:
            calib = yaml.safe_load(handle)
    except IOError:
        print('Failed reading calibration parameters from file %s' % calib_file)
        sys.exit(-1)

    if not isinstance(calib, dict):
        print('Invalid calibration file format in %s' % calib_file)
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

    print('Unsupported calibration file format in %s' % calib_file)
    sys.exit(-1)

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
    
    def get_child_frame_id(self):
        return self.tf.child_frame_id

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
    
    def get_child_frame_id(self):
        return self.tf.child_frame_id

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

class CamToCamTransformRefiner(Node):
    def __init__(self, cam_to_cam_cfg: CamToCamTransformConfig, filename: str = None):
        super().__init__('refine_cam_to_cam_transform_refiner')
        self.broadcaster = StaticTransformBroadcaster(self)
        self.cam_to_cam_cfg = cam_to_cam_cfg
        self.from_cam_name = self.cam_to_cam_cfg.from_camera_name
        self.to_cam_name = self.cam_to_cam_cfg.to_camera_name
        self.tf = None
        self.filename = filename
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
    
    def on_tf_static(self, msg: TFMessage):
        # Store all received transforms in a lookup dictionary
        if not hasattr(self, 'tf_lookup'):
            self.tf_lookup = {}

        for t in msg.transforms:
            key = (t.header.frame_id, t.child_frame_id)
            if key not in self.tf_lookup:  # Only log new transforms
                self.tf_lookup[key] = transform_to_matrix(t)
                self.get_logger().info(f"Buffered static TF: {key[0]} -> {key[1]}")
            else:
                self.tf_lookup[key] = transform_to_matrix(t)

    def _get_path_transform(self, frame_id, child_frame_id):
        """Helper to get a transform even if it's inverted in the lookup."""
        if (frame_id, child_frame_id) in self.tf_lookup:
            return self.tf_lookup[(frame_id, child_frame_id)]
        elif (child_frame_id, frame_id) in self.tf_lookup:
            return np.linalg.inv(self.tf_lookup[(child_frame_id, frame_id)])
        return None

    def calculate_transform(self):
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
    
    def export_transform(self, filename):
        if self.filename is not None:
            print ('Saving refined transform to %s' % os.path.abspath(self.filename))
            roll, pitch, yaw = self.cam_to_cam_cfg.get_euler()
            qx, qy, qz, qw = self.cam_to_cam_cfg.get_rotation()
            x, y, z = self.cam_to_cam_cfg.get_translation()
            with open(self.filename, 'w') as f:
                yaml.dump({
                    'frame_id': self.cam_to_cam_cfg.get_frame_id(),
                    'child_frame_id': self.cam_to_cam_cfg.get_child_frame_id(),
                    'translation': {
                        'x': x,
                        'y': y,
                        'z': z,
                    },
                    'rotation': {
                        'x': qx,
                        'y': qy,
                        'z': qz,
                        'w': qw,
                    },
                    'euler': {
                        'roll': float(roll),
                        'pitch': float(pitch),
                        'yaw': float(yaw),
                    },
                    'euler_degrees': {
                        'roll': math.degrees(roll),
                        'pitch': math.degrees(pitch),
                        'yaw': math.degrees(yaw),
                    }
                }, f)
    
    # Recompute transform based on current cam_to_cam config and publish it
    def update_and_publish_transform(self):
        r_depth_opt = f"{self.from_cam_name}_depth_optical_frame"
        l_depth_opt = f"{self.to_cam_name}_depth_optical_frame"
        
        # Get the internal RealSense transforms (should already be cached)
        r_depth_phys = f"{self.from_cam_name}_depth_frame"
        r_link = f"{self.from_cam_name}_link"
        r_color_phys = f"{self.from_cam_name}_color_frame"
        r_color_opt = f"{self.from_cam_name}_color_optical_frame"
        l_color_opt = f"{self.to_cam_name}_color_optical_frame"
        l_color_phys = f"{self.to_cam_name}_color_frame"
        l_link = f"{self.to_cam_name}_link"
        l_depth_phys = f"{self.to_cam_name}_depth_frame"
        
        T_r_dopt_to_dphys = self._get_path_transform(child_frame_id=r_depth_opt, frame_id=r_depth_phys)
        T_r_dphys_to_link = self._get_path_transform(child_frame_id=r_depth_phys, frame_id=r_link)
        T_r_link_to_cphys = self._get_path_transform(child_frame_id=r_link, frame_id=r_color_phys)
        T_r_cphys_to_copt  = self._get_path_transform(child_frame_id=r_color_phys, frame_id=r_color_opt)
        T_l_copt_to_cphys = self._get_path_transform(child_frame_id=l_color_opt, frame_id=l_color_phys)
        T_l_cphys_to_link = self._get_path_transform(child_frame_id=l_color_phys, frame_id=l_link)
        T_l_link_to_dphys = self._get_path_transform(child_frame_id=l_link, frame_id=l_depth_phys)
        T_l_dphys_to_dopt = self._get_path_transform(child_frame_id=l_depth_phys, frame_id=l_depth_opt)
        
        T_cam_to_cam = self.cam_to_cam_cfg.transform_to_matrix()
        T_right_internal = T_r_cphys_to_copt @ T_r_link_to_cphys @ T_r_dphys_to_link @ T_r_dopt_to_dphys
        T_left_internal  = T_l_dphys_to_dopt @ T_l_link_to_dphys @ T_l_cphys_to_link @ T_l_copt_to_cphys
        full_matrix = T_left_internal @ T_cam_to_cam @ T_right_internal
        
        stamp = self.get_clock().now().to_msg()
        self.tf = matrix_to_tf_stamped(full_matrix, l_depth_opt, r_depth_opt, stamp)
        self.broadcaster.sendTransform(self.tf)
    
    def have_tf(self):
        return self.tf is not None

class CamToBaseTransformRefiner(Node):
    def __init__(self, first_cam_cfg: CamToBaseTransformConfig, second_cam_cfg: CamToBaseTransformConfig, filename: str):
        super().__init__('cam_to_base_transform_refiner')
        self.broadcaster = StaticTransformBroadcaster(self)
        
        self.first_cam_cfg = first_cam_cfg
        self.second_cam_cfg = second_cam_cfg
        self.first_cam_name = self.first_cam_cfg.camera_name
        self.second_cam_name = self.second_cam_cfg.camera_name
        self.filename = filename

        self.tf_first = None
        self.tf_second = None
        self.tf_corrected = None
        self.icp_done = False
        self.tf_lookup = {}

        # PointCloud buffers for ICP
        self.cloud_first = None
        self.cloud_second = None
        
        self.first_depth_opt = f"{self.first_cam_name}_depth_optical_frame"
        self.second_depth_opt = f"{self.second_cam_name}_depth_optical_frame"
        self.manual_target = 'first'

        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)

        self.sub_tf_static = self.create_subscription(TFMessage, "/tf_static", self.on_tf_static, qos)
        
        # PointCloud Subscriptions
        self.sub_pc_first = self.create_subscription(PointCloud2, f"/camera/{self.first_cam_name}/depth/color/points", self.on_pc_first, 10)
        self.sub_pc_second = self.create_subscription(PointCloud2, f"/camera/{self.second_cam_name}/depth/color/points", self.on_pc_second, 10)

    def set_manual_target(self, camera_name: str):
        if camera_name == self.first_cam_name:
            self.manual_target = 'first'
        elif camera_name == self.second_cam_name:
            self.manual_target = 'second'
        else:
            raise ValueError(f"Camera '{camera_name}' not found. Valid values: {self.first_cam_name}, {self.second_cam_name}")

    def get_manual_target_config(self):
        return self.first_cam_cfg if self.manual_target == 'first' else self.second_cam_cfg

    def on_pc_first(self, msg):
        if not self.icp_done and self.cloud_first is None:
            self.cloud_first = self.ros_to_o3d(msg)
            print (f"Received first point cloud with {len(self.cloud_first.points)} points")

    def on_pc_second(self, msg):
        if not self.icp_done and self.cloud_second is None:
            self.cloud_second = self.ros_to_o3d(msg)
            print (f"Received second point cloud with {len(self.cloud_second.points)} points")

    def ros_to_o3d(self, ros_pc):
        points = np.array(pc2.read_points(ros_pc, skip_nans=True, field_names=["x", "y", "z"]))

        # sensor_msgs_py may return a structured array (dtype with fields x/y/z),
        # which Open3D cannot consume directly.
        if points.dtype.fields is not None:
            points = np.column_stack((points["x"], points["y"], points["z"]))

        if points.size == 0:
            points = np.empty((0, 3), dtype=np.float64)
        else:
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def crop_to_box(self, pcd):
        """
        Crops points to a specific volume relative to the robot base.
        Example: 1m x 1m area in front of the robot, above the table.
        """
        # min_bound = np.array([-0.15, -0.8, -0.5])
        # max_bound = np.array([0.15, -0.1, -0.05])
        # min_bound = np.array([-0.15, -0.5, -0.25])
        # max_bound = np.array([0.15, -0.1, -0.05])
        # min_bound = np.array([-0.5, -1.5, -0.5])
        # max_bound = np.array([0.5, 0.0, 0.5])
        min_bound = np.array([-0.18, -1.7, -0.5])
        max_bound = np.array([0.18, -0.0, 0.5])

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        return pcd.crop(bbox)

    def run_icp_refinement(self, T_first_base, T_second_base, point_to_plane=True):
        if self.cloud_first is None or self.cloud_second is None:
            print ("Point clouds not yet available for ICP refinement.")
            return None

        self.get_logger().info("Starting ICP Refinement...")
        
        # 1. Transform clouds to base frame using current best estimate
        target = self.cloud_first.transform(T_first_base)
        source = self.cloud_second.transform(T_second_base)
        
        target = self.crop_to_box(target)
        source = self.crop_to_box(source)
        
        voxel_size = 0.002 # 0.001 worked well for PTP
        target = target.voxel_down_sample(voxel_size=voxel_size)
        source = source.voxel_down_sample(voxel_size=voxel_size)
        
        threshold = 0.05 # 0.02 worked well too
        trans_init = np.eye(4) # Initial relative guess is identity because they are already in base
        
        if point_to_plane:
            radius_normal = voxel_size * 3
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        # We assume the calibration is close, so use a small search threshold (max_correspondence_distance)
        
        if point_to_plane:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
            )
        else:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
            )

        correction = reg_p2p.transformation
        self.get_logger().info(f"ICP converged. Fitness: {reg_p2p.fitness}")
        return correction

    def on_tf_static(self, msg: TFMessage):
        for t in msg.transforms:
            self.tf_lookup[(t.header.frame_id, t.child_frame_id)] = transform_to_matrix(t)

    def _get_path_transform(self, frame_id, child_frame_id):
        if (frame_id, child_frame_id) in self.tf_lookup:
            return self.tf_lookup[(frame_id, child_frame_id)]
        elif (child_frame_id, frame_id) in self.tf_lookup:
            return np.linalg.inv(self.tf_lookup[(child_frame_id, frame_id)])
        return None

    def calculate_transform(self):
        first_depth_opt = self.first_depth_opt
        first_depth_phys = f"{self.first_cam_name}_depth_frame"
        first_link = f"{self.first_cam_name}_link"
        first_color_phys = f"{self.first_cam_name}_color_frame"
        first_color_opt = f"{self.first_cam_name}_color_optical_frame"

        second_depth_opt = self.second_depth_opt
        second_depth_phys = f"{self.second_cam_name}_depth_frame"
        second_link = f"{self.second_cam_name}_link"
        second_color_phys = f"{self.second_cam_name}_color_frame"
        second_color_opt = f"{self.second_cam_name}_color_optical_frame"

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

    def calculate_and_refine(self):
        if self.tf_first is None or self.tf_second is None:
            self.calculate_transform()
            print ("Initial cam-to-base transforms calculated. Waiting for point clouds to run ICP refinement...")
            return

        try:
            T_first_to_base_raw = transform_to_matrix(self.tf_first)
            T_second_to_base_raw = transform_to_matrix(self.tf_second)
            
            # 2. Attempt ICP if we have clouds
            if not self.icp_done and self.cloud_first is not None and self.cloud_second is not None:
                print ("Running ICP refinement on the initial cam-to-base transforms...")
                T_corr = self.run_icp_refinement(T_first_to_base_raw, T_second_to_base_raw)
                print ("ICP correction:")
                print (T_corr)
                
                if T_corr is not None:
                    # Apply correction to the second camera (from depth to base)
                    # T_new = T_base_old * T_correction
                    T_second_to_base_refined = T_corr @ T_second_to_base_raw
                    
                    stamp = self.get_clock().now().to_msg()
                    self.tf_first = matrix_to_tf_stamped(T_first_to_base_raw, self.first_cam_cfg.get_frame_id(), self.first_depth_opt, stamp)
                    self.tf_second = matrix_to_tf_stamped(T_second_to_base_refined, self.second_cam_cfg.get_frame_id(), self.second_depth_opt, stamp)
                    
                    self.icp_done = True
                    self.get_logger().info("ICP Refinement applied to static transforms.")
                    if self.filename is not None:
                        T_second_refined = T_corr @ self.second_cam_cfg.transform_to_matrix()
                        self.tf_corrected = matrix_to_tf_stamped(
                            T=T_second_refined, 
                            frame_id=self.second_cam_cfg.get_frame_id(),
                            child_frame_id=self.second_cam_cfg.get_child_frame_id(),
                            stamp=stamp
                        )
                        self.export_transform(self.filename)
                    self.destroy_subscription(self.sub_pc_first)
                    self.destroy_subscription(self.sub_pc_second)
                else:
                    stamp = self.get_clock().now().to_msg()
                    self.tf_corrected = matrix_to_tf_stamped(
                        T=self.second_cam_cfg.transform_to_matrix(),
                        frame_id=self.second_cam_cfg.get_frame_id(),
                        child_frame_id=self.second_cam_cfg.get_child_frame_id(),
                        stamp=stamp
                    )
                    self.get_logger().warn("ICP failed to find a valid correction. Keeping original transform.")

        except Exception as e:
            self.get_logger().warn(f"Refinement error: {e}")

    def have_tf(self):
        return self.tf_first is not None and self.tf_second is not None

    def publish_transform(self):
        if self.tf_first is None or self.tf_second is None:
            self.calculate_and_refine()
            return

        if not self.icp_done:
            self.calculate_and_refine()
            return
        
        # Once ICP is done, just keep heartbeating the refined transforms
        now = self.get_clock().now().to_msg()
        self.tf_first.header.stamp = now
        self.tf_second.header.stamp = now
        self.broadcaster.sendTransform([self.tf_first, self.tf_second])

    # Recompute cam-to-base transforms from current config values and publish.
    def update_and_publish_transform(self):
        first_depth_opt = self.first_depth_opt
        first_depth_phys = f"{self.first_cam_name}_depth_frame"
        first_link = f"{self.first_cam_name}_link"
        first_color_phys = f"{self.first_cam_name}_color_frame"
        first_color_opt = f"{self.first_cam_name}_color_optical_frame"

        second_depth_opt = self.second_depth_opt
        second_depth_phys = f"{self.second_cam_name}_depth_frame"
        second_link = f"{self.second_cam_name}_link"
        second_color_phys = f"{self.second_cam_name}_color_frame"
        second_color_opt = f"{self.second_cam_name}_color_optical_frame"

        T_first_dopt_to_dphys = self._get_path_transform(child_frame_id=first_depth_opt, frame_id=first_depth_phys)
        T_first_dphys_to_link = self._get_path_transform(child_frame_id=first_depth_phys, frame_id=first_link)
        T_first_link_to_cphys = self._get_path_transform(child_frame_id=first_link, frame_id=first_color_phys)
        T_first_cphys_to_copt = self._get_path_transform(child_frame_id=first_color_phys, frame_id=first_color_opt)

        T_second_dopt_to_dphys = self._get_path_transform(child_frame_id=second_depth_opt, frame_id=second_depth_phys)
        T_second_dphys_to_link = self._get_path_transform(child_frame_id=second_depth_phys, frame_id=second_link)
        T_second_link_to_cphys = self._get_path_transform(child_frame_id=second_link, frame_id=second_color_phys)
        T_second_cphys_to_copt = self._get_path_transform(child_frame_id=second_color_phys, frame_id=second_color_opt)

        internals = [
            T_first_dopt_to_dphys, T_first_dphys_to_link, T_first_link_to_cphys, T_first_cphys_to_copt,
            T_second_dopt_to_dphys, T_second_dphys_to_link, T_second_link_to_cphys, T_second_cphys_to_copt,
        ]
        if any(m is None for m in internals):
            self.get_logger().warn("Not all RealSense static TFs available yet")
            return

        T_first_to_base = self.first_cam_cfg.transform_to_matrix()
        T_second_to_base = self.second_cam_cfg.transform_to_matrix()

        T_first_internal = T_first_cphys_to_copt @ T_first_link_to_cphys @ T_first_dphys_to_link @ T_first_dopt_to_dphys
        T_second_internal = T_second_cphys_to_copt @ T_second_link_to_cphys @ T_second_dphys_to_link @ T_second_dopt_to_dphys

        first_full_matrix = T_first_to_base @ T_first_internal
        second_full_matrix = T_second_to_base @ T_second_internal

        stamp = self.get_clock().now().to_msg()
        self.tf_first = matrix_to_tf_stamped(first_full_matrix, self.first_cam_cfg.get_frame_id(), first_depth_opt, stamp)
        self.tf_second = matrix_to_tf_stamped(second_full_matrix, self.second_cam_cfg.get_frame_id(), second_depth_opt, stamp)
        self.broadcaster.sendTransform([self.tf_first, self.tf_second])

    def export_manual_target_transform(self, filename):
        self.tf_corrected = self.get_manual_target_config().tf
        self.export_transform(filename)
    
    def export_transform(self, filename):
        if self.filename is not None:
            print ('Saving refined cam-to-base transform to %s' % os.path.abspath(self.filename))
            qx = self.tf_corrected.transform.rotation.x
            qy = self.tf_corrected.transform.rotation.y
            qz = self.tf_corrected.transform.rotation.z
            qw = self.tf_corrected.transform.rotation.w
            roll, pitch, yaw = euler_from_quaternion(self.tf_corrected.transform.rotation)
            x = self.tf_corrected.transform.translation.x
            y = self.tf_corrected.transform.translation.y
            z = self.tf_corrected.transform.translation.z
            with open(self.filename, 'w') as f:
                yaml.dump({
                    'frame_id': self.tf_corrected.header.frame_id,
                    'child_frame_id': self.tf_corrected.child_frame_id,
                    'translation': {
                        'x': x,
                        'y': y,
                        'z': z,
                    },
                    'rotation': {
                        'x': qx,
                        'y': qy,
                        'z': qz,
                        'w': qw,
                    },
                    'euler': {
                        'roll': float(roll),
                        'pitch': float(pitch),
                        'yaw': float(yaw),
                    },
                    'euler_degrees': {
                        'roll': math.degrees(roll),
                        'pitch': math.degrees(pitch),
                        'yaw': math.degrees(yaw),
                    }
                }, f)

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def print_status(status):
    sys.stdout.write('%-8s%-8s%-8s%-40s\r' % (status['mode'], status[status['mode']]['value'], status[status['mode']]['step'], status['message']))

parser = argparse.ArgumentParser(description='Refines the transform between two cameras.')
parser.add_argument('--refinement_type', type=str, default='manual', help='Type of refinement to perform (manual, auto)')
parser.add_argument('--tf_type', type=str, default='cam-to-base', help='Type of TF to publish: cam-to-cam or cam-to-base. Default: cam-to-base')
parser.add_argument('--from_cam_name', type=str, default='right', help='Name of the first camera (child frame)')
parser.add_argument('--to_cam_name', type=str, default='left', help='Name of the second camera (parent frame)')
parser.add_argument('--cam_to_cam_calibration', type=str, default='cam_to_cam.yaml', help='Calibration file for the first camera')
parser.add_argument('--first_to_base_calibration', type=str, default='/home/student-team/.ros2/easy_handeye2/calibrations/calibration_right_camera.calib', 
                    help='Calibration file for the first camera (for cam-to-base TF)')
# parser.add_argument('--first_to_base_calibration', type=str, default='/home/student-team/rgmc-2026/utils/easy_handeye2_ws/right_refined_cam_to_base.yaml', 
#                     help='Calibration file for the first camera (for cam-to-base TF)')
# parser.add_argument('--second_to_base_calibration', type=str, default='/home/student-team/rgmc-2026/utils/easy_handeye2_ws/right_refined_cam_to_base.yaml', 
#                     help='Calibration file for the second camera (for cam-to-base TF)')
parser.add_argument('--second_to_base_calibration', type=str, default='/home/student-team/.ros2/easy_handeye2/calibrations/calibration_left_camera.calib', 
                    help='Calibration file for the second camera (for cam-to-base TF)')
parser.add_argument('--file', type=str, default='test_right_refined_cam_to_base.yaml', help='If given, refined calibration is exported to the given file.')

if __name__ == '__main__':
    args = parser.parse_args()
    
    rclpy.init()
    cam_to_cam_tf = None
    cam_to_cam_broadcaster = None
    cam_to_base_broadcaster = None
    
    from_cam, to_cam = args.from_cam_name, args.to_cam_name
    if args.tf_type == 'cam-to-cam':
        print('Mode: cam-to-cam.')
        cam_to_cam_calib = args.cam_to_cam_calibration
        cam_to_cam_tf = CamToCamTransformConfig(from_cam, to_cam, cam_to_cam_calib)
        cam_to_cam_broadcaster = CamToCamTransformRefiner(cam_to_cam_tf, filename=args.file)
    else:
        print('Mode: cam-to-base.')
        first_to_base_calib = args.first_to_base_calibration
        second_to_base_calib = args.second_to_base_calibration
        first_cam_cfg = CamToBaseTransformConfig(from_cam, first_to_base_calib)
        second_cam_cfg = CamToBaseTransformConfig(to_cam, second_to_base_calib)
        cam_to_base_broadcaster = CamToBaseTransformRefiner(first_cam_cfg, second_cam_cfg, filename=args.file)

    active_broadcaster = cam_to_cam_broadcaster if args.tf_type == 'cam-to-cam' else cam_to_base_broadcaster
    if active_broadcaster is None:
        print("ERROR: Failed to initialize broadcaster")
        sys.exit(-1)
    
    # Wait for RealSense static TFs to arrive
    print("Waiting for RealSense static TFs...")
    for i in range(50):  # Wait up to 5 seconds
        rclpy.spin_once(active_broadcaster, timeout_sec=0.1)
        active_broadcaster.calculate_transform()
        if active_broadcaster.have_tf():
            break
    
    if not active_broadcaster.have_tf():
        print("ERROR: Could not compute initial transform. Make sure RealSense cameras are publishing.")
        sys.exit(-1)
    
    print("Initial transform computed successfully!")
    
    if args.tf_type == 'cam-to-cam':
        if cam_to_cam_tf is None or cam_to_cam_broadcaster is None:
            print("ERROR: cam-to-cam mode not initialized correctly")
            sys.exit(-1)
        if args.refinement_type == 'manual':
            manual_cfg = cam_to_cam_tf
            export_fn = cam_to_cam_broadcaster.export_transform

            x, y, z = manual_cfg.get_translation()
            roll, pitch, yaw = manual_cfg.get_euler()
            status = {'mode': 'pitch',
                          'x': {'value': x, 'step': 0.005},
                          'y': {'value': y, 'step': 0.005},
                          'z': {'value': z, 'step': 0.005},
                          'azimuth': {'value': math.degrees(yaw), 'step': 0.5},
                          'pitch': {'value': math.degrees(pitch), 'step': 0.5},
                          'roll': {'value': math.degrees(roll), 'step': 0.5},
                          'message': ''}

            print()
            print ('Press the following keys to change mode: x, y, z, (a)zimuth, (p)itch, (r)oll')
            print ('For each mode, press 6 to increase by step and 4 to decrease')
            print ('Press + to multiply step by 2 or - to divide')
            print ('Press Q to quit')
            status_keys = [key[0] for key in status.keys()]
            print ('%-8s%-8s%-8s%s' % ('Mode', 'value', 'step', 'message'))
            print_status(status)

            while rclpy.ok():
                kk = getch()
                status['message'] = ''
                try:
                    key_idx = status_keys.index(kk)
                    status['mode'] = list(status.keys())[key_idx]
                except ValueError as e:
                    if kk.upper() == 'Q':
                        # Save the refined transform before quitting
                        export_fn(args.file)
                        sys.stdout.write('\n')
                        exit(0)
                    elif kk == '4':
                        status[status['mode']]['value'] -= status[status['mode']]['step']
                    elif kk == '6':
                        status[status['mode']]['value'] += status[status['mode']]['step']
                    elif kk == '-':
                        status[status['mode']]['step'] /= 2.0
                    elif kk == '+':
                        status[status['mode']]['step'] *= 2.0
                    else:
                        status['message'] = 'Invalid key:' + kk

                print_status(status)
                manual_cfg.set_translation(status['x']['value'], status['y']['value'], status['z']['value'])
                manual_cfg.set_rotation_from_euler(
                    math.radians(status['roll']['value']),
                    math.radians(status['pitch']['value']),
                    math.radians(status['azimuth']['value'])
                )
                cam_to_cam_broadcaster.update_and_publish_transform()
                rclpy.spin_once(cam_to_cam_broadcaster, timeout_sec=0.001)  # Keep ROS alive
        else:
            print("Running auto publish for cam-to-cam transform...")
            while rclpy.ok():
                cam_to_cam_broadcaster.update_and_publish_transform()
                rclpy.spin_once(cam_to_cam_broadcaster, timeout_sec=0.01)
    else:
        if cam_to_base_broadcaster is None:
            print("ERROR: cam-to-base mode not initialized correctly")
            sys.exit(-1)
        if args.refinement_type == 'manual':
            cam_to_base_broadcaster.set_manual_target(from_cam)
            manual_cfg = cam_to_base_broadcaster.get_manual_target_config()
            export_fn = cam_to_base_broadcaster.export_manual_target_transform
            print(f"Manual cam-to-base refinement target: {from_cam}")

            x, y, z = manual_cfg.get_translation()
            roll, pitch, yaw = manual_cfg.get_euler()
            status = {'mode': 'pitch',
                          'x': {'value': x, 'step': 0.005},
                          'y': {'value': y, 'step': 0.005},
                          'z': {'value': z, 'step': 0.005},
                          'azimuth': {'value': math.degrees(yaw), 'step': 0.5},
                          'pitch': {'value': math.degrees(pitch), 'step': 0.5},
                          'roll': {'value': math.degrees(roll), 'step': 0.5},
                          'message': ''}

            print()
            print ('Press the following keys to change mode: x, y, z, (a)zimuth, (p)itch, (r)oll')
            print ('For each mode, press 6 to increase by step and 4 to decrease')
            print ('Press + to multiply step by 2 or - to divide')
            print ('Press Q to quit')
            status_keys = [key[0] for key in status.keys()]
            print ('%-8s%-8s%-8s%s' % ('Mode', 'value', 'step', 'message'))
            print_status(status)

            while rclpy.ok():
                kk = getch()
                status['message'] = ''
                try:
                    key_idx = status_keys.index(kk)
                    status['mode'] = list(status.keys())[key_idx]
                except ValueError as e:
                    if kk.upper() == 'Q':
                        # Save the refined transform before quitting
                        export_fn(args.file)
                        sys.stdout.write('\n')
                        exit(0)
                    elif kk == '4':
                        status[status['mode']]['value'] -= status[status['mode']]['step']
                    elif kk == '6':
                        status[status['mode']]['value'] += status[status['mode']]['step']
                    elif kk == '-':
                        status[status['mode']]['step'] /= 2.0
                    elif kk == '+':
                        status[status['mode']]['step'] *= 2.0
                    else:
                        status['message'] = 'Invalid key:' + kk

                print_status(status)
                manual_cfg.set_translation(status['x']['value'], status['y']['value'], status['z']['value'])
                manual_cfg.set_rotation_from_euler(
                    math.radians(status['roll']['value']),
                    math.radians(status['pitch']['value']),
                    math.radians(status['azimuth']['value'])
                )
                cam_to_base_broadcaster.update_and_publish_transform()
                rclpy.spin_once(cam_to_base_broadcaster, timeout_sec=0.001)  # Keep ROS alive
        else:
            print("Running automatic refinement with ICP. This may take a while...")
            while rclpy.ok():
                cam_to_base_broadcaster.publish_transform()
                rclpy.spin_once(cam_to_base_broadcaster, timeout_sec=0.01)
