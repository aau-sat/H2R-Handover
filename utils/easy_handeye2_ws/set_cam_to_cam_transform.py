# Computes the transform between two cameras based on their transforms to a common base frame.
# Execute easy_handeye2 calibration first to get the calibration files for both cameras, 
# then run this node to compute and publish the cam-to-cam transform as a static TF.

import rclpy
from rclpy.node import Node
import sys
from geometry_msgs.msg import TransformStamped
import argparse

import os
import math
import yaml
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import numpy as np

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
    print (f"Setting TF with frame_id='{frame_id}' and child_frame_id='{child_frame_id}'")
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

class CameraTransformConfig:
    def __init__(self, camera_name, calib_file):
        self.camera_name = camera_name
        try:
            calib = yaml.safe_load(open(calib_file, 'r'))
        except IOError as e:
            print ('Failed reading calibration parameters from file %s' % calib_file)
            sys.exit(-1)
        self.tf = TransformStamped()
        self.tf.header.frame_id = calib['parameters']['tracking_base_frame']
        self.tf.child_frame_id = calib['parameters']['robot_base_frame']
        self.tf.transform.translation.x = calib['transform']['translation']['x']
        self.tf.transform.translation.y = calib['transform']['translation']['y']
        self.tf.transform.translation.z = calib['transform']['translation']['z']
        self.tf.transform.rotation.x = calib['transform']['rotation']['x']
        self.tf.transform.rotation.y = calib['transform']['rotation']['y']
        self.tf.transform.rotation.z = calib['transform']['rotation']['z']
        self.tf.transform.rotation.w = calib['transform']['rotation']['w']
    
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

class CamToCamTransformBroadcaster(Node):
    def __init__(self, from_cam_cfg: CameraTransformConfig, to_cam_cfg: CameraTransformConfig, filename: str = None):
        super().__init__('cam_to_cam_transform_broadcaster')
        self.broadcaster = StaticTransformBroadcaster(self)
        timer_period = 1.0  # seconds
        self.from_cam_cfg = from_cam_cfg
        self.to_cam_cfg = to_cam_cfg
        self.frame_id = self.to_cam_cfg.get_frame_id()
        self.child_frame_id = self.from_cam_cfg.get_frame_id()
        self.tf = self._calculate_cam_to_cam_transform(filename)
        self.timer = self.create_timer(timer_period, self.publish_transform)
    
    def _calculate_cam_to_cam_transform(self, filename: str = None) -> TransformStamped:
        try:
            # Note: calibration is tracking_base (color_opt) -> robot_base (base)
            T_from_cam_to_base = self.from_cam_cfg.transform_to_matrix()# Tright
            T_to_cam_to_base = self.to_cam_cfg.transform_to_matrix() # Tleft
            
            # We need: right_color_optical_frame -> base -> left_color_optical_frame
            final_matrix = np.linalg.inv(T_to_cam_to_base) @ T_from_cam_to_base
            
            stamp = self.get_clock().now().to_msg()
            self.tf = matrix_to_tf_stamped(final_matrix, self.frame_id, self.child_frame_id, stamp)
            if filename is not None:
                print ('Successfully computed full cam-to-cam transform:')
                print (final_matrix)
                print ('Exporting to file %s' % os.path.abspath(filename))
                roll, pitch, yaw = euler_from_quaternion(self.tf.transform.rotation)
                print (f"Euler angles (radians): roll={roll}, pitch={pitch}, yaw={yaw}")
                with open(filename, 'w') as f:
                    yaml.dump({
                        'frame_id': self.tf.header.frame_id,
                        'child_frame_id': self.tf.child_frame_id,
                        'translation': {
                            'x': self.tf.transform.translation.x,
                            'y': self.tf.transform.translation.y,
                            'z': self.tf.transform.translation.z,
                        },
                        'rotation': {
                            'x': self.tf.transform.rotation.x,
                            'y': self.tf.transform.rotation.y,
                            'z': self.tf.transform.rotation.z,
                            'w': self.tf.transform.rotation.w,
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
            self.get_logger().info("Successfully computed full cam-to-cam transform.")
            return self.tf

        except Exception as e:
            self.get_logger().warn(f"Waiting for more TFs... {e}")
            return None

    def publish_transform(self):
        if self.tf is None:
            print("No transform to publish yet.")
            return
        self.tf.header.stamp = self.get_clock().now().to_msg()
        self.broadcaster.sendTransform(self.tf)

parser = argparse.ArgumentParser(description='Set the transform between two cameras based on their transforms to a common base frame.')
parser.add_argument('--from_cam_name', type=str, default='right', help='Name of the first camera (child frame)')
parser.add_argument('--to_cam_name', type=str, default='left', help='Name of the second camera (dominant frame)')
parser.add_argument('--from_cam_calibration', type=str, default='/home/student-team/.ros2/easy_handeye2/calibrations/calibration_right_camera.calib', help='Calibration file for the first camera')
parser.add_argument('--to_cam_calibration', type=str, default='/home/student-team/.ros2/easy_handeye2/calibrations/calibration_left_camera.calib', help='Calibration file for the second camera')
parser.add_argument('--file', type=str, default='cam_to_cam.yaml', help='If given, cam->cam calibration is exported to the given file.')

if __name__ == '__main__':
    args = parser.parse_args()
    
    from_cam, to_cam = args.from_cam_name, args.to_cam_name
    from_cam_calib, to_cam_calib = args.from_cam_calibration, args.to_cam_calibration
    
    from_cam_to_base_tf = CameraTransformConfig(from_cam, from_cam_calib)
    to_cam_to_base_tf = CameraTransformConfig(to_cam, to_cam_calib)

    rclpy.init()
    broadcaster = CamToCamTransformBroadcaster(from_cam_to_base_tf, to_cam_to_base_tf, filename=args.file)
    rclpy.spin(broadcaster)
