#!/usr/bin/env python3

"""Estimate a 3D hand pose from the `/hand_tracker` JSON stream.

The node consumes the hand tracker output, selects the active camera view,
samples depth either around the reported hand center or around all reported
landmarks, and publishes a smoothed pose in a target frame.
"""

import json

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


def quat_to_rot(qx, qy, qz, qw):
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return np.eye(3, dtype=float)

    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def rot_to_quat(rotation_matrix):
    trace = float(np.trace(rotation_matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation_matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
            qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
            qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
            qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
            qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
            qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
            qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
            qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
            qz = 0.25 * scale

    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm


def transform_to_matrix(transform):
    matrix = np.eye(4, dtype=float)
    matrix[0:3, 0:3] = quat_to_rot(
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    )
    matrix[0:3, 3] = [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ]
    return matrix


def matrix_to_pose_stamped(matrix, frame_id, stamp):
    pose_msg = PoseStamped()
    pose_msg.header.stamp = stamp
    pose_msg.header.frame_id = frame_id

    qx, qy, qz, qw = rot_to_quat(matrix[0:3, 0:3])
    pose_msg.pose.position.x = float(matrix[0, 3])
    pose_msg.pose.position.y = float(matrix[1, 3])
    pose_msg.pose.position.z = float(matrix[2, 3])
    pose_msg.pose.orientation.x = float(qx)
    pose_msg.pose.orientation.y = float(qy)
    pose_msg.pose.orientation.z = float(qz)
    pose_msg.pose.orientation.w = float(qw)
    return pose_msg


class HandPoseFromTrackerNode(Node):
    def __init__(self):
        super().__init__('hand_pose_from_tracker')

        self.bridge = CvBridge()
        self.last_time = None
        self.frames_lost = 0
        self.missing_tf_warn_count = 0
        self.pose_pub_count = 0

        self.left_depth = None
        self.right_depth = None
        self.left_depth_stamp = None
        self.right_depth_stamp = None
        self.left_camera_info = None
        self.right_camera_info = None
        self.static_tf_map = {}

        self.declare_parameter('hand_tracker_topic', '/hand_tracker')
        self.declare_parameter('camera_left_depth', '/camera/left/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_right_depth', '/camera/right/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_left_info', '/camera/left/aligned_depth_to_color/camera_info')
        self.declare_parameter('camera_right_info', '/camera/right/aligned_depth_to_color/camera_info')
        self.declare_parameter('depth_left_frame_id', 'left_color_optical_frame')
        self.declare_parameter('depth_right_frame_id', 'right_color_optical_frame')
        self.declare_parameter('target_frame', 'base')
        self.declare_parameter('pose_topic', '/tracked_hand_pose')
        self.declare_parameter('tracking_mode', 'center')
        self.declare_parameter('depth_roi_radius_px', 6)
        self.declare_parameter('landmark_min_valid_points', 5)
        self.declare_parameter('max_outlier_m', 0.25)
        self.declare_parameter('max_coast_frames', 12)
        self.declare_parameter('debug_status_period_s', 2.0)
        self.declare_parameter('tf_static_topic', '/tf_static')

        self.hand_tracker_topic = str(self.get_parameter('hand_tracker_topic').value or '/hand_tracker')
        self.left_depth_topic = str(self.get_parameter('camera_left_depth').value)
        self.right_depth_topic = str(self.get_parameter('camera_right_depth').value)
        self.left_info_topic = str(self.get_parameter('camera_left_info').value)
        self.right_info_topic = str(self.get_parameter('camera_right_info').value)
        self.left_depth_frame = str(self.get_parameter('depth_left_frame_id').value or 'left_color_optical_frame')
        self.right_depth_frame = str(self.get_parameter('depth_right_frame_id').value or 'right_color_optical_frame')
        self.target_frame = str(self.get_parameter('target_frame').value or 'base')
        self.pose_topic = str(self.get_parameter('pose_topic').value or '/tracked_hand_pose')
        self.tracking_mode = str(self.get_parameter('tracking_mode').value or 'center').lower().strip()
        self.depth_roi_radius_px = int(self.get_parameter('depth_roi_radius_px').value or 6)
        self.landmark_min_valid_points = int(self.get_parameter('landmark_min_valid_points').value or 5)
        self.max_outlier_m = float(self.get_parameter('max_outlier_m').value or 0.25)
        self.max_coast_frames = int(self.get_parameter('max_coast_frames').value or 12)
        self.debug_status_period_s = float(self.get_parameter('debug_status_period_s').value or 2.0)
        self.tf_static_topic = str(self.get_parameter('tf_static_topic').value or '/tf_static')

        if self.tracking_mode not in ('center', 'landmarks'):
            self.get_logger().warn(f"Invalid tracking_mode '{self.tracking_mode}'. Using 'center'.")
            self.tracking_mode = 'center'

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)

        sensor_qos = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        tf_qos = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.hand_tracker_sub = self.create_subscription(
            String,
            self.hand_tracker_topic,
            self.hand_tracker_callback,
            10,
        )
        self.left_depth_sub = self.create_subscription(Image, self.left_depth_topic, self.left_depth_callback, sensor_qos)
        self.right_depth_sub = self.create_subscription(Image, self.right_depth_topic, self.right_depth_callback, sensor_qos)
        self.left_info_sub = self.create_subscription(CameraInfo, self.left_info_topic, self.left_info_callback, 10)
        self.right_info_sub = self.create_subscription(CameraInfo, self.right_info_topic, self.right_info_callback, 10)
        self.tf_static_sub = self.create_subscription(TFMessage, self.tf_static_topic, self.tf_static_callback, tf_qos)
        self.status_timer = self.create_timer(self.debug_status_period_s, self.debug_status_callback)

        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2

        self.get_logger().info(
            'Hand pose tracker started with '
            f'tracking_mode={self.tracking_mode}, target_frame={self.target_frame}, '
            f'pose_topic={self.pose_topic}'
        )

    def tf_static_callback(self, msg):
        for transform_stamped in msg.transforms:
            self.static_tf_map[(transform_stamped.header.frame_id, transform_stamped.child_frame_id)] = transform_to_matrix(
                transform_stamped.transform
            )

    def left_depth_callback(self, msg):
        self.left_depth = self._depth_msg_to_array(msg)
        self.left_depth_stamp = msg.header.stamp

    def right_depth_callback(self, msg):
        self.right_depth = self._depth_msg_to_array(msg)
        self.right_depth_stamp = msg.header.stamp

    def left_info_callback(self, msg):
        self.left_camera_info = self._camera_info_from_msg(msg)

    def right_info_callback(self, msg):
        self.right_camera_info = self._camera_info_from_msg(msg)

    def _camera_info_from_msg(self, msg):
        return {
            'fx': float(msg.k[0]),
            'fy': float(msg.k[4]),
            'cx': float(msg.k[2]),
            'cy': float(msg.k[5]),
            'width': int(msg.width),
            'height': int(msg.height),
        }

    def _depth_msg_to_array(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if depth.dtype == np.uint16 or depth.dtype == np.uint32 or depth.dtype == np.int16 or depth.dtype == np.int32:
            return depth.astype(np.float32) / 1000.0
        return depth.astype(np.float32)

    def _depth_at_pixel(self, depth_image, x, y):
        if depth_image is None:
            return None

        h, w = depth_image.shape[:2]
        x = int(round(x))
        y = int(round(y))
        radius = max(0, self.depth_roi_radius_px)
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)

        patch = depth_image[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None

        return float(np.median(valid))

    def _pixel_to_3d(self, u, v, depth_m, camera_info):
        x_m = ((float(u) - camera_info['cx']) * depth_m) / camera_info['fx']
        y_m = ((float(v) - camera_info['cy']) * depth_m) / camera_info['fy']
        return np.array([x_m, y_m, depth_m], dtype=np.float32)

    def _normalize_landmark_point(self, landmark, width, height):
        if isinstance(landmark, dict):
            x = landmark.get('x')
            y = landmark.get('y')
        elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
            x, y = landmark[0], landmark[1]
        else:
            return None

        if x is None or y is None:
            return None

        x = float(x)
        y = float(y)
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return int(round(x * (width - 1))), int(round(y * (height - 1)))

        return int(round(x)), int(round(y))

    def _select_source_view(self, payload):
        active_view = payload.get('active_view')
        if active_view in ('left', 'right'):
            view_data = payload.get(active_view, {}) or {}
            if view_data.get('detected', False):
                return active_view, view_data

        for view_name in ('left', 'right'):
            view_data = payload.get(view_name, {}) or {}
            if view_data.get('detected', False):
                return view_name, view_data

        return None, None

    def _view_cache_for(self, view_name):
        if view_name == 'left':
            return self.left_depth, self.left_camera_info, self.left_depth_frame, self.left_depth_stamp
        if view_name == 'right':
            return self.right_depth, self.right_camera_info, self.right_depth_frame, self.right_depth_stamp
        return None, None, None, None

    def _estimate_from_center(self, depth_image, camera_info, center_px):
        if center_px is None or camera_info is None or depth_image is None:
            return None

        u, v = int(center_px[0]), int(center_px[1])
        depth_m = self._depth_at_pixel(depth_image, u, v)
        if depth_m is None:
            return None

        if not (0 <= u < camera_info['width'] and 0 <= v < camera_info['height']):
            return None

        return self._pixel_to_3d(u, v, depth_m, camera_info)

    def _estimate_from_landmarks(self, depth_image, camera_info, landmarks):
        if not landmarks or depth_image is None or camera_info is None:
            return None

        h, w = depth_image.shape[:2]
        points_3d = []

        for landmark in landmarks:
            pixel = self._normalize_landmark_point(landmark, w, h)
            if pixel is None:
                continue

            u, v = pixel
            if not (0 <= u < w and 0 <= v < h):
                continue

            depth_m = self._depth_at_pixel(depth_image, u, v)
            if depth_m is None:
                continue

            points_3d.append(self._pixel_to_3d(u, v, depth_m, camera_info))

        if len(points_3d) < self.landmark_min_valid_points:
            return None

        return np.median(np.stack(points_3d, axis=0), axis=0).astype(np.float32)

    def get_transform_target_from_source(self, target_frame, source_frame):
        if (target_frame, source_frame) in self.static_tf_map:
            return self.static_tf_map[(target_frame, source_frame)]
        if (source_frame, target_frame) in self.static_tf_map:
            return np.linalg.inv(self.static_tf_map[(source_frame, target_frame)])
        return None

    def transform_measurement_to_target(self, xyz, source_frame, stamp):
        if not self.target_frame or self.target_frame == source_frame:
            return True, xyz

        t_target_source = self.get_transform_target_from_source(self.target_frame, source_frame)
        if t_target_source is None:
            self.missing_tf_warn_count += 1
            if self.missing_tf_warn_count == 1 or self.missing_tf_warn_count % 30 == 0:
                self.get_logger().warn(
                    f'Missing static transform for {self.target_frame} <-> {source_frame}. '
                    f'Waiting for TF on {self.tf_static_topic}.'
                )
            return False, (0.0, 0.0, 0.0)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = source_frame
        pose.pose.position.x = float(xyz[0])
        pose.pose.position.y = float(xyz[1])
        pose.pose.position.z = float(xyz[2])
        pose.pose.orientation.w = 1.0

        pose_matrix = np.eye(4, dtype=float)
        pose_matrix[0:3, 3] = [
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ]

        transformed_matrix = t_target_source @ pose_matrix
        pose_target = matrix_to_pose_stamped(transformed_matrix, self.target_frame, stamp)
        return True, (
            pose_target.pose.position.x,
            pose_target.pose.position.y,
            pose_target.pose.position.z,
        )

    def _track_measurement(self, xyz_target, stamp):
        current_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9

        if self.last_time is None:
            self.last_time = current_time
            self.kf.statePost = np.array(
                [[xyz_target[0]], [xyz_target[1]], [xyz_target[2]], [0.0], [0.0], [0.0]],
                dtype=np.float32,
            )
            self.kf.statePre = self.kf.statePost.copy()
            self.frames_lost = 0
            return np.array(xyz_target, dtype=np.float32)

        dt = current_time - self.last_time
        self.last_time = current_time
        self.kf.transitionMatrix[0, 3] = dt
        self.kf.transitionMatrix[1, 4] = dt
        self.kf.transitionMatrix[2, 5] = dt

        prediction = self.kf.predict()
        pred_pos = np.array([prediction[0, 0], prediction[1, 0], prediction[2, 0]], dtype=np.float32)
        measurement = np.array(xyz_target, dtype=np.float32)

        if np.linalg.norm(pred_pos - measurement) > self.max_outlier_m:
            self.frames_lost += 1
            if self.frames_lost > self.max_coast_frames:
                self.get_logger().warn('Hand lost for too long. Resetting tracker.')
                self.last_time = None
                self.frames_lost = 0
                return None

            self.kf.statePost[3, 0] *= 0.8
            self.kf.statePost[4, 0] *= 0.8
            self.kf.statePost[5, 0] *= 0.8
            return np.array([self.kf.statePost[0, 0], self.kf.statePost[1, 0], self.kf.statePost[2, 0]], dtype=np.float32)

        self.frames_lost = 0
        estimated_state = self.kf.correct(measurement.reshape(3, 1))
        return np.array(
            [estimated_state[0, 0], estimated_state[1, 0], estimated_state[2, 0]],
            dtype=np.float32,
        )

    def hand_tracker_callback(self, msg):
        try:
            payload = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Failed to parse /hand_tracker JSON payload: {exc}')
            return

        source_view, view_data = self._select_source_view(payload)
        if source_view is None or view_data is None:
            return

        depth_image, camera_info, source_frame, depth_stamp = self._view_cache_for(source_view)
        if depth_image is None or camera_info is None:
            return

        center_px = payload.get('leading_hand_center') or view_data.get('active_hand_center')
        landmarks = view_data.get('landmarks')

        if self.tracking_mode == 'landmarks':
            xyz_camera = self._estimate_from_landmarks(depth_image, camera_info, landmarks)
            if xyz_camera is None:
                xyz_camera = self._estimate_from_center(depth_image, camera_info, center_px)
        else:
            xyz_camera = self._estimate_from_center(depth_image, camera_info, center_px)
            if xyz_camera is None:
                xyz_camera = self._estimate_from_landmarks(depth_image, camera_info, landmarks)

        if xyz_camera is None:
            return

        stamp = depth_stamp if depth_stamp is not None else self.get_clock().now().to_msg()

        ok, xyz_target = self.transform_measurement_to_target(xyz_camera, source_frame, stamp)
        if not ok:
            return

        smoothed_xyz = self._track_measurement(xyz_target, stamp)
        if smoothed_xyz is None:
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.target_frame if self.target_frame else source_frame
        pose_msg.pose.position.x = float(smoothed_xyz[0])
        pose_msg.pose.position.y = float(smoothed_xyz[1])
        pose_msg.pose.position.z = float(smoothed_xyz[2])
        pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)
        self.pose_pub_count += 1

        self.get_logger().info(
            f'Published hand pose #{self.pose_pub_count}: '
            f'view={source_view}, mode={self.tracking_mode}, '
            f'x={pose_msg.pose.position.x:.4f}, y={pose_msg.pose.position.y:.4f}, '
            f'z={pose_msg.pose.position.z:.4f}'
        )

    def debug_status_callback(self):
        self.get_logger().info(
            'Status: '
            f'left_depth={self.left_depth is not None}, right_depth={self.right_depth is not None}, '
            f'left_info={self.left_camera_info is not None}, right_info={self.right_camera_info is not None}, '
            f'tf_edges={len(self.static_tf_map)}, pub={self.pose_pub_count}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = HandPoseFromTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()