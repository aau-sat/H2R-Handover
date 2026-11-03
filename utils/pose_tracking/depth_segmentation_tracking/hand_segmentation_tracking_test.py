#!/usr/bin/env python3

"""
Fuse object pose estimates from two segmentation+depth streams.

The node subscribes to left/right segmentation masks and aligned depth images,
computes per-camera 3D centroids, transforms both centroids into a shared target
frame (typically base), fuses the measurements, and tracks the fused pose with a
Kalman filter.

The same is done for hand tracking 
"""

import threading

import cv2
import json
import message_filters
import numpy as np
import rclpy
from collections import deque
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        return np.eye(3, dtype=float)

    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=float)


def rot_to_quat(R: np.ndarray):
    tr = float(np.trace(R))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return qx / n, qy / n, qz / n, qw / n


def transform_to_matrix(t) -> np.ndarray:
    tx = t.translation.x
    ty = t.translation.y
    tz = t.translation.z

    qx = t.rotation.x
    qy = t.rotation.y
    qz = t.rotation.z
    qw = t.rotation.w

    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = quat_to_rot(qx, qy, qz, qw)
    T[0:3, 3] = [tx, ty, tz]
    return T


def pose_to_matrix(p: PoseStamped) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = quat_to_rot(
        p.pose.orientation.x,
        p.pose.orientation.y,
        p.pose.orientation.z,
        p.pose.orientation.w,
    )
    T[0:3, 3] = [
        p.pose.position.x,
        p.pose.position.y,
        p.pose.position.z,
    ]
    return T


def matrix_to_pose_stamped(T: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    qx, qy, qz, qw = rot_to_quat(T[0:3, 0:3])
    msg.pose.position.x = float(T[0, 3])
    msg.pose.position.y = float(T[1, 3])
    msg.pose.position.z = float(T[2, 3])
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


class DualSegmentationPoseEstimator(Node):
    def __init__(self):
        super().__init__('dual_segmentation_pose_estimator')

        self.bridge = CvBridge()
        self.camera_info_left = None
        self.camera_info_right = None
        self.last_time = None

        self.last_u_left = None
        self.last_v_left = None
        self.last_u_right = None
        self.last_v_right = None
        # per-view last pixel for hand mask based tracking
        self.last_u_left_hand = None
        self.last_v_left_hand = None
        self.last_u_right_hand = None
        self.last_v_right_hand = None

        self.frames_lost = 0
        self.frames_lost_hand = 0
        self.last_time_hand = None
        self.max_lost_frames_segmentation = 5

        self.static_tf_map = {}
        self.left_depth_frame = None
        self.right_depth_frame = None
        self.missing_tf_warn_count = 0
        self.sync_count = 0
        self.pose_pub_count = 0
        self.last_sync_count = 0
        self.last_pose_pub_count = 0

        self.handover_detected = False
        
        self.object_pose = None
        self.hand_pose = None
        self.segmentation_frames_lost_counter = 0
        self.last_valid_distance_m = None
        self.fallback_triggered = False
        self.fallback_start_time = None
        
        self.hand_msg_lock = threading.Lock()
        
        
        
        
        self.camera_info_left_count = 0
        self.camera_info_right_count = 0
        self.tf_static_msg_count = 0

        self.fps_window_size = 120
        self.camera_info_left_times = deque(maxlen=self.fps_window_size)
        self.camera_info_right_times = deque(maxlen=self.fps_window_size)
        self.tf_static_times = deque(maxlen=self.fps_window_size)
        self.sync_times = deque(maxlen=self.fps_window_size)

        self.declare_parameter('mask_left_topic', '/segmentation/left')
        self.declare_parameter('depth_left_topic', '/camera/left/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_left_topic', '/camera/left/aligned_depth_to_color/camera_info')
        self.declare_parameter('mask_right_topic', '/segmentation/right')
        self.declare_parameter('depth_right_topic', '/camera/right/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_right_topic', '/camera/right/aligned_depth_to_color/camera_info')
        self.declare_parameter('depth_left_frame_id', 'left_color_optical_frame')
        self.declare_parameter('depth_right_frame_id', 'right_color_optical_frame')

        self.declare_parameter('pose_topic', '/goal_pose')
        self.declare_parameter('target_frame', 'base')
        self.declare_parameter('tf_static_topic', '/tf_static')
        self.declare_parameter('handover_detected_topic', '/handover_detected')
        self.declare_parameter('hand_tracker_topic', '/hand_tracker')
        self.declare_parameter('hand_mask_left_topic', '/hand_segmentation/left')
        self.declare_parameter('hand_mask_right_topic', '/hand_segmentation/right')
        self.declare_parameter('hand_pose_mode', 'mask')
        self.declare_parameter('tracking_mode', 'center')
        
        

        self.declare_parameter('depth_roi_radius_px', 6)
        self.declare_parameter('landmark_min_valid_points', 5)
        self.declare_parameter('max_outlier_m_hand', 0.25)
        self.declare_parameter('sync_queue_size', 20)
        self.declare_parameter('sync_slop', 0.03)
        self.declare_parameter('min_contour_area', 50)
        self.declare_parameter('max_outlier_m', 0.5)
        self.declare_parameter('max_fused_disagreement_m', 0.25) # TODO testing other values
        self.declare_parameter('max_coast_frames', 12)
        self.declare_parameter('debug_status_period_s', 2.0)


        self.mask_left_topic = str(self.get_parameter('mask_left_topic').value or '/segmentation/left')
        self.depth_left_topic = str(
            self.get_parameter('depth_left_topic').value
            or '/camera/left/aligned_depth_to_color/image_raw'
        )
        self.camera_info_left_topic = str(
            self.get_parameter('camera_info_left_topic').value
            or '/camera/left/aligned_depth_to_color/camera_info'
        )
        self.mask_right_topic = str(self.get_parameter('mask_right_topic').value or '/segmentation/right')
        self.depth_right_topic = str(
            self.get_parameter('depth_right_topic').value
            or '/camera/right/aligned_depth_to_color/image_raw'
        )
        self.camera_info_right_topic = str(
            self.get_parameter('camera_info_right_topic').value
            or '/camera/right/aligned_depth_to_color/camera_info'
        )
        self.left_depth_frame = str(
            self.get_parameter('depth_left_frame_id').value
            or 'left_color_optical_frame'
        )
        self.right_depth_frame = str(
            self.get_parameter('depth_right_frame_id').value
            or 'right_color_optical_frame'
        )

        self.pose_topic = str(self.get_parameter('pose_topic').value or '/goal_pose')
        self.target_frame = str(self.get_parameter('target_frame').value or 'base')
        self.tf_static_topic = str(self.get_parameter('tf_static_topic').value or '/tf_static')
        self.handover_detected_topic = str(
            self.get_parameter('handover_detected_topic').value or '/handover_detected'
        )
        
        self.hand_tracker_topic = str(self.get_parameter('hand_tracker_topic').value or '/hand_tracker')
        self.hand_mask_left_topic = str(self.get_parameter('hand_mask_left_topic').value or '/hand_segmentation/left')
        self.hand_mask_right_topic = str(self.get_parameter('hand_mask_right_topic').value or '/hand_segmentation/right')
        self.hand_pose_mode = str(self.get_parameter('hand_pose_mode').value or 'json').lower().strip()
        self.tracking_mode = str(self.get_parameter('tracking_mode').value or 'center').lower().strip()
        self.depth_roi_radius_px = int(self.get_parameter('depth_roi_radius_px').value or 6)
        self.landmark_min_valid_points = int(self.get_parameter('landmark_min_valid_points').value or 5)
        self.max_outlier_m_hand = float(self.get_parameter('max_outlier_m_hand').value or 0.25)

        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value or 40)
        self.sync_slop = float(self.get_parameter('sync_slop').value or 0.03)
        self.min_contour_area = float(self.get_parameter('min_contour_area').value or 50)
        self.max_outlier_m = float(self.get_parameter('max_outlier_m').value or 0.5)
        self.max_fused_disagreement_m = float(
            self.get_parameter('max_fused_disagreement_m').value or 0.25
        )
        self.max_coast_frames = int(self.get_parameter('max_coast_frames').value or 12)
        self.debug_status_period_s = float(
            self.get_parameter('debug_status_period_s').value or 2.0
        )
        
        

        # Object tracker Kalman filter
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], np.float32)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2

        # Hand tracker Kalman filter 
        self.kf_hand = cv2.KalmanFilter(6, 3)
        self.kf_hand.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], np.float32)
        self.kf_hand.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf_hand.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kf_hand.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
        
        #self._initialize_kf_tuning(self.kf_hand)
        #self._initialize_kf_tuning(self.kf)

        tf_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.tf_static_sub = self.create_subscription(
            TFMessage,
            self.tf_static_topic,
            self.tf_static_callback,
            tf_qos,
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.info_left_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_left_topic,
            self.info_left_callback,
            30,
        )
        self.info_right_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_right_topic,
            self.info_right_callback,
            30,
        )
        self.handover_sub = self.create_subscription(
            Bool,
            self.handover_detected_topic,
            self.handover_detected_callback,
            10,
        )

        self.mask_left_sub = message_filters.Subscriber(
            self,
            Image,
            self.mask_left_topic,
            qos_profile=sensor_qos,
        )
        self.depth_left_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_left_topic,
            qos_profile=sensor_qos,
        )
        self.mask_right_sub = message_filters.Subscriber(
            self,
            Image,
            self.mask_right_topic,
            qos_profile=sensor_qos,
        )
        self.depth_right_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_right_topic,
            qos_profile=sensor_qos,
        )
        # Hand segmentation mask subscribers (synchronized together with the other image topics)
        self.hand_mask_left_sub = message_filters.Subscriber(
            self,
            Image,
            self.hand_mask_left_topic,
            qos_profile=sensor_qos,
        )
        self.hand_mask_right_sub = message_filters.Subscriber(
            self,
            Image,
            self.hand_mask_right_topic,
            qos_profile=sensor_qos,
        )
        # Subscribe to hand_tracker separately (String messages have no header, can't be synchronized)
        self.latest_hand_tracker_msg = None
        self.hand_tracker_sub = self.create_subscription(
            String, self.hand_tracker_topic, self.hand_tracker_callback, sensor_qos)

        # Synchronize only image messages with proper timestamps
        # Synchronize image topics including hand masks
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [
                self.mask_left_sub,
                self.depth_left_sub,
                self.mask_right_sub,
                self.depth_right_sub,
                self.hand_mask_left_sub,
                self.hand_mask_right_sub,
            ],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.ts.registerCallback(self.dual_image_callback)
        self.debug_timer = self.create_timer(self.debug_status_period_s, self.debug_status_callback)

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.hand_pose_pub = self.create_publisher(PoseStamped, '/hand_pose', 10)
        self.object_pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)

        self.get_logger().info(
            'Dual fusion estimator started with topics: '
            f'mask_left={self.mask_left_topic}, depth_left={self.depth_left_topic}, '
            f'mask_right={self.mask_right_topic}, depth_right={self.depth_right_topic}, '
            f'pose_out={self.pose_topic}, target_frame={self.target_frame}, '
            f'hand_mask_left={self.hand_mask_left_topic}, hand_mask_right={self.hand_mask_right_topic}, '
            f'handover_topic={self.handover_detected_topic}, '
            f'hand_tracker_topic={self.hand_tracker_topic} (async subscription), hand_pose_mode={self.hand_pose_mode}'
        )
        self.get_logger().info(
            f'Debug status timer enabled: period={self.debug_status_period_s:.1f}s, '
            # f'stable_pose_duration={self.stable_pose_duration_s:.2f}s, '
            # f'stable_pose_tolerance={self.stable_pose_tolerance_m:.4f}m, '
            # f'stable_pose_sample_count={self.stable_pose_sample_count}'
        )

    def _initialize_kf_tuning(self, kf):
        """
        Sets up the initial covariance matrices for the Kalman Filter.
        
        Splits the process noise to allow fast velocity changes while keeping 
        position tracking smooth.
        """
        # Q: Process Noise. Higher velocity noise reduces lag for sudden hand movements.
        q_pos = 1e-3
        q_vel = 1e-1  
        kf.processNoiseCov = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float32)

        # R: Measurement Noise. Trusting the 3D cameras slightly more.
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 5e-3
        
    def handover_detected_callback(self, msg: Bool):
        handover_now = bool(msg.data)
        self.handover_detected = handover_now
        #close topic
        self.destroy_subscription(self.handover_sub)
        self.get_logger().info('Handover detection subscription closed after first message')

    def tf_static_callback(self, msg):
        self.tf_static_msg_count += 1
        self.tf_static_times.append(self._now_sec())
        for t in msg.transforms:
            parent = t.header.frame_id
            child = t.child_frame_id
            self.static_tf_map[(parent, child)] = transform_to_matrix(t.transform)
        if (self.target_frame, self.left_depth_frame) in self.static_tf_map or (self.left_depth_frame, self.target_frame) in self.static_tf_map:
            if (self.target_frame, self.right_depth_frame) in self.static_tf_map or (self.right_depth_frame, self.target_frame) in self.static_tf_map:
                self.tf_static_sub.destroy()

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

    
    
    
    
    
    
    
    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _window_fps(self, times):
        if len(times) < 2:
            return 0.0
        dt = times[-1] - times[0]
        if dt <= 1e-9:
            return 0.0
        return (len(times) - 1) / dt

    def debug_status_callback(self):
        sync_delta = self.sync_count - self.last_sync_count
        pub_delta = self.pose_pub_count - self.last_pose_pub_count
        self.last_sync_count = self.sync_count
        self.last_pose_pub_count = self.pose_pub_count

        cam_left_fps = self._window_fps(self.camera_info_left_times)
        cam_right_fps = self._window_fps(self.camera_info_right_times)
        tf_static_fps = self._window_fps(self.tf_static_times)
        sync_fps = self._window_fps(self.sync_times)

        self.get_logger().info(
            'Status: '
            f'Handover={"YES" if self.handover_detected else "no"}, '
            f'Tracking mode={"HAND" if self.fallback_triggered else "OBJECT"}, '
            f'camera_info(left={self.camera_info_left_count}, right={self.camera_info_right_count}), '
            f'tf_static_msgs={self.tf_static_msg_count}, tf_edges={len(self.static_tf_map)}, '
            f'sync={self.sync_count} (+{sync_delta}), pub={self.pose_pub_count} (+{pub_delta}), '
            f'fps(cam_l={cam_left_fps:.2f}, cam_r={cam_right_fps:.2f}, '
            f'tf_static={tf_static_fps:.2f}, sync={sync_fps:.2f})'
        )

        if self.camera_info_left_count == 0:
            self.get_logger().warn(
                f'No messages yet on left camera info topic: {self.camera_info_left_topic}'
            )
        if self.camera_info_right_count == 0:
            self.get_logger().warn(
                f'No messages yet on right camera info topic: {self.camera_info_right_topic}'
            )

        if self.sync_count == 0 and self.camera_info_left_count > 0 and self.camera_info_right_count > 0:
            self.get_logger().warn(
                'No synchronized 4-topic callback yet. '
                'Consider increasing sync_slop (e.g. 0.03-0.08) or checking timestamps.'
            )

    def get_transform_target_from_source(self, target_frame, source_frame):
        if (target_frame, source_frame) in self.static_tf_map:
            return self.static_tf_map[(target_frame, source_frame)]
        if (source_frame, target_frame) in self.static_tf_map:
            return np.linalg.inv(self.static_tf_map[(source_frame, target_frame)])
        return None

    def info_left_callback(self, msg):
        self.camera_info_left_count += 1
        self.camera_info_left_times.append(self._now_sec())
        self.camera_info_left = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5],
            'width': msg.width,
            'height': msg.height,
        }
        if self.camera_info_left_count == 1:
            self.get_logger().info(f'Left camera info received from {self.camera_info_left_topic}')
            if self.info_left_sub is not None:
                self.destroy_subscription(self.info_left_sub)
                self.info_left_sub = None
                self.get_logger().info('Left camera info subscription closed after first message')

    def info_right_callback(self, msg):
        self.camera_info_right_count += 1
        self.camera_info_right_times.append(self._now_sec())
        self.camera_info_right = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5],
            'width': msg.width,
            'height': msg.height,
        }
        if self.camera_info_right_count == 1:
            self.get_logger().info(f'Right camera info received from {self.camera_info_right_topic}')
            if self.info_right_sub is not None:
                self.destroy_subscription(self.info_right_sub)
                self.info_right_sub = None
                self.get_logger().info('Right camera info subscription closed after first message')

    def extract_camera_measurement(self, cv_mask, cv_depth, camera_info, last_u, last_v, erosion_kernel=(5, 5), erosion_iters=2):
        binary_mask = cv_mask.astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, (0.0, 0.0, 0.0), (last_u, last_v), (0, 0)

        if last_u is None or last_v is None:
            ref_u = binary_mask.shape[1] // 2
            ref_v = binary_mask.shape[0] // 2
        else:
            ref_u = last_u
            ref_v = last_v

        closest_contour = None
        min_dist = float('inf')
        best_M = None

        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            cu = int(M['m10'] / M['m00'])
            cv = int(M['m01'] / M['m00'])
            dist = float((cu - ref_u) ** 2 + (cv - ref_v) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_contour = contour
                best_M = M

        if closest_contour is None or best_M is None:
            return False, (0.0, 0.0, 0.0), (last_u, last_v), (0, 0)

        u = int(best_M['m10'] / best_M['m00'])
        v = int(best_M['m01'] / best_M['m00'])

        clean_mask = np.zeros_like(binary_mask)
        cv2.drawContours(clean_mask, [closest_contour], -1, 255, thickness=cv2.FILLED)

        kernel = np.ones(erosion_kernel, np.uint8)
        inner_mask = cv2.erode(clean_mask, kernel, iterations=erosion_iters)
        valid_depth_mask = (inner_mask > 0) & (cv_depth > 0) & np.isfinite(cv_depth)
        segmented_depths = cv_depth[valid_depth_mask]

        if len(segmented_depths) == 0:
            return False, (0.0, 0.0, 0.0), (last_u, last_v), (u, v)

        z_m = float(np.median(segmented_depths)) / 1000.0
        fx, fy = camera_info['fx'], camera_info['fy']
        cx, cy = camera_info['cx'], camera_info['cy']

        x_m = ((u - cx) * z_m) / fx
        y_m = ((v - cy) * z_m) / fy

        return True, (x_m, y_m, z_m), (u, v), (u, v)

    def transform_measurement_to_target(self, xyz, source_frame, stamp):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = source_frame
        pose.pose.position.x = float(xyz[0])
        pose.pose.position.y = float(xyz[1])
        pose.pose.position.z = float(xyz[2])
        pose.pose.orientation.w = 1.0

        if not self.target_frame or self.target_frame == source_frame:
            return True, xyz

        T_target_source = self.get_transform_target_from_source(self.target_frame, source_frame)
        if T_target_source is None:
            self.missing_tf_warn_count += 1
            if self.missing_tf_warn_count == 1 or self.missing_tf_warn_count % 30 == 0:
                self.get_logger().warn(
                    f'Missing static transform for {self.target_frame} <-> {source_frame}. '
                    f'Waiting for TF on {self.tf_static_topic}.'
                )
            return False, (0.0, 0.0, 0.0)

        T_pose_source = pose_to_matrix(pose)
        T_pose_target = T_target_source @ T_pose_source
        pose_target = matrix_to_pose_stamped(T_pose_target, self.target_frame, stamp)

        return True, (
            pose_target.pose.position.x,
            pose_target.pose.position.y,
            pose_target.pose.position.z,
        )
        
    def hand_tracker_callback(self, msg: String):
        """Async callback to store the latest hand_tracker message (no header, can't be synchronized)."""
        with self.hand_msg_lock:
            self.latest_hand_tracker_msg = msg

    def estimate_object_pose(self, mask_left_msg, depth_left_msg, mask_right_msg, depth_right_msg):
        cv_mask_left = self.bridge.imgmsg_to_cv2(mask_left_msg, desired_encoding='passthrough')
        cv_depth_left = self.bridge.imgmsg_to_cv2(depth_left_msg, desired_encoding='passthrough')
        cv_mask_right = self.bridge.imgmsg_to_cv2(mask_right_msg, desired_encoding='passthrough')
        cv_depth_right = self.bridge.imgmsg_to_cv2(depth_right_msg, desired_encoding='passthrough')

        ok_left, xyz_left, uv_left_new, _ = self.extract_camera_measurement(
            cv_mask_left,
            cv_depth_left,
            self.camera_info_left,
            self.last_u_left,
            self.last_v_left,
        )
        ok_right, xyz_right, uv_right_new, _ = self.extract_camera_measurement(
            cv_mask_right,
            cv_depth_right,
            self.camera_info_right,
            self.last_u_right,
            self.last_v_right,
        )

        if ok_left:
            self.last_u_left, self.last_v_left = uv_left_new
        if ok_right:
            self.last_u_right, self.last_v_right = uv_right_new

        valid_measurements = []
        timestamp = depth_left_msg.header.stamp

        if ok_left:
            ok_left_tf, xyz_left_tf = self.transform_measurement_to_target(
                xyz_left,
                depth_left_msg.header.frame_id,
                timestamp,
            )
            if ok_left_tf:
                valid_measurements.append(np.array(xyz_left_tf, dtype=np.float32))

        if ok_right:
            ok_right_tf, xyz_right_tf = self.transform_measurement_to_target(
                xyz_right,
                depth_right_msg.header.frame_id,
                timestamp,
            )
            if ok_right_tf:
                valid_measurements.append(np.array(xyz_right_tf, dtype=np.float32))

        measurement_valid = len(valid_measurements) > 0
        fused_measurement = np.zeros(3, dtype=np.float32)

        if len(valid_measurements) == 1:
            fused_measurement = valid_measurements[0]
        elif len(valid_measurements) == 2:
            disagreement = np.linalg.norm(valid_measurements[0] - valid_measurements[1])
            if disagreement <= self.max_fused_disagreement_m:
                fused_measurement = 0.5 * (valid_measurements[0] + valid_measurements[1])
            else:
                measurement_valid = False
                self.get_logger().warn(
                    f'Left/right fused measurement disagreement too high: {disagreement:.3f} m',
                    throttle_duration_sec=1.0,
                )

        current_time = depth_left_msg.header.stamp.sec + depth_left_msg.header.stamp.nanosec * 1e-9

        if self.last_time is None:
            if not measurement_valid:
                return None, ok_left, ok_right
            self.last_time = current_time
            self.kf.statePost = np.array(
                [[fused_measurement[0]], [fused_measurement[1]], [fused_measurement[2]], [0.0], [0.0], [0.0]],
                np.float32,
            )
            self.kf.statePre = self.kf.statePost.copy()
            smoothed_x = float(fused_measurement[0])
            smoothed_y = float(fused_measurement[1])
            smoothed_z = float(fused_measurement[2])
        else:
            dt = current_time - self.last_time
            self.last_time = current_time

            self.kf.transitionMatrix[0, 3] = dt
            self.kf.transitionMatrix[1, 4] = dt
            self.kf.transitionMatrix[2, 5] = dt

            prediction = self.kf.predict()

            if measurement_valid:
                pred_pos = np.array([prediction[0, 0], prediction[1, 0], prediction[2, 0]], dtype=np.float32)
                if np.linalg.norm(pred_pos - fused_measurement) > self.max_outlier_m:
                    measurement_valid = False

            if measurement_valid:
                estimated_state = self.kf.correct(
                    np.array([[fused_measurement[0]], [fused_measurement[1]], [fused_measurement[2]]], np.float32)
                )
                smoothed_x = float(estimated_state[0, 0])
                smoothed_y = float(estimated_state[1, 0])
                smoothed_z = float(estimated_state[2, 0])
                self.frames_lost = 0
            else:
                self.frames_lost += 1
                if self.frames_lost > self.max_coast_frames:
                    self.get_logger().warn('Object lost for too long. Resetting tracker.')
                    self.last_time = None
                    self.frames_lost = 0
                    self.last_u_left = None
                    self.last_v_left = None
                    self.last_u_right = None
                    self.last_v_right = None
                    return None, ok_left, ok_right

                self.kf.statePost[3, 0] *= 0.8
                self.kf.statePost[4, 0] *= 0.8
                self.kf.statePost[5, 0] *= 0.8

                smoothed_x = float(self.kf.statePost[0, 0])
                smoothed_y = float(self.kf.statePost[1, 0])
                smoothed_z = float(self.kf.statePost[2, 0])

        object_pose_msg = PoseStamped()
        object_pose_msg.header.stamp = timestamp
        object_pose_msg.header.frame_id = self.target_frame if self.target_frame else depth_left_msg.header.frame_id
        object_pose_msg.pose.position.x = smoothed_x
        object_pose_msg.pose.position.y = smoothed_y
        object_pose_msg.pose.position.z = smoothed_z
        object_pose_msg.pose.orientation.x = 0.0
        object_pose_msg.pose.orientation.y = 0.0
        object_pose_msg.pose.orientation.z = 0.0
        object_pose_msg.pose.orientation.w = 1.0

        return object_pose_msg, ok_left, ok_right

    def _apply_sequential_kf_update(self, valid_measurements: list, dt: float, kf_instance) -> np.ndarray:
        """
        Applies multiple sensor measurements sequentially to the provided Kalman Filter.

        This approach avoids naive averaging and mathematically optimally fuses data 
        based on the filter's covariance matrices.

        Args:
            valid_measurements: List of valid 3D measurements (numpy arrays) from cameras.
            dt: Time delta since the last update in seconds.
            kf_instance: The Kalman filter instance to use (e.g., self.kf or self.kf_hand).

        Returns:
            np.ndarray: A 1D numpy array of shape (3,) representing the smoothed 
            position state [x, y, z]. Returns None if all measurements were outliers.
        """
        # Update transition matrix for the physical model (constant velocity)
        kf_instance.transitionMatrix[0, 3] = dt
        kf_instance.transitionMatrix[1, 4] = dt
        kf_instance.transitionMatrix[2, 5] = dt

        # 1. Predict phase (done only once per time step)
        prediction = kf_instance.predict()
        pred_pos = np.array([prediction[0, 0], prediction[1, 0], prediction[2, 0]], dtype=np.float32)
        
        # 2. Filter outliers and Correct phase (sequential)
        estimated_state = kf_instance.statePre  # Fallback to prediction if all measurements are outliers
        valid_corrections = 0

       # 2. Filter outliers and Correct phase (sequential)
        for i, measurement in enumerate(valid_measurements):
            # Outlier rejection based on euclidean distance from prediction
            if np.linalg.norm(pred_pos - measurement) <= self.max_outlier_m_hand:
                
                # Crucial Fix: For 2nd+ measurements, promote the posterior to prior
                if valid_corrections > 0:
                    kf_instance.statePre = kf_instance.statePost.copy()
                    kf_instance.errorCovPre = kf_instance.errorCovPost.copy()

                meas_mat = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], np.float32)
                estimated_state = kf_instance.correct(meas_mat)
                valid_corrections += 1

            # If no measurements were valid (all outliers), return None to trigger coasting
            if valid_corrections == 0:
                return None

        smoothed = np.array([estimated_state[0, 0], estimated_state[1, 0], estimated_state[2, 0]], dtype=np.float32)
        return smoothed

    def estimate_hand_pose(self, depth_left_msg, depth_right_msg, hand_mask_left_msg=None, hand_mask_right_msg=None):
        """
        Estimates the 3D hand pose by sequentially fusing measurements from two depth cameras.

        Args:
            depth_left_msg: The depth image message from the left camera.
            depth_right_msg: The depth image message from the right camera.

        Returns:
            tuple: A tuple containing:
                - PoseStamped: The estimated hand pose in the target frame, or None if lost.
                - str: A string indicating the source view used ('left', 'right', 'fused', or None).
        """
        # Read latest hand tracker payload (async subscription)
        if self.latest_hand_tracker_msg is None:
            return None, None

        try:
            with self.hand_msg_lock:
                payload = json.loads(self.latest_hand_tracker_msg.data) if self.latest_hand_tracker_msg.data else {}
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Failed to parse /hand_tracker JSON payload: {exc}')
            return None, None

        # Prepare per-view inputs
        left_view = payload.get('left', {}) or {}
        right_view = payload.get('right', {}) or {}

        # Mask-based mode: use hand segmentation masks and only the active view
        if self.hand_pose_mode == 'mask' and hand_mask_left_msg is not None and hand_mask_right_msg is not None:
            try:
                active_view, _ = self._select_source_view(payload)
            except Exception:
                active_view = None

            # convert masks and depths similar to object path
            cv_mask_left = self.bridge.imgmsg_to_cv2(hand_mask_left_msg, desired_encoding='passthrough')
            cv_depth_left = self.bridge.imgmsg_to_cv2(depth_left_msg, desired_encoding='passthrough')
            cv_mask_right = self.bridge.imgmsg_to_cv2(hand_mask_right_msg, desired_encoding='passthrough')
            cv_depth_right = self.bridge.imgmsg_to_cv2(depth_right_msg, desired_encoding='passthrough')

            valid_measurements = []
            source_view_used = None

            stamp = depth_left_msg.header.stamp if depth_left_msg is not None else self.get_clock().now().to_msg()

            # Only consider the active view; fallback to left or right if active not provided
            if active_view is None:
                # prefer leading_hand_center field or fallback to left
                active_view = payload.get('active_view') or ('left' if left_view else 'right')

            if active_view == 'left':
                ok, xyz, uv_new, _ = self.extract_camera_measurement(
                    cv_mask_left,
                    cv_depth_left,
                    self.camera_info_left,
                    self.last_u_left_hand,
                    self.last_v_left_hand,
                    erosion_kernel=(7, 7),
                    erosion_iters=3,
                )
                if ok:
                    self.last_u_left_hand, self.last_v_left_hand = uv_new
                    ok_tf, xyz_tf = self.transform_measurement_to_target(xyz, depth_left_msg.header.frame_id, stamp)
                    if ok_tf:
                        valid_measurements.append(np.array(xyz_tf, dtype=np.float32))
                        source_view_used = 'left'

            elif active_view == 'right':
                ok, xyz, uv_new, _ = self.extract_camera_measurement(
                    cv_mask_right,
                    cv_depth_right,
                    self.camera_info_right,
                    self.last_u_right_hand,
                    self.last_v_right_hand,
                    erosion_kernel=(7, 7),
                    erosion_iters=3,
                )
                if ok:
                    self.last_u_right_hand, self.last_v_right_hand = uv_new
                    ok_tf, xyz_tf = self.transform_measurement_to_target(xyz, depth_right_msg.header.frame_id, stamp)
                    if ok_tf:
                        valid_measurements.append(np.array(xyz_tf, dtype=np.float32))
                        source_view_used = 'right'

            # If no valid measurement from mask-mode, bail out
            if len(valid_measurements) == 0:
                return None, None

            # Kalman filter handling (hand-specific)
            current_time = stamp.sec + stamp.nanosec * 1e-9
            if self.last_time_hand is None:
                self.last_time_hand = current_time
                init_meas = valid_measurements[0]
                self.kf_hand.statePost = np.array(
                    [[init_meas[0]], [init_meas[1]], [init_meas[2]], [0.0], [0.0], [0.0]],
                    np.float32,
                )
                self.kf_hand.statePre = self.kf_hand.statePost.copy()
                smoothed = init_meas
                self.frames_lost_hand = 0
            else:
                dt = current_time - self.last_time_hand
                self.last_time_hand = current_time
                smoothed = self._apply_sequential_kf_update(valid_measurements, dt, self.kf_hand)
                if smoothed is None:
                    self.frames_lost_hand += 1
                    if self.frames_lost_hand > self.max_coast_frames:
                        self.get_logger().warn('Hand lost for too long. Resetting tracker.')
                        self.last_time_hand = None
                        self.frames_lost_hand = 0
                        return None, source_view_used
                    self.kf_hand.statePost[3, 0] *= 0.8
                    self.kf_hand.statePost[4, 0] *= 0.8
                    self.kf_hand.statePost[5, 0] *= 0.8
                    smoothed = np.array([self.kf_hand.statePost[0, 0], self.kf_hand.statePost[1, 0], self.kf_hand.statePost[2, 0]], dtype=np.float32)

            hand_pose_msg = PoseStamped()
            hand_pose_msg.header.stamp = stamp
            hand_pose_msg.header.frame_id = self.target_frame if self.target_frame else depth_left_msg.header.frame_id
            hand_pose_msg.pose.position.x = float(smoothed[0])
            hand_pose_msg.pose.position.y = float(smoothed[1])
            hand_pose_msg.pose.position.z = float(smoothed[2])
            hand_pose_msg.pose.orientation.w = 1.0

            return hand_pose_msg, source_view_used

        # Non-mask mode: use JSON landmarks/center as before

        current_time = stamp.sec + stamp.nanosec * 1e-9

        # Handle Kalman Filter initialization (hand-specific state)
        if self.last_time_hand is None:
            if len(valid_measurements) == 0:
                return None, source_view_used
            
            self.last_time_hand = current_time
            # Initialize with the first available valid measurement
            init_meas = valid_measurements[0] 
            self.kf_hand.statePost = np.array(
                [[init_meas[0]], [init_meas[1]], [init_meas[2]], [0.0], [0.0], [0.0]],
                np.float32,
            )
            self.kf_hand.statePre = self.kf_hand.statePost.copy()
            smoothed = init_meas
            self.frames_lost_hand = 0
            
        else:
            dt = current_time - self.last_time_hand
            self.last_time_hand = current_time

            smoothed = None
            if len(valid_measurements) > 0:
                smoothed = self._apply_sequential_kf_update(valid_measurements, dt, self.kf_hand)

            # If smoothed is None, all measurements were rejected as outliers or list was empty. Apply coasting.
            if smoothed is not None:
                self.frames_lost_hand = 0
            else:
                self.frames_lost_hand += 1
                if self.frames_lost_hand > self.max_coast_frames:
                    self.get_logger().warn('Hand lost for too long. Resetting tracker.')
                    self.last_time_hand = None
                    self.frames_lost_hand = 0
                    return None, source_view_used

                # Coasting logic: smoothly decay velocity
                self.kf_hand.statePost[3, 0] *= 0.8
                self.kf_hand.statePost[4, 0] *= 0.8
                self.kf_hand.statePost[5, 0] *= 0.8

                smoothed = np.array([self.kf_hand.statePost[0, 0], self.kf_hand.statePost[1, 0], self.kf_hand.statePost[2, 0]], dtype=np.float32)

        hand_pose_msg = PoseStamped()
        hand_pose_msg.header.stamp = stamp
        hand_pose_msg.header.frame_id = self.target_frame if self.target_frame else depth_left_msg.header.frame_id
        hand_pose_msg.pose.position.x = float(smoothed[0])
        hand_pose_msg.pose.position.y = float(smoothed[1])
        hand_pose_msg.pose.position.z = float(smoothed[2])
        hand_pose_msg.pose.orientation.w = 1.0

        return hand_pose_msg, source_view_used
    
    def add_offset_to_hand_pose(self, hand_pose_msg):
        hand_pose_msg.pose.position.x += float(self.last_valid_distance_m[0]) if self.last_valid_distance_m is not None else 0.0
        hand_pose_msg.pose.position.y += float(self.last_valid_distance_m[1]) if self.last_valid_distance_m is not None else 0.0
        hand_pose_msg.pose.position.z += float(self.last_valid_distance_m[2]) if self.last_valid_distance_m is not None else 0.0
        
        return hand_pose_msg

    def dual_image_callback(self, mask_left_msg, depth_left_msg, mask_right_msg, depth_right_msg, hand_mask_left_msg, hand_mask_right_msg):
        self.sync_count += 1
        self.sync_times.append(self._now_sec())

        if self.camera_info_left is None or self.camera_info_right is None:
            self.get_logger().warn('Waiting for both camera intrinsics...', throttle_duration_sec=2.0)
            return

        object_pose_msg, ok_left, ok_right = self.estimate_object_pose(
            mask_left_msg,
            depth_left_msg,
            mask_right_msg,
            depth_right_msg,
        )
        hand_pose_msg, source_view = self.estimate_hand_pose(depth_left_msg, depth_right_msg, hand_mask_left_msg, hand_mask_right_msg)
        if object_pose_msg is not None:
            self.object_pose_pub.publish(object_pose_msg)
        if hand_pose_msg is not None:
            self.hand_pose_pub.publish(hand_pose_msg)
        if object_pose_msg is not None and hand_pose_msg is not None:
            try:
                object_pos = np.array([
                    object_pose_msg.pose.position.x,
                    object_pose_msg.pose.position.y,
                    object_pose_msg.pose.position.z,
                ])
                hand_pos = np.array([
                    hand_pose_msg.pose.position.x,
                    hand_pose_msg.pose.position.y,
                    hand_pose_msg.pose.position.z,
                ])
                distance = object_pos - hand_pos
                self.last_valid_distance_m = distance
                distance_scalar = float(np.linalg.norm(distance))
                self.get_logger().debug(f'Distance between hand and object: {distance_scalar:.4f} m')
            except Exception as exc:
                self.get_logger().warn(f'Failed to calculate distance between hand and object: {exc}')
        if self.fallback_triggered:
            #self.get_logger().info(f'Fallback active')
            if hand_pose_msg is not None:
                hand_pose_msg = self.add_offset_to_hand_pose(hand_pose_msg)
                self.pose_pub.publish(hand_pose_msg)
                self.pose_pub_count += 1
                return  
            else:
                if object_pose_msg is not None:
                    self.pose_pub.publish(object_pose_msg)
                    self.pose_pub_count += 1
                return
            
        if not ok_left and not ok_right:
            self.get_logger().warn(
                'No valid segmentation measurement from either camera. Hand pose estimation may be inaccurate.',
                throttle_duration_sec=2.0,
            )
            self.segmentation_frames_lost_counter += 1
            if self.segmentation_frames_lost_counter > self.max_lost_frames_segmentation:
                self.fallback_triggered = True
                self.fallback_start_time = self._now_sec()
                self.get_logger().warn(
                    'Segmentation lost for too long. Falling back to hand pose on /goal_pose.'
                )
                if hand_pose_msg is not None:
                    hand_pose_msg = self.add_offset_to_hand_pose(hand_pose_msg)
                    self.pose_pub.publish(hand_pose_msg)
                    self.pose_pub_count += 1
                    return

        else:
            self.segmentation_frames_lost_counter = 0
            self.fallback_triggered = False
            if object_pose_msg is not None:
                self.pose_pub.publish(object_pose_msg)
                self.pose_pub_count += 1
            elif hand_pose_msg is not None:
                self.get_logger().warn(
                    'Object pose unavailable in normal path; publishing hand pose instead.',
                    throttle_duration_sec=1.0,
                )
                hand_pose_msg = self.add_offset_to_hand_pose(hand_pose_msg)
                self.pose_pub.publish(hand_pose_msg)
                self.pose_pub_count += 1
            else:
                self.get_logger().warn(
                    'Neither object nor hand pose available for publish in this cycle.',
                    throttle_duration_sec=1.0,
                )
        

def main(args=None):
    rclpy.init(args=args)
    node = DualSegmentationPoseEstimator()
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
