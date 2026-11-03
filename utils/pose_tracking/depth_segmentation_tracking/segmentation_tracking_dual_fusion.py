#!/usr/bin/env python3

"""
Fuse object pose estimates from two segmentation+depth streams.

The node subscribes to left/right segmentation masks and aligned depth images,
computes per-camera 3D centroids, transforms both centroids into a shared target
frame (typically base), fuses the measurements, and tracks the fused pose with a
Kalman filter.
"""

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
from std_msgs.msg import Bool
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

        self.frames_lost = 0

        self.static_tf_map = {}
        self.left_depth_frame = None
        self.right_depth_frame = None
        self.missing_tf_warn_count = 0
        self.sync_count = 0
        self.pose_pub_count = 0
        self.last_sync_count = 0
        self.last_pose_pub_count = 0

        # self.handover_detected = False
        # self.waiting_for_stable_pose = False
        # self.stable_pose_logged = False
        # self.stable_pose_ref = None
        # self.stable_pose_start_time = None
        # self.stable_pose_sample_count = 20
        # self.start_pose_samples = []
        # self.start_pose_logged = False
        # self.collecting_delivery_pose = False
        # self.delivery_pose_samples = []

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
        # self.declare_parameter('handover_detected_topic', '/handover_detected')
        # self.declare_parameter('stable_pose_duration_s', 3.0)
        # self.declare_parameter('stable_pose_tolerance_m', 0.01)
        # self.declare_parameter('stable_pose_sample_count', 20)

        self.declare_parameter('sync_queue_size', 20)
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('min_contour_area', 50)
        self.declare_parameter('max_outlier_m', 0.5)
        self.declare_parameter('max_fused_disagreement_m', 0.25)
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
        # self.handover_detected_topic = str(
        #     self.get_parameter('handover_detected_topic').value or '/handover_detected'
        # )

        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value or 40)
        self.sync_slop = float(self.get_parameter('sync_slop').value or 0.03)
        self.min_contour_area = float(self.get_parameter('min_contour_area').value or 50)
        self.max_outlier_m = float(self.get_parameter('max_outlier_m').value or 0.5)
        self.max_fused_disagreement_m = float(
            self.get_parameter('max_fused_disagreement_m').value or 0.25
        )
        self.max_coast_frames = int(self.get_parameter('max_coast_frames').value or 12)
        # self.stable_pose_duration_s = float(
        #     self.get_parameter('stable_pose_duration_s').value or 3.0
        # )
        # self.stable_pose_tolerance_m = float(
        #     self.get_parameter('stable_pose_tolerance_m').value or 0.01
        # )
        # self.stable_pose_sample_count = int(
        #     self.get_parameter('stable_pose_sample_count').value or 20
        # )
        # if self.stable_pose_sample_count < 1:
        #     self.stable_pose_sample_count = 1
        self.debug_status_period_s = float(
            self.get_parameter('debug_status_period_s').value or 2.0
        )

        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], np.float32)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2

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
        # self.handover_sub = self.create_subscription(
        #     Bool,
        #     self.handover_detected_topic,
        #     self.handover_detected_callback,
        #     10,
        # )

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

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [
                self.mask_left_sub,
                self.depth_left_sub,
                self.mask_right_sub,
                self.depth_right_sub,
            ],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.ts.registerCallback(self.dual_image_callback)
        self.debug_timer = self.create_timer(self.debug_status_period_s, self.debug_status_callback)

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)

        self.get_logger().info(
            'Dual fusion estimator started with topics: '
            f'mask_left={self.mask_left_topic}, depth_left={self.depth_left_topic}, '
            f'mask_right={self.mask_right_topic}, depth_right={self.depth_right_topic}, '
            f'pose_out={self.pose_topic}, target_frame={self.target_frame}, '
            # f'handover_topic={self.handover_detected_topic}'
        )
        self.get_logger().info(
            f'Debug status timer enabled: period={self.debug_status_period_s:.1f}s, '
            # f'stable_pose_duration={self.stable_pose_duration_s:.2f}s, '
            # f'stable_pose_tolerance={self.stable_pose_tolerance_m:.4f}m, '
            # f'stable_pose_sample_count={self.stable_pose_sample_count}'
        )

    # def handover_detected_callback(self, msg: Bool):
    #     handover_now = bool(msg.data)

    #     if handover_now and not self.handover_detected:
    #         self.get_logger().info(
    #             'handover_detected=true received. Starting post-handover stable pose check.'
    #         )
    #         self.waiting_for_stable_pose = True
    #         self.stable_pose_logged = False
    #         self.stable_pose_ref = None
    #         self.stable_pose_start_time = None
    #         self.start_pose_samples = []
    #         self.start_pose_logged = False
    #         self.collecting_delivery_pose = False
    #         self.delivery_pose_samples = []
    #     elif (not handover_now) and self.handover_detected:
    #         self.get_logger().info('handover_detected=false received. Stable pose check reset.')
    #         self.waiting_for_stable_pose = False
    #         self.stable_pose_logged = False
    #         self.stable_pose_ref = None
    #         self.stable_pose_start_time = None
    #         self.start_pose_samples = []
    #         self.start_pose_logged = False
    #         self.collecting_delivery_pose = False
    #         self.delivery_pose_samples = []

    #     self.handover_detected = handover_now

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
        }
        if self.camera_info_right_count == 1:
            self.get_logger().info(f'Right camera info received from {self.camera_info_right_topic}')
            if self.info_right_sub is not None:
                self.destroy_subscription(self.info_right_sub)
                self.info_right_sub = None
                self.get_logger().info('Right camera info subscription closed after first message')

    def extract_camera_measurement(self, cv_mask, cv_depth, camera_info, last_u, last_v):
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

        inner_mask = cv2.erode(clean_mask, np.ones((5, 5), np.uint8), iterations=2)
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

    # def _update_stable_pose_check(self, xyz_target, stamp):
    #     if not self.waiting_for_stable_pose or self.stable_pose_logged:
    #         return

    #     pos = np.array(xyz_target, dtype=np.float32)
    #     current_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9

    #     if not self.start_pose_logged:
    #         self.start_pose_samples.append(pos.copy())
    #         if len(self.start_pose_samples) >= self.stable_pose_sample_count:
    #             start_mean = np.mean(np.stack(self.start_pose_samples, axis=0), axis=0)
    #             start_location_data = {
    #                 "start_location_x_mm": float(start_mean[0]) * 1000.0,
    #                 "start_location_y_mm": float(start_mean[1]) * 1000.0,
    #                 "start_location_z_mm": float(start_mean[2]) * 1000.0,
    #             }
    #             self.get_logger().get_child("SUBMISSION").info(
    #                 f"{json.dumps(start_location_data)}"
    #             )
    #             self.start_pose_logged = True

    #     if self.stable_pose_ref is None:
    #         self.stable_pose_ref = pos
    #         self.stable_pose_start_time = current_time
    #         return

    #     if self.stable_pose_start_time is None:
    #         self.stable_pose_start_time = current_time
    #         return

    #     drift = float(np.linalg.norm(pos - self.stable_pose_ref))
    #     if drift <= self.stable_pose_tolerance_m:
    #         elapsed = current_time - self.stable_pose_start_time
    #         if elapsed >= self.stable_pose_duration_s:
    #             if not self.collecting_delivery_pose:
    #                 self.collecting_delivery_pose = True
    #                 self.delivery_pose_samples = []

    #             self.delivery_pose_samples.append(pos.copy())
    #             if len(self.delivery_pose_samples) >= self.stable_pose_sample_count:
    #                 delivery_mean = np.mean(np.stack(self.delivery_pose_samples, axis=0), axis=0)
    #                 self.stable_pose_logged = True
    #                 self.waiting_for_stable_pose = False
    #                 self.collecting_delivery_pose = False
    #                 delivery_location_data = {
    #                     "delivery_location_est_x_mm": float(delivery_mean[0]) * 1000.0,
    #                     "delivery_location_est_y_mm": float(delivery_mean[1]) * 1000.0,
    #                     "delivery_location_est_z_mm": float(delivery_mean[2]) * 1000.0,
    #                 }
    #                 self.get_logger().get_child("SUBMISSION").info(
    #                     f"{json.dumps(delivery_location_data)}"
    #                 )
    #     else:
    #         # Pose moved too much; restart stability window from current pose.
    #         self.stable_pose_ref = pos
    #         self.stable_pose_start_time = current_time
    #         self.collecting_delivery_pose = False
    #         self.delivery_pose_samples = []

    def dual_image_callback(self, mask_left_msg, depth_left_msg, mask_right_msg, depth_right_msg):
        self.sync_count += 1
        self.sync_times.append(self._now_sec())

        if self.camera_info_left is None or self.camera_info_right is None:
            self.get_logger().warn('Waiting for both camera intrinsics...', throttle_duration_sec=2.0)
            return

        cv_mask_left = self.bridge.imgmsg_to_cv2(mask_left_msg, desired_encoding='passthrough')
        cv_depth_left = self.bridge.imgmsg_to_cv2(depth_left_msg, desired_encoding='passthrough')
        cv_mask_right = self.bridge.imgmsg_to_cv2(mask_right_msg, desired_encoding='passthrough')
        cv_depth_right = self.bridge.imgmsg_to_cv2(depth_right_msg, desired_encoding='passthrough')

        ok_left, xyz_left, uv_left_new, uv_left_dbg = self.extract_camera_measurement(
            cv_mask_left,
            cv_depth_left,
            self.camera_info_left,
            self.last_u_left,
            self.last_v_left,
        )
        ok_right, xyz_right, uv_right_new, uv_right_dbg = self.extract_camera_measurement(
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
                return
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
                    return

                self.kf.statePost[3, 0] *= 0.8
                self.kf.statePost[4, 0] *= 0.8
                self.kf.statePost[5, 0] *= 0.8

                smoothed_x = float(self.kf.statePost[0, 0])
                smoothed_y = float(self.kf.statePost[1, 0])
                smoothed_z = float(self.kf.statePost[2, 0])

        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = self.target_frame if self.target_frame else depth_left_msg.header.frame_id
        pose_msg.pose.position.x = smoothed_x
        pose_msg.pose.position.y = smoothed_y
        pose_msg.pose.position.z = smoothed_z
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)
        self.pose_pub_count += 1
        # self._update_stable_pose_check((smoothed_x, smoothed_y, smoothed_z), timestamp)

        # self.get_logger().info(
        #     f'Published fused pose #{self.pose_pub_count}: '
        #     f'x={smoothed_x:.4f}, y={smoothed_y:.4f}, z={smoothed_z:.4f}, '
        #     f'left_ok={ok_left}, right_ok={ok_right}, '
        #     f'left_uv={uv_left_dbg}, right_uv={uv_right_dbg}'
        # )
        # check


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
