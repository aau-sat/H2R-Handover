#!/usr/bin/env python3

import json
import os
import socket
import threading

import message_filters
import numpy as np
import open3d as o3d  # type: ignore[reportMissingImports]
import rclpy
import time
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visual_properties.msg import Dimensions  # type: ignore[reportMissingImports]
import torch

import sys
sys.path.insert(0, '..')
from utils.pc_utils import *


def notify_popup(title, body):
    try:
        import subprocess

        env = os.environ.copy()
        env.setdefault('DISPLAY', ':0')
        if not env.get('DBUS_SESSION_BUS_ADDRESS'):
            runtime_bus = f"unix:path=/run/user/{os.getuid()}/bus"
            if os.path.exists(f"/run/user/{os.getuid()}/bus"):
                env['DBUS_SESSION_BUS_ADDRESS'] = runtime_bus
        subprocess.run(['notify-send', title, body], env=env, check=False)
    except Exception:
        pass


class GraspGenerator(Node):
    """ROS2 node that continuously builds filtered point clouds and sends them to the grasp service."""

    def __init__(self):
        super().__init__('grasp_pc_filter_tuner')

        self.use_two_views = self._param_bool('use_two_views', True)
        self.camera_name_first = self._param_str('camera_name_first', 'left')
        self.camera_name_second = self._param_str('camera_name_second', 'right')
        self.object_label = self._param_str('object_label', 'cup')
        self.object_label_topic = self._param_str('object_label_topic', '/detected_label')
        self.transparent = self._param_bool('transparent', True)
        self.dimensions_topic = self._param_str('dimensions_topic', '/estimated_dimensions')
        self.hand_tracker_topic = self._param_str('hand_tracker_topic', '/hand_tracker')
        self.handover_detected_topic = self._param_str('handover_detected_topic', '/handover_detected')
        self.request_period_sec = self._param_float('request_period_sec', 0.25)

        self.hand_knn_enabled = self._param_bool('hand_knn_enabled', True)
        self.hand_knn_threshold = self._param_float('hand_knn_threshold', 0.01)
        self.hand_knn_k = self._param_int('hand_knn_k', 4)
        self.hand_dbscan_enabled = self._param_bool('hand_dbscan_enabled', True)
        self.hand_dbscan_eps = self._param_float('hand_dbscan_eps', 0.05)
        self.hand_dbscan_min_points = self._param_int('hand_dbscan_min_points', 5)

        self.object_knn_enabled = self._param_bool('object_knn_enabled', True)
        self.object_knn_threshold = self._param_float('object_knn_threshold', 0.005)
        self.object_knn_k = self._param_int('object_knn_k', 10)
        self.object_dbscan_enabled = self._param_bool('object_dbscan_enabled', True)
        self.object_dbscan_eps = self._param_float('object_dbscan_eps', 0.02)
        self.object_dbscan_min_points = self._param_int('object_dbscan_min_points', 10)

        self.first_depth_topic = f'/camera/{self.camera_name_first}/aligned_depth_to_color/image_raw'
        self.second_depth_topic = f'/camera/{self.camera_name_second}/aligned_depth_to_color/image_raw'
        self.first_color_topic = f'/camera/{self.camera_name_first}/color/image_raw'
        self.second_color_topic = f'/camera/{self.camera_name_second}/color/image_raw'
        self.first_camera_info_topic = f'/camera/{self.camera_name_first}/aligned_depth_to_color/camera_info'
        self.second_camera_info_topic = f'/camera/{self.camera_name_second}/aligned_depth_to_color/camera_info'

        self.target_frame = 'base'
        self.source_frame_first = f'{self.camera_name_first}_color_optical_frame'
        self.source_frame_second = f'{self.camera_name_second}_color_optical_frame'

        self.bridge = CvBridge()

        self.latest_depth_first = None
        self.latest_depth_second = None
        self.latest_color_first = None
        self.latest_color_second = None

        self.left_hand_visibility_score = 0
        self.right_hand_visibility_score = 0
        self.preferred_single_view_side = 'left'
        self.visibility_counting_enabled = False

        self.latest_dimensions = None
        self.axis_default_world = self._param_array('axis_default_world', [0.0, 0.0, 1.0])

        self.first_camera_initialized = False
        self.second_camera_initialized = False

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.cb_group = ReentrantCallbackGroup()
        self.request_lock = threading.Lock()
        self.request_timer = None

        self.first_cam_info_sub = self.create_subscription(
            CameraInfo,
            self.first_camera_info_topic,
            self.first_camera_info_callback,
            qos_profile=qos_profile,
            callback_group=self.cb_group,
        )
        self.second_cam_info_sub = self.create_subscription(
            CameraInfo,
            self.second_camera_info_topic,
            self.second_camera_info_callback,
            qos_profile=qos_profile,
            callback_group=self.cb_group,
        )

        self.first_depth_sub = message_filters.Subscriber(self, Image, self.first_depth_topic, qos_profile=qos_profile)
        self.first_color_sub = message_filters.Subscriber(self, Image, self.first_color_topic, qos_profile=qos_profile)
        sync_topic_list_first = [self.first_depth_sub, self.first_color_sub]

        self.second_depth_sub = message_filters.Subscriber(self, Image, self.second_depth_topic, qos_profile=qos_profile)
        self.second_color_sub = message_filters.Subscriber(self, Image, self.second_color_topic, qos_profile=qos_profile)
        sync_topic_list_second = [self.second_depth_sub, self.second_color_sub]

        self.ts_first = message_filters.ApproximateTimeSynchronizer(
            sync_topic_list_first,
            queue_size=10,
            slop=0.05,
        )
        self.ts_first.registerCallback(self.sync_callback_first)

        self.ts_second = message_filters.ApproximateTimeSynchronizer(
            sync_topic_list_second,
            queue_size=10,
            slop=0.05,
        )
        self.ts_second.registerCallback(self.sync_callback_second)

        self.object_label_sub = self.create_subscription(
            String,
            self.object_label_topic,
            self.object_label_callback,
            10,
            callback_group=self.cb_group,
        )

        self.dimensions_sub = self.create_subscription(
            Dimensions,
            self.dimensions_topic,
            self.dimensions_callback,
            10,
            callback_group=self.cb_group,
        )

        self.handover_detected_sub = self.create_subscription(
            Bool,
            self.handover_detected_topic,
            self.handover_detected_callback,
            qos_profile=qos_profile,
            callback_group=self.cb_group,
        )

        if not self.use_two_views:
            self.hand_tracker_sub = self.create_subscription(
                String,
                self.hand_tracker_topic,
                self.hand_tracker_callback,
                10,
                callback_group=self.cb_group,
            )
        else:
            self.hand_tracker_sub = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_first = None
        self.tf_second = None

        self.grasp_socket_path = '/tmp/graspgen'
        self.sam_socket_path = '/tmp/sam3_socket'

        # Start a dedicated background thread that performs requests sequentially
        self._shutdown = False
        self.request_thread = threading.Thread(target=self._request_thread, daemon=True)
        self.request_thread.start()

        self.get_logger().info(
            'Pointcloud filter tuning node initialized with '
            f'hand(KNN={self.hand_knn_enabled}, DBSCAN={self.hand_dbscan_enabled}) '
            f'object(KNN={self.object_knn_enabled}, DBSCAN={self.object_dbscan_enabled})'
        )

    def _param_bool(self, name, default):
        value = self.declare_parameter(name, default).value
        return bool(default if value is None else value)

    def _param_float(self, name, default):
        value = self.declare_parameter(name, default).value
        return float(default if value is None else value)

    def _param_int(self, name, default):
        value = self.declare_parameter(name, default).value
        return int(default if value is None else value)

    def _param_str(self, name, default):
        value = self.declare_parameter(name, default).value
        return str(default if value is None else value)

    def _param_array(self, name, default):
        value = self.declare_parameter(name, default).value
        return default if value is None else value

    def handover_detected_callback(self, msg):
        if not bool(msg.data):
            return

        self.visibility_counting_enabled = True
        if self.handover_detected_sub is not None:
            self.destroy_subscription(self.handover_detected_sub)
            self.handover_detected_sub = None
        self.get_logger().info('handover_detected=true received; starting hand visibility scoring and removing handover subscription.')

    def dimensions_callback(self, msg):
        self.get_logger().info('Received dimensions update.')
        notify_popup('RGMC Vision', 'System is ready, grasp can be started')

        def _safe_float(value):
            try:
                value = float(value)
            except Exception:
                return None
            return value if np.isfinite(value) and value > 0 else None

        self.latest_dimensions = {
            'height': _safe_float(msg.height),
            'top_w': _safe_float(msg.top_w),
            'bot_w': _safe_float(msg.bot_w),
        }
        if self.dimensions_sub is not None:
            self.destroy_subscription(self.dimensions_sub)
            self.dimensions_sub = None

    def object_label_callback(self, msg):
        label = (msg.data or '').strip()
        if not label or label.lower() == 'none':
            return

        self.object_label = label
        self.get_logger().info(f'Updated object label from topic: {self.object_label}')
        if self.object_label_sub is not None:
            self.destroy_subscription(self.object_label_sub)
            self.object_label_sub = None

    def hand_tracker_callback(self, msg):
        if self.use_two_views or not self.visibility_counting_enabled:
            return

        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f'Failed to parse hand tracker payload: {exc}')
            return

        more_hand_visible = int(payload.get('more_hand_visible', 0))
        if more_hand_visible == 1:
            self.left_hand_visibility_score += 1
        elif more_hand_visible == 2:
            self.right_hand_visibility_score += 1
        else:
            self.left_hand_visibility_score += 1
            self.right_hand_visibility_score += 1

        self.preferred_single_view_side = (
            'left'
            if self.left_hand_visibility_score >= self.right_hand_visibility_score
            else 'right'
        )

    def _single_view_ready(self, side):
        if side == 'left':
            return (
                self.first_camera_initialized and
                self.latest_depth_first is not None and
                self.latest_color_first is not None and
                self.tf_first is not None
            )
        return (
            self.second_camera_initialized and
            self.latest_depth_second is not None and
            self.latest_color_second is not None and
            self.tf_second is not None
        )

    def _preferred_single_view(self):
        if self.preferred_single_view_side == 'right' and self._single_view_ready('right'):
            return 'right'
        if self.preferred_single_view_side == 'left' and self._single_view_ready('left'):
            return 'left'
        if self._single_view_ready('left'):
            return 'left'
        if self._single_view_ready('right'):
            return 'right'
        return None

    def first_camera_info_callback(self, msg):
        self.ffx = msg.k[0]
        self.ffy = msg.k[4]
        self.fcx = msg.k[2]
        self.fcy = msg.k[5]
        self.first_camera_initialized = True
        self.get_logger().info(f'Camera intrinsics received: fx={self.ffx}, fy={self.ffy}, cx={self.fcx}, cy={self.fcy}')
        if self.first_cam_info_sub is not None:
            self.destroy_subscription(self.first_cam_info_sub)
            self.first_cam_info_sub = None

    def second_camera_info_callback(self, msg):
        self.sfx = msg.k[0]
        self.sfy = msg.k[4]
        self.scx = msg.k[2]
        self.scy = msg.k[5]
        self.second_camera_initialized = True
        self.get_logger().info(f'Camera intrinsics received: fx={self.sfx}, fy={self.sfy}, cx={self.scx}, cy={self.scy}')
        if self.second_cam_info_sub is not None:
            self.destroy_subscription(self.second_cam_info_sub)
            self.second_cam_info_sub = None

    def sync_callback_first(self, depth_first_msg, color_first_msg):
        if self.tf_first is None:
            try:
                self.tf_first = self.tf_buffer.lookup_transform(
                    target_frame=self.target_frame,
                    source_frame=self.source_frame_first,
                    time=Time(),
                    timeout=Duration(nanoseconds=100_000_000),
                )
            except Exception as ex:
                self.get_logger().info(
                    f'Could not find transform from {self.source_frame_first} to {self.target_frame}: {ex}'
                )

        if self.tf_first is not None and self.tf_second is not None:
            self.tf_listener.unregister()

        try:
            self.latest_depth_first = self.bridge.imgmsg_to_cv2(depth_first_msg, '32FC1') / 1e3
            self.latest_color_first = self.bridge.imgmsg_to_cv2(color_first_msg, 'rgb8')
        except Exception as exc:
            self.get_logger().error(f'Error converting first depth image: {exc}')

    def sync_callback_second(self, depth_second_msg, color_second_msg):
        if self.tf_second is None:
            try:
                self.tf_second = self.tf_buffer.lookup_transform(
                    target_frame=self.target_frame,
                    source_frame=self.source_frame_second,
                    time=Time(),
                    timeout=Duration(nanoseconds=100_000_000),
                )
            except Exception as ex:
                self.get_logger().info(
                    f'Could not find transform from {self.source_frame_second} to {self.target_frame}: {ex}'
                )

        if self.tf_first is not None and self.tf_second is not None:
            self.tf_listener.unregister()

        try:
            self.latest_depth_second = self.bridge.imgmsg_to_cv2(depth_second_msg, '32FC1') / 1e3
            self.latest_color_second = self.bridge.imgmsg_to_cv2(color_second_msg, 'rgb8')
        except Exception as exc:
            self.get_logger().error(f'Error converting second depth image: {exc}')

    def _recv_exact(self, client, size):
        data = b''
        while len(data) < size:
            chunk = client.recv(size - len(data))
            if not chunk:
                raise ConnectionError('Connection closed while receiving data from segmentation service')
            data += chunk
        return data

    def request_segmentation_masks(self, color_image, label=None, bbox=None):
        if label is None:
            label = self.object_label

        h, w, _ = color_image.shape

        if bbox is None:
            x1, y1, x2, y2 = 0, 0, 0, 0
        else:
            x1, y1, x2, y2 = bbox

        label_bytes = label.encode()
        label_len = len(label_bytes)

        client = None
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self.sam_socket_path)

            header = np.array([h, w, x1, y1, x2, y2, label_len], dtype=np.int32)
            client.sendall(header.tobytes())
            client.sendall(label_bytes)
            client.sendall(color_image.tobytes())

            response = self._recv_exact(client, 8)
            object_ok = bool(np.frombuffer(response[:4], dtype=np.uint32)[0])
            hand_ok = bool(np.frombuffer(response[4:], dtype=np.uint32)[0])

            masks = []
            for _ in range(2):
                size_bytes = self._recv_exact(client, 4)
                payload_size = int(np.frombuffer(size_bytes, dtype=np.uint32)[0])
                mask_data = self._recv_exact(client, payload_size)
                masks.append(np.frombuffer(mask_data, dtype=np.uint8).reshape((h, w)))

            object_mask = masks[0] if len(masks) > 0 else None
            hand_mask = masks[1] if len(masks) > 1 else None

            self.get_logger().info(
                f'Segmentation masks received successfully: object_ok={object_ok}, hand_ok={hand_ok}'
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to connect to SAM3 server: {exc}')
            object_ok = False
            hand_ok = False
            object_mask = None
            hand_mask = None
        finally:
            if client is not None:
                client.close()

        if object_mask is not None:
            object_mask = object_mask.astype(bool).astype(np.uint8)
        if hand_mask is not None:
            hand_mask = hand_mask.astype(bool).astype(np.uint8)

        return {
            'object_ok': object_ok,
            'hand_ok': hand_ok,
            'object_mask': object_mask,
            'hand_mask': hand_mask,
        }

    def _transform_points_to_target(self, points, tf_msg):
        if points is None:
            return None

        points = np.asarray(points, dtype=np.float32)
        if len(points) == 0 or tf_msg is None:
            return points

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = Rotation.from_quat([
            tf_msg.transform.rotation.x,
            tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z,
            tf_msg.transform.rotation.w,
        ]).as_matrix()
        tf_matrix[:3, 3] = [
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z,
        ]
        point_cloud.transform(tf_matrix)
        return np.asarray(point_cloud.points)

    def _apply_dbscan(self, points, eps=0.02, min_points=10):
        if points is None or len(points) <= min_points:
            return points

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float32))
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

        if len(labels) == 0:
            return points

        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            return points

        counts = np.bincount(valid_labels)
        largest_cluster_label = np.argmax(counts)
        indices = np.where(labels == largest_cluster_label)[0]
        return points[indices]

    def _apply_knn_filter(self, points, threshold, k):
        if points is None:
            return None

        points = np.asarray(points, dtype=np.float32)
        if len(points) == 0 or k <= 0 or len(points) <= k:
            return points

        try:
            filtered = point_cloud_outlier_removal(torch.from_numpy(points).float(), threshold=threshold, K=k)
            if hasattr(filtered, 'detach'):
                filtered = filtered.detach()
            return filtered.cpu().numpy() if hasattr(filtered, 'cpu') else np.asarray(filtered, dtype=np.float32)
        except Exception as exc:
            self.get_logger().warn(f'KNN filtering failed, keeping original points: {exc}')
            return points

    def _filter_point_cloud(self, points, kind):
        if points is None:
            return None

        points = np.asarray(points, dtype=np.float32)
        if kind == 'hand':
            if self.hand_knn_enabled:
                points = self._apply_knn_filter(points, self.hand_knn_threshold, self.hand_knn_k)
            if self.hand_dbscan_enabled:
                points = self._apply_dbscan(points, eps=self.hand_dbscan_eps, min_points=self.hand_dbscan_min_points)
        else:
            if self.object_knn_enabled:
                points = self._apply_knn_filter(points, self.object_knn_threshold, self.object_knn_k)
            if self.object_dbscan_enabled:
                points = self._apply_dbscan(points, eps=self.object_dbscan_eps, min_points=self.object_dbscan_min_points)
        return points

    def _get_fixed_world_axis(self):
        try:
            default_axis = np.asarray(self.axis_default_world, dtype=np.float32)
        except Exception:
            default_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if np.linalg.norm(default_axis) < 1e-6:
            default_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        default_axis /= np.linalg.norm(default_axis)
        return default_axis

    def _build_completion_metadata(self, object_pc):
        if not self.transparent:
            return None

        dims = self.latest_dimensions or {}
        points = np.asarray(object_pc, dtype=np.float32) if object_pc is not None else np.zeros((0, 3), dtype=np.float32)
        center = points.mean(axis=0).astype(np.float32) if len(points) > 0 else np.zeros(3, dtype=np.float32)
        axis_world = self._get_fixed_world_axis()

        observed_height = None
        if len(points) > 0:
            axis_proj = points @ axis_world
            observed_height = float(np.percentile(axis_proj, 99.0) - np.percentile(axis_proj, 1.0))
            if not np.isfinite(observed_height) or observed_height <= 0:
                observed_height = float(axis_proj.max() - axis_proj.min()) if len(axis_proj) > 0 else None
            if observed_height is not None and (not np.isfinite(observed_height) or observed_height <= 0):
                observed_height = None

        dim_height = dims.get('height')
        if dim_height is not None:
            try:
                dim_height = float(dim_height)
            except Exception:
                dim_height = None
        if dim_height is not None and (not np.isfinite(dim_height) or dim_height <= 0):
            dim_height = None

        if dim_height is not None and observed_height is not None:
            if observed_height < 0.6 * dim_height:
                constrained_height = dim_height
            else:
                constrained_height = min(dim_height, observed_height)
        else:
            constrained_height = dim_height if dim_height is not None else observed_height

        if dims.get('top_w') is None or dims.get('bot_w') is None:
            self.get_logger().warn(
                f"Incomplete dimension metadata before send: height={dims.get('height')}, "
                f"top_w={dims.get('top_w')}, bot_w={dims.get('bot_w')}"
            )

        return {
            'transparent': True,
            'object_label': self.object_label,
            'axis_world': axis_world.tolist(),
            'center_world': center.tolist(),
            'height': dims.get('height'),
            'top_w': dims.get('top_w'),
            'bot_w': dims.get('bot_w'),
        }

    def _build_view_point_clouds(self, depth_image, color_image, fx, fy, cx, cy, tf_msg, view_name):
        masks = self.request_segmentation_masks(color_image, label=self.object_label)
        object_mask = masks['object_mask']
        hand_mask = masks['hand_mask']

        if not masks['object_ok'] or object_mask is None:
            self.get_logger().warn(f'Object mask unavailable for {view_name} view; skipping grasp generation')
            return None

        depth_image = depth_image.astype(np.float32).reshape(depth_image.shape)

        if masks['hand_ok'] and hand_mask is not None:
            try:
                hand_pc, object_pc = depth_and_segmentation_to_point_clouds(
                    depth_image=depth_image,
                    segmentation_mask_obj=object_mask,
                    segmentation_mask_hand=hand_mask,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    target_object_id=1,
                )
                self.get_logger().info(
                    f'Generated point clouds ({view_name}): hand={hand_pc.shape}, object={object_pc.shape}'
                )
            except Exception as exc:
                self.get_logger().error(f'Failed to generate {view_name}-view point clouds: {exc}')
                return None
        else:
            self.get_logger().warn(
                f'Hand mask unavailable for {view_name} view; creating empty scene point cloud'
            )
            try:
                _, object_pc = depth_and_segmentation_to_point_clouds_old(
                    depth_image=depth_image,
                    segmentation_mask=object_mask,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    target_object_id=1,
                    remove_object_from_scene=True,
                )
            except Exception as exc:
                self.get_logger().error(f'Failed to generate {view_name}-view object point cloud: {exc}')
                return None
            hand_pc = np.empty((0, 3), dtype=np.float32)

        if hand_pc is not None and len(hand_pc) > 0:
            hand_pc = self._filter_point_cloud(hand_pc, 'hand')
        else:
            hand_pc = np.empty((0, 3), dtype=np.float32)

        if object_pc is None or len(object_pc) == 0:
            self.get_logger().warn(f'Object mask produced no points for {view_name} view')
            return None

        object_pc = self._filter_point_cloud(object_pc, 'object')

        hand_pc = self._transform_points_to_target(hand_pc, tf_msg)
        object_pc = self._transform_points_to_target(object_pc, tf_msg)

        return hand_pc, object_pc

    def _get_view_candidates(self):
        first_view_ready = (
            self.first_camera_initialized and
            self.latest_depth_first is not None and
            self.latest_color_first is not None and
            self.tf_first is not None
        )
        second_view_ready = (
            self.second_camera_initialized and
            self.latest_depth_second is not None and
            self.latest_color_second is not None and
            self.tf_second is not None
        )

        if not self.use_two_views:
            preferred_view = self._preferred_single_view()
            if preferred_view == 'left' and first_view_ready:
                self.get_logger().info(
                    f'Single-view mode selecting left view based on hand visibility scores '
                    f'(left={self.left_hand_visibility_score}, right={self.right_hand_visibility_score})'
                )
                return [(
                    'first',
                    self.latest_depth_first,
                    self.latest_color_first,
                    self.ffx, self.ffy, self.fcx, self.fcy,
                    self.tf_first,
                )]
            if preferred_view == 'right' and second_view_ready:
                self.get_logger().info(
                    f'Single-view mode selecting right view based on hand visibility scores '
                    f'(left={self.left_hand_visibility_score}, right={self.right_hand_visibility_score})'
                )
                return [(
                    'second',
                    self.latest_depth_second,
                    self.latest_color_second,
                    self.sfx, self.sfy, self.scx, self.scy,
                    self.tf_second,
                )]

            if first_view_ready:
                return [(
                    'first',
                    self.latest_depth_first,
                    self.latest_color_first,
                    self.ffx, self.ffy, self.fcx, self.fcy,
                    self.tf_first,
                )]
            if second_view_ready:
                return [(
                    'second',
                    self.latest_depth_second,
                    self.latest_color_second,
                    self.sfx, self.sfy, self.scx, self.scy,
                    self.tf_second,
                )]
            return []

        view_candidates = []
        if first_view_ready:
            view_candidates.append((
                'first',
                self.latest_depth_first,
                self.latest_color_first,
                self.ffx, self.ffy, self.fcx, self.fcy,
                self.tf_first,
            ))
        if second_view_ready:
            view_candidates.append((
                'second',
                self.latest_depth_second,
                self.latest_color_second,
                self.sfx, self.sfy, self.scx, self.scy,
                self.tf_second,
            ))
        return view_candidates

    def send_grasp_request(self, scene_pc, object_pc, completion_meta=None):
        if not os.path.exists(self.grasp_socket_path):
            self.get_logger().warn(f'Socket {self.grasp_socket_path} not found; skipping grasp request')
            return None, None

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            client.connect(self.grasp_socket_path)

            meta_payload = b''
            if completion_meta is not None:
                meta_payload = json.dumps(completion_meta, separators=(',', ':')).encode('utf-8')

            scene_size = scene_pc.shape[0]
            object_size = object_pc.shape[0]

            header = np.array([scene_size, object_size, len(meta_payload)], dtype=np.int32)
            client.sendall(header.tobytes())
            client.sendall(scene_pc.astype(np.float32).tobytes())
            client.sendall(object_pc.astype(np.float32).tobytes())

            if meta_payload:
                client.sendall(meta_payload)

            response = b''
            while len(response) < 4:
                chunk = client.recv(4 - len(response))
                if not chunk:
                    break
                response += chunk
            if len(response) < 4:
                self.get_logger().error(f'Failed to receive grasp count, received {len(response)} bytes')
                return None, None

            grasp_count = int(np.frombuffer(response, dtype=np.int32)[0])
            self.get_logger().debug(f'Grasp count received: {grasp_count}')

            if grasp_count <= 0:
                self.get_logger().warn('Server returned zero grasps')
                return None, None

            grasps = []
            for _ in range(grasp_count):
                response = b''
                while len(response) < 64:
                    chunk = client.recv(64 - len(response))
                    if not chunk:
                        break
                    response += chunk

                if len(response) == 64:
                    grasp_matrix = np.frombuffer(response, dtype=np.float32).reshape((4, 4))
                    if np.all(grasp_matrix == 0):
                        self.get_logger().warn('Server returned no valid grasp (zero matrix)')
                        return None, None
                    grasps.append(grasp_matrix)
                else:
                    self.get_logger().error(f'Received {len(response)} bytes, expected 64')
                    return None, None

            grasp_labels = []
            for _ in range(grasp_count):
                label_bytes = b''
                while len(label_bytes) < 4:
                    chunk = client.recv(4 - len(label_bytes))
                    if not chunk:
                        break
                    label_bytes += chunk

                if len(label_bytes) == 4:
                    grasp_labels.append(int(np.frombuffer(label_bytes, dtype=np.int32)[0]))
                else:
                    self.get_logger().error(f'Received {len(label_bytes)} bytes for grasp label, expected 4')
                    return None, None

            return grasps, grasp_labels

        except Exception as exc:
            self.get_logger().warn(f'Socket communication error: {exc}')
            return None, None
        finally:
            client.close()

    def _request_loop(self):
        if not self.request_lock.acquire(blocking=False):
            return

        try:
            view_candidates = self._get_view_candidates()
            if not view_candidates:
                return

            view_clouds = []
            for view_name, depth_image, color_image, fx, fy, cx, cy, tf_msg in view_candidates:
                clouds = self._build_view_point_clouds(
                    depth_image=depth_image,
                    color_image=color_image,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    tf_msg=tf_msg,
                    view_name=view_name,
                )
                if clouds is not None:
                    view_clouds.append(clouds)

            if not view_clouds:
                return

            if len(view_clouds) == 1:
                hand_pc, object_pc = view_clouds[0]
            else:
                hand_pc = np.vstack([view_clouds[0][0], view_clouds[1][0]])
                object_pc = np.vstack([view_clouds[0][1], view_clouds[1][1]])

            if object_pc is None or len(object_pc) == 0:
                return

            completion_meta = self._build_completion_metadata(object_pc)
            self.send_grasp_request(hand_pc, object_pc, completion_meta)
        finally:
            self.request_lock.release()

    def _request_thread(self):
        """Background thread that continuously issues requests one after another.

        It queries for ready views and calls _request_loop synchronously; as soon as
        the previous call finishes (and receives grasps), it immediately issues
        the next request. When no view is ready, sleeps briefly to avoid busy-loop.
        """
        while rclpy.ok() and not getattr(self, '_shutdown', False):
            try:
                view_candidates = self._get_view_candidates()
                if not view_candidates:
                    time.sleep(0.05)
                    continue
                # perform one request (this blocks until send_grasp_request returns)
                try:
                    self._request_loop()
                except Exception as exc:
                    self.get_logger().warn(f'Request iteration failed: {exc}')
                    time.sleep(0.1)
            except Exception as exc:
                # Guard the thread from crashing
                try:
                    self.get_logger().error(f'Request thread top-level error: {exc}')
                except Exception:
                    pass
                time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenerator()

    executor = MultiThreadedExecutor()

    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()