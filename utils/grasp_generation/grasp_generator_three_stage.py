#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge
import socket
import os
import struct
import asyncio
import threading
from collections import deque
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from realsense2_camera_msgs.msg import RGBD
import cv2
import sensor_msgs.msg as sensor_msgs
import json
import time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.time import Time
from rclpy.duration import Duration
import open3d as o3d
from utils.pc_utils import *
import message_filters
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from grasp_generation.action import GenerateGrasp
from visual_properties.msg import Dimensions
from segmented_detection.msg import SegmentedObject, SegmentedDetection
from hand_detection.msg import HandDetection

try:
	from utils.config import asymmetric_container_priors, left_color_camera_topic_suffix, right_color_camera_topic_suffix
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from utils.config import asymmetric_container_priors, left_color_camera_topic_suffix, right_color_camera_topic_suffix

REQUEST_PREGENERATE = 1
REQUEST_HAND_FILTER = 2
REQUEST_REGENERATE = 3

HAND_LANDMARK_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

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
    """
    ROS2 action server that generates grasp poses using two-view approach with graceful fallback.
    
    Always uses two camera views to generate grasps. If one view's mask cannot be generated or is unavailable,
    gracefully falls back to using only the available view.
    
    Subscribes to:
        - /camera/{camera_name}/color/camera_info (sensor_msgs/CameraInfo): Camera intrinsic parameters for both views
        - /camera/{camera_name}/aligned_depth_to_color/image_raw (Image): Depth images for each camera view
        - /camera/{camera_name}/color/image_raw (Image): Color images for each camera view
        - /segmented_detection (segmented_detection/SegmentedDetection): Initial view-specific object mask and label
        - /handover_detected (Bool): Signal to indicate handover is being detected
        - /estimated_dimensions (Dimensions): Estimated object dimensions for grasp completion metadata
    """
        
    def __init__(self):
        super().__init__('grasp_generator_node')
        
        self.camera_name_first = self.declare_parameter('camera_name_first', 'left').get_parameter_value().string_value
        self.camera_name_second = self.declare_parameter('camera_name_second', 'right').get_parameter_value().string_value
        self.object_label_first = self.declare_parameter('object_label', 'cup').get_parameter_value().string_value
        self.object_label_second = self.object_label_first  # default to same label for second view; can be updated from segmented_object callback
        self.dimensions_topic = self.declare_parameter('dimensions_topic', '/estimated_dimensions').get_parameter_value().string_value
        self.segmented_object_topic = self.declare_parameter('segmented_object_topic', '/segmented_detection').get_parameter_value().string_value
        self.handover_detected_topic = self.declare_parameter('handover_detected_topic', '/handover_detected').get_parameter_value().string_value
        self.goal_pose_topic = self.declare_parameter('goal_pose_topic', '/goal_pose').get_parameter_value().string_value
        self.object_pose_deviation_threshold_m = float(
            self.declare_parameter('object_pose_deviation_threshold_m', 0.015).get_parameter_value().double_value
        )
        self.stage2_pose_timeout_sec = float(
            self.declare_parameter('stage2_pose_timeout_sec', 0.3).get_parameter_value().double_value
        )
        self.pose_baseline_window_size = int(
            self.declare_parameter('pose_baseline_window_size', 10).get_parameter_value().integer_value
        )
        self.stage2_sample_history_size = int(
            self.declare_parameter('stage2_sample_history_size', 60).get_parameter_value().integer_value
        )

        self.first_depth_topic = f'/camera/{self.camera_name_first}/aligned_depth_to_color/image_raw'
        self.second_depth_topic = f'/camera/{self.camera_name_second}/aligned_depth_to_color/image_raw'
        self.first_color_topic = f'/camera/{self.camera_name_first}/color/{left_color_camera_topic_suffix}'
        self.second_color_topic = f'/camera/{self.camera_name_second}/color/{right_color_camera_topic_suffix}'
        self.first_camera_info_topic = f'/camera/{self.camera_name_first}/aligned_depth_to_color/camera_info'
        self.second_camera_info_topic = f'/camera/{self.camera_name_second}/aligned_depth_to_color/camera_info'

        self.target_frame = 'base'
        self.source_frame_first = f'{self.camera_name_first}_color_optical_frame'
        self.source_frame_second = f'{self.camera_name_second}_color_optical_frame'

        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Storage for latest images for the second stage
        self.latest_depth_first = None
        self.latest_depth_second = None
        self.latest_color_first = None
        self.latest_color_second = None
        
        # first stage
        self.start_color_first = None
        self.start_depth_first = None
        self.start_mask_first = None
        self.start_color_second = None
        self.start_depth_second = None
        self.start_mask_second = None
        self._startup_pregeneration_ready = False
        self._startup_mask_attempted_first = False
        self._startup_mask_attempted_second = False
        # Latest YOLO-e masks per view (fallback when SAM3 fails)
        self._yoloe_mask_first: np.ndarray | None = None
        self._yoloe_mask_second: np.ndarray | None = None
        
        self.latest_dimensions = None
        self.object_is_box = False          # set from Dimensions.is_box
        self.object_initial_yaw_deg = None  # set from Dimensions.yaw_degree_robot_inital
        self.axis_default_world = self.declare_parameter('axis_default_world', [0.0, 0.0, 1.0]).get_parameter_value().double_array_value

        self.stage1_running = False
        self.stage1_complete = False
        self.stage2_running = False
        self.stage2_complete = False
        self.cached_startup_object_pc = None
        self.cached_final_grasps = None
        self.cached_final_labels = None
        self.cached_final_widths = None
        self.handover_detected = False
        self.stage2_armed_time = None
        self.latest_hand_detection = None
        self.state_lock = threading.Lock()
        self.hand_detection_lock = threading.Lock()
        self.frame_sample_lock = threading.Lock()
        self.goal_pose_lock = threading.Lock()
        self._stage1_thread = None
        self._stage2_thread = None
        self.frame_samples = deque(maxlen=max(1, self.stage2_sample_history_size))
        self.goal_pose_baseline_samples = deque(maxlen=max(1, self.pose_baseline_window_size))
        self.goal_pose_baseline = None
        self.latest_goal_pose_position = None
        self.fused_object_yaw = None
        self.fused_object_yaw_delta = None
        self.large_deviation_checked = False
        
        # Flag to prevent multiple simultaneous requests
        self.first_camera_initialized = False
        self.second_camera_initialized = False
        
        # create sensor qos profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
             depth=10
        )
        
        # Subscribers
        self.first_cam_info_sub = self.create_subscription(
            sensor_msgs.CameraInfo,
            self.first_camera_info_topic, 
            self.first_camera_info_callback, 
            qos_profile=qos_profile
        )
        self.second_cam_info_sub = self.create_subscription(
            sensor_msgs.CameraInfo,
            self.second_camera_info_topic,
            self.second_camera_info_callback,
            qos_profile=qos_profile
        )
        self.first_depth_sub = message_filters.Subscriber(self, Image, self.first_depth_topic, qos_profile=qos_profile)
        self.first_color_sub = message_filters.Subscriber(self, Image, self.first_color_topic, qos_profile=qos_profile)
        self.second_depth_sub = message_filters.Subscriber(self, Image, self.second_depth_topic, qos_profile=qos_profile)
        self.second_color_sub = message_filters.Subscriber(self, Image, self.second_color_topic, qos_profile=qos_profile)
        
        self.hand_detection_topic = self.declare_parameter('hand_tracker_topic', '/hand_tracker').get_parameter_value().string_value
        self.hand_detection_msg_filter = message_filters.Subscriber(self, HandDetection, self.hand_detection_topic, qos_profile=qos_profile)
        
        # Unified sync: all camera views + hand detection synchronized by timestamp
        sync_topic_list = [self.first_depth_sub, self.first_color_sub, 
                          self.second_depth_sub, self.second_color_sub,
                          self.hand_detection_msg_filter]
        
        self.ts_all = message_filters.ApproximateTimeSynchronizer(
            sync_topic_list,
            queue_size=10,
            slop=0.1  # 100ms sync tolerance for 5 messages
        )
        self.ts_all.registerCallback(self.sync_callback_all)

        self.dimensions_sub = self.create_subscription(
            Dimensions,
            self.dimensions_topic,
            self.dimensions_callback,
            10,
        )

        self.segmented_object_sub = self.create_subscription(
            SegmentedDetection,
            self.segmented_object_topic,
            self.segmented_object_callback,
            10,
        )

        self.handover_detected_sub = self.create_subscription(
            Bool,
            self.handover_detected_topic,
            self.handover_detected_callback,
            qos_profile=10,
        )

        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            self.goal_pose_topic,
            self.goal_pose_callback,
            10,
        )

        # --- YOLO-e segmentation mask subscribers (fallback for startup masks) ---
        _mask_first_topic = self.declare_parameter(
            'mask_first_topic', f'/segmentation/{self.camera_name_first}'
        ).get_parameter_value().string_value
        _mask_second_topic = self.declare_parameter(
            'mask_second_topic', f'/segmentation/{self.camera_name_second}'
        ).get_parameter_value().string_value
        self._yoloe_mask_sub_first = self.create_subscription(
            Image,
            _mask_first_topic,
            lambda msg: self._yoloe_mask_callback(msg, 'first'),
            10,
        )
        self._yoloe_mask_sub_second = self.create_subscription(
            Image,
            _mask_second_topic,
            lambda msg: self._yoloe_mask_callback(msg, 'second'),
            10,
        )
        self.get_logger().info(
            f'Subscribed to YOLO-e mask topics: first={_mask_first_topic}, second={_mask_second_topic}'
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_first = None
        self.tf_second = None

        self.grasp_socket_path = "/tmp/graspgen"
        self.sam_socket_path = "/tmp/sam3_socket"
        self.replanned_grasps_pub = self.create_publisher(GenerateGrasp.Result, '/replanned_grasps', 10)
        
        self.cb_group = ReentrantCallbackGroup()
        self.action_server = ActionServer(
            self,
            GenerateGrasp,
            'generate_grasp',
            execute_callback=self.execute_callback,
            callback_group=self.cb_group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def handover_detected_callback(self, msg):
        if not bool(msg.data):
            return
        
        start_stage2 = False
        with self.state_lock:
            self.handover_detected = True
            if self.stage2_armed_time is None:
                self.stage2_armed_time = time.monotonic()
            if not self.stage2_running and not self.stage2_complete:
                start_stage2 = True

        if self.handover_detected_sub is not None:
            self.destroy_subscription(self.handover_detected_sub)
            self.handover_detected_sub = None
        self.get_logger().info('handover_detected=true received; Stage 2 armed and handover subscription removed.')

        if start_stage2:
            self.get_logger().info('Launching Stage 2 hand filtering worker.')
            with self.state_lock:
                self.stage2_running = True
            self._stage2_thread = threading.Thread(target=self._run_stage2_hand_filtering, daemon=True)
            self._stage2_thread.start()

    def _destroy_goal_pose_subscription(self):
        if self.goal_pose_sub is None:
            return
        try:
            self.destroy_subscription(self.goal_pose_sub)
        finally:
            self.goal_pose_sub = None

    def goal_pose_callback(self, msg):
        position = np.array(
            [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(position)):
            return

        with self.goal_pose_lock:
            self.latest_goal_pose_position = position

            with self.state_lock:
                baseline_locked = self.handover_detected or self.stage2_running or self.stage2_complete

            if baseline_locked:
                return

            if self.goal_pose_baseline is None:
                self.goal_pose_baseline_samples.append(position)
                self.goal_pose_baseline = np.mean(np.asarray(self.goal_pose_baseline_samples), axis=0).astype(np.float32)
                return

            deviation = float(np.linalg.norm(position - self.goal_pose_baseline))
            baseline_update_gate = max(self.object_pose_deviation_threshold_m * 0.5, 1e-4)
            if deviation <= baseline_update_gate or len(self.goal_pose_baseline_samples) < self.pose_baseline_window_size:
                self.goal_pose_baseline_samples.append(position)
                self.goal_pose_baseline = np.mean(np.asarray(self.goal_pose_baseline_samples), axis=0).astype(np.float32)

    def dimensions_callback(self, msg):
        self.get_logger().info('Received dimensions update.')
        def _safe_float(value):
            try:
                value = float(value)
            except Exception:
                return None
            return value if np.isfinite(value) else None

        self.latest_dimensions = {
            'height': _safe_float(msg.height),
            'top_w': _safe_float(msg.top_w),
            'bot_w': _safe_float(msg.bot_w),
        }

        # Box flag and initial yaw from the dimensions message
        try:
            self.object_is_box = bool(msg.is_box)
        except AttributeError:
            self.object_is_box = False
            self.get_logger().warn('Dimensions message has no is_box field; defaulting to False.')

        try:
            self.object_initial_yaw_deg = int(msg.yaw_degree_robot_inital)
        except (AttributeError, ValueError, TypeError):
            self.object_initial_yaw_deg = None
            self.get_logger().warn(
                'Dimensions message has no yaw_degree_robot_inital field or it is invalid; '
                'defaulting to None.'
            )

        self.get_logger().info(
            f'Dimensions received: is_box={self.object_is_box}, '
            f'yaw_initial={self.object_initial_yaw_deg} deg'
        )
        self.dimensions_sub.destroy()
    
    def _update_latest_hand_detection(self, msg):
        """Update latest hand detection (called from sync callback)."""
        with self.hand_detection_lock:
            self.latest_hand_detection = msg

    def _latest_hand_detection_snapshot(self):
        with self.hand_detection_lock:
            return self.latest_hand_detection

    def _latest_frame_sample_snapshot(self):
        with self.frame_sample_lock:
            if not self.frame_samples:
                return None
            return self.frame_samples[-1]

    def _frame_sample_history_snapshot(self):
        with self.frame_sample_lock:
            return list(self.frame_samples)

    def _latest_goal_pose_snapshot(self):
        with self.goal_pose_lock:
            baseline = None if self.goal_pose_baseline is None else self.goal_pose_baseline.copy()
            position = None if self.latest_goal_pose_position is None else self.latest_goal_pose_position.copy()
        return baseline, position

    def _object_pose_deviation(self):
        baseline, position = self._latest_goal_pose_snapshot()
        if baseline is None or position is None:
            return None
        return float(np.linalg.norm(position - baseline))

    def segmented_object_callback(self, msg):
        left_msg = msg.left
        right_msg = msg.right
        left_view = str(left_msg.view).strip()
        right_view = str(right_msg.view).strip()
        left_label = str(left_msg.segmentation_label).strip()
        right_label = str(right_msg.segmentation_label).strip()

        try:
            left_mask = self.bridge.imgmsg_to_cv2(left_msg.segmentation_mask, desired_encoding='mono8')
            right_mask = self.bridge.imgmsg_to_cv2(right_msg.segmentation_mask, desired_encoding='mono8')
            left_mask = (np.asarray(left_mask) > 0).astype(np.uint8)
            right_mask = (np.asarray(right_mask) > 0).astype(np.uint8)
        except Exception as exc:
            self.get_logger().error(f'Failed to decode startup segmented mask: {exc}')
            return

        if left_view == self.camera_name_first:
            self.object_label_first = left_label
            self.start_mask_first = left_mask
            self._startup_mask_attempted_first = True
            self.get_logger().info(f"Received object label '{self.object_label_first}' from /segmented_detection message.")
            self.get_logger().info(f'Received startup mask for first view: {self.camera_name_first}')
        if right_view == self.camera_name_second:
            self.object_label_second = right_label
            self.start_mask_second = right_mask
            self._startup_mask_attempted_second = True
            self.get_logger().info(f"Received object label '{self.object_label_second}' from /segmented_detection message.")
            self.get_logger().info(f'Received startup mask for second view: {self.camera_name_second}')

        self._startup_prepare_masks()

    def _startup_prepare_masks(self):
        if self._startup_pregeneration_ready:
            return

        # Request the missing startup mask for the opposite view, once.
        # Use single_prompt mode for object-only segmentation during pregeneration.
        if self.start_mask_first is not None and self.start_mask_second is None:
            if self.start_color_second is not None and not self._startup_mask_attempted_second:
                self._startup_mask_attempted_second = True
                masks = self.request_segmentation_masks(self.start_color_second, label=self.object_label_second, single_prompt=True)
                object_mask = masks.get('object_mask')
                if masks.get('object_ok') and object_mask is not None:
                    self.start_mask_second = object_mask.astype(np.uint8)
                    self.get_logger().info(f'Startup mask generated for second view: {self.camera_name_second}')
                else:
                    self.get_logger().warn(f'Startup mask generation failed for second view: {self.camera_name_second}')
                    # Fallback: use latest YOLO-e mask if available
                    yoloe = self._yoloe_mask_second
                    if yoloe is not None and np.any(yoloe):
                        self.start_mask_second = yoloe
                        self.get_logger().info(
                            f'SAM3 failed for second view; using latest YOLO-e mask as fallback.'
                        )
                    else:
                        # YOLO-e not yet available; reset flag to retry on next frame
                        self._startup_mask_attempted_second = False
                        self.get_logger().warn(
                            f'No YOLO-e mask available for second view yet; will retry on next frame.'
                        )
        elif self.start_mask_second is not None and self.start_mask_first is None:
            if self.start_color_first is not None and not self._startup_mask_attempted_first:
                self._startup_mask_attempted_first = True
                masks = self.request_segmentation_masks(self.start_color_first, label=self.object_label_first, single_prompt=True)
                object_mask = masks.get('object_mask')
                if masks.get('object_ok') and object_mask is not None:
                    self.start_mask_first = object_mask.astype(np.uint8)
                    self.get_logger().info(f'Startup mask generated for first view: {self.camera_name_first}')
                else:
                    self.get_logger().warn(f'Startup mask generation failed for first view: {self.camera_name_first}')
                    # Fallback: use latest YOLO-e mask if available
                    yoloe = self._yoloe_mask_first
                    if yoloe is not None and np.any(yoloe):
                        self.start_mask_first = yoloe
                        self.get_logger().info(
                            f'SAM3 failed for first view; using latest YOLO-e mask as fallback.'
                        )
                    else:
                        # YOLO-e not yet available; reset flag to retry on next frame
                        self._startup_mask_attempted_first = False
                        self.get_logger().warn(
                            f'No YOLO-e mask available for first view yet; will retry on next frame.'
                        )

        have_start_frames = (
            self.start_color_first is not None and
            self.start_depth_first is not None and
            self.start_color_second is not None and
            self.start_depth_second is not None
        )
        have_at_least_one_mask = (self.start_mask_first is not None) or (self.start_mask_second is not None)

        if have_start_frames and have_at_least_one_mask:
            self._startup_pregeneration_ready = True
            with self.state_lock:
                if not self.stage1_running and not self.stage1_complete:
                    self.stage1_running = True
                    self._stage1_thread = threading.Thread(target=self._run_stage1_pregeneration, daemon=True)
                    self._stage1_thread.start()
            self.get_logger().info('Startup prep complete: start frames and startup masks are ready.')

    def _yoloe_mask_callback(self, msg: Image, view: str):
        """Buffer the latest YOLO-e segmentation mask for the given view ('first' or 'second')."""
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            # Binarise: any non-zero pixel → 1
            binary_mask = (np.asarray(mask) > 0).astype(np.uint8)
            if view == 'first':
                self._yoloe_mask_first = binary_mask
            else:
                self._yoloe_mask_second = binary_mask
        except Exception as e:
            self.get_logger().warning(f'Failed to decode YOLO-e mask for {view} view: {e}')

    def _build_object_point_cloud_for_view(self, depth_image, color_image, mask, fx, fy, cx, cy, tf_msg, view_name):
        if depth_image is None or color_image is None or mask is None or tf_msg is None:
            return None

        depth_image = depth_image.astype(np.float32).reshape(depth_image.shape)
        try:
            _, object_pc = depth_and_segmentation_to_point_clouds_old(
                depth_image=depth_image,
                segmentation_mask=mask,
                fx=fx, fy=fy, cx=cx, cy=cy,
                target_object_id=1,
                remove_object_from_scene=True,
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to generate startup object point cloud for {view_name} view: {exc}')
            return None

        if object_pc is None or len(object_pc) == 0:
            return None

        object_pc = point_cloud_outlier_removal(torch.from_numpy(object_pc).float(), threshold=0.005, K=10).cpu().numpy()
        if len(object_pc) > 50:
            object_pc = self._apply_dbscan(object_pc)

        object_pc = self._transform_points_to_target(object_pc, tf_msg)
        return object_pc

    def _landmark_to_pixel(self, landmark, width, height):
        if landmark is None:
            return None

        x_value = float(landmark.x)
        y_value = float(landmark.y)

        if 0.0 <= x_value <= 1.5 and 0.0 <= y_value <= 1.5:
            x_px = int(round(x_value * (width - 1)))
            y_px = int(round(y_value * (height - 1)))
        else:
            x_px = int(round(x_value))
            y_px = int(round(y_value))

        x_px = int(np.clip(x_px, 0, width - 1))
        y_px = int(np.clip(y_px, 0, height - 1))
        return x_px, y_px

    def _landmark_to_camera_point(self, landmark, depth_image, fx, fy, cx, cy, width, height):
        pixel = self._landmark_to_pixel(landmark, width, height)
        if pixel is None:
            return None

        x_px, y_px = pixel
        depth_value = float(depth_image[y_px, x_px])
        if not np.isfinite(depth_value) or depth_value <= 0.0:
            return None

        x_cam = (float(x_px) - cx) * depth_value / fx
        y_cam = (float(y_px) - cy) * depth_value / fy
        z_cam = depth_value
        return np.array([x_cam, y_cam, z_cam], dtype=np.float32)

    def _sample_line_segment(self, start_point, end_point):
        start_point = np.asarray(start_point, dtype=np.float32)
        end_point = np.asarray(end_point, dtype=np.float32)
        segment = end_point - start_point
        segment_length = float(np.linalg.norm(segment))
        if not np.isfinite(segment_length) or segment_length <= 1e-6:
            return [start_point]

        sample_count = max(2, int(np.ceil(segment_length / 0.001)) + 1)
        return [start_point * (1.0 - alpha) + end_point * alpha for alpha in np.linspace(0.0, 1.0, sample_count, dtype=np.float32)]

    def _build_hand_point_cloud_from_landmarks(self, depth_image, hand_view, fx, fy, cx, cy, tf_msg, view_name, inflate_factor=1.7):
        if depth_image is None or tf_msg is None or hand_view is None:
            return None

        landmarks = list(getattr(hand_view, 'landmarks', []) or [])
        if not landmarks:
            return None

        depth_image = depth_image.astype(np.float32).reshape(depth_image.shape)
        height, width = depth_image.shape[:2]

        reconstructed_points = []
        landmark_points = {}

        for index, landmark in enumerate(landmarks):
            point = self._landmark_to_camera_point(landmark, depth_image, fx, fy, cx, cy, width, height)
            if point is None:
                continue
            landmark_points[index] = point
            reconstructed_points.append(point)

        if not landmark_points:
            self.get_logger().warn(f'No valid landmark depths for {view_name} view; falling back to SAM3.')
            return None

        if inflate_factor is not None and float(inflate_factor) > 1.0:
            fingertip_indices = (4, 7, 8, 11, 12, 15, 16, 19, 20)
            tip_radius_m = 0.004 * float(inflate_factor)
            tip_angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)

            for tip_index in fingertip_indices:
                tip_point = landmark_points.get(tip_index)
                if tip_point is None:
                    continue

                for angle in tip_angles:
                    radial_offset = np.array(
                        [np.cos(angle), np.sin(angle), 0.0],
                        dtype=np.float32,
                    ) * tip_radius_m
                    reconstructed_points.append(tip_point + radial_offset)

        for start_index, end_index in HAND_LANDMARK_CONNECTIONS:
            if start_index not in landmark_points or end_index not in landmark_points:
                continue
            reconstructed_points.extend(
                self._sample_line_segment(landmark_points[start_index], landmark_points[end_index])
            )

        if not reconstructed_points:
            return None

        hand_pc = np.asarray(reconstructed_points, dtype=np.float32)
        if len(hand_pc) > 50:
            hand_pc = self._apply_dbscan(hand_pc, eps=0.03, min_points=5)
        hand_pc = point_cloud_outlier_removal(torch.from_numpy(hand_pc).float(), threshold=0.02, K=2).cpu().numpy()
        if hand_pc is None or len(hand_pc) == 0:
            return None

        return self._transform_points_to_target(hand_pc, tf_msg)

    def _build_hand_point_cloud_for_view(self, depth_image, color_image, hand_view, fx, fy, cx, cy, tf_msg, view_name):
        reconstructed_hand_pc = self._build_hand_point_cloud_from_landmarks(
            depth_image=depth_image,
            hand_view=hand_view,
            fx=fx, fy=fy, cx=cx, cy=cy,
            tf_msg=tf_msg,
            view_name=view_name,
        )
        if reconstructed_hand_pc is not None and len(reconstructed_hand_pc) > 0:
            return reconstructed_hand_pc

        if depth_image is None or color_image is None or tf_msg is None:
            return None

        self.get_logger().info(f'Falling back to SAM3 hand mask for {view_name} view.')
        masks = self.request_segmentation_masks(color_image, label='hand', single_prompt=True)
        hand_mask = masks.get('object_mask')
        if not masks.get('object_ok') or hand_mask is None:
            self.get_logger().warn(f'Hand mask unavailable for {view_name} view; skipping hand cloud build')
            return None

        depth_image = depth_image.astype(np.float32).reshape(depth_image.shape)
        try:
            _, hand_pc = depth_and_segmentation_to_point_clouds_old(
                depth_image=depth_image,
                segmentation_mask=hand_mask,
                fx=fx, fy=fy, cx=cx, cy=cy,
                target_object_id=1,
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to generate hand point cloud for {view_name} view: {exc}')
            return None

        if hand_pc is None or len(hand_pc) == 0:
            return None

        if len(hand_pc) > 50:
            hand_pc = self._apply_dbscan(hand_pc, eps=0.05, min_points=5)
        hand_pc = point_cloud_outlier_removal(torch.from_numpy(hand_pc).float(), threshold=0.01, K=4).cpu().numpy()
        if hand_pc is None or len(hand_pc) == 0:
            return None
        return self._transform_points_to_target(hand_pc, tf_msg)

    def _collect_startup_object_point_cloud(self):
        view_clouds = []

        if self.start_mask_first is not None and self.start_depth_first is not None and self.start_color_first is not None and self.tf_first is not None:
            cloud = self._build_object_point_cloud_for_view(
                depth_image=self.start_depth_first,
                color_image=self.start_color_first,
                mask=self.start_mask_first,
                fx=self.ffx, fy=self.ffy, cx=self.fcx, cy=self.fcy,
                tf_msg=self.tf_first,
                view_name='first',
            )
            if cloud is not None and len(cloud) > 0:
                view_clouds.append(cloud)

        if self.start_mask_second is not None and self.start_depth_second is not None and self.start_color_second is not None and self.tf_second is not None:
            cloud = self._build_object_point_cloud_for_view(
                depth_image=self.start_depth_second,
                color_image=self.start_color_second,
                mask=self.start_mask_second,
                fx=self.sfx, fy=self.sfy, cx=self.scx, cy=self.scy,
                tf_msg=self.tf_second,
                view_name='second',
            )
            if cloud is not None and len(cloud) > 0:
                view_clouds.append(cloud)

        if not view_clouds:
            return None
        if len(view_clouds) == 1:
            return view_clouds[0]
        return np.vstack(view_clouds)

    def _collect_latest_hand_point_cloud(self, sample=None):
        view_clouds = []

        if sample is None:
            hand_detections = self._latest_hand_detection_snapshot()
            # copy latest frames
            color_first = self.latest_color_first.copy() if self.latest_color_first is not None else None
            depth_first = self.latest_depth_first.copy() if self.latest_depth_first is not None else None
            color_second = self.latest_color_second.copy() if self.latest_color_second is not None else None
            depth_second = self.latest_depth_second.copy() if self.latest_depth_second is not None else None
        else:
            hand_detections = sample.get('hand_detection')
            color_first = sample.get('color_first')
            depth_first = sample.get('depth_first')
            color_second = sample.get('color_second')
            depth_second = sample.get('depth_second')
        
        # debug: export color images
        # if color_first is not None:
        #     cv2.imwrite('latest_color_first.png', color_first)
        # if color_second is not None:
        #     cv2.imwrite('latest_color_second.png', color_second)

        if depth_first is not None and color_first is not None and self.tf_first is not None and self.first_camera_initialized:
            if hand_detections is not None and hand_detections.left_has_active and len(hand_detections.left) > 0:
                hand_view_first = hand_detections.left[0] # top hand prioritized for first view
            else:
                hand_view_first = None
            cloud = self._build_hand_point_cloud_for_view(
                depth_image=depth_first,
                color_image=color_first,
                hand_view=hand_view_first,
                fx=self.ffx, fy=self.ffy, cx=self.fcx, cy=self.fcy,
                tf_msg=self.tf_first,
                view_name='first',
            )
            if cloud is not None and len(cloud) > 0:
                view_clouds.append(cloud)

        if depth_second is not None and color_second is not None and self.tf_second is not None and self.second_camera_initialized:
            if hand_detections is not None and hand_detections.right_has_active and len(hand_detections.right) > 0:
                hand_view_second = hand_detections.right[0] # top hand prioritized for second view
            else:
                hand_view_second = None
            cloud = self._build_hand_point_cloud_for_view(
                depth_image=depth_second,
                color_image=color_second,
                hand_view=hand_view_second,
                fx=self.sfx, fy=self.sfy, cx=self.scx, cy=self.scy,
                tf_msg=self.tf_second,
                view_name='second',
            )
            if cloud is not None and len(cloud) > 0:
                view_clouds.append(cloud)

        if not view_clouds:
            return None
        if len(view_clouds) == 1:
            return view_clouds[0]
        return np.vstack(view_clouds)

    def _score_hand_object_contact(self, hand_pc, object_pc):
        if hand_pc is None or object_pc is None or len(hand_pc) == 0 or len(object_pc) == 0:
            return float('inf')

        hand_points = np.asarray(hand_pc, dtype=np.float32)
        object_points = np.asarray(object_pc, dtype=np.float32)
        if len(hand_points) > 300:
            stride = max(1, len(hand_points) // 300)
            hand_points = hand_points[::stride]
        if len(object_points) > 2000:
            stride = max(1, len(object_points) // 2000)
            object_points = object_points[::stride]

        try:
            tree = cKDTree(object_points)
            distances, _ = tree.query(hand_points, k=1)
        except Exception as exc:
            self.get_logger().warn(f'Failed to score hand/object contact sample: {exc}')
            return float('inf')

        if distances is None or len(distances) == 0:
            return float('inf')
        return float(np.percentile(distances, 10.0))

    def _select_fallback_hand_point_cloud(self):
        samples = self._frame_sample_history_snapshot()
        if not samples:
            return None

        # return latest available hand point cloud
        best_sample = samples[-1]
        best_hand_pc = self._collect_latest_hand_point_cloud(sample=best_sample)
        # with self.state_lock:
        #     object_pc = None if self.cached_startup_object_pc is None else self.cached_startup_object_pc.copy()
        # best_hand_pc = None
        # best_score = float('inf')

        # for sample in reversed(samples):
        #     hand_pc = self._collect_latest_hand_point_cloud(sample=sample)
        #     if hand_pc is None or len(hand_pc) == 0:
        #         continue

        #     score = self._score_hand_object_contact(hand_pc, object_pc)
        #     if best_hand_pc is None or score < best_score:
        #         best_score = score
        #         best_hand_pc = hand_pc

        # if best_hand_pc is not None:
        #     self.get_logger().info(f'Using fallback Stage 2 sample with hand/object contact score {best_score:.4f}m.')
        return best_hand_pc

    def _run_stage1_pregeneration(self):
        while self.latest_dimensions is None:
            time.sleep(0.01)
        self.get_logger().info('Stage 1 pregeneration worker started.')
        while True:
            with self.state_lock:
                if self.stage1_complete:
                    self.stage1_running = False
                    return

            object_pc = self._collect_startup_object_point_cloud()
            if object_pc is None or len(object_pc) == 0:
                time.sleep(0.05)
                continue
            with self.state_lock:
                self.cached_startup_object_pc = object_pc.copy()

            completion_meta = self._build_completion_metadata(object_pc)
            response = self.send_grasp_request(
                request_type=REQUEST_PREGENERATE,
                scene_pc=None,
                object_pc=object_pc,
                completion_meta=completion_meta,
            )

            with self.state_lock:
                self.stage1_running = False
                self.stage1_complete = bool(response.get('status'))
            if self.stage1_complete:
                self.get_logger().info('Stage 1 pregeneration completed successfully.')
            else:
                self.get_logger().error('Stage 1 pregeneration failed.')
            return

    def _run_stage2_hand_filtering(self):
        self.get_logger().info('Stage 2 hand filtering worker started.')
        while True:
            with self.state_lock:
                if self.stage2_complete:
                    self.stage2_running = False
                    return
                stage1_ready = self.stage1_complete
                armed_time = self.stage2_armed_time

            if not stage1_ready:
                print('Stage 1 not complete yet; waiting before starting Stage 2 hand filtering...')
                time.sleep(0.05)
                continue

            if armed_time is None:
                armed_time = time.monotonic()
                with self.state_lock:
                    self.stage2_armed_time = armed_time

            elapsed_sec = time.monotonic() - armed_time
            deviation = self._object_pose_deviation()
            pose_triggered = (
                deviation is not None and
                deviation >= self.object_pose_deviation_threshold_m
            )
            timed_out = elapsed_sec >= self.stage2_pose_timeout_sec

            if pose_triggered:
                self.get_logger().info(
                    f'Object pose deviation {deviation:.4f}m exceeded '
                    f'{self.object_pose_deviation_threshold_m:.4f}m; running Stage 2 on synchronized sample.'
                )
                # NOTE: goal_pose subscription is intentionally kept alive here so that
                # _monitor_large_deviation can continue receiving /goal_pose updates
                # until the 20 cm threshold is reached.  The subscription is torn down
                # by the monitor itself once it exits.
                hand_pc = self._collect_latest_hand_point_cloud(sample=self._latest_frame_sample_snapshot())
            elif timed_out:
                if deviation is None:
                    self.get_logger().warn(
                        f'No usable /goal_pose baseline/current pose after {elapsed_sec:.2f}s; '
                        'using best recent hand/object-contact sample for Stage 2.'
                    )
                else:
                    self.get_logger().warn(
                        f'No object pose deviation after {elapsed_sec:.2f}s '
                        f'(current={deviation:.4f}m, threshold={self.object_pose_deviation_threshold_m:.4f}m); '
                        'using best recent hand/object-contact sample for Stage 2.'
                    )
                # Same as above: keep subscription alive for the large-deviation monitor.
                hand_pc = self._select_fallback_hand_point_cloud()
                if hand_pc is None or len(hand_pc) == 0:
                    hand_pc = self._collect_latest_hand_point_cloud(sample=self._latest_frame_sample_snapshot())
            else:
                time.sleep(0.01)
                continue

            # Preserve the previous behavior: Stage 2 can still call the service with an
            # empty scene if no hand cloud can be recovered from the selected sample.

            response = self.send_grasp_request(
                request_type=REQUEST_HAND_FILTER,
                scene_pc=hand_pc,
                object_pc=None,
                completion_meta=None,
            )

            success = bool(response.get('status'))
            final_grasps = response.get('grasps') if success else np.zeros((0, 4, 4), dtype=np.float32)
            final_labels = response.get('labels') if success else []
            final_widths = response.get('widths') if success else []

            with self.state_lock:
                self.stage2_running = False
                self.stage2_complete = True
                self.cached_final_grasps = final_grasps
                self.cached_final_labels = final_labels
                self.cached_final_widths = final_widths
            if success:
                self.get_logger().info('Stage 2 hand filtering completed successfully.')
            else:
                self.get_logger().error('Stage 2 hand filtering failed.')

            # Start the 20 cm deviation monitor only for box-shaped objects.
            # For non-box objects the rotation estimation is not needed.
            if self.object_is_box:
                self.get_logger().info(
                    'Object is a box (is_box=True); starting large-deviation rotation monitor.'
                )
                threading.Thread(target=self._monitor_large_deviation, daemon=True).start()
            else:
                self.get_logger().info(
                    'Object is not a box (is_box=False); skipping large-deviation rotation monitor.'
                )
                # Still clean up the goal pose subscription since the monitor won't do it.
                self._destroy_goal_pose_subscription()
            return
    
    # ------------------------------------------------------------------
    # Post-stage-2:15 cm deviation monitor & rotation estimation
    # ------------------------------------------------------------------

    # Maximum time (seconds) to wait for a 15 cm deviation before giving up.
    _LARGE_DEVIATION_MONITOR_TIMEOUT_SEC = 20.0
    _LARGE_DEVIATION_THRESHOLD_M = 0.15
    _ORIENTATION_CHANGE_THRESHOLD_DEG = 30.0

    def _monitor_large_deviation(self):
        """Periodically check object deviation after Stage 2 completes.

        Runs at 20 Hz. Once the object has moved ≥ 15 cm from the baseline
        position recorded before handover, unsubscribes from /goal_pose and
        re-segments the object inline to estimate its yaw in the robot base
        frame.

        The /goal_pose subscription is kept alive until this monitor exits
        (either on success or timeout) so that `latest_goal_pose_position`
        keeps receiving fresh values.
        """
        self.get_logger().info('Large-deviation monitor started (20 Hz, threshold=0.20 m).')
        start_time = time.monotonic()
        try:
            while True:
                # Bail out if another code path already set the flag.
                with self.state_lock:
                    if self.large_deviation_checked:
                        return

                # Timeout guard — avoid spinning forever if /goal_pose stops.
                if time.monotonic() - start_time > self._LARGE_DEVIATION_MONITOR_TIMEOUT_SEC:
                    with self.state_lock:
                        self.large_deviation_checked = True
                    self.get_logger().warn(
                        f'Large-deviation monitor timed out after '
                        f'{self._LARGE_DEVIATION_MONITOR_TIMEOUT_SEC:.0f}s; '
                        'rotation estimation skipped.'
                    )
                    return

                with self.goal_pose_lock:
                    current_pose = (
                        None if self.latest_goal_pose_position is None
                        else self.latest_goal_pose_position.copy()
                    )
                    baseline = (
                        None if self.goal_pose_baseline is None
                        else self.goal_pose_baseline.copy()
                    )

                if current_pose is not None and baseline is not None:
                    deviation = float(np.linalg.norm(current_pose - baseline))
                    if deviation >= self._LARGE_DEVIATION_THRESHOLD_M:
                        with self.state_lock:
                            self.large_deviation_checked = True
                        self._destroy_goal_pose_subscription()
                        self.get_logger().info(
                            f'Object deviation {deviation:.4f} m reached {self._LARGE_DEVIATION_THRESHOLD_M * 100:.0f} cm threshold; '
                            'estimating rotation.'
                        )
                        self._estimate_rotation_at_large_deviation()
                        return

                time.sleep(0.05)
        finally:
            # Always clean up the subscription when the monitor exits, regardless
            # of whether the threshold was reached or a timeout/exception occurred.
            self._destroy_goal_pose_subscription()

    def _build_fused_object_point_cloud_for_rotation(self, color_first, depth_first, color_second, depth_second):
        """Re-segment both views and return a fused object point cloud in the robot base frame.

        Returns None if neither view yields a valid point cloud.
        """
        view_clouds = []

        for color_img, depth_img, fx, fy, cx, cy, tf_msg, view_name, cam_ok in [
            (
                color_first, depth_first,
                getattr(self, 'ffx', None), getattr(self, 'ffy', None),
                getattr(self, 'fcx', None), getattr(self, 'fcy', None),
                self.tf_first, 'first', self.first_camera_initialized,
            ),
            (
                color_second, depth_second,
                getattr(self, 'sfx', None), getattr(self, 'sfy', None),
                getattr(self, 'scx', None), getattr(self, 'scy', None),
                self.tf_second, 'second', self.second_camera_initialized,
            ),
        ]:
            if not cam_ok or color_img is None or depth_img is None or tf_msg is None:
                continue
            if fx is None or fy is None or cx is None or cy is None:
                continue

            object_label = self.object_label_first if view_name == 'first' else self.object_label_second
            masks = self.request_segmentation_masks(
                color_img, label=object_label, single_prompt=True
            )
            if not masks.get('object_ok') or masks.get('object_mask') is None:
                self.get_logger().warn(
                    f'Re-segmentation failed for {view_name} view; skipping.'
                )
                continue

            object_mask = masks['object_mask'].astype(np.uint8)
            cloud = self._build_object_point_cloud_for_view(
                depth_image=depth_img,
                color_image=color_img,
                mask=object_mask,
                fx=fx, fy=fy, cx=cx, cy=cy,
                tf_msg=tf_msg,
                view_name=view_name,
            )
            if cloud is not None and len(cloud) > 0:
                view_clouds.append(cloud)

        if not view_clouds:
            return None
        if len(view_clouds) == 1:
            return view_clouds[0]
        return np.vstack(view_clouds)

    def _estimate_rotation_at_large_deviation(self):
        """Re-segment the object at the 20 cm deviation point and compute its yaw.

        Yaw convention (robot base frame, standard ROS +X forward / +Y left):
            0 deg  → object faces towards the robot
            +90 deg → object faces right
            -90 deg → object faces left
        """
        self.get_logger().info('Estimating object rotation at large deviation ...')
        self.fused_object_yaw_delta = None

        # Snapshot current frames
        color_first  = self.latest_color_first.copy()  if self.latest_color_first  is not None else None
        depth_first  = self.latest_depth_first.copy()  if self.latest_depth_first  is not None else None
        color_second = self.latest_color_second.copy() if self.latest_color_second is not None else None
        depth_second = self.latest_depth_second.copy() if self.latest_depth_second is not None else None

        fused_points = self._build_fused_object_point_cloud_for_rotation(
            color_first, depth_first, color_second, depth_second
        )

        if fused_points is None or len(fused_points) < 4:
            self.get_logger().warn(
                'Not enough points for OBB fitting; rotation estimation skipped.'
            )
            return

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(fused_points.astype(np.float64))
            try:
                obb = pcd.get_minimal_oriented_bounding_box()
            except AttributeError:
                obb = pcd.get_oriented_bounding_box()

            R = np.asarray(obb.R)          # 3×3 rotation matrix of the OBB
            extent = np.asarray(obb.extent) # (ex, ey, ez) half-extents

            # Identify the OBB column most aligned with the world vertical (Z)
            h_idx = int(np.argmax(np.abs(R[2, :])))

            # The remaining two columns are horizontal axes; pick the longer one
            others = [i for i in range(3) if i != h_idx]
            major_h_idx = others[int(np.argmax(extent[others]))]
            h_axis = R[:, major_h_idx]  # 3-vector in the robot base frame

            # Project onto the horizontal plane and compute yaw
            # Convention: 0° -> +Y (towards robot), +90° -> -X (right),
            # -90° -> +X (left)
            yaw_deg = -float(np.degrees(np.arctan2(h_axis[0], h_axis[1])))
            # Fold into [-90, +90] because the axis has no intrinsic direction
            yaw_deg = ((yaw_deg + 90.0) % 180.0) - 90.0

            self.fused_object_yaw = yaw_deg
            self.get_logger().info(
                f'Object rotation estimated at 20 cm deviation: '
                f'{self.fused_object_yaw:.2f} degrees'
            )

            initial_yaw = self.object_initial_yaw_deg
            if initial_yaw is not None:
                try:
                    initial_yaw = float(initial_yaw)
                    yaw_delta = self.fused_object_yaw - initial_yaw
                    # Keep delta in [-90, +90] under the same axis-direction ambiguity.
                    yaw_delta = ((yaw_delta + 90.0) % 180.0) - 90.0
                    self.fused_object_yaw_delta = yaw_delta
                    self.get_logger().info(
                        f'Yaw comparison (initial vs deviation): '
                        f'initial={initial_yaw:.2f} deg, '
                        f'deviation={self.fused_object_yaw:.2f} deg, '
                        f'delta={yaw_delta:.2f} deg'
                    )

                    if self.object_is_box and abs(yaw_delta) > self._ORIENTATION_CHANGE_THRESHOLD_DEG:
                        self.get_logger().info(
                            f'Orientation change {abs(yaw_delta):.2f} deg exceeds '
                            f'{self._ORIENTATION_CHANGE_THRESHOLD_DEG:.2f} deg; '
                            'running Stage 3 regeneration.'
                        )
                        response = self.send_grasp_request(
                            request_type=REQUEST_REGENERATE,
                            rotation_deg=yaw_delta,
                        )
                        if bool(response.get('status')):
                            grasp_matrices = response.get('grasps') or []
                            grasp_labels = response.get('labels') or []
                            grasp_widths = response.get('widths') or []

                            poses_msg = PoseArray()
                            poses_msg.header.stamp = self.get_clock().now().to_msg()
                            poses_msg.header.frame_id = self.target_frame
                            for grasp_matrix in grasp_matrices:
                                poses_msg.poses.append(self.matrix_to_pose(grasp_matrix))

                            replanned_msg = GenerateGrasp.Result()
                            replanned_msg.grasp_goals = poses_msg
                            replanned_msg.grasp_labels = grasp_labels
                            replanned_msg.grasp_widths = grasp_widths
                            replanned_msg.status = len(grasp_matrices) > 0
                            self.replanned_grasps_pub.publish(replanned_msg)
                            self.get_logger().info(
                                f'Published {len(grasp_matrices)} replanned grasps to /replanned_grasps.'
                            )
                        else:
                            self.get_logger().warn('Stage 3 regeneration request returned no grasps.')
                except (TypeError, ValueError):
                    self.get_logger().warn(
                        'Initial yaw is not numeric; cannot compute yaw delta.'
                    )
            else:
                self.get_logger().warn(
                    'Initial yaw is unavailable; cannot compute yaw delta.'
                )
        except Exception as exc:
            self.get_logger().error(f'OBB fitting failed during rotation estimation: {exc}')

    def first_camera_info_callback(self, msg):
        self.ffx = msg.k[0]
        self.ffy = msg.k[4]
        self.fcx = msg.k[2]
        self.fcy = msg.k[5]
        self.first_camera_initialized = True
        self.get_logger().info(f"Camera intrinsics received: fx={self.ffx}, fy={self.ffy}, cx={self.fcx}, cy={self.fcy}")
        # Unsubscribe after getting intrinsics
        self.destroy_subscription(self.first_cam_info_sub)

    def second_camera_info_callback(self, msg):
        self.sfx = msg.k[0]
        self.sfy = msg.k[4]
        self.scx = msg.k[2]
        self.scy = msg.k[5]
        self.second_camera_initialized = True
        self.get_logger().info(f"Camera intrinsics received: fx={self.sfx}, fy={self.sfy}, cx={self.scx}, cy={self.scy}")
        # Unsubscribe after getting intrinsics
        if self.second_cam_info_sub is not None:
            self.destroy_subscription(self.second_cam_info_sub)
            self.second_cam_info_sub = None

    def sync_callback_all(self, depth_first_msg, color_first_msg, depth_second_msg, color_second_msg, hand_detection_msg):
        """Unified callback: processes both camera views synchronized with hand detection."""
        # Lookup transforms on first call
        if self.tf_first is None:
            try:
                self.tf_first = self.tf_buffer.lookup_transform(
                    target_frame=self.target_frame,
                    source_frame=self.source_frame_first,
                    time=Time(),
                    timeout=Duration(nanoseconds=100_000_000)
                )
            except Exception as ex:
                self.get_logger().info(
                    f'Could not find transform from {self.source_frame_first} to {self.target_frame}: {ex}'
                )
        
        if self.tf_second is None:
            try:
                self.tf_second = self.tf_buffer.lookup_transform(
                    target_frame=self.target_frame,
                    source_frame=self.source_frame_second,
                    time=Time(),
                    timeout=Duration(nanoseconds=100_000_000)
                )
            except Exception as ex:
                self.get_logger().info(
                    f'Could not find transform from {self.source_frame_second} to {self.target_frame}: {ex}'
                )
        
        # Unregister listener once both transforms are found
        if self.tf_first is not None and self.tf_second is not None:
            self.tf_listener.unregister()
        
        # Process first camera view
        try:
            self.latest_depth_first = self.bridge.imgmsg_to_cv2(depth_first_msg, "32FC1") / 1e3
            self.latest_color_first = self.bridge.imgmsg_to_cv2(color_first_msg, "rgb8")
            if self.start_depth_first is None:
                self.start_depth_first = np.copy(self.latest_depth_first)
            if self.start_color_first is None:
                self.start_color_first = np.copy(self.latest_color_first)
        except Exception as e:
            self.get_logger().error(f'Error converting first camera images: {str(e)}')
        
        # Process second camera view
        try:
            self.latest_depth_second = self.bridge.imgmsg_to_cv2(depth_second_msg, "32FC1") / 1e3
            self.latest_color_second = self.bridge.imgmsg_to_cv2(color_second_msg, "rgb8")
            if self.start_depth_second is None:
                self.start_depth_second = np.copy(self.latest_depth_second)
            if self.start_color_second is None:
                self.start_color_second = np.copy(self.latest_color_second)
        except Exception as e:
            self.get_logger().error(f'Error converting second camera images: {str(e)}')
        
        # self.latest_hand_detection = hand_detection_msg
        # Update hand detection (guaranteed synchronized by timestamp)
        if len(hand_detection_msg.left) > 0 or len(hand_detection_msg.right) > 0:
            self._update_latest_hand_detection(hand_detection_msg)

            sample = {
                'stamp': self.get_clock().now().nanoseconds,
                'depth_first': np.copy(self.latest_depth_first) if self.latest_depth_first is not None else None,
                'color_first': np.copy(self.latest_color_first) if self.latest_color_first is not None else None,
                'depth_second': np.copy(self.latest_depth_second) if self.latest_depth_second is not None else None,
                'color_second': np.copy(self.latest_color_second) if self.latest_color_second is not None else None,
                'hand_detection': hand_detection_msg,
            }

            with self.frame_sample_lock:
                self.frame_samples.append(sample)

    def _recv_exact(self, client, size):
        data = b""
        while len(data) < size:
            chunk = client.recv(size - len(data))
            if not chunk:
                raise ConnectionError('Connection closed while receiving data from segmentation service')
            data += chunk
        return data

    def request_segmentation_masks(self, color_image, label, bbox=None, single_prompt=False):
        """Request segmentation masks from SAM3 service.
        
        Args:
            color_image: Input color image (H x W x 3)
            label: Object label string. If None, uses self.object_label
            bbox: Bounding box (x1, y1, x2, y2) or None
            single_prompt: If True, request only object mask. If False, request object+hand masks (default).
        """

        # Format label based on prompt mode
        if single_prompt:
            formatted_label = f"single:{label}"
        else:
            formatted_label = label

        h, w, _ = color_image.shape

        if bbox is None:
            x1, y1, x2, y2 = 0, 0, 0, 0
        else:
            x1, y1, x2, y2 = bbox

        label_bytes = formatted_label.encode()
        label_len = len(label_bytes)

        client = None
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self.sam_socket_path)

            # Send Header: H, W, x1, y1, x2, y2, label_len
            header = np.array([h, w, x1, y1, x2, y2, label_len], dtype=np.int32)
            client.sendall(header.tobytes())
            client.sendall(label_bytes)

            # Send Image
            client.sendall(color_image.tobytes())

            # Parse response based on prompt mode
            if single_prompt:
                # Single prompt: read 1 ok flag (4 bytes) + 1 mask
                ok_bytes = self._recv_exact(client, 4)
                object_ok = bool(np.frombuffer(ok_bytes, dtype=np.uint32)[0])
                hand_ok = False  # Not applicable for single prompt
                
                # Read single mask
                size_bytes = self._recv_exact(client, 4)
                payload_size = int(np.frombuffer(size_bytes, dtype=np.uint32)[0])
                mask_data = self._recv_exact(client, payload_size)
                object_mask = np.frombuffer(mask_data, dtype=np.uint8).reshape((h, w))
                hand_mask = None
            else:
                # Dual prompt: read 2 ok flags (8 bytes) + 2 masks
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
                f'Segmentation masks received successfully: object_ok={object_ok}, hand_ok={hand_ok}, single_prompt={single_prompt}'
            )

        except Exception as e:
            self.get_logger().error(f"Failed to connect to SAM3 server: {e}")
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

        # if object_mask is not None:
        #     cv2.imwrite('debug_sam3_request_object_mask.png', object_mask * 255)
        # if hand_mask is not None:
        #     cv2.imwrite('debug_sam3_request_hand_mask.png', hand_mask * 255)

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
        """Apply DBSCAN clustering to filter out outliers from the point cloud."""
        if points is None or len(points) <= min_points: 
            return points
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        
        if len(labels) == 0:
            return points
        
        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            return points
        
        # Keep only the largest cluster
        counts = np.bincount(valid_labels)
        largest_cluster_label = np.argmax(counts)
        
        indices = np.where(labels == largest_cluster_label)[0]
        return points[indices]

    def _tf_msg_to_matrix(self, tf_msg):
        tf_matrix = np.eye(4, dtype=np.float32)
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
        return tf_matrix

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
        dims = self.latest_dimensions or {}
        points = np.asarray(object_pc, dtype=np.float32) if object_pc is not None else np.zeros((0, 3), dtype=np.float32)
        center = points.mean(axis=0).astype(np.float32) if len(points) > 0 else np.zeros(3, dtype=np.float32)
        axis_world = self._get_fixed_world_axis()

        observed_height = None
        if len(points) > 0:
            # Estimate object extent along the fixed world axis from point cloud support.
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

        # check for any prior contained in object_label
        reconstruct_flag = True
        for prior in asymmetric_container_priors:
            if prior in self.object_label_first or prior in self.object_label_second:
                reconstruct_flag = False
                break
        
        # reconstruction only implemented for cup/glass-like objects
        # self.get_logger().info('Reconstruction flag set to {} based on object labels: first="{}", second="{}"'.format(
        #     reconstruct_flag, self.object_label_first, self.object_label_second
        # ))
        # NAD CHECK !!!
        if reconstruct_flag and not 'bag' in self.object_label_first and not 'bag' in self.object_label_second:
            selected_reconstruct_label = 'glass' if 'glass' in self.object_label_first or 'glass' in self.object_label_second else 'cup'
        else:
            selected_reconstruct_label = ''
            reconstruct_flag = False
        
        # print(f"Completion metadata: reconstruct_flag={reconstruct_flag}, selected_reconstruct_label='{selected_reconstruct_label}', ")
        return {
            'reconstruct': reconstruct_flag,
            'object_label': selected_reconstruct_label,
            'axis_world': axis_world.tolist(),
            'center_world': center.tolist(),
            'height': constrained_height,
            'top_w': dims.get('top_w'),
            'bot_w': dims.get('bot_w'),
        }

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        result = GenerateGrasp.Result()

        while True:
            with self.state_lock:
                if self.stage2_complete and self.cached_final_grasps is not None:
                    grasp_matrices = np.asarray(self.cached_final_grasps, dtype=np.float32)
                    grasp_labels = list(self.cached_final_labels or [])
                    grasp_widths = list(self.cached_final_widths or [0.0])
                    break
            
            await asyncio.sleep(0.01)

        poses_msg = PoseArray()
        poses_msg.header.stamp = self.get_clock().now().to_msg()
        poses_msg.header.frame_id = self.target_frame
        for grasp_matrix in grasp_matrices:
            pose_msg = self.matrix_to_pose(grasp_matrix)
            poses_msg.poses.append(pose_msg)
            pos = grasp_matrix[:3, 3]
            self.get_logger().info(
                f'Grasp pose chosen - Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]'
            )

        if len(grasp_widths) > 0:
            grasp_width_np = np.asarray(grasp_widths, dtype=np.float32)
            self.get_logger().info(
                f'Received {len(grasp_widths)} grasp widths from service: '
                f'min={float(np.min(grasp_width_np)):.4f}m, '
                f'max={float(np.max(grasp_width_np)):.4f}m'
            )

        result.grasp_goals = poses_msg
        result.grasp_labels = grasp_labels
        result.grasp_widths = grasp_widths
        result.status = len(grasp_matrices) > 0
        if result.status:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result
    
    def goal_callback(self, goal_request):
        """Accept new goals."""
        self.get_logger().info('Received new grasp generation request.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle cancellation requests."""
        self.get_logger().info('Received cancellation request.')
        return CancelResponse.ACCEPT
    
    def send_grasp_request(self, request_type, scene_pc=None, object_pc=None, completion_meta=None, rotation_deg=None):
        """Send a staged grasp request via Unix socket."""
        retry_delay_sec = 0.1

        if scene_pc is None:
            scene_pc = np.zeros((0, 3), dtype=np.float32)
        if object_pc is None:
            object_pc = np.zeros((0, 3), dtype=np.float32)

        while True:
            if not os.path.exists(self.grasp_socket_path):
                self.get_logger().warn(
                    f'Socket {self.grasp_socket_path} not found. Retrying in {retry_delay_sec:.1f}s...'
                )
                time.sleep(retry_delay_sec)
                continue

            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

            try:
                client.connect(self.grasp_socket_path)

                meta_payload = b""
                if completion_meta is not None:
                    meta_payload = json.dumps(completion_meta, separators=(',', ':')).encode('utf-8')
                elif rotation_deg is not None:
                    meta_payload = json.dumps(
                        {'rotation_deg': float(rotation_deg)},
                        separators=(',', ':'),
                    ).encode('utf-8')

                scene_size = scene_pc.shape[0]
                object_size = object_pc.shape[0]

                header = np.array([request_type, scene_size, object_size, len(meta_payload)], dtype=np.int32)
                client.sendall(header.tobytes())

                if scene_size > 0:
                    client.sendall(scene_pc.astype(np.float32).tobytes())

                if object_size > 0:
                    client.sendall(object_pc.astype(np.float32).tobytes())

                if meta_payload:
                    client.sendall(meta_payload)

                if request_type == REQUEST_PREGENERATE:
                    status_bytes = self._recv_exact(client, 4)
                    status = bool(np.frombuffer(status_bytes, dtype=np.int32)[0])
                    return {'status': status}

                response = self._recv_exact(client, 4)
                grasp_count = int(np.frombuffer(response, dtype=np.int32)[0])
                self.get_logger().info(f'Grasp count received: {grasp_count}')

                if grasp_count <= 0:
                    return {'status': False, 'grasps': None, 'labels': None, 'widths': None}

                grasps = []
                for _ in range(grasp_count):
                    pose_bytes = self._recv_exact(client, 64)
                    grasp_matrix = np.frombuffer(pose_bytes, dtype=np.float32).reshape((4, 4))
                    grasps.append(grasp_matrix)

                labels = []
                for _ in range(grasp_count):
                    label_bytes = self._recv_exact(client, 4)
                    labels.append(int(np.frombuffer(label_bytes, dtype=np.int32)[0]))

                widths_bytes = self._recv_exact(client, grasp_count * 4)
                widths = np.frombuffer(widths_bytes, dtype=np.float32).astype(np.float32).tolist()

                return {'status': True, 'grasps': grasps, 'labels': labels, 'widths': widths}

            except KeyboardInterrupt:
                self.get_logger().info('Client shutting down due to keyboard interrupt.')
                client.close()
                return {'status': False, 'grasps': None, 'labels': None, 'widths': None}
            except Exception as e:
                self.get_logger().warn(
                    f'Socket communication error: {str(e)}. Retrying in {retry_delay_sec:.1f}s...'
                )
                client.close()
                time.sleep(retry_delay_sec)
    
    def matrix_to_pose(self, matrix):
        """
        Convert a 4x4 transformation matrix to a PoseStamped message.
        
        Args:
            matrix: 4x4 numpy array representing the pose
            
        Returns:
            PoseStamped message
        """
        pose_msg = Pose()
        
        # Extract position (translation vector)
        pose_msg.position.x = float(matrix[0, 3])
        pose_msg.position.y = float(matrix[1, 3])
        pose_msg.position.z = float(matrix[2, 3])
        
        # Extract orientation (rotation matrix to quaternion)
        rotation_matrix = matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # Returns [x, y, z, w]
        
        pose_msg.orientation.x = float(quat[0])
        pose_msg.orientation.y = float(quat[1])
        pose_msg.orientation.z = float(quat[2])
        pose_msg.orientation.w = float(quat[3])
        
        return pose_msg


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
