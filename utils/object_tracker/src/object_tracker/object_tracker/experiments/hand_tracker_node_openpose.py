import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Bool, String


try:
	from utils.config import object_priors, symmetric_container_priors, other_priors, vlm_prompt, color_camera_topic_suffix
except ModuleNotFoundError:
	sys.path.append(str(Path("/home/student-team/rgmc-2026/")))
	from utils.config import object_priors, symmetric_container_priors, other_priors, vlm_prompt, color_camera_topic_suffix

def _rotation_matrix_from_quaternion(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )

def _skew_symmetric(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


class HandTrackerNode(Node):
    def __init__(self):
        super().__init__('hand_tracker')

        self.MAX_HISTORY = 40
        PROXIMITY_THRESH_CENTER = 40.0 #45.0
        PROXIMITY_THRESH_BBOX = 10.0
        EPIPOLAR_GATE_THRESH = 40.0
        TRACKING_THRESH = 100.0
        self.PROXIMITY_WINDOW = 6
        self.palm_indices = [0, 1, 2, 5, 9, 13, 17]
        self.finger_indices = [3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
        self.hand_indices = self.palm_indices + self.finger_indices
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]
        
        self.bridge = CvBridge()

        self.declare_parameter('mode', 'dual')
        self.declare_parameter('proximity_logic', 'or')
        self.declare_parameter('proximity_reference', 'bbox') # 'center' as object center, or 'bbox' as its edges
        self.declare_parameter('camera_left_color', '/camera/left'+color_camera_topic_suffix)
        self.declare_parameter('camera_right_color', '/camera/right'+color_camera_topic_suffix)
        self.declare_parameter('object_detections_topic', '/object_detections')
        self.declare_parameter('debug_visualization', True)
        self.declare_parameter('show_debug_window', True)
        self.declare_parameter('epipolar_gate_threshold_px', EPIPOLAR_GATE_THRESH)
        self.declare_parameter('tracking_threshold_px', TRACKING_THRESH)
        self.declare_parameter('openpose_model_folder', '/usr/local/src/openpose/models/')
        self.declare_parameter('openpose_python_path', '')
        self.declare_parameter('openpose_net_resolution', '-1x368')
        self.declare_parameter('openpose_hand_net_resolution', '368x368')
        self.declare_parameter('openpose_disable_blending', True)
        self.declare_parameter('openpose_render_pose', 0)
        self.declare_parameter('openpose_num_gpu', 1)
        self.declare_parameter('openpose_keypoint_threshold', 0.05)

        self.mode = str(self.get_parameter('mode').value or 'dual').lower().strip()
        self.proximity_logic = str(self.get_parameter('proximity_logic').value or 'or').lower().strip()
        self.proximity_reference = str(self.get_parameter('proximity_reference').value or 'center').lower().strip()
        self.proximity_threshold_px = PROXIMITY_THRESH_CENTER if self.proximity_reference == 'center' else PROXIMITY_THRESH_BBOX
        self.tracking_threshold_px = float(self.get_parameter('tracking_threshold_px').value or TRACKING_THRESH)
        self.epipolar_gate_threshold_px = float(self.get_parameter('epipolar_gate_threshold_px').value or EPIPOLAR_GATE_THRESH)
        self.topic_l_rgb = str(self.get_parameter('camera_left_color').value or '/camera/left'+color_camera_topic_suffix)
        self.topic_r_rgb = str(self.get_parameter('camera_right_color').value or '/camera/right'+color_camera_topic_suffix)
        self.object_detections_topic = str(
            self.get_parameter('object_detections_topic').value or '/object_detections'
        )
        self.debug_visualization = bool(
            self.get_parameter('debug_visualization').value or self.get_parameter('show_debug_window').value
        )
        self.debug_window_name = 'hand_tracker_debug'
        self.openpose_model_folder = str(self.get_parameter('openpose_model_folder').value or '')
        self.openpose_python_path = str(self.get_parameter('openpose_python_path').value or '')
        self.openpose_keypoint_threshold = float(
            self.get_parameter('openpose_keypoint_threshold').value or 0.05
        )

        if self.mode not in ('left', 'right', 'dual'):
            self.get_logger().warn(f"Invalid mode '{self.mode}'. Using 'dual'.")
            self.mode = 'dual'
        if self.proximity_logic not in ('and', 'or'):
            self.get_logger().warn(f"Invalid proximity_logic '{self.proximity_logic}'. Using 'or'.")
            self.proximity_logic = 'or'
        if self.proximity_reference not in ('center', 'bbox'):
            self.get_logger().warn(f"Invalid proximity_reference '{self.proximity_reference}'. Using 'center'.")
            self.proximity_reference = 'center'

        self.openpose = self._create_openpose_wrapper()

        self.pub_message = self.create_publisher(String, '/hand_tracker', 10)
        self.handover_detected_pub = self.create_publisher(Bool, 'handover_detected', 10)

        # Try to load intrinsics from file, otherwise we'll subscribe to CameraInfo topics
        self.camera_intrinsics = {}
        self.camera_poses = {}
        self.stereo_models = {}

        # Subscribe to TF static for static transforms and to camera info topics
        self.get_logger().info('Waiting for TF / camera_info to build stereo models...')
        self.tf_sub = self.create_subscription(TFMessage, '/tf_static', self._tf_callback, 10)
        # subscribe to camera info topics for intrinsics
        left_info = f"/camera/left/color/camera_info"
        right_info = f"/camera/right/color/camera_info"
        self.cam_info_sub_left = self.create_subscription(CameraInfo, left_info, self._camera_info_callback_left, 10)
        self.cam_info_sub_right = self.create_subscription(CameraInfo, right_info, self._camera_info_callback_right, 10)

        sync_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub_l_rgb = message_filters.Subscriber(self, Image, self.topic_l_rgb, qos_profile=sync_qos)
        self.sub_r_rgb = message_filters.Subscriber(self, Image, self.topic_r_rgb, qos_profile=sync_qos)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_l_rgb, self.sub_r_rgb], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info(
            f"ApproximateTimeSynchronizer initialized: "
            f"left_topic={self.topic_l_rgb}, right_topic={self.topic_r_rgb}, "
            f"queue_size=10, slop=0.1s"
        )

        self.object_detections_sub = self.create_subscription(
            String,
            self.object_detections_topic,
            self.object_detections_callback,
            sync_qos,
        )

        self.latest_bbox_data = {}
        self.obj_history_left = deque(maxlen=self.MAX_HISTORY)
        self.obj_history_right = deque(maxlen=self.MAX_HISTORY)
        self.obj_bbox_history_left = deque(maxlen=self.MAX_HISTORY)
        self.obj_bbox_history_right = deque(maxlen=self.MAX_HISTORY)
        self.active_state = {
            'left': {
                'started': False,
                'active_hand_center': None,
                'lead': False,
                'landmarks': None,
            },
            'right': {
                'started': False,
                'active_hand_center': None,
                'lead': False,
                'landmarks': None,
            },
        }
        self.leading_view = None
        self.leading_hand_center = None
        self.handover_detected_once = False
        self.handover_log_already_done = False

        self.get_logger().info(
            'Hand Tracker Initialized with internal handover detection '
            f'(mode={self.mode}, proximity_logic={self.proximity_logic}, '
            f'proximity_reference={self.proximity_reference}, '
            f'debug_visualization={self.debug_visualization}, '
            f'epipolar_gate_threshold_px={self.epipolar_gate_threshold_px}).'
        )

    def _camera_info_callback_left(self, msg: CameraInfo):
        self._camera_info_callback(msg, 'left')

    def _create_openpose_wrapper(self):
        if self.openpose_python_path:
            sys.path.append(self.openpose_python_path)
        env_python_path = os.environ.get('OPENPOSE_PYTHON_PATH')
        if env_python_path:
            sys.path.append(env_python_path)
        for candidate_path in (
            '/usr/local/python',
            '/usr/local/openpose/build/python',
            str(Path.home() / 'openpose/build/python'),
        ):
            if Path(candidate_path).exists():
                sys.path.append(candidate_path)

        try:
            from openpose import pyopenpose as op
        except ImportError:
            try:
                import pyopenpose as op
            except ImportError as exc:
                raise ImportError(
                    'Could not import OpenPose Python bindings. Set the ROS parameter '
                    '`openpose_python_path` or the OPENPOSE_PYTHON_PATH environment variable '
                    'to the directory containing pyopenpose/openpose.py.'
                ) from exc

        model_folder = Path(self.openpose_model_folder).expanduser()
        if not model_folder.exists():
            raise FileNotFoundError(
                f'OpenPose model folder not found: {model_folder}. '
                'Set `openpose_model_folder` to your OpenPose models directory.'
            )

        params = {
            'model_folder': str(model_folder),
            'hand': True,
            'face': False,
            'net_resolution': str(self.get_parameter('openpose_net_resolution').value),
            'hand_net_resolution': str(self.get_parameter('openpose_hand_net_resolution').value),
            'disable_blending': bool(self.get_parameter('openpose_disable_blending').value),
            'render_pose': int(self.get_parameter('openpose_render_pose').value),
            'num_gpu': int(self.get_parameter('openpose_num_gpu').value),
        }

        wrapper = op.WrapperPython()
        wrapper.configure(params)
        wrapper.start()
        self.openpose_module = op
        self.get_logger().info(
            'OpenPose initialized for body-guided left/right hand detection '
            f'with model_folder={model_folder}'
        )
        return wrapper

    def _camera_info_callback_right(self, msg: CameraInfo):
        self._camera_info_callback(msg, 'right')

    def _camera_info_callback(self, msg: CameraInfo, side: str):
        try:
            k_matrix = np.array([[msg.k[0], msg.k[1], msg.k[2]], [msg.k[3], msg.k[4], msg.k[5]], [msg.k[6], msg.k[7], msg.k[8]]], dtype=np.float64)
        except Exception as exc:
            self.get_logger().warn(f'Failed to parse CameraInfo for {side}: {exc}')
            return

        self.camera_intrinsics[side] = {'K': k_matrix, 'dist': np.zeros(5, dtype=np.float64)}
        self.get_logger().info(f'Loaded camera intrinsics from topic for {side} camera.')

        # If both intrinsics and poses exist, finalize stereo models
        self._maybe_finalize_stereo_models()

    def _tf_callback(self, msg: TFMessage):
        # TFMessage contains a list of transforms
        for t in msg.transforms:
            child = t.child_frame_id
            if child in ('left_color_optical_frame', 'right_color_optical_frame'):
                try:
                    trans = t.transform.translation
                    rot = t.transform.rotation
                    matrix = np.eye(4, dtype=np.float64)
                    matrix[:3, :3] = _rotation_matrix_from_quaternion(rot.x, rot.y, rot.z, rot.w)
                    matrix[:3, 3] = [float(trans.x), float(trans.y), float(trans.z)]
                    side = 'left' if 'left' in child else 'right'
                    self.camera_poses[side] = matrix
                    self.get_logger().info(f'Loaded camera-to-base transform for {side} from TF.')
                except Exception as exc:
                    self.get_logger().warn(f'Failed parsing TF for {child}: {exc}')

        # If both intrinsics and camera poses are now available, finalize
        self._maybe_finalize_stereo_models()

    def _maybe_finalize_stereo_models(self):
        # Check readiness
        poses_ready = isinstance(self.camera_poses, dict) and ('left' in self.camera_poses and 'right' in self.camera_poses)
        intr_ready = self.camera_intrinsics is not None and ('left' in self.camera_intrinsics and 'right' in self.camera_intrinsics) and self.camera_intrinsics['left'] is not None and self.camera_intrinsics['right'] is not None

        if not poses_ready or not intr_ready:
            return

        # Build stereo models
        self.stereo_models = self._build_stereo_models()
        self.get_logger().info('Stereo models precomputed from TF and CameraInfo topics.')

        # Destroy subscriptions to avoid ongoing callbacks
        self.destroy_subscription(self.tf_sub)
        self.destroy_subscription(self.cam_info_sub_left)
        self.destroy_subscription(self.cam_info_sub_right)

    def _build_stereo_models(self):
        models = {}
        if self.camera_intrinsics is None:
            return models

        for leading_view, other_view in (('left', 'right'), ('right', 'left')):
            if leading_view not in self.camera_poses or other_view not in self.camera_poses:
                continue

            k_lead = self.camera_intrinsics[leading_view]['K']
            k_other = self.camera_intrinsics[other_view]['K']
            t_base_lead = self.camera_poses[leading_view]
            t_base_other = self.camera_poses[other_view]
            t_other_lead = np.linalg.inv(t_base_other) @ t_base_lead

            rotation = t_other_lead[:3, :3]
            translation = t_other_lead[:3, 3]
            fundamental = np.linalg.inv(k_other).T @ _skew_symmetric(translation) @ rotation @ np.linalg.inv(k_lead)

            models[leading_view] = {
                'other_view': other_view,
                'fundamental': fundamental,
            }

        if not models:
            self.get_logger().warn('Stereo models could not be precomputed; epipolar gating will be disabled.')

        return models

    def object_detections_callback(self, msg):
        try:
            self.latest_bbox_data = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            self.get_logger().warn('Failed to parse /object_detections JSON payload.')
            self.latest_bbox_data = {}

    def _bbox_center(self, bbox):
        if not bbox or len(bbox) != 4:
            return None

        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return None

        return (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

    def _update_object_history(self):
        left_bbox = self.latest_bbox_data.get('left_bbox')
        right_bbox = self.latest_bbox_data.get('right_bbox')
        left_center = self._bbox_center(left_bbox)
        right_center = self._bbox_center(right_bbox)

        if left_center is not None:
            self.obj_history_left.append(left_center)
        if right_center is not None:
            self.obj_history_right.append(right_center)
        if left_bbox is not None:
            self.obj_bbox_history_left.append(left_bbox)
        if right_bbox is not None:
            self.obj_bbox_history_right.append(right_bbox)

    def serialize_landmarks(self, points, image_width, image_height):
        serialized = []
        for point in points:
            if point is None:
                serialized.append({'x': None, 'y': None, 'z': None, 'confidence': 0.0})
                continue

            x_px, y_px, confidence = point
            serialized.append(
                {
                    'x': float(x_px) / float(max(image_width - 1, 1)),
                    'y': float(y_px) / float(max(image_height - 1, 1)),
                    'z': 0.0,
                    'confidence': float(confidence),
                }
            )

        return serialized

    def _center_from_indices(self, points, indices):
        selected_points = [points[index] for index in indices if index < len(points)]
        if not selected_points:
            return None

        return (
            int(np.mean([point[0] for point in selected_points])),
            int(np.mean([point[1] for point in selected_points])),
        )

    def _distance(self, first, second):
        return math.hypot(float(first[0] - second[0]), float(first[1] - second[1]))

    # gates with allowed radius around the reference point
    def _distance_with_gate(self, first, second):
        if first is None or second is None:
            return float('inf')

        distance = self._distance(first, second)
        self.get_logger().info(f"Distance from hand to reference: {distance:.2f}px (first: {first}, second: {second})")
        return distance if distance <= self.tracking_threshold_px else float('inf')

    def _point_to_bbox_distance(self, point, bbox):
        if point is None or bbox is None:
            return float('inf')

        x, y = float(point[0]), float(point[1])
        x1, y1, x2, y2 = bbox
        dx = max(x1 - x, 0.0, x - x2)
        dy = max(y1 - y, 0.0, y - y2)
        return math.hypot(dx, dy)

    def weighted_recent_center(self, points, window):
        if not points:
            return None

        recent = np.array(list(points)[-window:], dtype=np.float32)
        if recent.shape[0] == 0:
            return None

        weights = np.arange(1, recent.shape[0] + 1, dtype=np.float32)
        weights /= np.sum(weights)
        center = np.sum(recent * weights[:, None], axis=0)
        return float(center[0]), float(center[1])

    def weighted_recent_bbox(self, bboxes, window):
        if not bboxes:
            return None

        recent = np.array(list(bboxes)[-window:], dtype=np.float32)
        if recent.shape[0] == 0:
            return None

        weights = np.arange(1, recent.shape[0] + 1, dtype=np.float32)
        weights /= np.sum(weights)
        weighted_bbox = np.sum(recent * weights[:, None], axis=0)
        return tuple(float(value) for value in weighted_bbox)

    def _extract_hand_candidates(self, hand_keypoints, side_name, image_width, image_height):
        if hand_keypoints is None:
            return []

        hand_array = np.asarray(hand_keypoints)
        if hand_array.size == 0:
            return []

        if hand_array.ndim == 2:
            hand_array = hand_array[np.newaxis, :, :]

        hands_data = []
        for person_idx, keypoints in enumerate(hand_array):
            points = []
            valid_points = []
            visible_palm_count = 0
            visible_finger_count = 0

            for index, keypoint in enumerate(keypoints):
                if len(keypoint) < 3:
                    points.append(None)
                    continue

                x_px = float(keypoint[0])
                y_px = float(keypoint[1])
                confidence = float(keypoint[2])
                in_image = 0.0 <= x_px < image_width and 0.0 <= y_px < image_height
                if confidence >= self.openpose_keypoint_threshold and in_image:
                    point = (x_px, y_px, confidence)
                    points.append(point)
                    valid_points.append(point)
                    if index in self.palm_indices:
                        visible_palm_count += 1
                    elif index in self.finger_indices:
                        visible_finger_count += 1
                else:
                    points.append(None)

            if not valid_points:
                continue

            center_points = [points[index] for index in self.hand_indices if index < len(points) and points[index] is not None]
            if not center_points:
                continue

            hand_center = (
                int(np.mean([point[0] for point in center_points])),
                int(np.mean([point[1] for point in center_points])),
            )
            visible_landmark_count = visible_palm_count + visible_finger_count
            visibility_score = float(visible_palm_count) + (5.0 * float(visible_finger_count))

            hands_data.append(
                {
                    'hand_center': hand_center,
                    'landmarks': points,
                    'landmarks_serialized': self.serialize_landmarks(points, image_width, image_height),
                    'visible_landmark_count': visible_landmark_count,
                    'visibility_score': visibility_score,
                    'handedness': side_name,
                    'person_index': int(person_idx),
                }
            )

        return hands_data

    def process_camera_view(self, rgb_msg, view_name=None):
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        h, w = cv_rgb.shape[:2]

        datum = self.openpose_module.Datum()
        datum.cvInputData = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
        self.openpose.emplaceAndPop(self.openpose_module.VectorDatum([datum]))

        hands_data = []
        hand_keypoints = getattr(datum, 'handKeypoints', None)
        if hand_keypoints is not None and len(hand_keypoints) >= 2:
            # OpenPose stores handKeypoints[0] as left hand and handKeypoints[1] as right hand.
            hands_data.extend(self._extract_hand_candidates(hand_keypoints[0], 'Left', w, h))
            hands_data.extend(self._extract_hand_candidates(hand_keypoints[1], 'Right', w, h))

        return hands_data, cv_rgb

    def _detect_handover_for_view(self, view_name, hands_data):
        if not hands_data:
            return False, None

        if self.proximity_reference == 'bbox':
            reference = self.weighted_recent_bbox(
                self.obj_bbox_history_left if view_name == 'left' else self.obj_bbox_history_right,
                self.PROXIMITY_WINDOW,
            )
            if reference is None:
                return False, None
            distance_fn = lambda hand: self._point_to_bbox_distance(hand['hand_center'], reference)
        else:
            reference = self.weighted_recent_center(
                self.obj_history_left if view_name == 'left' else self.obj_history_right,
                self.PROXIMITY_WINDOW,
            )
            if reference is None:
                return False, None
            distance_fn = lambda hand: self._distance(hand['hand_center'], reference)

        best_hand = min(hands_data, key=distance_fn)
        best_distance = distance_fn(best_hand)
        best_hand = dict(best_hand)
        best_hand['best_distance'] = best_distance
        # debug
        # self.get_logger().info(f"Best hand for {view_name} view: center={best_hand['hand_center']}, best_distance={best_distance:.2f}")

        if best_distance < self.proximity_threshold_px:
            return True, best_hand

        return False, best_hand

    def _activate_view(self, view_name, hand_data):
        state = self.active_state[view_name]
        state['started'] = True
        state['active_hand_center'] = hand_data['hand_center']
        state['lead'] = view_name == self.leading_view
        state['landmarks'] = hand_data['landmarks_serialized']

        if view_name == self.leading_view:
            self.leading_hand_center = hand_data['hand_center']

    def _activate_view_with_center(self, view_name, hand_center):
        state = self.active_state[view_name]
        state['started'] = hand_center is not None
        state['active_hand_center'] = hand_center
        state['lead'] = view_name == self.leading_view
        state['landmarks'] = None

        if view_name == self.leading_view:
            self.leading_hand_center = hand_center

    def _epipolar_distance(self, fundamental_matrix, source_point, target_point):
        if fundamental_matrix is None or source_point is None or target_point is None:
            return float('inf')

        source_x, source_y = float(source_point[0]), float(source_point[1])
        target_x, target_y = float(target_point[0]), float(target_point[1])

        source_h = np.array([source_x, source_y, 1.0], dtype=np.float64)
        line = fundamental_matrix @ source_h
        denominator = math.hypot(float(line[0]), float(line[1]))
        if denominator == 0.0:
            return float('inf')

        return abs(float(line[0]) * target_x + float(line[1]) * target_y + float(line[2])) / denominator

    def _line_segment_in_image(self, line, image_shape):
        height, width = image_shape[:2]
        a, b, c = [float(value) for value in line]
        points = []

        if abs(b) > 1e-9:
            for x in (0.0, float(width - 1)):
                y = -(a * x + c) / b
                if 0.0 <= y <= height - 1:
                    points.append((int(round(x)), int(round(y))))

        if abs(a) > 1e-9:
            for y in (0.0, float(height - 1)):
                x = -(b * y + c) / a
                if 0.0 <= x <= width - 1:
                    points.append((int(round(x)), int(round(y))))

        unique_points = []
        for point in points:
            if point not in unique_points:
                unique_points.append(point)

        if len(unique_points) < 2:
            return None

        return unique_points[0], unique_points[1]

    def _draw_line_in_image(self, image, line, color, thickness=1):
        segment = self._line_segment_in_image(line, image.shape)
        if segment is None:
            return False

        start, end = segment
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
        return True

    def _draw_epipolar_debug(self, debug_images):
        if self.leading_view not in self.stereo_models or self.leading_hand_center is None:
            return

        stereo_model = self.stereo_models[self.leading_view]
        other_view = stereo_model['other_view']
        target_image = debug_images.get(other_view)
        source_image = debug_images.get(self.leading_view)
        if target_image is None or source_image is None:
            return

        source_x, source_y = float(self.leading_hand_center[0]), float(self.leading_hand_center[1])
        source_h = np.array([source_x, source_y, 1.0], dtype=np.float64)
        line = stereo_model['fundamental'] @ source_h
        line_norm = math.hypot(float(line[0]), float(line[1]))
        if line_norm == 0.0:
            return

        cv2.circle(source_image, self.leading_hand_center, 11, (0, 255, 255), 2, cv2.LINE_AA)

        gate_offset = self.epipolar_gate_threshold_px * line_norm
        self._draw_line_in_image(target_image, line, (0, 255, 255), 2)
        self._draw_line_in_image(target_image, (line[0], line[1], line[2] + gate_offset), (80, 180, 255), 1)
        self._draw_line_in_image(target_image, (line[0], line[1], line[2] - gate_offset), (80, 180, 255), 1)

        cv2.putText(
            target_image,
            f'epipolar gate +/-{self.epipolar_gate_threshold_px:.0f}px',
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    
    def _gate_by_distance(self, candidates, reference_center):
        if candidates is None:
            return None

        gated_candidates = []
        for hand_data in candidates:
            hand_center = hand_data.get('hand_center')
            distance = self._distance_with_gate(hand_center, reference_center)
            if distance != float('inf'):
                gated_candidates.append((distance, hand_data))

        if not gated_candidates:
            self.get_logger().info(
                f"No hand candidates within tracking threshold {self.tracking_threshold_px:.2f}px "
                f"of reference {reference_center}."
            )
            return None

        return min(
            gated_candidates,
            key=lambda item: item[0],
        )[1]

    def _select_hand_for_view(self, view_name, hands_data):
        state = self.active_state[view_name]
        if not state['started'] or not hands_data:
            return None

        reference_center = state['active_hand_center']
        if view_name != self.leading_view and self.leading_view in self.stereo_models and self.leading_hand_center is not None:
            stereo_model = self.stereo_models[self.leading_view]
            accepted_hands = []
            for hand_data in hands_data:
                line_distance = self._epipolar_distance(
                    stereo_model['fundamental'],
                    self.leading_hand_center,
                    hand_data['hand_center'],
                )
                if line_distance <= self.epipolar_gate_threshold_px:
                    accepted_hands.append(hand_data)
                else:
                    self.get_logger().warn(
                        f"Rejected hand at {hand_data['hand_center']} for {view_name} view due to epipolar distance {line_distance:.2f} exceeding threshold (leading hand: {self.leading_hand_center})."
                    )

            if not accepted_hands:
                return None

            # Use reference center if available for secondary filtering, otherwise use visibility score
            if reference_center is not None:
                return self._gate_by_distance(accepted_hands, reference_center)
            else:
                return max(
                    accepted_hands,
                    key=lambda hand: hand['visibility_score'],
                )

        if reference_center is None:
            return max(hands_data, key=lambda hand: hand['visibility_score'])

        return self._gate_by_distance(hands_data, reference_center)
    
    def _choose_leading_view(self, left_candidate, right_candidate, cv_img_l=None, cv_img_r=None):
        if left_candidate is None:
            self.get_logger().info('Left candidate is None, selecting right view as leading.')
            return 'right', right_candidate
        if right_candidate is None:
            self.get_logger().info('Right candidate is None, selecting left view as leading.')
            return 'left', left_candidate
        
        handedness_l = left_candidate.get('handedness')
        handedness_r = right_candidate.get('handedness')

        if handedness_l == 'Left' and handedness_r == 'Left':
            self.get_logger().info('Both candidates are Left hands, selecting right view as leading to avoid ambiguity.')
            return 'right', right_candidate
        if handedness_l == 'Right' and handedness_r == 'Right':
            self.get_logger().info('Both candidates are Right hands, selecting left view as leading to avoid ambiguity.')
            return 'left', left_candidate
        
        # one view detection is missing - take the other detection
        if handedness_l is None and handedness_r == 'Left':
            self.get_logger().info('Left candidate is None, selecting right view as leading.')
            return 'right', right_candidate
        if handedness_l is None and handedness_r == 'Right':
            self.get_logger().info('Left candidate is None, selecting left view as leading.')
            return 'left', left_candidate
        if handedness_l == 'Right' and handedness_r is None:
            self.get_logger().info('Right candidate is None, selecting left view as leading.')
            return 'left', left_candidate
        if handedness_l == 'Left' and handedness_r is None:
            self.get_logger().info('Right candidate is None, selecting right view as leading.')
            return 'right', right_candidate

        self.get_logger().info("OpenPose candidates disagree or are ambiguous. Selecting leading view based on visibility score.")
        
        candidates = [('left', left_candidate), ('right', right_candidate)]
        return max(candidates, key=lambda item: (item[1]['visibility_score']))#, -item[1].get('best_distance', float('inf'))))

    def _update_view_role(self, leading_view):
        self.leading_view = leading_view
        for view_name, state in self.active_state.items():
            state['lead'] = view_name == leading_view

    def _build_view_payload(self, hand_data):
        selected = hand_data if hand_data is not None else None

        return {
            'detected': selected is not None,
            'active_hand_center': selected['hand_center'] if selected is not None else None,
            'landmarks': selected['landmarks_serialized'] if selected is not None else None,
            'handedness': selected.get('handedness') if selected is not None else None,
        }

    def _enabled_views(self):
        if self.mode == 'left':
            return ['left']
        if self.mode == 'right':
            return ['right']
        return ['left', 'right']

    def _publish_handover_detected_once(self):
        if self.handover_detected_once:
            return

        out_msg = Bool()
        out_msg.data = True
        self.handover_detected_pub.publish(out_msg)
        self.handover_detected_once = True

        if not self.handover_log_already_done:
            self.handover_log_already_done = True
            msg = {"t_human_first_contact_ms": ((time.time_ns()) / 1e6)}
            self.get_logger().get_child("SUBMISSION").info(f"{json.dumps(msg)}")

        self.get_logger().info(
            'Handover detected and published once. Tracking started. '
            f'leading_view={self.leading_view}, leading_hand_center={self.leading_hand_center}'
        )

    def _get_center_for_handover_fallback(self, view_name):
        bbox_key = 'left_bbox' if view_name == 'left' else 'right_bbox'
        bbox_center = self._bbox_center(self.latest_bbox_data.get(bbox_key))
        if bbox_center is not None:
            return bbox_center

        history = self.obj_history_left if view_name == 'left' else self.obj_history_right
        history_center = self.weighted_recent_center(history, self.PROXIMITY_WINDOW)
        if history_center is None:
            return None
        return (int(history_center[0]), int(history_center[1]))

    def _combine_handover_flags(self, left_flag, right_flag):
        if self.mode == 'left':
            return left_flag
        if self.mode == 'right':
            return right_flag

        if self.proximity_logic == 'and':
            return left_flag and right_flag

        return left_flag or right_flag

    def sync_callback(self, l_rgb, r_rgb):
        # Validate that both messages are received from ApproximateTimeSynchronizer
        if l_rgb is None or r_rgb is None:
            self.get_logger().warn("sync_callback received None message(s)")
            return
        
        hands_l, cv_img_l = self.process_camera_view(l_rgb, 'left')
        hands_r, cv_img_r = self.process_camera_view(r_rgb, 'right')

        if not self.handover_detected_once:
            # Handover detection phase
            self._update_object_history()

            left_handover, left_candidate = self._detect_handover_for_view('left', hands_l)
            right_handover, right_candidate = self._detect_handover_for_view('right', hands_r)

            handover_detected_now = self._combine_handover_flags(left_handover, right_handover)
            if handover_detected_now:
                leading_view, leading_candidate = self._choose_leading_view(left_candidate, right_candidate, cv_img_l, cv_img_r)
                if leading_view is None:
                    return

                self._update_view_role(leading_view)
                self._activate_view(leading_view, leading_candidate)

                for view_name in self._enabled_views():
                    if view_name == leading_view:
                        continue

                    candidate = left_candidate if view_name == 'left' else right_candidate
                    if candidate is not None:
                        self._activate_view(view_name, candidate)
                    else:
                        fallback_center = self._get_center_for_handover_fallback(view_name)
                        self._activate_view_with_center(view_name, fallback_center)

                self._publish_handover_detected_once()
            return
        else:
            # Tracking phase (post-handover)
            selected_left = self._select_hand_for_view('left', hands_l)
            if selected_left is not None:
                self._activate_view('left', selected_left)

            selected_right = self._select_hand_for_view('right', hands_r)
            if selected_right is not None:
                self._activate_view('right', selected_right)

            message = {}
            message['active_view'] = self.leading_view
            if 'left' in self._enabled_views() and selected_left is not None:
                message['left'] = self._build_view_payload(selected_left)
            else:
                message['left'] = {
                    'detected': False,
                    'active_hand_center': None,
                    'landmarks': None,
                }
            if 'right' in self._enabled_views() and selected_right is not None:
                message['right'] = self._build_view_payload(selected_right)
            else:
                message['right'] = {
                    'detected': False,
                    'active_hand_center': None,
                    'landmarks': None,
                }

            if 'left' not in message and 'right' not in message:
                #return
                pass

            # Add timestamp from RGB image
            message['timestamp'] = {
                'sec': int(l_rgb.header.stamp.sec),
                'nanosec': int(l_rgb.header.stamp.nanosec),
            }

            message_json = String()
            message_json.data = json.dumps(message)
            self.pub_message.publish(message_json)

            if self.debug_visualization:
                debug_left = self.draw_debug_hands(cv_img_l, hands_l, selected_left, 'left')
                debug_right = self.draw_debug_hands(cv_img_r, hands_r, selected_right, 'right')
                self._draw_epipolar_debug({'left': debug_left, 'right': debug_right})

                cv2.putText(
                    debug_left,
                    'Left camera',
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_right,
                    'Right camera',
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                target_height = max(debug_left.shape[0], debug_right.shape[0])
                if debug_left.shape[0] < target_height:
                    pad = target_height - debug_left.shape[0]
                    debug_left = cv2.copyMakeBorder(debug_left, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                if debug_right.shape[0] < target_height:
                    pad = target_height - debug_right.shape[0]
                    debug_right = cv2.copyMakeBorder(debug_right, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                separator = np.zeros((target_height, 20, 3), dtype=np.uint8)
                debug_view = np.hstack([debug_left, separator, debug_right])
                cv2.imshow(self.debug_window_name, debug_view)
                cv2.waitKey(1)

            self.get_logger().info(
                f"left={message.get('left', {}).get('active_hand_center')}"
                + f" | right={message.get('right', {}).get('active_hand_center')}"
            )

    def draw_debug_hands(self, image, hands_data, selected_hand, view_name):
        debug_image = image.copy()

        cv2.putText(
            debug_image,
            f'{view_name.upper()} VIEW',
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for hand_data in hands_data:
            is_selected = selected_hand is not None and hand_data['hand_center'] == selected_hand['hand_center']
            hand_color = (0, 255, 0) if is_selected else (0, 165, 255)
            points = hand_data.get('landmarks') or []

            for start_idx, end_idx in self.hand_connections:
                if start_idx >= len(points) or end_idx >= len(points):
                    continue
                start = points[start_idx]
                end = points[end_idx]
                if start is None or end is None:
                    continue
                cv2.line(
                    debug_image,
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    hand_color,
                    2,
                    cv2.LINE_AA,
                )

            for point in points:
                if point is None:
                    continue
                cv2.circle(debug_image, (int(point[0]), int(point[1])), 3, hand_color, -1)

            cv2.circle(debug_image, hand_data['hand_center'], 8, hand_color, -1)

            cv2.putText(
                debug_image,
                f'{view_name.upper()} ACTIVE ({hand_data.get("handedness", "?")})'
                if is_selected
                else f'HAND ({hand_data.get("handedness", "?")})',
                (hand_data['hand_center'][0] + 8, hand_data['hand_center'][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                hand_color,
                2,
                cv2.LINE_AA,
            )

        return debug_image


def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
