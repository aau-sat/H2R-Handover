import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
import message_filters
import mediapipe as mp
import numpy as np
import yaml
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Bool, String
from google.protobuf.json_format import MessageToDict
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
        self.PROXIMITY_THRESH = 40.0 #45.0
        self.PROXIMITY_WINDOW = 6
        self.EPIPOLAR_GATE_THRESH = 45.0
        
        # Body landmarks (from Pose model)
        self.body_landmarks = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
        }
        
        # Hand landmarks (from Pose model)
        self.hand_landmarks_pose = {
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
        }
        
        # All hand landmarks indices for visibility/proximity calculations
        self.hand_indices = [17, 18, 19, 20, 21, 22]
        
        self.bridge = CvBridge()

        self.declare_parameter('mode', 'dual')
        self.declare_parameter('proximity_logic', 'or')
        self.declare_parameter('camera_left_color', '/camera/left/color/image_raw')
        self.declare_parameter('camera_right_color', '/camera/right/color/image_raw')
        self.declare_parameter('object_detections_topic', '/object_detections')
        self.declare_parameter('debug_visualization', True)
        self.declare_parameter('show_debug_window', True)
        self.declare_parameter('epipolar_gate_threshold_px', self.EPIPOLAR_GATE_THRESH)
        self.declare_parameter('pose_model_path', 'pose_landmarker.task')

        self.mode = str(self.get_parameter('mode').value or 'dual').lower().strip()
        self.proximity_logic = str(self.get_parameter('proximity_logic').value or 'or').lower().strip()
        self.topic_l_rgb = str(self.get_parameter('camera_left_color').value or '/camera/left/color/image_raw')
        self.topic_r_rgb = str(self.get_parameter('camera_right_color').value or '/camera/right/color/image_raw')
        self.object_detections_topic = str(
            self.get_parameter('object_detections_topic').value or '/object_detections'
        )
        self.debug_visualization = bool(
            self.get_parameter('debug_visualization').value or self.get_parameter('show_debug_window').value
        )
        self.debug_window_name = 'hand_tracker_debug'
        self.epipolar_gate_threshold_px = float(
            self.get_parameter('epipolar_gate_threshold_px').value or self.EPIPOLAR_GATE_THRESH
        )
        pose_model_path = str(self.get_parameter('pose_model_path').value)

        if self.mode not in ('left', 'right', 'dual'):
            self.get_logger().warn(f"Invalid mode '{self.mode}'. Using 'dual'.")
            self.mode = 'dual'
        if self.proximity_logic not in ('and', 'or'):
            self.get_logger().warn(f"Invalid proximity_logic '{self.proximity_logic}'. Using 'or'.")
            self.proximity_logic = 'or'

        # Initialize Pose Landmarker (holistic model)
        try:
            model_path = Path(pose_model_path)
            if not model_path.exists():
                self.get_logger().error(
                    f"Pose Landmarker model not found at {pose_model_path}. "
                )
                raise FileNotFoundError(f"Model file not found: {pose_model_path}")
            
            base_options = python.BaseOptions(model_asset_path=str(pose_model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            self.get_logger().info(f"Pose Landmarker initialized successfully from {pose_model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Pose Landmarker: {e}")
            raise

        self.pub_message = self.create_publisher(String, '/hand_tracker', 10)
        self.handover_detected_pub = self.create_publisher(Bool, 'handover_detected', 10)

        # Try to load intrinsics from file, otherwise we'll subscribe to CameraInfo topics
        self.camera_intrinsics = {}
        self.camera_poses = {}

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

        self.object_detections_sub = self.create_subscription(
            String,
            self.object_detections_topic,
            self.object_detections_callback,
            sync_qos,
        )

        self.latest_bbox_data = {}
        self.obj_history_left = deque(maxlen=self.MAX_HISTORY)
        self.obj_history_right = deque(maxlen=self.MAX_HISTORY)
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
            f'debug_visualization={self.debug_visualization}, '
            f'epipolar_gate_threshold_px={self.epipolar_gate_threshold_px}).'
        )

    def _camera_info_callback_left(self, msg: CameraInfo):
        self._camera_info_callback(msg, 'left')

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
        left_center = self._bbox_center(self.latest_bbox_data.get('left_bbox'))
        right_center = self._bbox_center(self.latest_bbox_data.get('right_bbox'))

        if left_center is not None:
            self.obj_history_left.append(left_center)
        if right_center is not None:
            self.obj_history_right.append(right_center)

    def serialize_landmarks(self, landmarks):
        return [
            {'x': float(lm.x), 'y': float(lm.y), 'z': float(lm.z)}
            for lm in landmarks.landmark
        ]

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

    def process_camera_view(self, rgb_msg, view_name=None):
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        h, w = cv_rgb.shape[:2]

        # Convert to MediaImage for Pose Landmarker
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_rgb)
        pose_results = self.pose_landmarker.detect(mp_image)
        
        hands_data = []

        if pose_results.pose_landmarks:
            for pose_idx, pose_landmarks in enumerate(pose_results.pose_landmarks):
                # Extract body landmarks for handedness detection
                body_lms = {name: pose_landmarks[idx] for name, idx in self.body_landmarks.items()}
                
                # Extract hand landmarks
                hand_lms_dict = {}
                for name, idx in self.hand_landmarks_pose.items():
                    hand_lms_dict[name] = pose_landmarks[idx]
                
                # Determine handedness based on wrist positions relative to shoulders
                handedness = self._determine_handedness_from_body(body_lms)
                
                # Get hand center from hand landmarks
                hand_points = [
                    (int(pose_landmarks[idx].x * w), int(pose_landmarks[idx].y * h))
                    for idx in self.hand_indices
                ]
                
                # Count visible hand landmarks
                visible_hand_count = sum(
                    1
                    for idx in self.hand_indices
                    if pose_landmarks[idx].presence > 0.5 and 0 <= int(pose_landmarks[idx].x * w) < w and 0 <= int(pose_landmarks[idx].y * h) < h
                )
                
                visibility_score = float(visible_hand_count) * 5.0  # Weight hand visibility higher
                hand_center = self._center_from_indices(hand_points, list(range(len(hand_points))))
                
                if hand_center is None or visible_hand_count < 2:
                    continue
                
                # Serialize hand landmarks (only hand part from pose results)
                hand_landmarks_serialized = [
                    {'x': float(pose_landmarks[idx].x), 'y': float(pose_landmarks[idx].y), 'z': float(pose_landmarks[idx].z)}
                    for idx in self.hand_indices
                ]
                
                # Store body landmarks as well for debug visualization
                body_landmarks_serialized = [
                    {'x': float(body_lms[name].x), 'y': float(body_lms[name].y), 'z': float(body_lms[name].z)}
                    for name in self.body_landmarks.keys()
                ]
                
                hands_data.append(
                    {
                        'hand_center': hand_center,
                        'landmarks': pose_landmarks,  # Full pose landmarks for drawing
                        'hand_landmarks_serialized': hand_landmarks_serialized,
                        'body_landmarks_serialized': body_landmarks_serialized,
                        'visible_landmark_count': visible_hand_count,
                        'visibility_score': visibility_score,
                        'handedness': handedness,
                        'hand_indices': self.hand_indices,
                        'body_landmarks_dict': self.body_landmarks,
                    }
                )

        return hands_data, cv_rgb

    def _determine_handedness_from_body(self, body_lms):
        """
        Determine handedness (Left or Right) based on body landmarks.
        Uses shoulder and wrist positions to infer which hand each set of landmarks belongs to.
        """
        left_shoulder = body_lms.get('left_shoulder')
        right_shoulder = body_lms.get('right_shoulder')
        left_wrist = body_lms.get('left_wrist')
        right_wrist = body_lms.get('right_wrist')
        
        if not all([left_shoulder, right_shoulder, left_wrist, right_wrist]):
            return 'Unknown'
        
        # Left hand has left wrist on the left side of center line (between shoulders)
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2.0
        left_wrist_offset = abs(left_wrist.x - shoulder_center_x)
        right_wrist_offset = abs(right_wrist.x - shoulder_center_x)
        
        # Whichever wrist is more visible/prominent determines the hand orientation
        # For now, we detect left hand if left wrist is more prominent
        if left_wrist.visibility > 0.5 and left_wrist.visibility > right_wrist.visibility:
            if left_wrist.x < shoulder_center_x:
                return 'Left'
        
        if right_wrist.visibility > 0.5 and right_wrist.visibility >= left_wrist.visibility:
            if right_wrist.x > shoulder_center_x:
                return 'Right'
        
        # Fallback: determine based on which hand landmarks are most visible
        return 'Right'

    def _detect_handover_for_view(self, view_name, hands_data):
        object_center = self.weighted_recent_center(
            self.obj_history_left if view_name == 'left' else self.obj_history_right,
            self.PROXIMITY_WINDOW,
        )

        if object_center is None or not hands_data:
            return False, None, None

        best_hand = min(hands_data, key=lambda hand: self._distance(hand['hand_center'], object_center))
        best_distance = self._distance(best_hand['hand_center'], object_center)
        best_hand = dict(best_hand)
        best_hand['best_distance'] = best_distance

        if best_distance < self.PROXIMITY_THRESH:
            return True, best_hand, best_distance

        return False, best_hand, best_distance

    def _activate_view(self, view_name, hand_data):
        state = self.active_state[view_name]
        state['started'] = True
        state['active_hand_center'] = hand_data['hand_center']
        state['lead'] = view_name == self.leading_view
        state['landmarks'] = hand_data['hand_landmarks_serialized']

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

    def _epipolar_distance(self, fundamental_matrix, point):
        if fundamental_matrix is None or point is None:
            return float('inf')

        x_coord, y_coord = float(point[0]), float(point[1])
        homogeneous_point = np.array([x_coord, y_coord, 1.0], dtype=np.float64)
        line = fundamental_matrix @ homogeneous_point
        denominator = math.hypot(float(line[0]), float(line[1]))
        if denominator == 0.0:
            return float('inf')

        return abs(float(line[0]) * x_coord + float(line[1]) * y_coord + float(line[2])) / denominator

    def _select_hand_for_view(self, view_name, hands_data):
        state = self.active_state[view_name]
        if not state['started'] or not hands_data:
            return None

        reference_center = state['active_hand_center']
        if view_name != self.leading_view and self.leading_view in self.stereo_models and self.leading_hand_center is not None:
            stereo_model = self.stereo_models[self.leading_view]
            accepted_hands = []
            for hand_data in hands_data:
                line_distance = self._epipolar_distance(stereo_model['fundamental'], hand_data['hand_center'])
                if line_distance <= self.epipolar_gate_threshold_px:
                    accepted_hands.append((line_distance, hand_data))
                else:
                    self.get_logger().warn(
                        f"Rejected hand at {hand_data['hand_center']} for {view_name} view due to epipolar distance {line_distance:.2f} exceeding threshold."
                    )

            if not accepted_hands:
                return None

            return min(
                accepted_hands,
                key=lambda item: (
                    item[0],
                    -item[1]['visibility_score'],
                ),
            )[1]

        if reference_center is None:
            return max(hands_data, key=lambda hand: hand['visibility_score'])

        return min(hands_data, key=lambda hand: self._distance(hand['hand_center'], reference_center))
    
    def _choose_leading_view(self, left_candidate, right_candidate):
        if left_candidate is None:
            self.get_logger().info('Left candidate is None, selecting right view as leading.')
            return 'right', right_candidate
        if right_candidate is None:
            self.get_logger().info('Right candidate is None, selecting left view as leading.')
            return 'left', left_candidate
        if left_candidate['handedness'] == 'Left' and right_candidate['handedness'] == 'Left':
            self.get_logger().info('Both candidates are Left hands, selecting right view as leading to avoid ambiguity.')
            return 'right', right_candidate
        if left_candidate['handedness'] == 'Right' and right_candidate['handedness'] == 'Right':
            self.get_logger().info('Both candidates are Right hands, selecting left view as leading to avoid ambiguity.')
            return 'left', left_candidate
        
        candidates = [('left', left_candidate), ('right', right_candidate)]

        if not candidates:
            return None, None

        return max(candidates, key=lambda item: (item[1]['visibility_score'], -item[1].get('best_distance', float('inf'))))

    # def _choose_leading_view(self, left_candidate, right_candidate):
    #     candidates = []
    #     if left_candidate is not None:
    #         candidates.append(('left', left_candidate))
    #     if right_candidate is not None:
    #         candidates.append(('right', right_candidate))

    #     if not candidates:
    #         return None, None

    #     return max(candidates, key=lambda item: (item[1]['visibility_score']))#, -item[1].get('best_distance', float('inf'))))

    def _update_view_role(self, leading_view):
        self.leading_view = leading_view
        for view_name, state in self.active_state.items():
            state['lead'] = view_name == leading_view

    def _update_active_view(self, view_name, hands_data):
        state = self.active_state[view_name]
        if not state['started'] or not hands_data:
            return None

        active_center = state['active_hand_center']
        if active_center is None:
            selected_hand = hands_data[0]
        else:
            selected_hand = min(hands_data, key=lambda hand: self._distance(hand['hand_center'], active_center))

        state['active_hand_center'] = selected_hand['hand_center']
        state['lead'] = view_name == self.leading_view
        state['landmarks'] = selected_hand['hand_landmarks_serialized']

        if view_name == self.leading_view:
            self.leading_hand_center = selected_hand['hand_center']

        return selected_hand

    def _build_view_payload(self, hand_data):
        selected = hand_data if hand_data is not None else None

        return {
            'detected': selected is not None,
            'active_hand_center': selected['hand_center'] if selected is not None else None,
            'landmarks': selected['hand_landmarks_serialized'] if selected is not None else None,
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
        hands_l, cv_img_l = self.process_camera_view(l_rgb, 'left')
        hands_r, cv_img_r = self.process_camera_view(r_rgb, 'right')

        if not self.handover_detected_once:
            # Handover detection phase
            self._update_object_history()

            left_handover, left_candidate, _ = self._detect_handover_for_view('left', hands_l)
            right_handover, right_candidate, _ = self._detect_handover_for_view('right', hands_r)

            handover_detected_now = self._combine_handover_flags(left_handover, right_handover)
            if handover_detected_now:
                leading_view, leading_candidate = self._choose_leading_view(left_candidate, right_candidate)
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
            message['leading_hand_center'] = self.leading_hand_center
            if 'left' in self._enabled_views() and selected_left is not None:
                message['left'] = self._build_view_payload(selected_left)
            if 'right' in self._enabled_views() and selected_right is not None:
                message['right'] = self._build_view_payload(selected_right)

            if 'left' not in message and 'right' not in message:
                return

            message_json = String()
            message_json.data = json.dumps(message)
            self.pub_message.publish(message_json)

            if self.debug_visualization:
                debug_left = self.draw_debug_hands(cv_img_l, hands_l, selected_left, 'left')
                debug_right = self.draw_debug_hands(cv_img_r, hands_r, selected_right, 'right')

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
        h, w = debug_image.shape[:2]

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
            body_color = (255, 0, 0) if is_selected else (0, 255, 255)
            
            # Draw body landmarks
            if 'body_landmarks_serialized' in hand_data:
                for body_name, body_lm in zip(hand_data['body_landmarks_dict'].keys(), hand_data['body_landmarks_serialized']):
                    x = int(body_lm['x'] * w)
                    y = int(body_lm['y'] * h)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(debug_image, (x, y), 6, body_color, -1)
            
            # Draw hand landmarks
            if 'landmarks' in hand_data:
                pose_lms = hand_data['landmarks']
                for idx in hand_data['hand_indices']:
                    lm = pose_lms[idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    if 0 <= x < w and 0 <= y < h and lm.presence > 0.5:
                        cv2.circle(debug_image, (x, y), 5, hand_color, -1)
            
            # Draw hand center and label
            cv2.circle(debug_image, hand_data['hand_center'], 8, hand_color, -1)

            cv2.putText(
                debug_image,
                f'{view_name.upper()} ACTIVE' if is_selected else f'HAND ({hand_data["handedness"]})',
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