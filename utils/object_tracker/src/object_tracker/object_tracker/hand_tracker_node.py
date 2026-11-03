from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import message_filters
import cv2
import json
import mediapipe as mp
from collections import deque, defaultdict
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import sys
import os
import importlib.util
try:
	from utils.config import color_camera_topic_suffix
except ModuleNotFoundError:
	sys.path.append(str(Path(__file__).resolve().parents[8]))
	from utils.config import color_camera_topic_suffix


class HandTrackerNode(Node):
	def __init__(self):
		super().__init__('dual_hand_detector')

		# ------------------- PARAMETERS -------------------
		self.MAX_HISTORY = 40
		self.PROXIMITY_THRESH = 45.0
		self.PROXIMITY_WINDOW = 6
		self.finger_tip_indices = [4, 8, 12, 16, 20]

		self.bridge = CvBridge()

		# Initialize MediaPipe Hands
		# Kept in memory here so we don't reload the model on every single frame
		self.mp_hands = mp.solutions.hands
		self.hands1 = self.mp_hands.Hands(
			static_image_mode=False,
			max_num_hands=2,
			min_detection_confidence=0.5
		)
		self.hands2 = self.mp_hands.Hands(
			static_image_mode=False,
			max_num_hands=2,
			min_detection_confidence=0.5
		)

		self.obj_history = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))
		self.hand_history_1 = deque(maxlen=self.MAX_HISTORY)
		self.hand_history_2 = deque(maxlen=self.MAX_HISTORY)
		self.active_hand_id_1 = None
		self.active_hand_id_2 = None

		# --- NEW: Initialize MediaPipe Drawing utilities ---
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_drawing_styles = mp.solutions.drawing_styles

		# Declare the input topics
		self.declare_parameter('camera_topic_left', '/camera/left'+color_camera_topic_suffix)
		self.camera_topic_left = str(self.get_parameter('camera_topic_left').value or '/camera/left/color/image_raw')
		self.declare_parameter('camera_topic_right', '/camera/right'+color_camera_topic_suffix)
		self.camera_topic_right = str(self.get_parameter('camera_topic_right').value or '/camera/right/color/image_raw')
		self.declare_parameter('publish_mode', 'continuous')
		self.declare_parameter('handover_detected_topic', '/handover_detected')
		self.declare_parameter('handover_exit_extra_frames', 10)
		self.publish_mode = str(self.get_parameter('publish_mode').value or 'continuous').lower().strip()
		self.handover_detected_topic = str(self.get_parameter('handover_detected_topic').value or '/handover_detected')
		self.handover_exit_extra_frames = int(self.get_parameter('handover_exit_extra_frames').value or 6)
		if self.publish_mode not in ('continuous', 'exit'):
			self.get_logger().warn(f"Invalid publish_mode '{self.publish_mode}'. Using 'continuous'.")
			self.publish_mode = 'continuous'
		if self.handover_exit_extra_frames < 0:
			self.get_logger().warn(
				f"Invalid handover_exit_extra_frames '{self.handover_exit_extra_frames}'. Using 0."
			)
			self.handover_exit_extra_frames = 0
		self.exit_mode_enabled = self.publish_mode == 'exit'
		self.handover_detected = False
		self.handover_exit_countdown = None
		self.exit_requested = False

		# output topic
		self.pub_message = self.create_publisher(String, '/hand_tracker', 10)

		# self.pub_vis1 = self.create_publisher(Image, '/hand_tracker/hand_det_left', 10)
		# self.pub_vis2 = self.create_publisher(Image, '/hand_tracker/hand_det_right', 10)

		input_qos = QoSProfile(
			history=HistoryPolicy.KEEP_LAST,
			depth=10,
			reliability=ReliabilityPolicy.BEST_EFFORT,
			durability=DurabilityPolicy.VOLATILE,
		)

		# Set up message filters for synchronization
		self.sub1 = message_filters.Subscriber(self, Image, self.camera_topic_left, qos_profile=input_qos)
		self.sub2 = message_filters.Subscriber(self, Image, self.camera_topic_right, qos_profile=input_qos)
		self.handover_detected_sub = self.create_subscription(
			Bool,
			self.handover_detected_topic,
			self.handover_detected_callback,
			input_qos,
		)

		# ApproximateTimeSynchronizer aligns the messages based on their timestamps.
		# queue_size=10, slop=0.1 means frames within 0.1 seconds of each other are paired.
		self.ts = message_filters.ApproximateTimeSynchronizer(
			[self.sub1, self.sub2], queue_size=10, slop=0.1
		)
		self.ts.registerCallback(self.sync_callback)

		self.get_logger().info(
			f"Hand tracker node started: publish_mode={self.publish_mode}, "
			f"handover_detected_topic={self.handover_detected_topic}, "
			f"handover_exit_extra_frames={self.handover_exit_extra_frames}"
		)

	def handover_detected_callback(self, msg):
		if not bool(msg.data):
			return

		self.handover_detected = True
		if not self.exit_mode_enabled:
			return

		if self.handover_detected and self.handover_exit_countdown is None:
			self.handover_exit_countdown = self.handover_exit_extra_frames
			self.get_logger().info(
				f'Handover detected. Publishing for {self.handover_exit_countdown} more synced frames before exit.'
			)

	def _should_stop_publishing(self):
		if not self.exit_mode_enabled:
			return False
		if not self.handover_detected:
			return False
		return self.handover_exit_countdown is not None and self.handover_exit_countdown <= 0

	def _update_exit_countdown(self):
		if not self.exit_mode_enabled or not self.handover_detected or self.handover_exit_countdown is None:
			return
		if self.handover_exit_countdown > 0:
			self.handover_exit_countdown -= 1
		if self.handover_exit_countdown == 0:
			self.get_logger().info('Exit countdown reached zero. Shutting down hand tracker node.')
			rclpy.shutdown()

	def calculate_hand_area(self, hand_landmarks, img_height, img_width):
		"""
		Calculates the bounding box area of the hand in pixel space.
		"""
		# MediaPipe returns normalized coordinates (0.0 to 1.0)
		x_coords = [landmark.x for landmark in hand_landmarks.landmark]
		y_coords = [landmark.y for landmark in hand_landmarks.landmark]
		
		min_x, max_x = min(x_coords), max(x_coords)
		min_y, max_y = min(y_coords), max(y_coords)

		# Convert back to pixel widths and heights
		width_px = (max_x - min_x) * img_width
		height_px = (max_y - min_y) * img_height
		
		return width_px * height_px

	def calculate_hand_information_score(self, hand_landmarks, img_height, img_width, handedness_score=0.5):
		"""
		Estimate how informative a hand view is for finger-collision filtering.
		Higher score means more visible keypoints and stronger model confidence.
		"""
		visible_landmarks = 0
		visible_tips = 0

		for idx, landmark in enumerate(hand_landmarks.landmark):
			x_px = landmark.x * img_width
			y_px = landmark.y * img_height
			inside = 0.0 <= x_px < img_width and 0.0 <= y_px < img_height
			if not inside:
				continue

			visible_landmarks += 1
			if idx in self.finger_tip_indices:
				visible_tips += 1

		# Finger tips are weighted higher because grasp collision filtering depends on them most.
		keypoint_score = float(visible_landmarks) + 2.0 * float(visible_tips)
		confidence_bonus = 5.0 * float(np.clip(handedness_score, 0.0, 1.0))
		return keypoint_score + confidence_bonus

	def sync_callback(self, msg1, msg2):
		if self.exit_requested or self._should_stop_publishing():
			self.exit_requested = True
			return

		# 1. Convert ROS Image messages to OpenCV format (RGB)
		try:
			cv_img1 = self.bridge.imgmsg_to_cv2(msg1, desired_encoding='rgb8')
			cv_img2 = self.bridge.imgmsg_to_cv2(msg2, desired_encoding='rgb8')

			h1, w1, _ = cv_img1.shape
			h2, w2, _ = cv_img2.shape
		except Exception as e:
			self.get_logger().error(f"Failed to convert images: {e}")
			return

		h1, w1, _ = cv_img1.shape
		h2, w2, _ = cv_img2.shape

		active_hand_center_1 = None
		active_palm_center_1 = None
		active_hand_center_2 = None
		active_palm_center_2 = None

		# Copy images so we can draw on them safely
		annotated_img1 = cv_img1.copy()
		annotated_img2 = cv_img2.copy()

		# 2. Process both images with MediaPipe
		res1 = self.hands1.process(cv_img1)
		res2 = self.hands2.process(cv_img2)

		det1 = res1.multi_hand_landmarks is not None
		det2 = res2.multi_hand_landmarks is not None

		centers_1, palm_centers_1, info_scores_1 = [], [], []
		centers_2, palm_centers_2, info_scores_2 = [], [], []
		# left
		if det1:
			for hand_landmarks in res1.multi_hand_landmarks:
				self.mp_drawing.draw_landmarks(
					annotated_img1,
					hand_landmarks,
					self.mp_hands.HAND_CONNECTIONS,
					self.mp_drawing_styles.get_default_hand_landmarks_style(),
					self.mp_drawing_styles.get_default_hand_connections_style())
			self.calculate_hand_centers(res1, w1, h1, centers_1, palm_centers_1, info_scores_1)
			# pick hand with highest keypoint+confidence information score
			best_idx = info_scores_1.index(max(info_scores_1))
			if self.active_hand_id_1 != best_idx:
				self.hand_history_1.clear()
				self.active_hand_id_1 = best_idx
			active_hand_center_1 = centers_1[best_idx]
			if best_idx < len(palm_centers_1):
				active_palm_center_1 = palm_centers_1[best_idx]
			self.hand_history_1.append(active_hand_center_1)
			if active_hand_center_1:
				cv2.circle(annotated_img1, active_hand_center_1, 6, (255, 0, 0), -1)

		# right
		if det2:
			for hand_landmarks in res2.multi_hand_landmarks:
				self.mp_drawing.draw_landmarks(
					annotated_img2,
					hand_landmarks,
					self.mp_hands.HAND_CONNECTIONS,
					self.mp_drawing_styles.get_default_hand_landmarks_style(),
					self.mp_drawing_styles.get_default_hand_connections_style())
			self.calculate_hand_centers(res2, w2, h2, centers_2, palm_centers_2, info_scores_2)
			# pick hand with highest keypoint+confidence information score
			best_idx = info_scores_2.index(max(info_scores_2))
			if self.active_hand_id_2 != best_idx:
				self.hand_history_2.clear()
				self.active_hand_id_2 = best_idx
			active_hand_center_2 = centers_2[best_idx]
			if best_idx < len(palm_centers_2):
				active_palm_center_2 = palm_centers_2[best_idx]
			self.hand_history_2.append(active_hand_center_2)
			if active_hand_center_2:
				cv2.circle(annotated_img2, active_hand_center_2, 6, (255, 0, 0), -1)

		# 3. Output the Debug Message (Same as before)
		if not det1 and not det2:
			return

		if det1 and det2:
			frame_str = "BOTH 1 and 2"
		elif det1:
			frame_str = "Frame 1 ONLY"
		else:
			frame_str = "Frame 2 ONLY"

		score1 = max(info_scores_1) if len(info_scores_1) > 0 else 0.0
		score2 = max(info_scores_2) if len(info_scores_2) > 0 else 0.0

		more_visible = 0
		if score1 > score2:
			visibility_str = "Frame 1 MORE"
			more_visible = 1
		elif score2 > score1:
			visibility_str = "Frame 2 MORE"
			more_visible = 2
		else:
			visibility_str = "Frame 1 ~ Frame 2"

		self.get_logger().info(
			f"Hand detected in: [{frame_str}] | {visibility_str} "
			f"(score_left={score1:.2f}, score_right={score2:.2f})"
		)

		cv2.putText(annotated_img1, f'[{frame_str}] | {visibility_str}', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,0,180), 2, cv2.LINE_AA)
		cv2.putText(annotated_img2, f'[{frame_str}] | {visibility_str}', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,0,180), 2, cv2.LINE_AA)
		
		# Convert back to ROS messages
		annotated_msg1 = self.bridge.cv2_to_imgmsg(annotated_img1, encoding='rgb8')
		annotated_msg2 = self.bridge.cv2_to_imgmsg(annotated_img2, encoding='rgb8')

		# Keep the original timestamps so they stay perfectly synced in RViz/rqt
		annotated_msg1.header = msg1.header
		annotated_msg2.header = msg2.header
		# self.pub_vis1.publish(annotated_msg1)
		# self.pub_vis2.publish(annotated_msg2)

		# publish the real message
		left = {
			"detected": det1,
			"active_palm_center": active_palm_center_1,
			"active_hand_center": active_hand_center_1
		}
		right = {
			"detected": det2,
			"active_palm_center": active_palm_center_2,
			"active_hand_center": active_hand_center_2
		}
		message = {
			"left": left,
			"right": right,
			"more_hand_visible": more_visible # 0 - no preference, 1 - left, 2 - right
		}
		message_json = String()
		message_json.data = json.dumps(message)
		
		self.pub_message.publish(message_json)
		self._update_exit_countdown()

	def calculate_hand_centers(self, hand_results, w, h, centers, palm_centers, info_scores):
		if hand_results.multi_hand_landmarks:
			for idx, lm in enumerate(hand_results.multi_hand_landmarks):

				fingertip_positions = []
				for i in self.finger_tip_indices:
					l = lm.landmark[i]
					x_pix = int(l.x * w)
					y_pix = int(l.y * h)
					fingertip_positions.append((x_pix, y_pix))

				cx = int(np.mean([p[0] for p in fingertip_positions]))
				cy = int(np.mean([p[1] for p in fingertip_positions]))
				centers.append((cx, cy))

				# Separate palm center (wrist + MCP joints), computed in parallel to keep existing logic unchanged.
				palm_indices = [0, 5, 9, 13, 17]
				palm_positions = []
				for i in palm_indices:
					l = lm.landmark[i]
					x_pix = int(l.x * w)
					y_pix = int(l.y * h)
					palm_positions.append((x_pix, y_pix))
				pcx = int(np.mean([p[0] for p in palm_positions]))
				pcy = int(np.mean([p[1] for p in palm_positions]))
				palm_centers.append((pcx, pcy))

				handedness_score = 0.5
				if (
					hand_results.multi_handedness is not None
					and idx < len(hand_results.multi_handedness)
					and len(hand_results.multi_handedness[idx].classification) > 0
				):
					handedness_score = float(hand_results.multi_handedness[idx].classification[0].score)

				info_scores.append(
					self.calculate_hand_information_score(lm, h, w, handedness_score=handedness_score)
				)

def main(args=None):
	rclpy.init(args=args)
	node = HandTrackerNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()