import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import message_filters

import cv2
import mediapipe as mp
from collections import deque, defaultdict
from ultralytics import YOLO
import numpy as np
import math
import json
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class HandoverDetectorNode(Node):

	def __init__(self):
		super().__init__("handover_detector_node")

		self.MAX_HISTORY = 40
		self.PROXIMITY_THRESH = 45.0
		self.PROXIMITY_WINDOW = 6

		self.obj_history_left = deque(maxlen=self.MAX_HISTORY)
		self.obj_history_right = deque(maxlen=self.MAX_HISTORY)
		self.hand_history_left = deque(maxlen=self.MAX_HISTORY)
		self.hand_history_right = deque(maxlen=self.MAX_HISTORY)

		# Parameters
		self.declare_parameter('mode', 'dual')
		self.declare_parameter('proximity_logic', 'or')
		self.declare_parameter('publish_mode', 'continuous')

		self.mode = str(self.get_parameter('mode').value or 'left').lower().strip()
		self.proximity_logic = str(self.get_parameter('proximity_logic').value or 'and').lower().strip()
		self.publish_mode = str(self.get_parameter('publish_mode').value or 'continuous').lower().strip()

		if self.mode not in ('left', 'right', 'dual'):
			self.get_logger().warn(f"Invalid mode '{self.mode}'. Using 'left'.")
			self.mode = 'left'
		if self.proximity_logic not in ('and', 'or'):
			self.get_logger().warn(f"Invalid proximity_logic '{self.proximity_logic}'. Using 'and'.")
			self.proximity_logic = 'and'
		if self.publish_mode not in ('continuous', 'exit'):
			self.get_logger().warn(f"Invalid publish_mode '{self.publish_mode}'. Using 'continuous'.")
			self.publish_mode = 'continuous'

		# 1. Create the Handover Detected Publisher
		self.handover_detected_pub = self.create_publisher(Bool, 'handover_detected', 10)

		input_qos = QoSProfile(
			history=HistoryPolicy.KEEP_LAST,
			depth=10,
			reliability=ReliabilityPolicy.BEST_EFFORT,
			durability=DurabilityPolicy.VOLATILE,
		)

		# Topic names
		self.hand_tracker_topic = "/hand_tracker"
		self.bounding_boxes_topic = "/object_detections"

		# Track detection state for exit mode
		self.handover_detected_once = False

		# Create synchronized subscribers using message_filters
		if self.mode == 'left':
			self._setup_left_syncer(input_qos)
		elif self.mode == 'right':
			self._setup_right_syncer(input_qos)
		else:  # dual
			self._setup_left_syncer(input_qos)
			self._setup_right_syncer(input_qos)

		self.handover_log_already_done = False
		self.get_logger().info(
			f"Handover Detection Node started: mode={self.mode}, "
			f"proximity_logic={self.proximity_logic}, publish_mode={self.publish_mode}"
		)

	def _setup_left_syncer(self, qos_profile):
		"""Setup message filter synchronizer for left view."""
		self.hand_tracker_left_sub = message_filters.Subscriber(
			self, String, self.hand_tracker_topic, qos_profile=qos_profile
		)
		self.bounding_boxes_left_sub = message_filters.Subscriber(
			self, String, self.bounding_boxes_topic, qos_profile=qos_profile
		)
		self.ts_left = message_filters.ApproximateTimeSynchronizer(
			[self.hand_tracker_left_sub, self.bounding_boxes_left_sub],
			queue_size=10,
			slop=0.05,
			allow_headerless=True
		)
		self.ts_left.registerCallback(self.callback_left_sync)

	def _setup_right_syncer(self, qos_profile):
		"""Setup message filter synchronizer for right view."""
		self.hand_tracker_right_sub = message_filters.Subscriber(
			self, String, self.hand_tracker_topic, qos_profile=qos_profile
		)
		self.bounding_boxes_right_sub = message_filters.Subscriber(
			self, String, self.bounding_boxes_topic, qos_profile=qos_profile
		)
		self.ts_right = message_filters.ApproximateTimeSynchronizer(
			[self.hand_tracker_right_sub, self.bounding_boxes_right_sub],
			queue_size=10,
			slop=0.05
		)
		self.ts_right.registerCallback(self.callback_right_sync)

	def callback_left_sync(self, hand_tracker_msg, bounding_boxes_msg):
		"""Synchronized callback for left view."""
		if self.handover_detected_once and self.publish_mode == 'exit':
			return
		hand_tracker_data = json.loads(hand_tracker_msg.data)
		bounding_boxes_data = json.loads(bounding_boxes_msg.data)
		
		# Update left history
		hand_data = hand_tracker_data.get('left', {})
		bbox = bounding_boxes_data.get('left_bbox')
		
		if bbox is not None:
			x1, y1, x2, y2 = bbox
			cx = (x1 + x2) // 2
			cy = (y1 + y2) // 2
			self.obj_history_left.append((cx, cy))
		
		hand_center = hand_data.get('active_hand_center')
		if hand_center is not None:
			self.hand_history_left.append(hand_center)
		
		# Process based on mode
		if self.mode == 'left':
			proximity_detected = self.detect_proximity(self.obj_history_left, self.hand_history_left)
			self._publish_handover_status(proximity_detected)
		elif self.mode == 'dual':
			self._maybe_publish_dual_view()

	def callback_right_sync(self, hand_tracker_msg, bounding_boxes_msg):
		"""Synchronized callback for right view."""
		if self.handover_detected_once and self.publish_mode == 'exit':
			return
		hand_tracker_data = json.loads(hand_tracker_msg.data)
		bounding_boxes_data = json.loads(bounding_boxes_msg.data)
		
		# Update right history
		hand_data = hand_tracker_data.get('right', {})
		bbox = bounding_boxes_data.get('right_bbox')
		
		if bbox is not None:
			x1, y1, x2, y2 = bbox
			cx = (x1 + x2) // 2
			cy = (y1 + y2) // 2
			self.obj_history_right.append((cx, cy))
		
		hand_center = hand_data.get('active_hand_center')
		if hand_center is not None:
			self.hand_history_right.append(hand_center)
		
		# Process based on mode
		if self.mode == 'right':
			proximity_detected = self.detect_proximity(self.obj_history_right, self.hand_history_right)
			self._publish_handover_status(proximity_detected)
		elif self.mode == 'dual':
			self._maybe_publish_dual_view()

	def _maybe_publish_dual_view(self):
		"""For dual view mode, publish only when both views have sufficient data."""
		proximity_detected_left = self.detect_proximity(self.obj_history_left, self.hand_history_left)
		proximity_detected_right = self.detect_proximity(self.obj_history_right, self.hand_history_right)

		# Apply configured logic
		if self.proximity_logic == 'and':
			handover_detected = proximity_detected_left and proximity_detected_right
		else:  # or
			handover_detected = proximity_detected_left or proximity_detected_right

		self._publish_handover_status(handover_detected)

	def _publish_handover_status(self, handover_detected):
		"""Publish handover status and handle exit mode."""
		out_msg = Bool()
		out_msg.data = handover_detected
		self.handover_detected_pub.publish(out_msg)

		if handover_detected and not self.handover_log_already_done:
			self.handover_log_already_done = True
			msg = {"t_human_first_contact_ms": ((time.time_ns()) / 1e6)}
			self.get_logger().get_child("SUBMISSION").info(f"{json.dumps(msg)}")

		if handover_detected and self.publish_mode == 'exit':
			if not self.handover_detected_once:
				self.handover_detected_once = True
				self.get_logger().info(
					'Handover detected. publish_mode=exit: shutting down after first publish.'
				)
				rclpy.shutdown()

	def detect_proximity(self, object_history, hand_history):
		if len(object_history) > 1 and len(hand_history) > 1:
			hand_center_w = self.weighted_recent_center(hand_history, self.PROXIMITY_WINDOW)
			obj_center_w = self.weighted_recent_center(object_history, self.PROXIMITY_WINDOW)
			if hand_center_w is not None and obj_center_w is not None:
				distance = math.hypot(
					hand_center_w[0] - obj_center_w[0],
					hand_center_w[1] - obj_center_w[1]
				)
				if distance < self.PROXIMITY_THRESH:
					return True
		return False

	def weighted_recent_center(self, points, window):
		"""Weighted mean of the most recent 2D points, emphasizing latest samples."""
		if not points:
			return None

		#self.get_logger().info(f'aaaaa : {list(points)[-window:]}')
		recent = np.array(list(points)[-window:], dtype=np.float32)
		if recent.shape[0] == 0:
			return None

		weights = np.arange(1, recent.shape[0] + 1, dtype=np.float32)
		weights /= np.sum(weights)
		center = np.sum(recent * weights[:, None], axis=0)
		return float(center[0]), float(center[1])


# =============================================================
#                         ENTRY POINT
# =============================================================
def main():
	rclpy.init()
	node = HandoverDetectorNode()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()