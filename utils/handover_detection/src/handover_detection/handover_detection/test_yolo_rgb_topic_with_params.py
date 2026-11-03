import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge

import cv2
import mediapipe as mp
from collections import deque, defaultdict
from ultralytics import YOLO
import numpy as np
import math
import json


class HandObjectTracker(Node):

	def __init__(self):
		super().__init__("hand_object_tracker")
		self.declare_parameter("camera_name", "left")
		camera_name = self.get_parameter("camera_name").get_parameter_value().string_value
		self.get_logger().info(f"Focusing on camera with name: {camera_name}")

		# ------------------- PARAMETERS -------------------
		self.MAX_HISTORY = 40
		self.THRESH =40.0
		self.DRAW_POINTS = True
		self.DRAW_ARROWS = True

		self.CONFIDENCE_THRESHOLD_DETECTED_OBJECTS_OUTPUT = 0.8 #threshold of confidence for objects to appear in 'detected_object' topic

		self.finger_tip_indices = [4, 8, 12, 16, 20]
		self.frame_idx = 0

		# ------------------- MODELS -------------------
		self.model = YOLO("models/yolov8n.pt")
		self.get_logger().info(f"YOLO loaded with classes: {self.model.names}")

		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(
			static_image_mode=False,
			max_num_hands=2,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5
		)

		# ------------------- TRACKING DATA -------------------
		self.obj_history = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))
		self.hand_history = deque(maxlen=self.MAX_HISTORY)
		self.active_hand_id = None

		included_ids = [39, 40, 41, 45, 46, 47, 48, 49, 53, 54, 73, 75]
		self.filtered_ids = [
			i for i in included_ids
			# i for i, name in self.model.names.items()
			# if name not in excluded_names
		]

		# ------------------- ROS PUBLISHERS -------------------
		self.handover_detected_ = False
		self.handover_detected_pub = self.create_publisher(Bool, "handover_detected", 10)
		self.objects_pub = self.create_publisher(String, "detected_objects", 10)

		# ------------------- ROS SUBSCRIBER -------------------
		self.bridge = CvBridge()
		self.subscription = self.create_subscription(
			Image,
			f'/camera/{camera_name}/color/image_raw',
			self.image_callback,
			10
		)

		self.get_logger().info("HandObjectTracker initialized.")


	# =============================================================
	#                         MAIN CALLBACK
	# =============================================================
	def image_callback(self, msg):
		frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		timestamp = msg.header.stamp
		# Convert the Time object (sec/nanosec) to a readable string format: <seconds>.<nanoseconds>, unix time (seconds since 1/1/1970 UTC)
		timestamp_str = f"{timestamp.sec}.{timestamp.nanosec:09d}"


		self.frame_idx += 1

		h, w, _ = frame.shape
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# ------------------- HAND DETECTION -------------------
		hand_results = self.hands.process(rgb)
		centers, areas = [], []
		active_hand_center = None

		if hand_results.multi_hand_landmarks:
			for lm in hand_results.multi_hand_landmarks:

				fingertip_positions = []
				for i in self.finger_tip_indices:
					l = lm.landmark[i]
					x_pix = int(l.x * w)
					y_pix = int(l.y * h)
					fingertip_positions.append((x_pix, y_pix))

				cx = int(np.mean([p[0] for p in fingertip_positions]))
				cy = int(np.mean([p[1] for p in fingertip_positions]))
				centers.append((cx, cy))

				xs = [l.x * w for l in lm.landmark]
				ys = [l.y * h for l in lm.landmark]
				areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))

			# pick biggest hand
			biggest_idx = areas.index(max(areas))
			if self.active_hand_id != biggest_idx:
				self.hand_history.clear()
				self.active_hand_id = biggest_idx

			active_hand_center = centers[self.active_hand_id]
			self.hand_history.append(active_hand_center)

		if active_hand_center and self.DRAW_POINTS:
			cv2.circle(frame, active_hand_center, 6, (255, 0, 0), -1)

		# ------------------- YOLO OBJECT TRACKING -------------------
		results = self.model.track(frame, persist=True, verbose=False)

		# JSON list of detected objects
		detected_objects = []

		closest_obj_id = None
		closest_obj_cls = None
		closest_box = None
		closest_obj_conf = None

		if results:
			result = results[0]
			if result.boxes.id is not None:
				for box in result.boxes:

					cls_id = int(box.cls[0])
					if cls_id not in self.filtered_ids:
						continue

					x1, y1, x2, y2 = map(int, box.xyxy[0])
					cx = (x1 + x2) // 2
					cy = (y1 + y2) // 2
					obj_id = int(box.id[0])

					# store history
					self.obj_history[obj_id].append((cx, cy))

					confidence = box.conf.item()

					# by default, we output all detected objects with confidence higher than threshold
					if confidence >= self.CONFIDENCE_THRESHOLD_DETECTED_OBJECTS_OUTPUT:
						# add object to JSON list
						detected_objects.append({
							"id": obj_id,
							"class": self.model.names[cls_id],
							"xy":  [cx, cy],
							"bbox": [x1, y1, x2, y2],
							"conf": confidence,
							"img_timestamp": timestamp_str
						})


		# ------------------- TRAJECTORY-BASED MATCHING -------------------
		if (
			active_hand_center is not None
			and len(self.hand_history) > 1
			and results
		):
			best_score = float("inf")

			for box in result.boxes:

				cls_id = int(box.cls[0])
				if cls_id not in self.filtered_ids:
					continue

				obj_id = int(box.id[0])
				pts = self.obj_history[obj_id]

				if len(pts) < 2:
					continue

				N = min(len(self.hand_history), len(pts))
				hand_traj = list(self.hand_history)[-N:]
				obj_traj = list(pts)[-N:]

				# average point-to-point distance between trajectories
				score = sum(
					math.hypot(
						hand_traj[i][0] - obj_traj[i][0],
						hand_traj[i][1] - obj_traj[i][1]
					)
					for i in range(N)
				) / N

				if score < best_score:
					best_score = score
					closest_obj_id = obj_id
					closest_obj_cls = cls_id
					x1, y1, x2, y2 = map(int, box.xyxy[0])
					closest_box = (x1, y1, x2, y2)
					closest_obj_conf = box.conf.item()

		# ------------------- DRAWING + MOTION PUBLISHING -------------------
		motion_detected = False

		if closest_obj_id is not None:
			pts = self.obj_history[closest_obj_id]
			x1, y1, x2, y2 = closest_box
			cx = (x1 + x2) // 2
			cy = (y1 + y2) // 2

			# however, if we have a closest_obj_id, we only output this one to published detected_objects while ignoring the others
			detected_objects = []
			detected_objects.append({
				"id": closest_obj_id,
				"class": self.model.names[closest_obj_cls],
				"xy":  [cx, cy],
				"bbox": [x1, y1, x2, y2],
				"conf": closest_obj_conf,
				"img_timestamp": timestamp_str
			})

			color = (0, 255, 0)

			if len(pts) > 1:
				total_movement = sum(
					math.hypot(
						pts[i][0] - pts[i - 1][0],
						pts[i][1] - pts[i - 1][1]
					)
					for i in range(1, len(pts))
				)
				if total_movement > self.THRESH:
					motion_detected = True
					color = (0, 0, 255)

			# draw bbox + labels
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
			cv2.putText(
				frame,
				f"{self.model.names[closest_obj_cls]} (ID {closest_obj_id})",
				(x1, y1 - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				color,
				2
			)

			if motion_detected:
				cv2.putText(
					frame,
					"DETECTED",
					(x1, y1 - 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.8,
					(0, 0, 255),
					2
				)
				self.handover_detected_ = True

			# draw trajectory
			if self.DRAW_ARROWS and len(pts) > 1:
				for i in range(1, len(pts)):
					cv2.line(frame, pts[i - 1], pts[i], color, 2)

		# Publish the Boolean motion flag
		self.handover_detected_pub.publish(Bool(data=self.handover_detected_))

		# Publish JSON string of detected objects, either contains all detected objects bigger than threshold or closest object if exists
		if len(detected_objects) != 0:
			self.objects_pub.publish(
				String(data=json.dumps(detected_objects))
			)

		# ------------------- DISPLAY -------------------
		cv2.imshow("Hand + Object Tracking ROS", frame)
		cv2.waitKey(1)


# =============================================================
#                         ENTRY POINT
# =============================================================
def main():
	rclpy.init()
	node = HandObjectTracker()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
