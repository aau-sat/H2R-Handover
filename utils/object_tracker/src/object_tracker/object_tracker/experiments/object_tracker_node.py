import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import message_filters
from cv_bridge import CvBridge
from ultralytics import YOLO, YOLOWorld
import json
import mediapipe
from collections import deque, defaultdict
import cv2
import numpy as np

# reimplementation with gemini...

class ObjectTrackerNode(Node):
	def __init__(self):
		super().__init__('object_tracker_node')

		# --- Parameters ---
		# Model to use
		self.declare_parameter('use_yolo_world', True)
		self.use_yolo_world = self.get_parameter('use_yolo_world').get_parameter_value().bool_value

		# Object classes to track
		if self.use_yolo_world:
			self.declare_parameter('target_classes', ["glas", "cup", "mug", "bottle", "plate", "bowl", "food container"])
			self.target_classes = self.get_parameter('target_classes').value
		else:
			self.declare_parameter('target_classes', [39, 40, 41, 45, 46, 47, 48, 49, 53, 54, 73, 75])
			self.target_classes = self.get_parameter('target_classes').value
		
		# How many frames to "hold" a detection if YOLO loses it
		self.declare_parameter('max_coast_frames', 10) 
		self.max_coast_frames = self.get_parameter('max_coast_frames').value

		# Should cropped images be published?
		self.declare_parameter('publish_crops', False)
		self.publish_crops = self.get_parameter('publish_crops').get_parameter_value().bool_value

		# Crop padding (like HTML, crop is expanded by that padding on either side)
		self.declare_parameter('crop_padding', 20)
		self.crop_padding = self.get_parameter('crop_padding').value

		# Should hand detection be performed?
		self.declare_parameter('hand_detection', True)
		self.hand_detection = self.get_parameter('hand_detection').get_parameter_value().bool_value

		# Publish debug topics and logs?
		self.declare_parameter('debug_topics', True)
		self.debug_topics = self.get_parameter('debug_topics').get_parameter_value().bool_value

		# Topics to listen to
		self.declare_parameter('camera_topic_left', ' ') #'/camera/left/color/image_raw'
		self.camera_topic_left_defined = bool(self.get_parameter('camera_topic_left').get_parameter_value().string_value.strip())
		self.camera_topic_left = self.get_parameter('camera_topic_left').value
		
		self.declare_parameter('camera_topic_right', ' ')
		self.camera_topic_right_defined = bool(self.get_parameter('camera_topic_right').get_parameter_value().string_value.strip())
		self.camera_topic_right = self.get_parameter('camera_topic_right').value

		self.both_camera_topics_defined = self.camera_topic_left_defined and self.camera_topic_right_defined

		# --- Publishers ---
		self.json_pub = self.create_publisher(String, '/object_tracker/detected_objects', 10)
		self.pub_vis1 = self.create_publisher(Image, '/object_tracker/det_left', 10)
		self.pub_vis2 = self.create_publisher(Image, '/object_tracker/det_right', 10)
		self.pub_vis1_crop = self.create_publisher(Image, '/object_tracker/crop/left', 10)
		self.pub_vis2_crop = self.create_publisher(Image, '/object_tracker/crop/right', 10)
		
		# --- Subscribers & Synchronizer ---
		custom_image_stream_qos = QoSProfile(
			history=HistoryPolicy.KEEP_LAST,
			depth=1, # Keep only the absolute latest frame to prevent processing lag
			reliability=ReliabilityPolicy.RELIABLE, # Crucial: prevents fragment dropping
			durability=DurabilityPolicy.VOLATILE
		)

		if self.both_camera_topics_defined:
			sub_cam1 = message_filters.Subscriber(self, Image, self.camera_topic_left, qos_profile=custom_image_stream_qos)
			sub_cam2 = message_filters.Subscriber(self, Image, self.camera_topic_right, qos_profile=custom_image_stream_qos)

			self.ts = message_filters.ApproximateTimeSynchronizer(
				[sub_cam1, sub_cam2], 
				queue_size=10, 
				slop=0.05
			)
			self.ts.registerCallback(self.sync_callback)
		else:
			if self.camera_topic_left_defined:
				sub_cam1 = self.create_subscription(Image, self.camera_topic_left, self.camera_1_callback, custom_image_stream_qos)
			elif self.camera_topic_right_defined:
				sub_cam2 = self.create_subscription(Image, self.camera_topic_right, self.camera_2_callback, custom_image_stream_qos)
			else:
				self.get_logger().fatal("No camera topic defined!!! Time to crash... or get stuck... IDK")
				rclpy.shutdown()

		# --- Model and prerequisites ---
		self.bridge = CvBridge()
		
		if self.use_yolo_world:
			self.get_logger().info("Loading YOLO World model for tracking...")
			self.model1 = YOLOWorld('yolov8x-world.pt')
			self.model2 = YOLOWorld('yolov8x-world.pt')
			self.model1.set_classes(self.target_classes)
			self.model2.set_classes(self.target_classes)
		else:
			self.get_logger().info("Loading YOLO model for tracking...")
			self.model1 = YOLO('yolo11x.pt')
			self.model2 = YOLO('yolo11x.pt')

		# --- State History ---
		# Dictionaries to hold the current state of each camera. Key = class_id
		self.state_cam1 = {}
		self.state_cam2 = {}

		# Classes detected by both cameras at the same time once
		self.confirmed_classes = set()

		#Hand stuff
		if self.hand_detection:
			self.active_hand_id = None
			self.finger_tip_indices = [4, 8, 12, 16, 20]
			self.hand_history_1 = deque(maxlen=self.max_coast_frames)
			self.hand_history_2 = deque(maxlen=self.max_coast_frames)
			self.active_hand_id_1 = None
			self.active_hand_id_2 = None

			self.mp_hands = mediapipe.solutions.hands
			self.hands_1 = self.mp_hands.Hands(
				static_image_mode=False,
				max_num_hands=2,
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5
			)
			self.hands_2 = self.mp_hands.Hands(
				static_image_mode=False,
				max_num_hands=2,
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5
			)
		
		self.get_logger().info("Object Tracker Node started. Ready.")

	# Callbacks
	def camera_1_callback(self, msg):
		self.process_camera_messages(msg, None)

	def camera_2_callback(self, msg):
		self.process_camera_messages(None, msg)

	def sync_callback(self, msg_cam1, msg_cam2):
		self.process_camera_messages(msg_cam1, msg_cam2)

	# processes camera messages
	def process_camera_messages(self, msg_cam1, msg_cam2):
		active_cam_1 = False
		active_cam_2 = False
		if msg_cam1 is not None:
			active_cam_1 = True
			cv_img1 = self.bridge.imgmsg_to_cv2(msg_cam1, desired_encoding='bgr8')
		if msg_cam2 is not None:
			active_cam_2 = True
			cv_img2 = self.bridge.imgmsg_to_cv2(msg_cam2, desired_encoding='bgr8')
		
		if self.use_yolo_world:
			if active_cam_1:
				results1 = self.model1.track(cv_img1, 
					persist=True, 
					batch=10, 
					max_det=10, 
					stream_buffer=True, 
					conf=0.2, 
					tracker="bytetrack.yaml",
					verbose=self.debug_topics)[0] 
			if active_cam_2:
				results2 = self.model2.track(cv_img2, 
					persist=True, 
					batch=10, 
					max_det=10, 
					stream_buffer=True, 
					conf=0.2, 
					tracker="bytetrack.yaml",
					verbose=self.debug_topics)[0]
		else:
			if active_cam_1:
				results1 = self.model1.track(cv_img1, persist=True, verbose=self.debug_topics, classes=self.target_classes)[0]
			if active_cam_2:
				results2 = self.model2.track(cv_img2, persist=True, verbose=self.debug_topics, classes=self.target_classes)[0]
		
		if active_cam_1:
			self.update_state(self.state_cam1, results1, msg_cam1.header.stamp)
		if active_cam_2:
			self.update_state(self.state_cam2, results2, msg_cam2.header.stamp)

		# --- CONFIRMATION LOGIC ---
		# 1. Find classes actively detected right now in both cameras
		active_now_1 = {c_id for c_id, data in self.state_cam1.items() if data['is_detected_now']} if active_cam_1 else set()
		active_now_2 = {c_id for c_id, data in self.state_cam2.items() if data['is_detected_now']} if active_cam_2 else set()
		
		if active_cam_1 and active_cam_2: # we need to cross match as this piece of code was designed for both streams
			# 2. Add overlapping detections to our confirmed set
			simultaneous_detections = active_now_1.intersection(active_now_2) #only the ones that are in both
			self.confirmed_classes.update(simultaneous_detections) #merge new detections from both into confirmed classes
		else: # only one stream
			self.confirmed_classes = active_now_1.union(active_now_2) #if only one stream we take those

		# 3. Get all classes currently being tracked/coasted across both cameras
		tracked_anywhere = set(self.state_cam1.keys()).union(set(self.state_cam2.keys())) #current detections
		
		# 4. Prune the confirmed list (remove classes that have completely aged out)
		self.confirmed_classes.intersection_update(tracked_anywhere) #update confirmed by removing all items that are not in tracked
		
		# 5. Our final valid list is the intersection of what is tracked vs what is confirmed
		valid_classes = tracked_anywhere.intersection(self.confirmed_classes) #intersect tracked with confirmed
		# --------------------------

		# Map only the VALID classes to a persistent output ID
		fusion_id_map = {cls_id: match_id for match_id, cls_id in enumerate(valid_classes)}

		# Debug
		annotated1 = None
		annotated2 = None
		if self.debug_topics:
			if active_cam_1:
				annotated1 = self.draw_debug_frame(cv_img1.copy(), self.state_cam1, self.state_cam2, valid_classes, fusion_id_map, active_cam_2)
			if active_cam_2:
				annotated2 = self.draw_debug_frame(cv_img2.copy(), self.state_cam2, self.state_cam1, valid_classes, fusion_id_map, active_cam_1)

		# ------------------- CROPPED IMAGES -------------------
		if self.publish_crops and (active_cam_1 and active_cam_2 and len(results1.boxes) > 0 and len(results2.boxes) > 0) or (active_cam_1 and not active_cam_2) or (not active_cam_1 and active_cam_2):
			# left
			if active_cam_1:
				out_crop1 = self.crop_object(cv_img1, results1)
				self.pub_vis1_crop.publish(out_crop1)
			# right
			if active_cam_2:
				out_crop2 = self.crop_object(cv_img2, results2)
				self.pub_vis2_crop.publish(out_crop2)

		# ------------------- HAND DETECTION -------------------
		if self.hand_detection:
			#left
			if active_cam_1:
				active_hand_center_1, active_palm_center_1 = self.detect_hand(cv_img1, annotated1, self.hands_1, self.hand_history_1, self.active_hand_id_1)
			#right
			if active_cam_2:
				active_hand_center_2, active_palm_center_2 = self.detect_hand(cv_img2, annotated2, self.hands_2, self.hand_history_2, self.active_hand_id_2)

		# Publish debug
		if self.debug_topics:
			if active_cam_1:
				self.pub_vis1.publish(self.bridge.cv2_to_imgmsg(annotated1, encoding="bgr8"))
			if active_cam_2:
				self.pub_vis2.publish(self.bridge.cv2_to_imgmsg(annotated2, encoding="bgr8"))

		# Output
		output_data = []

		for cls_id in valid_classes:
			class_name = self.model1.names[cls_id]
			match_id = fusion_id_map[cls_id]
			
			left_data = self.state_cam1.get(cls_id, {
				"xy": [], "bbox": [], "confidence": 0.0, "timestamp": None, "is_detected_now": False
			})
			right_data = self.state_cam2.get(cls_id, {
				"xy": [], "bbox": [], "confidence": 0.0, "timestamp": None, "is_detected_now": False
			})
			left = None
			right = None
			if active_cam_1:
				left = {
					"xy": left_data.get("xy", []),
					"bbox": left_data.get("bbox", []),
					"confidence": left_data.get("confidence", 0.0),
					"timestamp": left_data.get("timestamp", None),
					"is_detected": left_data.get("is_detected_now", False),
					"active_hand_center": [int(active_hand_center_1[0]), int(active_hand_center_1[1])] if self.hand_detection and active_hand_center_1 is not None else None,
					"active_palm_center": [int(active_palm_center_1[0]), int(active_palm_center_1[1])] if self.hand_detection and active_palm_center_1 is not None else None,
				},
			if active_cam_2:
				right = {
					"xy": right_data.get("xy", []),
					"bbox": right_data.get("bbox", []),
					"confidence": right_data.get("confidence", 0.0),
					"timestamp": right_data.get("timestamp", None),
					"is_detected": right_data.get("is_detected_now", False),
					"active_hand_center": [int(active_hand_center_2[0]), int(active_hand_center_2[1])] if self.hand_detection and active_hand_center_2 is not None else None,
					"active_palm_center": [int(active_palm_center_2[0]), int(active_palm_center_2[1])] if self.hand_detection and active_palm_center_2 is not None else None,
				}

			obj_dict = {
				"id": match_id,
				"class_name": class_name,
				"left": left,
				"right": right
			}
			output_data.append(obj_dict)

		if output_data:
			msg = String()
			msg.data = json.dumps(output_data)
			self.json_pub.publish(msg)

	# Updates states
	def update_state(self, state_dict, results, timestamp):
		"""
		Updates the camera's state dictionary. Ages out old detections and inserts new ones.
		"""
		# 1. Age all existing tracks (increment missed frames)
		for cls_id in list(state_dict.keys()):
			state_dict[cls_id]['missed_frames'] += 1
			state_dict[cls_id]['is_detected_now'] = False
			
			# Remove if it has been gone for too long
			if state_dict[cls_id]['missed_frames'] > self.max_coast_frames:
				del state_dict[cls_id]

		# 2. Update with fresh detections from the current frame
		if results.boxes is not None:
			for box in results.boxes:
				c_id = int(box.cls[0].item())
				
				# YOLO tracking IDs might be None if the tracker is still initializing
				track_id = int(box.id[0].item()) if box.id is not None else -1
				
				x, y, w, h = box.xywh[0].tolist()
				coords = box[0].xyxy[0].tolist()
				x1, y1, x2, y2 = map(int, coords)
				conf = float(box.conf[0].item())
				
				# If there are multiple objects of the same class, keep the highest confidence one
				if c_id in state_dict and state_dict[c_id]['is_detected_now']:
					if conf <= state_dict[c_id]['confidence']:
						continue 

				# Overwrite/Update the state for this class
				state_dict[c_id] = {
					'xy': [x, y],
					'bbox': [x1, y1, x2, y2],
					'confidence': conf,
					'track_id': track_id,
					'timestamp': {"sec": timestamp.sec, "nanosec": timestamp.nanosec},
					'missed_frames': 0,
					'is_detected_now': True 
				}

	# Crops the object based on the bounding boxes
	def crop_object(self, cv_img, results):
		coords = results.boxes[0].xyxy[0].tolist()
		x1, y1, x2, y2 = map(int, coords)

		x1 = max(0, x1 - self.crop_padding)
		y1 = max(0, y1 - self.crop_padding)
		x2 = min(cv_img.shape[1], x2 + self.crop_padding)
		y2 = min(cv_img.shape[0], y2 + self.crop_padding)

		crop = cv_img[y1:y2, x1:x2]
		return self.bridge.cv2_to_imgmsg(crop, encoding='bgr8')

	# Detects the hand
	def detect_hand(self, cv_img, annotated_image, hands, hand_history, active_hand_id):
		h, w, _ = cv_img.shape
		rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		hand_results = hands.process(rgb)
		centers, areas = [], []
		active_hand_center = None
		palm_centers = []
		active_palm_center = None

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

				xs = [l.x * w for l in lm.landmark]
				ys = [l.y * h for l in lm.landmark]
				areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))

			# pick biggest hand
			biggest_idx = areas.index(max(areas))
			if active_hand_id != biggest_idx:
				hand_history.clear()
				active_hand_id = biggest_idx

			active_hand_center = centers[active_hand_id]
			if biggest_idx < len(palm_centers):
				active_palm_center = palm_centers[biggest_idx]
			hand_history.append(active_hand_center)

		if annotated_image is not None and active_hand_center:
			cv2.circle(annotated_image, active_hand_center, 6, (255, 0, 0), -1)

		return active_hand_center, active_palm_center

	# Draws the debug frames
	def draw_debug_frame(self, img, state_this, state_other, valid_classes, fusion_id_map, active_other: bool):
		"""
		Manually draws bounding boxes using OpenCV based on tracking state.
		Colors: for both: Green (Both), Yellow (Only This), Red (Coasting/Not detected now), for one stream yellow is also green
		"""
		for cls_id, data in state_this.items():
			# Skip drawing if it hasn't been confirmed yet
			if cls_id not in valid_classes:
				continue
	
			x1, y1, x2, y2 = data['bbox']
			conf = data['confidence']
			track_id = data['track_id']
			is_det_this = data['is_detected_now']
			
			# Check if the other camera sees it right now
			is_det_other = False
			if cls_id in state_other:
				is_det_other = state_other[cls_id]['is_detected_now']
			
			# Determine Color (BGR format in OpenCV)
			if not is_det_this:
				color = (0, 0, 255) # Red
			elif not active_other or (is_det_this and is_det_other):
				color = (0, 255, 0) # Green
			else:
				color = (0, 255, 255) # Yellow
			
			cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
			
			# Create Label: [F:0] cup | T:4 | 0.85
			class_name = self.model1.names[cls_id]
			fusion_id = fusion_id_map[cls_id]
			label = f"[F:{fusion_id}] {class_name} | T:{track_id} | {conf:.2f}"
			
			# Draw text background for readability
			(text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
			cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
			cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

			cv2.putText(img, 'YOLO_WORLD' if self.use_yolo_world else 'YOLO', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,0,180), 1, cv2.LINE_AA)
			
		return img

def main(args=None):
	rclpy.init(args=args)
	node = ObjectTrackerNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()