import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import json
from std_msgs.msg import String

class DepthClipperNode(Node):
	def __init__(self):
		super().__init__('hand_clipper_node')
		
		# Parameters
		self.declare_parameter('camera_topic_rgb', '/camera/left/color/image_raw')
		camera_topic_rgb = self.get_parameter('camera_topic_rgb').value
		self.camera_topic_rgb = camera_topic_rgb if isinstance(camera_topic_rgb, str) else '/camera/left/color/image_raw'
		self.view_side = 'right' if '/right/' in self.camera_topic_rgb else 'left'

		self.declare_parameter('camera_topic_d', '/camera/left/aligned_depth_to_color/image_raw')
		camera_topic_d = self.get_parameter('camera_topic_d').value
		self.camera_topic_d = camera_topic_d if isinstance(camera_topic_d, str) else '/camera/left/aligned_depth_to_color/image_raw'

		self.declare_parameter('depth_clipped_topic', '/depth_clipped')
		depth_clipped_topic = self.get_parameter('depth_clipped_topic').value
		self.depth_clipped_topic = depth_clipped_topic if isinstance(depth_clipped_topic, str) else '/depth_clipped'

		self.declare_parameter('depth_threshold_mm', 2000) # mm
		depth_threshold_mm = self.get_parameter('depth_threshold_mm').value
		self.threshold = int(depth_threshold_mm) if depth_threshold_mm is not None else 2000

		# Subscribe to hand tracker
		self.declare_parameter('hand_tracker_topic', '/hand_tracker')
		hand_tracker_topic = self.get_parameter('hand_tracker_topic').value
		self.hand_tracker_topic = hand_tracker_topic if isinstance(hand_tracker_topic, str) else '/hand_tracker'
		self.hand_tracker_sub = self.create_subscription(String, self.hand_tracker_topic, self.hand_tracker_callback, 10)
		self.latest_hand_info = {}

		# Depth-proximity parameters
		self.declare_parameter('depth_tolerance_mm', 100)
		self.declare_parameter('proximity_radius_px', 120)
		self.declare_parameter('hand_persist_frames', 10)
		depth_tolerance_mm = self.get_parameter('depth_tolerance_mm').value
		proximity_radius_px = self.get_parameter('proximity_radius_px').value
		hand_persist_frames = self.get_parameter('hand_persist_frames').value
		self.depth_tolerance = int(depth_tolerance_mm) if depth_tolerance_mm is not None else 100
		self.proximity_radius = int(proximity_radius_px) if proximity_radius_px is not None else 120
		self.hand_persist_frames = int(hand_persist_frames) if hand_persist_frames is not None else 10

		self.cached_hand_state = {
			'left': {'center': None, 'miss_frames': 0},
			'right': {'center': None, 'miss_frames': 0},
		}

		self.replacement_color = [128,128,128] # bgr
		self.bridge = CvBridge()

		# Topics
		custom_image_stream_qos = QoSProfile(
			history=HistoryPolicy.KEEP_LAST,
			depth=1, # Keep only the absolute latest frame to prevent processing lag
			reliability=ReliabilityPolicy.RELIABLE, # Crucial: prevents fragment dropping
			durability=DurabilityPolicy.VOLATILE
		)
		self.color_sub = message_filters.Subscriber(self, Image, self.camera_topic_rgb, qos_profile=custom_image_stream_qos)
		self.depth_sub = message_filters.Subscriber(self, Image, self.camera_topic_d, qos_profile=custom_image_stream_qos)

		# Synchronize topics within a 0.1s window
		self.ts = message_filters.ApproximateTimeSynchronizer(
			[self.color_sub, self.depth_sub], 
			queue_size=20, 
			slop=0.1
		)
		self.ts.registerCallback(self.image_callback)

		# Publisher
		self.clipped_pub = self.create_publisher(Image, self.depth_clipped_topic, 10)
		
		self.get_logger().info("Depth Clipper Node started. Clipping pixels further than {}mm".format(
			self.get_parameter('depth_threshold_mm').value))

	def image_callback(self, color_msg, depth_msg):
		# Convert ROS Image messages to OpenCV/NumPy format
		# color: bgr8 (3 channels), depth: 16UC1 (1 channel, mm)
		cv_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
		_ = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')


		# If we have a latest hand position from the tracker, use it to compute clipping
		# Expect tracker message structure: {"left": {"detected":bool, "active_hand_center":[x,y], ...}, "right":{...}, "more_hand_visible":0|1|2}
		hand_side = self.view_side
		hand_data = self.latest_hand_info.get(hand_side, {})
		tracked_center = hand_data.get('active_hand_center')
		is_detected = bool(hand_data.get('detected', tracked_center is not None))

		cached = self.cached_hand_state[hand_side]
		hand_center = None
		if is_detected and tracked_center is not None and len(tracked_center) >= 2:
			hand_center = tracked_center
			cached['center'] = [int(tracked_center[0]), int(tracked_center[1])]
			cached['miss_frames'] = 0
		elif cached['center'] is not None and cached['miss_frames'] < self.hand_persist_frames:
			hand_center = cached['center']
			cached['miss_frames'] += 1
		else:
			cached['center'] = None
			cached['miss_frames'] = 0

		if hand_center is None:
			# fallback: publish original image unchanged
			clipped_msg = self.bridge.cv2_to_imgmsg(cv_color, encoding='bgr8')
			clipped_msg.header = color_msg.header
			self.clipped_pub.publish(clipped_msg)
			return

		hx, hy = int(hand_center[0]), int(hand_center[1])
		h, w = cv_color.shape[:2]
		if hx < 0 or hy < 0 or hx >= w or hy >= h:
			clipped_msg = self.bridge.cv2_to_imgmsg(cv_color, encoding='bgr8')
			clipped_msg.header = color_msg.header
			self.clipped_pub.publish(clipped_msg)
			return

		y_idxs, x_idxs = np.ogrid[0:h, 0:w]
		dist_sq = (x_idxs - hx) ** 2 + (y_idxs - hy) ** 2
		spatial_mask = dist_sq <= (self.proximity_radius ** 2)

		keep_mask = spatial_mask

		# Replace everything outside the hand-centered circle.
		cv_color[~keep_mask] = self.replacement_color

		# Publishing
		clipped_msg = self.bridge.cv2_to_imgmsg(cv_color, encoding='bgr8')
		clipped_msg.header = color_msg.header # Preserve timestamps and frame_id
		self.clipped_pub.publish(clipped_msg)

	def hand_tracker_callback(self, msg: String):
		try:
			data = json.loads(msg.data)
		except Exception:
			return

		# store left/right dicts (may contain None centers)
		self.latest_hand_info['left'] = data.get('left', {})
		self.latest_hand_info['right'] = data.get('right', {})

def main(args=None):
	rclpy.init(args=args)
	node = DepthClipperNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()