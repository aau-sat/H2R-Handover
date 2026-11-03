import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np

class DepthClipperNode(Node):
	def __init__(self):
		super().__init__('depth_clipper_node')
		
		# Parameters
		self.declare_parameter('camera_topic_rgb', '/camera/left/color/image_raw')
		self.camera_topic_rgb = self.get_parameter('camera_topic_rgb').value

		self.declare_parameter('camera_topic_d', '/camera/left/aligned_depth_to_color/image_raw')
		self.camera_topic_d = self.get_parameter('camera_topic_d').value

		self.declare_parameter('depth_clipped_topic', '/depth_clipped')
		self.depth_clipped_topic = self.get_parameter('depth_clipped_topic').value

		self.declare_parameter('depth_threshold_mm', 1750) # mm
		self.threshold = self.get_parameter('depth_threshold_mm').value

		self.replacement_color = [0, 0, 0] #[128,128,128] # bgr
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
		cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')

		# Numpy processing of mask from depth information
		# Warning: Kernel is likely shit and can definitely be improved. I just found that online but haven't looked into it further!
		mask = ((cv_depth > self.threshold) | (cv_depth == 0)).astype(np.uint8)
		kernel = np.ones((15, 15), np.uint8)

		#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.erode(mask, kernel, iterations = 1)

		# Replace masked pixels with gray
		cv_color[mask.astype(bool)] = self.replacement_color

		# Publishing
		clipped_msg = self.bridge.cv2_to_imgmsg(cv_color, encoding='bgr8')
		clipped_msg.header = color_msg.header # Preserve timestamps and frame_id
		self.clipped_pub.publish(clipped_msg)

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