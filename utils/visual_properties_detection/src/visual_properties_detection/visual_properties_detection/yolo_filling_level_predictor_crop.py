import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import json
import cv2
import numpy as np
from ultralytics import YOLO

# Same as yolo_filling_level_predictor but using the crop topics from the all-cups rosbags as input

class YoloFillingLevelPredictorNode(Node):
	def __init__(self):
		super().__init__('image_string_processor')

		self.bridge = CvBridge()

		# 1. Declare parameters with default topic names
		self.declare_parameter('image_left_topic', '/object_tracker/crop/left')
		self.declare_parameter('image_right_topic', '/object_tracker/crop/right')
		#self.declare_parameter('bbox_topic', '/object_detections')
		self.declare_parameter('publishing_topic', '/filling_estimation')
		self.declare_parameter('model_path', 'filling-level-predictor-3.pt')

		# 2. Retrieve parameter values
		imgl_topic = self.get_parameter('image_left_topic').get_parameter_value().string_value
		imgr_topic = self.get_parameter('image_right_topic').get_parameter_value().string_value
		#bbox_topic = self.get_parameter('bbox_topic').get_parameter_value().string_value
		publishing_topic = self.get_parameter('publishing_topic').get_parameter_value().string_value
		model_path = self.get_parameter('model_path').get_parameter_value().string_value

		self.get_logger().info(f"YoloFillingLevelPredictorNode started. Listening to:\n- {imgl_topic}\n- {imgr_topic}\n- ")

		# 3. Initialize state variables to cache the latest messages
		self.latest_imagel = None
		self.latest_imager = None
		#self.latest_bbox = None
		# 4. Create Subscribers
		self.sub_imagel = self.create_subscription(
			Image,
			imgl_topic,
			self.imagel_callback,
			10  # QoS profile depth
		)
		
		self.sub_imager = self.create_subscription(
			Image,
			imgr_topic,
			self.imager_callback,
			10
		)
		
		#self.sub_bbox = self.create_subscription(
		#	String,
		#	bbox_topic,
		#	self.bbox_callback,
		#	10
		#)
		self.filling_level_pub = self.create_publisher(String, publishing_topic, 10)

		# Model initialization
		self.model = YOLO(model_path)

	def imagel_callback(self, msg: Image):
		self.latest_imagel = msg
		self.process_data()

	def imager_callback(self, msg: Image):
		self.latest_imager = msg
		self.process_data()

	#def bbox_callback(self, msg: String):
	#	self.latest_bbox = msg.data
	#	self.process_data()

	def process_data(self):
		"""
		This function is triggered every time ANY topic receives a new message.
		It ensures all streams have received at least one message before running.
		"""
		# Ensure we have received data from all topics
		if self.latest_imagel is None or self.latest_imager is None:
			return
		
		# Convert json
		# data = json.loads(self.latest_bbox)
		# left_bbox = data['left_bbox']
		# right_bbox = data['right_bbox']
			
		# images
		cv_img_l = self.bridge.imgmsg_to_cv2(self.latest_imagel, desired_encoding='bgr8')
		cv_img_r = self.bridge.imgmsg_to_cv2(self.latest_imager, desired_encoding='bgr8')

		# crop
		# bbox is x1,y1,x2,y2 and we slice [y1:y2, x1:x2]
		#if (left_bbox is not None and right_bbox is not None):
		#crop_l = cv_img_l[int(left_bbox[1]):int(left_bbox[3]), int(left_bbox[0]):int(left_bbox[2])]
		#crop_r = cv_img_r[int(right_bbox[1]):int(right_bbox[3]), int(right_bbox[0]):int(right_bbox[2])]

		crop_l = cv_img_l
		crop_r = cv_img_r

		# cv2.imshow("Left Crop", crop_l)
		# cv2.waitKey(1)
		

		# inference
		class_name_l, confidence_l = self.infer_image(crop_l)
		class_name_r, confidence_r = self.infer_image(crop_r)
		
		self.get_logger().info(f"Inference: [Left: {class_name_l} | {confidence_l * 100:.2f}%] [Right: {class_name_r} | {confidence_r * 100:.2f}%]")

		msg_l = {'class' : class_name_l, 'conf' : confidence_l}
		msg_r = {'class' : class_name_r, 'conf' : confidence_r}
		msg = {'left': msg_l, 'right': msg_r}
		out_msg = String()
		out_msg.data = json.dumps(msg)
		self.filling_level_pub.publish(out_msg)

		# -------------------------------------

		# Clear the variables if you want to require fresh messages from 
		# EVERY topic before processing again (rather than reusing the last known good frame).
		self.latest_image1 = None
		self.latest_image2 = None
		self.latest_string = None

	def infer_image(self, cv_img):
		results = self.model(cv_img)

		result = results[0]

		# Get the index of the highest confidence prediction
		top_class_index = result.probs.top1
		
		# Map that index to the actual class name ('full', 'half', or 'empty')
		class_name = result.names[top_class_index]
		
		# Get the confidence score (convert tensor to standard Python float)
		confidence = result.probs.top1conf.item()
		return class_name, confidence

def main(args=None):
	rclpy.init(args=args)
	node = YoloFillingLevelPredictorNode()
	
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		node.get_logger().info("Keyboard Interrupt detected, shutting down.")
	finally:
		node.destroy_node()
		rclpy.try_shutdown()

if __name__ == '__main__':
	main()