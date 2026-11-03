import rclpy
from rclpy.node import Any, Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import json
import cv2
import numpy as np

import base64
import mimetypes
import requests

# run outside vllm serve google/gemma-4-E4B-it   --max-model-len 32768   --gpu-memory-utilization 0.4

# Uncomment if you plan to convert ROS Images to OpenCV formats
# from cv_bridge import CvBridge 

class GemmaFillingLevelPredictorNode(Node):
	def __init__(self):
		super().__init__('image_string_processor')

		self.bridge = CvBridge()

		# 1. Declare parameters with default topic names
		self.declare_parameter('image_left_topic', '/camera/left/color/image_raw')
		self.declare_parameter('image_right_topic', '/camera/right/color/image_raw')
		self.declare_parameter('bbox_topic', '/object_detections')
		self.declare_parameter('publishing_topic', '/filling_estimation')

		# 2. Retrieve parameter values
		imgl_topic = self.get_parameter('image_left_topic').get_parameter_value().string_value
		imgr_topic = self.get_parameter('image_right_topic').get_parameter_value().string_value
		bbox_topic = self.get_parameter('bbox_topic').get_parameter_value().string_value
		publishing_topic = self.get_parameter('publishing_topic').get_parameter_value().string_value

		self.get_logger().info(f"GemmaFillingLevelPredictorNode started. Listening to:\n- {imgl_topic}\n- {imgr_topic}\n- {bbox_topic}")

		# 3. Initialize state variables to cache the latest messages
		self.latest_imagel = None
		self.latest_imager = None
		self.latest_bbox = None
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
		
		self.sub_bbox = self.create_subscription(
			String,
			bbox_topic,
			self.bbox_callback,
			10
		)
		self.filling_level_pub = self.create_publisher(String, publishing_topic, 10)

	def imagel_callback(self, msg: Image):
		self.latest_imagel = msg
		self.process_data()

	def imager_callback(self, msg: Image):
		self.latest_imager = msg
		self.process_data()

	def bbox_callback(self, msg: String):
		self.latest_bbox = msg.data
		self.process_data()

	def process_data(self):
		"""
		This function is triggered every time ANY topic receives a new message.
		It ensures all streams have received at least one message before running.
		"""
		# Ensure we have received data from all topics
		if self.latest_imagel is None or self.latest_imager is None or self.latest_bbox is None:
			return
		
		# Convert json
		data = json.loads(self.latest_bbox)
		left_bbox = data['left_bbox']
		right_bbox = data['right_bbox']
		left_yolo_class = data['left_class']
		right_yolo_class = data['right_class']
			
		# images
		cv_img_l = self.bridge.imgmsg_to_cv2(self.latest_imagel, desired_encoding='bgr8')
		cv_img_r = self.bridge.imgmsg_to_cv2(self.latest_imager, desired_encoding='bgr8')

		# crop
		# bbox is x1,y1,x2,y2 and we slice [y1:y2, x1:x2]
		if (left_bbox is not None and right_bbox is not None):
			crop_l = cv_img_l[int(left_bbox[1]):int(left_bbox[3]), int(left_bbox[0]):int(left_bbox[2])]
			crop_r = cv_img_r[int(right_bbox[1]):int(right_bbox[3]), int(right_bbox[0]):int(right_bbox[2])]

			# cv2.imshow("Left Crop", crop_l)
			# cv2.waitKey(1)

			# inference
			res_l = self.infer_image(crop_l)
			res_r = self.infer_image(crop_r)
			
			self.get_logger().info(f"Inference: [Left: {res_l} ] [Right: {res_r} ]")

			msg_l = {'class' : res_l}
			msg_r = {'class' : res_r}
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
		out = self.run_gemma(cv_img, "You see a cup that may be empty or contain material/liquid. Respond with only one value: 0% if empty, 50% if it is half full, 100% if it is full")
		return out

	def _encode_cv2_image_to_data_url(self, cv2_image: Any, image_format: str = ".jpg") -> str:

		ok, buffer = cv2.imencode(image_format, cv2_image)
		if not ok:
			raise ValueError("Could not encode cv2 image")

		

		fmt = image_format.lower()
		if fmt == ".png":
			mime = "image/png"
		elif fmt == ".webp":
			mime = "image/webp"
		else:
			mime = "image/jpeg"

		encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
		return f"data:{mime};base64,{encoded}"


	def run_gemma(
	 	self,
		cv2_image: Any,
		prompt: str,
		model: str = "google/gemma-4-E4B-it",
		url: str = "http://localhost:8000/v1/chat/completions",
		max_tokens: int = 64,
		image_format: str = ".jpg",
		) -> str:
		image_data_url = self._encode_cv2_image_to_data_url(cv2_image, image_format=image_format)

		payload = {
			"model": model,
			"messages": [
				{
					"role": "user",
					"content": [
						{"type": "text", "text": prompt},
						{"type": "image_url", "image_url": {"url": image_data_url}},
					],
				}
			],
			"max_tokens": max_tokens,
		}
		response = requests.post(url, json=payload, timeout=120)
		response.raise_for_status()
		data = response.json()
		return data["choices"][0]["message"]["content"]
	


def main(args=None):
	rclpy.init(args=args)
	node = GemmaFillingLevelPredictorNode()
	
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		node.get_logger().info("Keyboard Interrupt detected, shutting down.")
	finally:
		node.destroy_node()
		rclpy.try_shutdown()

if __name__ == '__main__':
	main()