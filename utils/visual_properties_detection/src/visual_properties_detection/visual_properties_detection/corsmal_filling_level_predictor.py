import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image as PILImage

# skeleton by gemini: non-blocking async processing
class CorsmalFillingLevelPredictorNode(Node):
	def __init__(self):
		super().__init__('corsmal_filling_level_predictor_node')

		# Parameters
		self.declare_parameter('crop_topic', '/object_tracker/crop/left')
		self.crop_topic = self.get_parameter('crop_topic').value

		self.bridge = CvBridge()
		
		# 1. Threading: Use a separate group for the heavy processing
		self.processing_group = MutuallyExclusiveCallbackGroup()
		
		# 2. QoS: Keep only the 1 latest message
		qos = rclpy.qos.QoSProfile(
			depth=1,
			reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT
		)

		self.subscription = self.create_subscription(
			Image, self.crop_topic, self.listener_callback, qos)
		
		# 3. Storage for the latest data
		self.latest_msg = None
		
		# 4. Processing Loop: Check for new data every 0.1s (or faster)
		# It will only run one at a time because of the callback group
		self.timer = self.create_timer(
			0.1, self.inference_loop, callback_group=self.processing_group)
		
		# Model initialization
		self.class_names = ["Empty", "50% (Half-full)", "90% (Full)", "Opaque / Unknown"]

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.model = models.resnet18(pretrained=False, num_classes=4)
		self.model.load_state_dict(torch.load("model.t7", weights_only=False)['model'])
		# move to gpu
		self.model.to(self.device)

		# Set the model to evaluation mode
		self.model.eval() 

		# Resizing to 224x224 and applying standard ImageNet normalization
		self.transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((224, 224)), 
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

	def listener_callback(self, msg):
		# High speed reception: just swap the pointer and leave
		self.latest_msg = msg

	def inference_loop(self):
		if self.latest_msg is None:
			return

		# --- DO HEAVY INFERENCE HERE ---
		cv_img = self.bridge.imgmsg_to_cv2(self.latest_msg, desired_encoding='bgr8')
		self.latest_msg = None # clear to avoid processing same msg twice
		
		tensor = self.transform(cv_img).unsqueeze(0)
		tensor = tensor.to(self.device) # move to GPU
		with torch.no_grad():
			output = self.model(tensor)
		probabilities = torch.nn.functional.softmax(output[0], dim=0)
		predicted_class_idx = torch.argmax(probabilities).item()
		self.get_logger().info(f"Inference: {probabilities} | {self.class_names[predicted_class_idx]}")

def main():
	rclpy.init()
	node = CorsmalFillingLevelPredictorNode()
	# 5. Use MultiThreadedExecutor so the timer doesn't block the subscriber
	executor = MultiThreadedExecutor()
	rclpy.spin(node, executor=executor)