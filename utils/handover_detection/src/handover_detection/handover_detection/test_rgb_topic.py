# basic MediaPipe overlay
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import mediapipe as mp
import cv2

class ImageListener(Node):
	def __init__(self):
		super().__init__('image_listener')
		self.bridge = CvBridge()
		self.subscription = self.create_subscription(
			Image,
			'/rgb',
			self.callback,
			10)
		self.mp_hands = mp.solutions.hands
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_styles = mp.solutions.drawing_styles
		# Create a Hands object
		self.hands = self.mp_hands.Hands(
			static_image_mode=False,      # Use for video (set True for single images)
			max_num_hands=2,              # Detect up to 2 hands
			min_detection_confidence=0.2, # Confidence thresholds
			min_tracking_confidence=0.2
		)

	def callback(self, msg):
		# Convert ROS Image message to OpenCV image
		cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

		# Convert to RGB (MediaPipe uses RGB)
		rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

		# Process the frame
		results = self.hands.process(rgb_frame)

		# Draw the hand skeleton
		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				self.mp_drawing.draw_landmarks(
					cv_image,  # image to draw on
					hand_landmarks,
					self.mp_hands.HAND_CONNECTIONS,
					self.mp_styles.get_default_hand_landmarks_style(),
					self.mp_styles.get_default_hand_connections_style()
				)

		# Show or process the image
		cv2.imshow("Image with hand detection", cv_image)
		cv2.waitKey(1)

def main():
	rclpy.init()
	node = ImageListener()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
