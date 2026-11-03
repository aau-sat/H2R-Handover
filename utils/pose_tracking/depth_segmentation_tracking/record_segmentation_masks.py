import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageRecorder(Node):
    """
    A ROS 2 Node that subscribes to an image topic and saves frames to disk.
    
    Attributes:
        bridge (CvBridge): Converter between ROS and OpenCV images.
        save_path (str): Directory where images will be stored.
        count (int): Counter for file naming.
    """

    def __init__(self):
        """Initializes the node, creates the subscription, and prepares the directory."""
        super().__init__('image_recorder')
        
        # Parameters
        self.declare_parameter('topic_name', '/segmentation_mask')
        self.declare_parameter('folder_name', 'recorded_masks')
        
        topic = self.get_parameter('topic_name').get_parameter_value().string_value
        self.save_path = self.get_parameter('folder_name').get_parameter_value().string_value
        
        # Setup environment
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            self.get_logger().info(f"Created directory: {self.save_path}")

        self.bridge = CvBridge()
        self.count = 0
        
        # Subscription
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10)
        
        self.get_logger().info(f"Recording from {topic} to {self.save_path}...")

    def image_callback(self, msg: Image):
        """
        Converts ROS Image message to OpenCV format and saves it as a PNG.

        Args:
            msg (Image): The incoming ROS image message.
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Save image
            file_name = os.path.join(self.save_path, f"frame_{self.count:04d}.png")
            cv2.imwrite(file_name, cv_image)
            
            self.get_logger().info(f"Saved: {file_name}")
            self.count += 1
            
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {str(e)}")

def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)
    node = ImageRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()