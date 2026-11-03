import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters

class DualImageSubscriber(Node):
    """
    A ROS 2 node that subscribes to two image topics and displays them.

    This node uses message_filters to synchronize the two incoming image
    streams and OpenCV to display them in separate windows.

    Attributes:
        bridge (CvBridge): Utility to convert ROS images to OpenCV images.
        image1_sub (message_filters.Subscriber): Subscriber for the first image topic.
        image2_sub (message_filters.Subscriber): Subscriber for the second image topic.
        ts (message_filters.ApproximateTimeSynchronizer): Synchronizer for the two topics.
    """

    def __init__(self):
        """Initializes the DualImageSubscriber node and sets up the synchronizer."""
        super().__init__('dual_image_subscriber')
        self.bridge = CvBridge()

        # Using message_filters to ensure images are processed simultaneously 
        self.image1_sub = message_filters.Subscriber(self, Image, '/camera/cam_left/color/image_raw')
        self.image2_sub = message_filters.Subscriber(self, Image, '/camera/cam_right/color/image_raw')

        # ApproximateTimeSynchronizer is best practice for real-world sensor data
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image1_sub, self.image2_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

    def image_callback(self, msg1: Image, msg2: Image) -> None:
        """
        Callback function to upscale and display synchronized images.

        Args:
            msg1 (Image): The image message from the first topic.
            msg2 (Image): The image message from the second topic.
        """
        try:
            # Convert ROS to OpenCV
            cv_image1 = self.bridge.imgmsg_to_cv2(msg1, "bgr8")
            cv_image2 = self.bridge.imgmsg_to_cv2(msg2, "bgr8")

            # Upscale both images by a factor of 2 using linear interpolation
            res1 = cv2.resize(cv_image1, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            res2 = cv2.resize(cv_image2, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

            # Displaying upscaled images
            cv2.imshow("Camera 1 (2x Scale)", res1)
            cv2.imshow("Camera 2 (2x Scale)", res2)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

def main(args=None) -> None:
    """
    Main entry point for the node.

    Args:
        args (list, optional): Command line arguments. Defaults to None.
    """
    rclpy.init(args=args)
    node = DualImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received, shutting down.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()