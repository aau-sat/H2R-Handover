import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import glob

class ImagePlayer(Node):
    """
    A ROS 2 Node that plays back images with an on-screen display (OSD) of metadata.

    Attributes:
        bridge (CvBridge): Converter between OpenCV and ROS images.
        fps (float): Frames per second for playback.
        image_files (list): Sorted list of image paths.
    """

    def __init__(self):
        """Initializes the node and sets up the OSD configuration."""
        super().__init__('image_player')
        
        # Parameters
        self.declare_parameter('folder_name', 'recorded_masks')
        self.declare_parameter('fps', 10.0)
        
        folder = self.get_parameter('folder_name').get_parameter_value().string_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value
        
        self.image_files = sorted(glob.glob(os.path.join(folder, "*.png")))
        
        if not self.image_files:
            self.get_logger().error(f"No images found in {folder}!")
            return

        self.bridge = CvBridge()
        self.index = 0
        self.publisher_ = self.create_publisher(Image, 'playback_images', 10)
        
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f"Playing {len(self.image_files)} images at {self.fps} FPS.")

    def timer_callback(self):
        """Processes images and adds text overlays before displaying."""
        if self.index >= len(self.image_files):
            self.index = 0
            
        img_path = self.image_files[self.index]
        cv_image = cv2.imread(img_path)
        
        if cv_image is not None:
            # Add Overlay Text
            info_text = f"Frame: {self.index + 1}/{len(self.image_files)} | FPS: {self.fps}"
            
            # cv2.putText(img, text, org, font, scale, color, thickness)
            cv2.putText(cv_image, info_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display image
            cv2.imshow("Image Playback", cv_image)
            cv2.waitKey(1)
            
            # Publish processed image
            msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.publisher_.publish(msg)
            
            self.index += 1

def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = ImagePlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()