import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLOWorld
import cv2

class YoloWorldNode(Node):
    def __init__(self):
        super().__init__('yolo_world_node')
        
        # Initialize YOLO-World
        self.model = YOLOWorld('yolov8x-world.pt')
        self.model.set_classes(["glas", "cup", "mug", "bottle", "plate", "bowl", "food container"]) # Define your targets here
        

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        #self.sub = self.create_subscription(Image, '/yolo/clipped_by_depth', self.image_callback, 10)
        self.pub = self.create_publisher(Image, '/yolo/detections', 10)
        
        self.get_logger().info("YOLO-World Node is ready.")

    def image_callback(self, msg):
        # Convert ROS to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run tracking (persist=True keeps IDs across frames)
        results = self.model.track(cv_img, 
                                   persist=True, 
                                   batch=10, 
                                   max_det=10, 
                                   stream_buffer=True, 
                                   conf=0.2, 
                                   tracker="bytetrack.yaml")
        
        # Visualize
        annotated_frame = results[0].plot()
        
        # Convert back to ROS and publish
        out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.pub.publish(out_msg)

def main():
    rclpy.init()
    rclpy.spin(YoloWorldNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()