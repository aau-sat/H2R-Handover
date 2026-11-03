import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_world_node')
        
        # Initialize YOLO-World
        self.model = YOLO('yolo11x.pt')
        #self.model = YOLOWorld('yolov8l-world.pt')
        #self.model.set_classes(["glas", "cup", "mug", "bottle", "plate", "bowl", "food container"]) # Define your targets here
        
        self.model.info()

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/left/color/image_raw', self.image_callback, 10)
        #self.sub = self.create_subscription(Image, '/yolo/clipped_by_depth', self.image_callback, 10)

        self.pub = self.create_publisher(Image, '/yolo/detections', 10)
        self.pubcrop = self.create_publisher(Image, '/yolo/dections/crop', 10)
        
        self.get_logger().info("YOLO-World Node is ready.")

    def image_callback(self, msg):
        # Convert ROS to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run tracking (persist=True keeps IDs across frames)
        results = self.model.track(cv_img, persist=True, classes=[41,40,45,75])
        # results = self.model.track(cv_img, 
        #                            persist=True, 
        #                            batch=10, 
        #                            max_det=5, 
        #                            stream_buffer=True, 
        #                            conf=0.3, 
        #                            tracker="bytetrack.yaml")

        # cropping for test
        main_result = results[0]
        coords = main_result.boxes[0].xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(cv_img.shape[1], x2 + padding)
        y2 = min(cv_img.shape[0], y2 + padding)

        crop = cv_img[y1:y2, x1:x2]
        out_crop = self.bridge.cv2_to_imgmsg(crop, encoding='bgr8')
        self.pubcrop.publish(out_crop)
        
        # Visualize
        annotated_frame = results[0].plot()
        
        # Convert back to ROS and publish
        out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.pub.publish(out_msg)

def main():
    rclpy.init()
    rclpy.spin(YoloNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()