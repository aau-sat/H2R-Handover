import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2

class DoubleTrackerYoloNode(Node):
    def __init__(self):
        super().__init__('double_tracker_yolo_node')

        self.bridge = CvBridge()

        self.publeft = self.create_publisher(Image, '/yolo/det_left', 10)
        self.pubright = self.create_publisher(Image, '/yolo/det_right', 10)

        self.bridge = CvBridge()
        self.model = YOLO('yolo11l.pt') # Load model once

        # Subscribers
        self.subleft = Subscriber(self, Image, '/camera/cam_left/color/image_raw')
        self.subright = Subscriber(self, Image, '/camera/cam_right/color/image_raw')

        # Synchronizer: queue_size=10, slop=0.1 seconds
        self.ts = ApproximateTimeSynchronizer([self.subleft, self.subright], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

    def image_callback(self, msg_left, msg_right):

        # Convert to OpenCV
        cv_left = self.bridge.imgmsg_to_cv2(msg_left, "bgr8")
        cv_right = self.bridge.imgmsg_to_cv2(msg_right, "bgr8")

        # Run Inference
        res_left = self.model(cv_left, verbose=False, classes=[41,40,45,75])[0]
        res_right = self.model(cv_right, verbose=False, classes=[41,40,45,75])[0]

        annotated_left = res_left.plot()
        annotated_right = res_right.plot()

        det_left = len(res_left.boxes) > 0
        det_right = len(res_right.boxes) > 0

        # Logging Logic
        if det_left and det_right:
            self.get_logger().info("Object detected in BOTH streams")
        elif det_left:
            self.get_logger().info("Object detected in Stream A only")
        elif det_right:
            self.get_logger().info("Object detected in Stream B only")
        else:
            self.get_logger().info("No object detected", throttle_duration_sec=1.0)

        out_msg_left = self.bridge.cv2_to_imgmsg(annotated_left, encoding='bgr8')
        out_msg_right = self.bridge.cv2_to_imgmsg(annotated_right, encoding='bgr8')
        self.publeft.publish(out_msg_left)
        self.pubright.publish(out_msg_right)

def main():
    rclpy.init()
    rclpy.spin(DoubleTrackerYoloNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()