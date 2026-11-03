import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import onnxruntime as ort

# Roboflow & Supervision Imports
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

class YoloWorldRoboflowNode(Node):
    def __init__(self):
        super().__init__('yolo_world_roboflow_node')
        self.get_logger().info("Using device: " + ort.get_device())
        
        # 1. Initialize YOLO-World via Inference
        # Options: "yolo_world/s", "yolo_world/m", "yolo_world/l"
        self.model = YOLOWorld(model_id="yolo_world/s")
        self.classes = ["glas", "cup", "mug", "bottle", "plate", "bowl", "food container"]
        self.model.set_classes(self.classes)
        self.conf_threshold = 0.3
        
        # 2. Initialize Supervision Tools
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=50,
            frame_rate=30
        )
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        # 3. ROS2 Setup
        self.bridge = CvBridge()
        # qos_profile = QoSProfile(
        #     reliability = ReliabilityPolicy.BEST_EFFORT,
        #     history = HistoryPolicy.KEEP_LAST,
        #     depth = 1
        # )
        self.sub = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.image_callback, 
            10)
        self.pub = self.create_publisher(Image, '/yolo/detections', 10)
        
        self.get_logger().info("YOLO-World Node (Roboflow + Supervision) Started")

    def image_callback(self, msg):
        # Convert ROS to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 4. Inference
        # Note: Set confidence lower for zero-shot models if needed
        results = self.model.infer(frame, confidence=self.conf_threshold)
        detections = sv.Detections.from_inference(results)
        
        # 5. Tracking (updates detections with tracker_id)
        detections = self.tracker.update_with_detections(detections)
        
        # 6. Annotation
        # Create labels: "Class Name #ID"
        labels = [
            f"{self.classes[class_id]} #{tracker_id}"
            for class_id, tracker_id 
            in zip(detections.class_id, detections.tracker_id)
        ]
        
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # 7. Publish
        out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.pub.publish(out_msg)

def main():
    rclpy.init()
    rclpy.spin(YoloWorldRoboflowNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()

