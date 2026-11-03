import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

class YoloWorldTracker(Node):
    def __init__(self):
        super().__init__('yolo_world_tracker')

        # --- CONFIGURATION ---
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('output_topic', '/yolo_world/tracked_image')
        #self.declare_parameter('target_classes', ["glas", "cup", "mug", "bottle", "plate", "bowl", "food container"]) # Define what you want to track
        self.declare_parameter('confidence_threshold', 0.1)
        
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        #self.target_classes = self.get_parameter('target_classes').get_parameter_value().string_array_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        # --- ROS HANDLERS ---
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )
        self.publisher_ = self.create_publisher(Image, output_topic, 10)
        self.bridge = CvBridge()

        # --- YOLO WORLD & TRACKER SETUP ---
        self.get_logger().info("Loading YOLO-World model...")
        # Initialize YOLO-World (requires ROBOFLOW_API_KEY in environment)
        self.model = YOLOWorld(model_id="yolo_world/l")
        self.model.set_classes([["glas", "cup", "mug", "bottle", "plate", "bowl", "food container", "drink container"]])
        
        # Initialize Tracker (ByteTrack)
        self.tracker = sv.ByteTrack()

        # Initialize Annotators
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        self.get_logger().info(f"YOLO-World Tracker initialized. Tracking stuff")

    def image_callback(self, msg):
        try:
            # 1. Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 2. Inference (Detection)
            results = self.model.infer(frame, confidence=self.conf_threshold)
            
            # 3. Convert results to Supervision Detections
            detections = sv.Detections.from_inference(results)

            # 4. Update Tracker
            # This assigns unique tracker_id to each detection across frames
            detections = self.tracker.update_with_detections(detections)

            # 5. Annotation
            # Create labels that include the Tracker ID
            labels = [
                f"#{tracker_id} {self.target_classes[class_id]} {confidence:0.2f}"
                for class_id, tracker_id, confidence
                in zip(detections.class_id, detections.tracker_id, detections.confidence)
            ]

            annotated_frame = self.box_annotator.annotate(
                scene=frame.copy(), 
                detections=detections
            )
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, 
                detections=detections, 
                labels=labels
            )

            # 6. Publish the output image
            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.publisher_.publish(out_msg)

            # Optional: Display locally for debugging
            # cv2.imshow("YOLO-World Tracking", annotated_frame)
            # cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = YoloWorldTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()