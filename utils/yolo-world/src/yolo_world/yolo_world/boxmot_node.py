import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLOWorld
import cv2
import numpy as np
from boxmot import create_tracker
# from torchvision.models.detection import (
#     fasterrcnn_resnet50_fpn_v2,
#     FasterRCNN_ResNet50_FPN_V2_Weights as Weights
# )

class BoxmotNode(Node):
    def __init__(self):
        super().__init__('boxmot_yolo_world_node')
        
        # Initialize Detector (YOLO-World)
        self.model = YOLOWorld('yolov8l-world.pt')
        self.model.set_classes(["glas", "cup", "mug", "bottle", "plate", "bowl", "food container"])

        # alternatively, from sample code
        # Load detector with pretrained weights and preprocessing transforms
        # weights = Weights.DEFAULT
        # detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
        # detector.to(device).eval()
        # transform = weights.transforms()
        
        # Initialize Boxmot Tracker (downloads if not found)
        self.tracker = create_tracker(
            tracker_type='botsort',      # Options: 'deepocsort', 'botsort', 'strongsort', 'bytetrack', 'ocsort'
            tracker_config=None,         # Uses the default config for the selected tracker
            reid_weights='osnet_x0_25_msmt17.pt', 
            device='cuda:0',             # Change to 'cpu' if you are not using a GPU
            half=False
        )

        self.bridge = CvBridge()
        ### Change which topic to listen to here!
        self.sub = self.create_subscription(Image, '/yolo/clipped_by_depth', self.image_callback, 10)
        #self.sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Image, '/yolo/detections', 10)
        
        self.get_logger().info("Boxmot + YOLO-World Node is ready.")

    def image_callback(self, msg):
        # Convert ROS to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run Detection ONLY (Bypass Ultralytics tracking)
        results = self.model.predict(cv_img, conf=0.3, verbose=False)
        result = results[0]
        
        # Extract Detections for Boxmot
        # Boxmot expects an (N, 6) numpy array: [x1, y1, x2, y2, conf, cls]
        dets = np.empty((0, 6))
        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            dets = np.column_stack((boxes, confs, clss))
        
        # Update Boxmot Tracker
        # 'tracks' returns an (N, 8) numpy array: [x1, y1, x2, y2, track_id, conf, cls, original_detection_index]
        tracks = self.tracker.update(dets, cv_img)
        
        # Visualize
        annotated_frame = cv_img.copy()
        if tracks is not None and len(tracks) > 0:
            self.get_logger().info(str(len(tracks)))#temp log

            # others
            for track in tracks[1:]:
                x1, y1, x2, y2, track_id, conf, cls, ind = track
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                
                # Generate a unique color based on the track_id
                color = (int(track_id * 30 % 255), int(track_id * 70 % 255), int(track_id * 110 % 255))
                
                # Draw the bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw the label with Track ID
                class_name = self.model.names[int(cls)]
                label = f"ID:{int(track_id)} {class_name} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # first one
            track = tracks[0]
            x1, y1, x2, y2, track_id, conf, cls, ind = track
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            
            # Generate a unique color based on the track_id
            color = (0,255,0)
            
            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw the label with Track ID
            class_name = self.model.names[int(cls)]
            label = f"ID:{int(track_id)} {class_name} {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # cache for next run
            self.last_tracks = tracks
            
        else: # no tracks found, "keep" old ones
            self.get_logger().info("none found") #temp log
            color = (0,0,255)
            x1, y1, x2, y2, track_id, conf, cls, ind = self.last_tracks[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw the label with Track ID
            class_name = self.model.names[int(cls)]
            label = f"ID:{int(track_id)} {class_name} {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert back to ROS and publish
        out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.pub.publish(out_msg)

def main():
    rclpy.init()
    rclpy.spin(BoxmotNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()