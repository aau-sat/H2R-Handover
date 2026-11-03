import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import message_filters
import cv2
import json
import mediapipe as mp
from collections import deque, defaultdict
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from pathlib import Path
import sys

# Attempt to import config, fallback to manual path if necessary
try:
    from utils.config import color_camera_topic_suffix
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[8]))
    try:
        from utils.config import color_camera_topic_suffix
    except:
        color_camera_topic_suffix = "/color/image_raw"

class HandTrackerNode(Node):
    def __init__(self):
        super().__init__('depth_aware_hand_tracker')

        # ------------------- PARAMETERS -------------------
        self.MAX_HISTORY = 40
        self.finger_tip_indices = [4, 8, 12, 16, 20]
        self.bridge = CvBridge()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands1 = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.hands2 = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Topic Declarations
        self.declare_parameter('camera_left_color', '/camera/left/color/image_raw')
        self.declare_parameter('camera_left_depth', '/camera/left/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_right_color', '/camera/right/color/image_raw')
        self.declare_parameter('camera_right_depth', '/camera/right/aligned_depth_to_color/image_raw')
        
        self.topic_l_rgb = self.get_parameter('camera_left_color').value
        self.topic_l_depth = self.get_parameter('camera_left_depth').value
        self.topic_r_rgb = self.get_parameter('camera_right_color').value
        self.topic_r_depth = self.get_parameter('camera_right_depth').value

        # Handover and Exit Logic
        self.declare_parameter('publish_mode', 'continuous')
        self.publish_mode = self.get_parameter('publish_mode').value
        self.handover_detected = False

        # Optional debug visualization window
        self.declare_parameter('show_debug_window', False)
        self.show_debug_window = bool(self.get_parameter('show_debug_window').value)
        self.debug_window_name = 'hand_tracker_debug'
        
        # Publishers
        self.pub_message = self.create_publisher(String, '/hand_tracker', 10)

        # QoS for Syncing
        sync_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Synchronized Subscribers (4-way sync)
        self.sub_l_rgb = message_filters.Subscriber(self, Image, self.topic_l_rgb, qos_profile=sync_qos)
        self.sub_l_depth = message_filters.Subscriber(self, Image, self.topic_l_depth, qos_profile=sync_qos)
        self.sub_r_rgb = message_filters.Subscriber(self, Image, self.topic_r_rgb, qos_profile=sync_qos)
        self.sub_r_depth = message_filters.Subscriber(self, Image, self.topic_r_depth, qos_profile=sync_qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_l_rgb, self.sub_l_depth, self.sub_r_rgb, self.sub_r_depth],
            queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("Depth-Aware Hand Tracker Initialized.")

    def get_average_depth(self, hand_landmarks, depth_img, w, h):
        """Calculates the average depth of the hand using all landmarks."""
        depth_values = []
        for lm in hand_landmarks.landmark:
            # Map normalized coords to pixel coords
            x, y = int(lm.x * w), int(lm.y * h)
            # Boundary check
            if 0 <= x < w and 0 <= y < h:
                d = depth_img[y, x]
                if d > 0:  # Ignore invalid depth readings
                    depth_values.append(float(d))
        
        return np.mean(depth_values) if depth_values else float('inf')

    def process_camera_view(self, rgb_msg, depth_msg, mp_model):
        """Processes a single camera view and returns info for the closest hand."""
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
        h, w = cv_rgb.shape[:2]

        results = mp_model.process(cv_rgb)
        
        hands_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate coordinates
                pts = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]
                
                # Finger tips center (active_hand_center)
                tips = [pts[i] for i in self.finger_tip_indices]
                f_center = (int(np.mean([p[0] for p in tips])), int(np.mean([p[1] for p in tips])))
                
                # Palm center
                palm_idx = [0, 5, 9, 13, 17]
                palms = [pts[i] for i in palm_idx]
                p_center = (int(np.mean([p[0] for p in palms])), int(np.mean([p[1] for p in palms])))
                
                # Depth selection
                avg_depth = self.get_average_depth(hand_landmarks, cv_depth, w, h)
                
                hands_data.append({
                    "depth": avg_depth,
                    "hand_center": f_center,
                    "palm_center": p_center,
                    "landmarks": hand_landmarks
                })

            # Sort by depth (closest first)
            hands_data.sort(key=lambda x: x["depth"])
            return True, hands_data, cv_rgb
        
        return False, [], cv_rgb

    def draw_debug_hands(self, image, hands_data, active_index):
        debug_image = image.copy()

        for idx, hand_data in enumerate(hands_data):
            is_active = idx == active_index
            hand_color = (0, 255, 0) if is_active else (0, 165, 255)
            palm_color = (0, 255, 255) if is_active else (255, 0, 255)

            hand_center = hand_data["hand_center"]
            palm_center = hand_data["palm_center"]
            landmarks = hand_data["landmarks"]

            self.mp_drawing.draw_landmarks(
                debug_image,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            if hand_center is not None:
                cv2.circle(debug_image, hand_center, 8, hand_color, -1)
                cv2.putText(
                    debug_image,
                    'ACTIVE' if is_active else 'HAND',
                    (hand_center[0] + 8, hand_center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    hand_color,
                    2,
                    cv2.LINE_AA,
                )

            if palm_center is not None:
                cv2.circle(debug_image, palm_center, 8, palm_color, -1)

        return debug_image

    def sync_callback(self, l_rgb, l_depth, r_rgb, r_depth):
        # Process Left
        det_l, hands_l, cv_img_l = self.process_camera_view(l_rgb, l_depth, self.hands1)
        # Process Right
        det_r, hands_r, cv_img_r = self.process_camera_view(r_rgb, r_depth, self.hands2)

        if not det_l and not det_r:
            return

        best_hand_l = hands_l[0] if det_l and hands_l else None
        best_hand_r = hands_r[0] if det_r and hands_r else None

        # Determine which camera has the closer hand
        depth_l = best_hand_l["depth"] if det_l else float('inf')
        depth_r = best_hand_r["depth"] if det_r else float('inf')

        more_hand_visible = 0  # Default: no preference
        if depth_l < depth_r:
            more_hand_visible = 1  # Left is closer
        elif depth_r < depth_l:
            more_hand_visible = 2  # Right is closer

        # Prepare JSON data - matching the format of hand_tracker_node.py
        message = {
            "left": {
                "detected": det_l,
                "active_palm_center": best_hand_l["palm_center"] if det_l else None,
                "active_hand_center": best_hand_l["hand_center"] if det_l else None
            },
            "right": {
                "detected": det_r,
                "active_palm_center": best_hand_r["palm_center"] if det_r else None,
                "active_hand_center": best_hand_r["hand_center"] if det_r else None
            },
            "more_hand_visible": more_hand_visible
        }

        # Publish
        message_json = String()
        message_json.data = json.dumps(message)
        self.pub_message.publish(message_json)

        if self.show_debug_window:
            debug_left = self.draw_debug_hands(cv_img_l, hands_l, 0 if det_l else -1)
            debug_right = self.draw_debug_hands(cv_img_r, hands_r, 0 if det_r else -1)

            cv2.putText(debug_left, 'Left camera', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_right, 'Right camera', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            target_height = max(debug_left.shape[0], debug_right.shape[0])
            if debug_left.shape[0] < target_height:
                pad = target_height - debug_left.shape[0]
                debug_left = cv2.copyMakeBorder(debug_left, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if debug_right.shape[0] < target_height:
                pad = target_height - debug_right.shape[0]
                debug_right = cv2.copyMakeBorder(debug_right, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            separator = np.zeros((target_height, 20, 3), dtype=np.uint8)
            debug_view = np.hstack([debug_left, separator, debug_right])
            cv2.imshow(self.debug_window_name, debug_view)
            cv2.waitKey(1)

        # Logging for Debug
        self.get_logger().info(f"L_depth: {depth_l:.1f}mm | R_depth: {depth_r:.1f}mm | More visible: {more_hand_visible}")

def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerNode()
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