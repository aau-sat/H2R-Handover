import json
import socket
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


SOCKET_PATH = "/tmp/sam3_socket"


class SamSocketBridge(Node):
    """
    A ROS 2 node that bridges camera images and object detections to a single shot
    SAM3 segmentation service.
    
    Subscribes to camera images and detection results, then sends them to the segmentation
    service via Unix and publishes to /segmentation_mask. (single shot service, with bounding boxes)
    """
    def __init__(self):
        """
        Initializes the SamSocketBridge node, setting up ROS 2 publishers/subscribers.
        """
        super().__init__("sam_socket_bridge")

        self.bridge = CvBridge()

        # Declare parameters for topic names
        self.declare_parameter('image_topic', '/camera/left/color/image_raw')
        self.declare_parameter('detection_topic', '/detected_objects')
        self.declare_parameter('mask_topic', '/segmentation_mask')
        
        # Get parameter values
        self.image_topic = self.get_parameter('image_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.mask_topic = self.get_parameter('mask_topic').value

        # Latest detection with bbox and label
        self.latest_detection = None
        self.last_valid_detection = None

        # ROS 2 subscriptions and publishers
        self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.create_subscription(String, self.detection_topic, self.det_cb, 10)
        self.mask_pub = self.create_publisher(Image, self.mask_topic, 10)

        self.get_logger().info(f"SAM Socket Bridge started (using Unix socket service)")
        self.get_logger().info(f"Subscribing to:")
        self.get_logger().info(f"  - {self.image_topic}")
        self.get_logger().info(f"  - {self.detection_topic}")
        self.get_logger().info(f"Publishing to: {self.mask_topic}")

    def det_cb(self, msg):
        """
        Callback for the detected objects topic.

        Parses the JSON payload to extract the best detection with bbox and label.

        Args:
            msg (std_msgs.msg.String): The incoming message containing JSON formatted detection data.
        """
        try:
            detections = json.loads(msg.data)
            if detections:
                # Get the detection with highest confidence
                best = max(detections, key=lambda x: x.get("conf", 0.0))
                
                # Extract bbox and label
                bbox = best.get("bbox")  # Expected format: [x1, y1, x2, y2]
                label = best.get("class")
                
                if bbox and label and len(bbox) == 4:
                    self.latest_detection = {
                        "bbox": bbox,
                        "label": label,
                        "conf": best.get("conf", 0.0)
                    }
                    self.last_valid_detection = self.latest_detection
                    self.get_logger().info(
                        f"Updated detection: {label} [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}] "
                        f"(conf: {best.get('conf', 0.0):.2f})"
                    )
                else:
                    self.latest_detection = None
            else:
                self.latest_detection = None
        except Exception as e:
            self.get_logger().error(f"Detection parse error: {e}")

    def image_cb(self, msg):
        """
        Callback for the incoming camera image topic.

        Converts the image to RGB, pairs it with the latest detection (bbox + label),
        and sends it to the segmentation service.

        Args:
            msg (sensor_msgs.msg.Image): The incoming image message.
        """
        # Use latest detection or fallback to last valid one
        detection = self.latest_detection or self.last_valid_detection

        if not detection:
            return

        try:
            # Convert image to RGB numpy array
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = frame[:, :, ::-1]  # BGR -> RGB

            h, w = frame.shape[:2]
            bbox = detection["bbox"]
            label = detection["label"]
            x1, y1, x2, y2 = map(int, bbox)

            # Send to segmentation service and get mask
            mask = self.call_segmentation_service(frame, h, w, x1, y1, x2, y2, label)

            if mask is not None and mask.size > 0:
                # Convert mask to ROS message
                mask_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
                mask_msg.header = msg.header  # Keep original timestamp
                self.mask_pub.publish(mask_msg)
                self.get_logger().info(f"Published mask for '{label}' at [{x1}, {y1}, {x2}, {y2}]")
            else:
                self.get_logger().warn(f"Empty mask received for '{label}'")

        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")

    def call_segmentation_service(self, frame, h, w, x1, y1, x2, y2, label):
        """
        Sends image and detection data to the socket segmentation service.

        Args:
            frame (np.ndarray): RGB image array (H, W, 3)
            h (int): Image height
            w (int): Image width
            x1, y1, x2, y2 (int): Bounding box coordinates
            label (str): Object class label

        Returns:
            np.ndarray: Segmentation mask (H, W) or empty array on failure
        """
        try:
            # Connect to Unix socket
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(SOCKET_PATH)

            # Encode label
            label_bytes = label.encode()
            label_len = len(label_bytes)

            # Pack header: h, w, x1, y1, x2, y2, label_len (7 integers = 28 bytes)
            header = np.array([h, w, x1, y1, x2, y2, label_len], dtype=np.int32)
            client.sendall(header.tobytes())
            
            client.sendall(label_bytes)
            img_bytes = frame.astype(np.uint8).tobytes()
            client.sendall(img_bytes)

            # Receive mask
            mask_size = h * w
            mask_data = b""
            while len(mask_data) < mask_size:
                chunk = client.recv(mask_size - len(mask_data))
                if not chunk:
                    break
                mask_data += chunk

            client.close()

            if len(mask_data) == mask_size:
                mask = np.frombuffer(mask_data, dtype=np.uint8).reshape((h, w))
                return mask
            else:
                self.get_logger().error(f"Incomplete mask received: {len(mask_data)}/{mask_size} bytes")
                return np.array([])

        except Exception as e:
            self.get_logger().error(f"Segmentation service call error: {e}")
            return np.array([])


def main():
    rclpy.init()
    node = SamSocketBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
