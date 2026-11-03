import json
import os
import queue
import time
import threading
import pickle
import zmq
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
from cv_bridge import CvBridge


_CV2_UI_LOCK = threading.Lock()


def get_timestamp_str(header):
    """
    Extracts the timestamp from a ROS 2 header and formats it as a string.

    Args:
        header (std_msgs.msg.Header): The header of the incoming ROS 2 message.

    Returns:
        str: The timestamp formatted as 'seconds.nanoseconds'.
    """
    return f"{header.stamp.sec}.{header.stamp.nanosec}"


class SamBridgeAsync(Node):
    """
    A ROS 2 node acting as an asynchronous bridge for SAM (Segment Anything Model) inference.
    
    Pairs incoming camera images with the latest object detection labels and exchanges 
    them with a backend inference server via ZeroMQ.
    """
    def __init__(
        self,
        camera="left",
        node_name="sam_bridge_async",
        image_topic="/camera/left/color/image_raw",
        detection_topic="/detected_objects",
        hand_tracker_topic="/hand_tracker",
        mask_topic="/segmentation/left",
        push_addr="tcp://localhost:5555",
        sub_addr="tcp://localhost:5556",
        target_fps=30.0,
        track_hand=False,
        show_visualization=False,
    ):
        """
        Initializes the SamBridgeAsync node, setting up ZMQ sockets, 
        ROS 2 publishers/subscribers, and synchronization primitives.
        """
        super().__init__(node_name)

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # Declare parameters for topic names
        self.declare_parameter('camera', camera)
        self.declare_parameter('image_topic', image_topic)
        self.declare_parameter('detection_topic', detection_topic)
        self.declare_parameter('hand_tracker_topic', hand_tracker_topic)
        self.declare_parameter('mask_topic', mask_topic)
        self.declare_parameter('reinit_mask_topic', 'segmentation_mask_reinit')
        self.declare_parameter('push_addr', push_addr)
        self.declare_parameter('sub_addr', sub_addr)
        self.declare_parameter('target_fps', float(target_fps))
        self.declare_parameter('track_hand', bool(track_hand))
        self.declare_parameter('show_visualization', bool(show_visualization))
        
        # Get parameter values
        self.camera = str(self.get_parameter('camera').value or camera)
        self.image_topic = str(self.get_parameter('image_topic').value or image_topic)
        self.detection_topic = str(self.get_parameter('detection_topic').value or detection_topic)
        self.hand_tracker_topic = str(self.get_parameter('hand_tracker_topic').value or hand_tracker_topic)
        self.mask_topic = str(self.get_parameter('mask_topic').value or mask_topic)
        self.reinit_mask_topic = str(
            self.get_parameter('reinit_mask_topic').value or 'segmentation_mask_reinit'
        )
        self.push_addr = str(self.get_parameter('push_addr').value or push_addr)
        self.sub_addr = str(self.get_parameter('sub_addr').value or sub_addr)
        self.target_fps = float(self.get_parameter('target_fps').value or target_fps)
        self.track_hand = bool(
            self.get_parameter('track_hand').value
            if self.get_parameter('track_hand').value is not None
            else track_hand
        )
        self.show_visualization = bool(
            self.get_parameter('show_visualization').value
            if self.get_parameter('show_visualization').value is not None
            else show_visualization
        )

        self.window_name = f"SAM Bridge Outgoing [{self.camera}]"
        self._visualization_enabled = self.show_visualization
        self._viz_queue = None
        self._viz_stop = None
        self._viz_thread = None

        if self._visualization_enabled:
            has_display = bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
            if not has_display:
                self._visualization_enabled = False
                self.get_logger().warn(
                    "Visualization requested, but no DISPLAY/WAYLAND_DISPLAY found. "
                    "Disabling imshow."
                )
            else:
                self._viz_queue = queue.Queue(maxsize=1)
                self._viz_stop = threading.Event()
                self._viz_thread = threading.Thread(
                    target=self._visualization_loop,
                    daemon=True,
                )
                self._viz_thread.start()

        # Latest / fallback detection
        self.latest_label = None
        self.last_valid_label = None
        self.latest_bbox = None
        self.last_valid_bbox = None
        self.latest_id = None
        self.last_valid_id = None
        self.latest_hand_center = None
        self.latest_hand_tracker_msg = None

        self.header_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_max = 200  

        # ZMQ 
        ctx = zmq.Context()

        self.push = ctx.socket(zmq.PUSH)
        self.push.connect(self.push_addr)
        self.push.setsockopt(zmq.SNDHWM, 5)   # Small buffer -> drops when overloaded
        self.push.setsockopt(zmq.LINGER, 0)

        # Receiver
        self.sub = ctx.socket(zmq.SUB)
        self.sub.connect(self.sub_addr)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.setsockopt(zmq.RCVHWM, 50)
        self.sub.setsockopt(zmq.LINGER, 0)
        
        # Throttling Setup
        self.min_interval = 1.0 / self.target_fps
        self.last_send_time = self.get_clock().now()
        
        # create sensor qos profile
        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile=qos_profile)
        self.create_subscription(String, self.detection_topic, self.det_cb, qos_profile=qos_profile)
        if self.track_hand:
            self.create_subscription(String, self.hand_tracker_topic, self.hand_tracker_cb, qos_profile=qos_profile)
        self.mask_pub = self.create_publisher(Image, self.mask_topic, 10)
        self.reinit_mask_pub = self.create_publisher(Header, self.reinit_mask_topic, 10)

        # Mask receiver thread
        self.recv_thread = threading.Thread(target=self.mask_receiver_loop, daemon=True)
        self.recv_thread.start()

        self.get_logger().info("SAM Bridge Async started (timestamp-based)")
        self.get_logger().info(f"Camera selector: {self.camera}")
        self.get_logger().info(f"ZMQ connect push: {self.push_addr}")
        self.get_logger().info(f"ZMQ connect sub:  {self.sub_addr}")
        self.get_logger().info(f"Subscribing to:")
        self.get_logger().info(f"  - {self.image_topic}")
        self.get_logger().info(f"  - {self.detection_topic}")
        if self.track_hand:
            self.get_logger().info(f"  - {self.hand_tracker_topic} (hand tracking enabled)")
        self.get_logger().info(f"Publishing to: {self.mask_topic}")
        self.get_logger().info(f"Publishing reinit flags to: {self.reinit_mask_topic}")

    def _visualization_loop(self):
        """Serialize all OpenCV window calls to avoid UI thread contention."""
        if self._viz_stop is None or self._viz_queue is None:
            return

        viz_stop = self._viz_stop
        viz_queue = self._viz_queue

        try:
            with _CV2_UI_LOCK:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            while rclpy.ok() and not viz_stop.is_set():
                try:
                    vis = viz_queue.get(timeout=0.1)
                except queue.Empty:
                    with _CV2_UI_LOCK:
                        cv2.waitKey(1)
                    continue

                with _CV2_UI_LOCK:
                    cv2.imshow(self.window_name, vis)
                    cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Visualization loop error: {e}")
            self._visualization_enabled = False

    def _enqueue_visualization(self, vis):
        if not self._visualization_enabled or self._viz_queue is None:
            return

        try:
            self._viz_queue.put_nowait(vis)
        except queue.Full:
            try:
                _ = self._viz_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._viz_queue.put_nowait(vis)
            except queue.Full:
                pass

    def destroy_node(self):
        if self._viz_stop is not None:
            self._viz_stop.set()
        if self._viz_thread is not None and self._viz_thread.is_alive():
            self._viz_thread.join(timeout=1.0)
        if self._visualization_enabled:
            with _CV2_UI_LOCK:
                try:
                    cv2.destroyWindow(self.window_name)
                except Exception:
                    pass
        return super().destroy_node()

    def det_cb(self, msg):
        """
        Callback for the detected objects topic.

        Parses the JSON payload to extract the most confident detection label
        and its bounding box.

        Args:
            msg (std_msgs.msg.String): The incoming message containing JSON formatted detection data.
        """
        try:
            detections = json.loads(msg.data)

            # Accept either a list of detections or a single detection object.
            if isinstance(detections, dict):
                detections = [detections]
            if not isinstance(detections, list):
                self.get_logger().warn("Detection payload is not a list/dict; ignoring message.")
                return

            with self.lock:
                if detections:
                    best = max(detections, key=lambda x: x.get("conf", 0.0))

                    # Support both formats:
                    # 1) Camera-keyed: {"left": {...}, "right": {...}, "class_name": ...}
                    # 2) Flat: {"class": ..., "bbox": [...], ...}
                    camera_specific_data = best.get(self.camera)
                    if not isinstance(camera_specific_data, dict):
                        camera_specific_data = best

                    label = (
                        best.get("class_name")
                        or best.get("class")
                        or best.get("label")
                        or camera_specific_data.get("class_name")
                        or camera_specific_data.get("class")
                        or camera_specific_data.get("label")
                    )

                    obj_id = best.get("id", camera_specific_data.get("id", None))
                    bbox = camera_specific_data.get("bbox", None)

                    hand_center = camera_specific_data.get(
                        "active_hand_center",
                        best.get("active_hand_center", None),
                    )
                    if label:
                        self.latest_label = label
                        self.last_valid_label = label
                    if isinstance(bbox, list) and len(bbox) == 4:
                        
                        self.latest_bbox = bbox
                        self.last_valid_bbox = bbox
                    if isinstance(hand_center, list) and len(hand_center) == 2:
                        self.latest_hand_center = hand_center
                    else:
                        self.latest_hand_center = None
                    if obj_id is not None:
                        self.latest_id = obj_id
                        self.last_valid_id = obj_id
                else:
                    self.latest_label = None
                    self.latest_bbox = None
                    self.latest_id = None
                    self.latest_hand_center = None
        except Exception as e:
            self.get_logger().error(f"Detection parse error: {e}")

    def hand_tracker_cb(self, msg):
        """Store the latest /hand_tracker JSON payload for the active camera view."""
        with self.lock:
            self.latest_hand_tracker_msg = msg

    def _landmark_point_to_pixel(self, landmark, width, height):
        if isinstance(landmark, dict):
            x_value = landmark.get('x')
            y_value = landmark.get('y')
        elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
            x_value, y_value = landmark[0], landmark[1]
        else:
            return None

        if x_value is None or y_value is None:
            return None

        x_value = float(x_value)
        y_value = float(y_value)

        if -1.0 <= x_value <= 1.5 and -1.0 <= y_value <= 1.5:
            pixel_x = int(round(x_value * width))
            pixel_y = int(round(y_value * height))
        else:
            pixel_x = int(round(x_value))
            pixel_y = int(round(y_value))

        return pixel_x, pixel_y

    def _bbox_from_landmarks(self, landmarks, width, height, padding_ratio=0.20, min_padding_px=24):
        if not isinstance(landmarks, list) or not landmarks:
            return None

        points = []
        for landmark in landmarks:
            point = self._landmark_point_to_pixel(landmark, width, height)
            if point is not None:
                points.append(point)

        if not points:
            return None

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_min = min(x_values)
        y_min = min(y_values)
        x_max = max(x_values)
        y_max = max(y_values)

        span = max(x_max - x_min, y_max - y_min)
        padding = max(int(round(span * padding_ratio)), min_padding_px)

        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width - 1, x_max + padding)
        y_max = min(height - 1, y_max + padding)

        if x_max <= x_min:
            x_max = min(width - 1, x_min + max(1, min_padding_px))
        if y_max <= y_min:
            y_max = min(height - 1, y_min + max(1, min_padding_px))

        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def _build_hand_tracker_prompt(self, payload, width, height):
        if not isinstance(payload, dict):
            return None, None, None

        view_data = payload.get(self.camera)
        if not isinstance(view_data, dict):
            return None, None, None

        hand_center = view_data.get('active_hand_center')
        landmarks = view_data.get('landmarks')
        bbox = self._bbox_from_landmarks(landmarks, width, height)

        if bbox is None and isinstance(hand_center, list) and len(hand_center) == 2:
            center_x, center_y = hand_center
            center_x = int(round(float(center_x)))
            center_y = int(round(float(center_y)))
            half_side = max(24, int(round(min(width, height) * 0.10)))
            bbox = [
                max(0, center_x - half_side),
                max(0, center_y - half_side),
                min(width - 1, center_x + half_side),
                min(height - 1, center_y + half_side),
            ]

        return hand_center, bbox, landmarks

    def image_cb(self, msg):
        """
        Callback for the incoming camera image topic.

        Converts the image to RGB, pairs it with the latest label, and sends it via ZMQ.
        Uses the image timestamp as the unique identifier to associate the resulting mask.

        Args:
            msg (sensor_msgs.msg.Image): The incoming image message.
        """
        
        current_time = self.get_clock().now()
        elapsed_sec = (current_time - self.last_send_time).nanoseconds / 1e9
        
        if elapsed_sec < self.min_interval:
            return  # Drop frame to maintain 12 Hz

        label = None
        obj_id = None
        bbox = None
        hand_center = None
        hand_landmarks = None

        if self.track_hand:
            with self.lock:
                latest_hand_tracker_msg = self.latest_hand_tracker_msg

            if latest_hand_tracker_msg is None or not latest_hand_tracker_msg.data:
                return

            try:
                hand_tracker_payload = json.loads(latest_hand_tracker_msg.data)
            except Exception as e:
                self.get_logger().error(f"Hand tracker parse error: {e}")
                return

            hand_center, bbox, hand_landmarks = self._build_hand_tracker_prompt(
                hand_tracker_payload,
                msg.width,
                msg.height,
            )

            if hand_center is None or bbox is None:
                return

            label = "Hand"
            obj_id = 1
        else:
            with self.lock:
                label = self.latest_label or self.last_valid_label
                obj_id = self.latest_id or self.last_valid_id
                bbox = self.latest_bbox or self.last_valid_bbox
                hand_center = self.latest_hand_center

            if not label:
                return

        self.last_send_time = current_time
        try:
            bgr_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = bgr_frame[:, :, ::-1]  # BGR -> RGB (this is what gets sent)

            timestamp_key = get_timestamp_str(msg.header)

            # Cache header so mask gets the correct timestamp
            with self.cache_lock:
                self.header_cache[timestamp_key] = msg.header
                # Simple cache bound
                if len(self.header_cache) > self.cache_max:
                    # Drop oldest
                    oldest = next(iter(self.header_cache))
                    self.header_cache.pop(oldest, None)

            payload = {
                "timestamp_key": timestamp_key,
                "h": frame.shape[0],
                "w": frame.shape[1],
                "obj_id": obj_id,
                "label": label,
                "bbox": bbox,
                "active_hand_center": hand_center,
                "hand_landmarks": hand_landmarks,
                "image": frame,
            }

            # Visualize the outgoing frame with the same prompt sent to SAM.
            if self.show_visualization:
                vis = bgr_frame.copy()
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        vis,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                if isinstance(hand_center, list) and len(hand_center) == 2:
                    hx, hy = [int(v) for v in hand_center]
                    cv2.circle(vis, (hx, hy), 5, (255, 0, 0), -1)
                self._enqueue_visualization(vis)

            try:
                self.get_logger().info(
                    f"Sending frame {timestamp_key} with id {obj_id}, label '{label}', bbox {bbox}, hand_center {hand_center}"
                )
                self.push.send(pickle.dumps(payload), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

        except Exception as e:
            self.get_logger().error(f"Image send error: {e}")

    def mask_receiver_loop(self):
        """
        Background thread loop to receive processed masks via ZMQ and publish them to ROS 2.
        
        Re-attaches the original cached ROS 2 header to the mask to maintain 
        temporal synchronization across the system.
        """
        while rclpy.ok():
            try:
                data = self.sub.recv()  
                resp = pickle.loads(data)

                # Use new key
                timestamp_key = resp["timestamp_key"]
                h = resp["h"]
                w = resp["w"]
                mask_bytes = resp["mask"]
                reinit = bool(resp.get("reinit", False) or resp.get("reinitialized", False))

                mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(h, w)
                mask_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")

                with self.cache_lock:
                    header = self.header_cache.pop(timestamp_key, None)

                if header is not None:
                    mask_msg.header = header
                else:
                    mask_msg.header.stamp = self.get_clock().now().to_msg()
                
                stamp = mask_msg.header.stamp
                ns = stamp.sec * 1_000_000_000 + stamp.nanosec
                
                if not hasattr(self, "last_ns"):
                    self.last_ns = 0
                if ns < self.last_ns:
                    self.get_logger().error(f"TIME WENT BACKWARDS: {ns} < {self.last_ns}")
                self.last_ns = ns
                
                self.mask_pub.publish(mask_msg)

                if reinit:
                    reinit_msg = Header()
                    reinit_msg.stamp = mask_msg.header.stamp
                    # Minimalistic encoding: use frame_id to transport boolean state.
                    reinit_msg.frame_id = "true"
                    self.reinit_mask_pub.publish(reinit_msg)
                    self.get_logger().info(
                        f"Published reinit header for timestamp {timestamp_key} on {self.reinit_mask_topic}"
                    )

            except Exception as e:
                self.get_logger().error(f"Mask recv error: {e}")
                time.sleep(0.01)


def main():
    """
    Main entry point for the node.
    """
    rclpy.init()
    node = SamBridgeAsync()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()