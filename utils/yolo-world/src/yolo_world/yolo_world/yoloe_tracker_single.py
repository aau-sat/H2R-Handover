import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import traceback
from pathlib import Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

try:
    from ultralytics import YOLOE
except ImportError:
    YOLOE = None


class YoloESegmentationTracker(Node):
    """ROS 2 Node for YOLOE segmentation and bounding box tracking using JSON output."""

    def __init__(self):
        """Initializes the YoloESegmentationTracker node, parameters, and ROS publishers/subscribers."""
        super().__init__("yoloe_segmentation_tracker")

        # config
        self.declare_parameter("input_topic", "/camera/left/color/image_raw")
        self.declare_parameter("output_topic", "/segmentation/left")
        self.declare_parameter("json_topic", "/object_detections")
        self.declare_parameter("model_path", "/home/student-team/rgmc-2026/utils/yolo-world/src/yolo_world/resource/yoloe-11l-seg.pt")
        self.declare_parameter("target_labels", ["wine glass", "cup"])
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.5)
        self.declare_parameter("use_tracker", True)
        self.declare_parameter("tracker_config", "bytetrack.yaml")
        self.declare_parameter("latest_only_mode", True)
        self.declare_parameter("input_queue_depth", 1)
        self.declare_parameter("enable_visualization", True)
        self.declare_parameter("visualization_window_name", "YOLOE Single Tracker")
        self.declare_parameter("visualization_mask_alpha", 0.45)
        self.declare_parameter("visualization_scale", 1.0)

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        json_topic = self.get_parameter("json_topic").get_parameter_value().string_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.json_topic = json_topic
        self.enable_visualization = self.get_parameter("enable_visualization").get_parameter_value().bool_value
        self.visualization_window_name = self.get_parameter("visualization_window_name").get_parameter_value().string_value
        self.visualization_mask_alpha = self.get_parameter("visualization_mask_alpha").get_parameter_value().double_value
        self.visualization_scale = self.get_parameter("visualization_scale").get_parameter_value().double_value
        self.latest_only_mode = self.get_parameter("latest_only_mode").get_parameter_value().bool_value
        input_queue_depth = self.get_parameter("input_queue_depth").get_parameter_value().integer_value
        input_queue_depth = max(1, int(input_queue_depth))
        self.input_queue_depth = 1 if self.latest_only_mode else input_queue_depth
        
        target_labels = self.get_parameter("target_labels").get_parameter_value().string_array_value
        self.target_labels = [label.strip() for label in target_labels if label.strip()]
        self.target_labels_lc = {label.lower() for label in self.target_labels}
        
        self.conf_threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.iou_threshold = self.get_parameter("iou_threshold").get_parameter_value().double_value
        self.use_tracker = self.get_parameter("use_tracker").get_parameter_value().bool_value
        self.tracker_config = self.get_parameter("tracker_config").get_parameter_value().string_value

        if not self.target_labels:
            self.target_labels = ["cup"]
            self.target_labels_lc = {"cup"}

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.input_queue_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ros handlers
        self.subscription = self.create_subscription(Image, input_topic, self.image_callback, input_qos)
        self.mask_publisher_ = self.create_publisher(Image, output_topic, 10)
        self.json_publisher_ = self.create_publisher(String, json_topic, 10)
        self.bridge = CvBridge()
        self.input_seen = False
        self.input_probe_sub = self.create_subscription(Image, input_topic, self._input_probe_cb, input_qos)

        # debug stuff
        self.frame_count = 0
        self.mask_publish_count = 0
        self.json_publish_count = 0
        self.error_count = 0
        self.last_frame_stamp = None
        self.last_result_count = 0
        self.no_target_count = 0
        self.visualization_available = self.enable_visualization
        self.debug_timer = self.create_timer(5.0, self._debug_status_timer_cb)

        self.get_logger().info(
            "Startup config | "
            f"input={self.input_topic}, mask_out={self.output_topic}, json_out={self.json_topic}, "
            f"conf={self.conf_threshold}, iou={self.iou_threshold}, use_tracker={self.use_tracker}, "
            f"tracker={self.tracker_config}, labels={self.target_labels}, "
            f"latest_only={self.latest_only_mode}, input_queue_depth={self.input_queue_depth}, "
            f"vis={self.enable_visualization}, vis_window={self.visualization_window_name}, "
            f"vis_alpha={self.visualization_mask_alpha}, vis_scale={self.visualization_scale}"
        )

        # yoloe setup
        self.get_logger().info("Loading YOLOE segmentation model...")
        self.model = self._load_model(model_path)
        self._configure_text_prompts()

        self.get_logger().info(
            f"YOLOE initialized. Target labels: {self.target_labels}. "
            f"Mask topic: {output_topic} | JSON topic: {json_topic}"
        )

        if self.enable_visualization:
            self.get_logger().info(f"Visualization enabled. Window: {self.visualization_window_name}")

    def _debug_status_timer_cb(self):
        self.get_logger().info(
            "Heartbeat | "
            f"input_seen={self.input_seen}, frames={self.frame_count}, last_results={self.last_result_count}, "
            f"no_target={self.no_target_count}, "
            f"mask_published={self.mask_publish_count}, json_published={self.json_publish_count}, "
            f"errors={self.error_count}"
        )

    def _input_probe_cb(self, msg):
        if self.input_seen:
            return
        self.input_seen = True
        self.get_logger().info(
            f"First input received on {self.input_topic} at "
            f"{int(msg.header.stamp.sec)}.{int(msg.header.stamp.nanosec):09d}"
        )
        if self.input_probe_sub is not None:
            self.destroy_subscription(self.input_probe_sub)
            self.input_probe_sub = None

    def _load_model(self, model_path):
        """Loads the YOLOE model from the specified path.

        Args:
            model_path (str): The raw string path to the model weights.

        Returns:
            ultralytics.YOLOE: The loaded model instance.
            
        Raises:
            RuntimeError: If YOLOE is not available in the installed ultralytics package.
        """
        if YOLOE is None:
            raise RuntimeError(
                "ultralytics.YOLOE is not available in the installed Ultralytics version. "
                "Please install a version that includes YOLOE support."
            )

        resolved_model_path = self._resolve_model_path(model_path)
        self.get_logger().info(f"Using ultralytics.YOLOE with pretrained model: {resolved_model_path}")
        return YOLOE(resolved_model_path)

    def _resolve_model_path(self, model_path):
        """Resolves the absolute path to the model weights.

        Args:
            model_path (str): The initial path provided via ROS parameters.

        Returns:
            str: The fully resolved and verified absolute path.
            
        Raises:
            FileNotFoundError: If the model weights cannot be found.
        """
        raw_path = Path(model_path).expanduser()

        if raw_path.exists():
            return str(raw_path)

        module_dir = Path(__file__).resolve().parent
        package_relative = module_dir / raw_path
        if package_relative.exists():
            return str(package_relative)

        cache_candidate = Path.home() / ".cache" / "ultralytics" / raw_path.name
        if cache_candidate.exists():
            return str(cache_candidate)

        raise FileNotFoundError(f"YOLOE model weights not found: '{model_path}'.")

    def _configure_text_prompts(self):
        """Configures the model's text prompts if supported by the architecture."""
        if hasattr(self.model, "set_classes"):
            self.model.set_classes(self.target_labels)
            self.get_logger().info(f"Configured prompt classes: {self.target_labels}")
        else:
            self.get_logger().warn("Model does not expose set_classes(). Text-prompt configuration skipped.")

    def _predict(self, frame):
        """Executes the tracking or prediction inference on a single frame.

        Args:
            frame (numpy.ndarray): The BGR image frame from the camera.

        Returns:
            list: A list of ultralytics Result objects.
        """
        if self.use_tracker:
            return self.model.track(
                source=frame,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                tracker=self.tracker_config,
                verbose=False,
            )

        return self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

    def _class_name(self, names, class_id):
        """Retrieves the string label for a given class ID.

        Args:
            names (dict or list): The class names mapping from the model.
            class_id (int): The numerical ID of the detected class.

        Returns:
            str: The string representation of the class.
        """
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _label_matches(self, detected_label):
        """Checks if a detected label is in the configured target list.

        Args:
            detected_label (str): The label detected by the model.

        Returns:
            bool: True if the label matches a target, False otherwise.
        """
        return detected_label.lower() in self.target_labels_lc

    def _largest_target_index(self, result):
        if result is None or result.boxes is None or result.boxes.xyxy is None or result.boxes.cls is None:
            return None

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        best_idx = None
        best_area = -1.0
        for idx, (box, cls_id) in enumerate(zip(boxes_xyxy, class_ids)):
            label = self._class_name(result.names, int(cls_id))
            if not self._label_matches(label):
                continue

            x1, y1, x2, y2 = box
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            if area > best_area:
                best_area = area
                best_idx = idx

        return best_idx

    def _build_segmentation_mask(self, result, height, width, selected_index=None):
        """Generates a binary segmentation mask for targeted objects.

        Args:
            result (ultralytics.engine.results.Results): The inference result.
            height (int): The height of the original frame.
            width (int): The width of the original frame.

        Returns:
            numpy.ndarray: A mono8 OpenCV image array representing the mask.
        """
        mask_image = np.zeros((height, width), dtype=np.uint8)

        if (
            result is None
            or selected_index is None
            or result.masks is None
            or result.boxes is None
            or result.boxes.cls is None
        ):
            return mask_image

        masks_data = result.masks.data
        class_ids = result.boxes.cls
        names = result.names

        detection_count = min(len(masks_data), len(class_ids))
        if selected_index >= detection_count:
            return mask_image

        label = self._class_name(names, int(class_ids[selected_index]))
        if not self._label_matches(label):
            return mask_image

        instance_mask = masks_data[selected_index]
        if hasattr(instance_mask, "cpu"):
            instance_mask = instance_mask.cpu().numpy()
        else:
            instance_mask = np.asarray(instance_mask)

        if instance_mask.shape[:2] != (height, width):
            instance_mask = cv2.resize(instance_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        mask_image[instance_mask > 0.5] = 255

        return mask_image

    def _build_json_message(self, result, selected_index=None):
        """Constructs a JSON string message containing bounding box detections.

        Args:
            result (ultralytics.engine.results.Results): The inference result containing bounding boxes.

        Returns:
            std_msgs.msg.String: The ROS 2 String message containing the JSON payload.
        """
        bbox_payload = None
        if (
            result is not None
            and selected_index is not None
            and result.boxes is not None
            and result.boxes.xyxy is not None
        ):
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            if selected_index < len(boxes_xyxy):
                label = self._class_name(result.names, int(class_ids[selected_index]))
                if self._label_matches(label):
                    x1, y1, x2, y2 = map(int, boxes_xyxy[selected_index])
                    bbox_payload = [x1, y1, x2, y2]

        msg = String()
        msg.data = json.dumps({"bbox": bbox_payload})
        return msg

    def _overlay_mask(self, image_bgr, mask_mono):
        if mask_mono is None:
            return image_bgr

        mask_bool = mask_mono > 0
        if not np.any(mask_bool):
            return image_bgr

        overlay = image_bgr.copy()
        overlay[mask_bool] = (0, 255, 0)

        alpha = float(self.visualization_mask_alpha)
        alpha = max(0.0, min(1.0, alpha))
        blended = cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0.0)
        return blended

    def _draw_detection_box(self, image_bgr, json_payload):
        bbox = json_payload.get("bbox", None)
        if bbox is None:
            return

        x1, y1, x2, y2 = bbox
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(image_bgr, "BIGGEST", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    def _publish_visualization(self, frame, mask_image, json_payload):
        if not self.visualization_available:
            return

        try:
            vis = frame.copy()
            vis = self._overlay_mask(vis, mask_image)
            self._draw_detection_box(vis, json_payload)
            cv2.putText(vis, "SINGLE", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if self.visualization_scale > 0 and self.visualization_scale != 1.0:
                vis = cv2.resize(
                    vis,
                    None,
                    fx=self.visualization_scale,
                    fy=self.visualization_scale,
                    interpolation=cv2.INTER_LINEAR,
                )

            cv2.imshow(self.visualization_window_name, vis)
            cv2.waitKey(1)
        except Exception as exc:
            self.visualization_available = False
            self.get_logger().warn(
                f"Visualization disabled due to OpenCV display error: {exc}",
                throttle_duration_sec=5.0,
            )

    def image_callback(self, msg):
        """Callback function to process incoming camera frames and publish results.

        Args:
            msg (sensor_msgs.msg.Image): The incoming ROS image message.
        """
        try:
            self.frame_count += 1
            self.last_frame_stamp = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))

            if self.frame_count == 1 or self.frame_count % 30 == 0:
                self.get_logger().info(
                    f"Received frame #{self.frame_count} stamp={self.last_frame_stamp[0]}.{self.last_frame_stamp[1]:09d}"
                )

            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            results = self._predict(frame)
            self.last_result_count = len(results) if results else 0
            result = results[0] if results else None
            height, width = frame.shape[:2]

            selected_index = self._largest_target_index(result)
            if selected_index is None:
                self.no_target_count += 1
                self.get_logger().warn(
                    f"No matching target detections on frame #{self.frame_count}. Publishing zero mask and null bbox.",
                    throttle_duration_sec=2.0,
                )
            
            # segmentation mask
            mask_image = self._build_segmentation_mask(result, height, width, selected_index)
            mask_msg = self.bridge.cv2_to_imgmsg(mask_image, encoding="mono8")
            mask_msg.header = msg.header
            self.mask_publisher_.publish(mask_msg)
            self.mask_publish_count += 1
            
            # json message
            json_msg = self._build_json_message(result, selected_index)
            self.json_publisher_.publish(json_msg)
            self.json_publish_count += 1

            if self.enable_visualization:
                try:
                    json_payload = json.loads(json_msg.data)
                except Exception:
                    json_payload = {"bbox": None}
                self._publish_visualization(frame, mask_image, json_payload)

            if self.frame_count % 30 == 0:
                try:
                    bbox_state = "set" if json.loads(json_msg.data).get("bbox") is not None else "null"
                except Exception:
                    bbox_state = "unknown"
                self.get_logger().info(
                    f"Frame #{self.frame_count} processed | bbox={bbox_state}, "
                    f"mask_published={self.mask_publish_count}, json_published={self.json_publish_count}"
                )

        except Exception as exc:
            self.error_count += 1
            self.get_logger().error(
                f"Error processing frame #{self.frame_count} (stamp={self.last_frame_stamp}): {exc}\n"
                f"{traceback.format_exc()}"
            )

    def destroy_node(self):
        if self.enable_visualization:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None):
    """Main execution entry point for the ROS 2 node."""
    rclpy.init(args=args)
    node = YoloESegmentationTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()