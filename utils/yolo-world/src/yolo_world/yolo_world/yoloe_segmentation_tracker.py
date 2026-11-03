import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLOE
except ImportError:
    YOLOE = None


class YoloESegmentationTracker(Node):
    def __init__(self):
        super().__init__("yoloe_segmentation_tracker")

        # --- CONFIGURATION ---
        self.declare_parameter("input_topic", "/camera/left/color/image_raw")
        self.declare_parameter("output_topic", "/segmentation/left")
        self.declare_parameter("model_path", "/home/student-team/rgmc-2026/utils/yolo-world/src/yolo_world/resource/yoloe-11l-seg.pt")
        self.declare_parameter("target_labels", ["wine glass", "cup"])
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.5)
        self.declare_parameter("use_tracker", True)
        self.declare_parameter("tracker_config", "bytetrack.yaml")

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
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

        # --- ROS HANDLERS ---
        self.subscription = self.create_subscription(Image, input_topic, self.image_callback, 10)
        self.publisher_ = self.create_publisher(Image, output_topic, 10)
        self.bridge = CvBridge()

        # --- YOLOE SETUP ---
        self.get_logger().info("Loading YOLOE segmentation model...")
        self.model = self._load_model(model_path)
        self._configure_text_prompts()

        self.get_logger().info(
            "YOLOE segmentation tracker initialized. "
            f"Target labels: {self.target_labels}. Output topic: {output_topic}"
        )

    def _load_model(self, model_path):
        if YOLOE is None:
            raise RuntimeError(
                "ultralytics.YOLOE is not available in the installed Ultralytics version. "
                "Please install a version that includes YOLOE support."
            )

        resolved_model_path = self._resolve_model_path(model_path)
        self.get_logger().info(f"Using ultralytics.YOLOE with pretrained model: {resolved_model_path}")
        return YOLOE(resolved_model_path)

    def _resolve_model_path(self, model_path):
        raw_path = Path(model_path).expanduser()

        # Accept absolute paths and paths relative to current working directory.
        if raw_path.exists():
            return str(raw_path)

        # Try a path relative to this package module for convenience.
        module_dir = Path(__file__).resolve().parent
        package_relative = module_dir / raw_path
        if package_relative.exists():
            return str(package_relative)

        # Try Ultralytics cache location where downloaded weights are usually stored.
        cache_candidate = Path.home() / ".cache" / "ultralytics" / raw_path.name
        if cache_candidate.exists():
            return str(cache_candidate)

        raise FileNotFoundError(
            f"YOLOE model weights not found: '{model_path}'. "
            "Download the weights file and pass its absolute path via model_path, for example: "
            "ros2 run yolo_world yoloe_segmentation_tracker --ros-args "
            "-p model_path:=/absolute/path/to/yoloe-26l-seg.pt"
        )

    def _configure_text_prompts(self):
        if hasattr(self.model, "set_classes"):
            self.model.set_classes(self.target_labels)
            self.get_logger().info(f"Configured prompt classes: {self.target_labels}")
        else:
            self.get_logger().warn(
                "Model does not expose set_classes(). Text-prompt configuration skipped."
            )

    def _predict(self, frame):
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
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _label_matches(self, detected_label):
        return detected_label.lower() in self.target_labels_lc

    def _build_segmentation_mask(self, result, height, width):
        mask_image = np.zeros((height, width), dtype=np.uint8)

        if result.masks is None or result.boxes is None or result.boxes.cls is None:
            return mask_image

        masks_data = result.masks.data
        class_ids = result.boxes.cls
        names = result.names

        detection_count = min(len(masks_data), len(class_ids))
        for i in range(detection_count):
            class_id = int(class_ids[i])
            label = self._class_name(names, class_id)
            if not self._label_matches(label):
                continue

            instance_mask = masks_data[i]
            if hasattr(instance_mask, "cpu"):
                instance_mask = instance_mask.cpu().numpy()
            else:
                instance_mask = np.asarray(instance_mask)

            if instance_mask.shape[:2] != (height, width):
                instance_mask = cv2.resize(instance_mask, (width, height), interpolation=cv2.INTER_NEAREST)

            mask_image[instance_mask > 0.5] = 255

        return mask_image

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            results = self._predict(frame)

            if not results:
                return

            result = results[0]
            height, width = frame.shape[:2]
            mask_image = self._build_segmentation_mask(result, height, width)

            out_msg = self.bridge.cv2_to_imgmsg(mask_image, encoding="mono8")
            out_msg.header = msg.header
            self.publisher_.publish(out_msg)

        except Exception as exc:
            self.get_logger().error(f"Error processing frame: {exc}")


def main(args=None):
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