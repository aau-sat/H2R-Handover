import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import numpy as np
import json
import traceback
from pathlib import Path
from typing import Dict, Optional
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

try:
    from ultralytics import YOLOE
except ImportError:
    YOLOE = None

try:
    from utils.config import symmetric_container_priors, asymmetric_container_priors, other_priors, object_priors, left_color_camera_topic_suffix, right_color_camera_topic_suffix
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path("/home/student-team/rgmc-2026/")))
    from utils.config import symmetric_container_priors, asymmetric_container_priors, other_priors, object_priors, left_color_camera_topic_suffix, right_color_camera_topic_suffix

class DoubleTrackerYoloNode(Node):
    """Synchronizes two camera streams and runs YOLOE segmentation/tracking on both."""

    def __init__(self):
        super().__init__("double_tracker_yolo_node")

        # --- CONFIGURATION ---
        self.declare_parameter("left_input_topic", "/camera/left/color/" + left_color_camera_topic_suffix)
        self.declare_parameter("right_input_topic", "/camera/right/color/" + right_color_camera_topic_suffix)
        self.declare_parameter("left_output_topic", "/segmentation/left")
        self.declare_parameter("right_output_topic", "/segmentation/right")
        self.declare_parameter("json_topic", "/object_detections")
        # self.declare_parameter("label_output_topic", "/detected_label")
        self.declare_parameter("tracking_strategy", "largest")  # Options: "largest" (default), "id", "best_confidence"
        self.declare_parameter("label_publish_only_if_subscriber", False)
        self.declare_parameter('handover_detected_topic', '/handover_detected')
        # Center ROI for initial object detection (before target_id is set).
        # Format: [center_x, center_y, width, height] as normalized fractions [0.0-1.0] or absolute pixels.
        # If only [width, height] is provided, the window is centered in the image.
        # If empty, no masking is applied.
        self.declare_parameter("left_crop_roi", [])  # Default to centered central window
        self.declare_parameter("right_crop_roi", [])  # Default to centered central window
        self.declare_parameter("sync_queue_size", 20)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("target_labels", object_priors)
        
        self.declare_parameter("model_name", "yoloe-26l-seg.pt")
        self.declare_parameter("confidence_threshold", 0.40)
        self.declare_parameter("iou_threshold", 0.5)
        self.declare_parameter("use_tracker", True)
        self.declare_parameter("tracker_config", "bytetrack.yaml")
        self.declare_parameter("latest_only_mode", True)
        self.declare_parameter("input_queue_depth", 1)
        self.declare_parameter("enable_visualization", True)
        self.declare_parameter("visualization_window_name", "YOLOE Dual Tracker")
        self.declare_parameter("visualization_mask_alpha", 0.45)
        self.declare_parameter("visualization_scale", 1.0)
        
        # Per-camera tracking state for ID-based selection strategy.
        self.tracked_target_id: Dict[str, Optional[int]] = {"left": None, "right": None}
        self.tracked_target_bbox: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        # Track if model has been reinitialized based on first detection
        self.model_reinitialized_based_on_label = False
        self.detected_object_category = None  # Will store one of: 'symmetric', 'asymmetric', 'other'

        left_input_topic = self.get_parameter("left_input_topic").get_parameter_value().string_value
        right_input_topic = self.get_parameter("right_input_topic").get_parameter_value().string_value
        left_output_topic = self.get_parameter("left_output_topic").get_parameter_value().string_value
        right_output_topic = self.get_parameter("right_output_topic").get_parameter_value().string_value
        json_topic = self.get_parameter("json_topic").get_parameter_value().string_value
        # label_output_topic = self.get_parameter("label_output_topic").get_parameter_value().string_value
        self.left_input_topic = left_input_topic
        self.right_input_topic = right_input_topic
        self.left_output_topic = left_output_topic
        self.right_output_topic = right_output_topic
        self.json_topic = json_topic
        # self.label_output_topic = label_output_topic
        # self.label_publish_only_if_subscriber = self.get_parameter(
        #     "label_publish_only_if_subscriber"
        # ).get_parameter_value().bool_value
        self.handover_detected_topic = self.get_parameter('handover_detected_topic').get_parameter_value().string_value
        self.tracking_strategy = self.get_parameter("tracking_strategy").get_parameter_value().string_value.strip().lower()
        left_roi = self.get_parameter("left_crop_roi").get_parameter_value().double_array_value
        right_roi = self.get_parameter("right_crop_roi").get_parameter_value().double_array_value
        self.left_crop_roi = list(left_roi) if left_roi else []
        self.right_crop_roi = list(right_roi) if right_roi else []
        self.enable_visualization = self.get_parameter("enable_visualization").get_parameter_value().bool_value
        self.visualization_window_name = self.get_parameter("visualization_window_name").get_parameter_value().string_value
        self.visualization_mask_alpha = self.get_parameter("visualization_mask_alpha").get_parameter_value().double_value
        self.visualization_scale = self.get_parameter("visualization_scale").get_parameter_value().double_value
        self.latest_only_mode = self.get_parameter("latest_only_mode").get_parameter_value().bool_value
        input_queue_depth = self.get_parameter("input_queue_depth").get_parameter_value().integer_value
        input_queue_depth = max(1, int(input_queue_depth))
        self.input_queue_depth = 1 if self.latest_only_mode else input_queue_depth
        sync_queue_size = self.get_parameter("sync_queue_size").get_parameter_value().integer_value
        sync_slop = self.get_parameter("sync_slop").get_parameter_value().double_value
        self.sync_slop = sync_slop
        self.sync_queue_size_effective = 1 if self.latest_only_mode else max(1, int(sync_queue_size))
        self.model_name = self.get_parameter("model_name").get_parameter_value().string_value

        target_labels = self.get_parameter("target_labels").get_parameter_value().string_array_value
        self.target_labels = [label.strip() for label in target_labels if label.strip()]
        self.conf_threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.iou_threshold = self.get_parameter("iou_threshold").get_parameter_value().double_value
        self.use_tracker = self.get_parameter("use_tracker").get_parameter_value().bool_value
        self.tracker_config = self.get_parameter("tracker_config").get_parameter_value().string_value
        self.label_chosen = False  # To track if we've published a label at least once

        if not self.target_labels:
            self.target_labels = ["cup"]

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.input_queue_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # output_qos = QoSProfile(
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=1,
        #     reliability=ReliabilityPolicy.RELIABLE,
        #     durability=DurabilityPolicy.VOLATILE,
        # )

        # ros config
        self.bridge = CvBridge()
        self.left_mask_pub = self.create_publisher(Image, left_output_topic, 10) #output_qos)
        self.right_mask_pub = self.create_publisher(Image, right_output_topic, 10) #output_qos)
        self.json_pub = self.create_publisher(String, json_topic, 10) #output_qos)
        # self.label_pub = self.create_publisher(String, label_output_topic, 10)

        self.subleft = Subscriber(self, Image, left_input_topic, qos_profile=input_qos)
        self.subright = Subscriber(self, Image, right_input_topic, qos_profile=input_qos)
        self.ts = ApproximateTimeSynchronizer(
            [self.subleft, self.subright],
            queue_size=self.sync_queue_size_effective,
            slop=sync_slop,
        )
        self.ts.registerCallback(self.image_callback)

        # handover_detected stuff
        self.sub_handover_detected = self.create_subscription(Bool, self.handover_detected_topic, self.callback_handover_detected, 10)

        self.is_handover_detected = False # when this is true, no more publish the bounding boxes

        # check for incoming data to confirm topics are correct
        # self.left_input_seen = False
        # self.right_input_seen = False
        # self.left_probe_sub = self.create_subscription(Image, left_input_topic, self._left_input_probe_cb, input_qos)
        # self.right_probe_sub = self.create_subscription(Image, right_input_topic, self._right_input_probe_cb, input_qos)

        # debug stuff
        self.synced_pair_count = 0
        self.left_mask_publish_count = 0
        self.right_mask_publish_count = 0
        self.json_publish_count = 0
        # self.label_publish_count = 0
        self.error_count = 0
        self.left_only_result_count = 0
        self.right_only_result_count = 0
        self.both_result_count = 0
        self.none_result_count = 0
        self.last_sync_delta_ms = None
        self.debug_timer = self.create_timer(5.0, self._debug_status_timer_cb)

        self.get_logger().info(
            "Startup config | "
            f"left_in={self.left_input_topic}, right_in={self.right_input_topic}, "
            f"left_out={self.left_output_topic}, right_out={self.right_output_topic}, "
            # f"json_out={self.json_topic}, label_out={self.label_output_topic}, "
            f"sync_queue={self.sync_queue_size_effective}, sync_slop={self.sync_slop}, conf={self.conf_threshold}, "
            f"iou={self.iou_threshold}, use_tracker={self.use_tracker}, tracker={self.tracker_config}, "
            f"latest_only={self.latest_only_mode}, input_queue_depth={self.input_queue_depth}, "
            f"labels={self.target_labels}, vis={self.enable_visualization}, "
            f"vis_window={self.visualization_window_name}, vis_alpha={self.visualization_mask_alpha}, "
            f"vis_scale={self.visualization_scale}, "
            # f"label_publish_only_if_subscriber={self.label_publish_only_if_subscriber}, "
            f"target labels={self.target_labels}"
        )

        self.visualization_available = self.enable_visualization

        # model setup
        self.get_logger().info("Loading YOLOE segmentation model(s)...")
        self.model_left = self._load_model(self.model_name)
        # Keep tracker states independent between camera streams.
        self.model_right = self._load_model(self.model_name)
        self._configure_text_prompts(self.model_left)
        self._configure_text_prompts(self.model_right)

        self.get_logger().info(
            f"Dual YOLOE tracker initialized. Labels: {self.target_labels}. "
            f"Left mask: {left_output_topic} | Right mask: {right_output_topic} | "
            # f"JSON: {json_topic} | Label: {label_output_topic}"
        )

        if self.enable_visualization:
            self.get_logger().info(
                f"Visualization enabled. Window: {self.visualization_window_name}"
            )

    def _debug_status_timer_cb(self):
        self.get_logger().debug(
            "Heartbeat | "
            # f"left_in_seen={self.left_input_seen}, right_in_seen={self.right_input_seen}, "
            f"synced_pairs={self.synced_pair_count}, sync_delta_ms={self.last_sync_delta_ms}, "
            f"both={self.both_result_count}, left_only={self.left_only_result_count}, "
            f"right_only={self.right_only_result_count}, none={self.none_result_count}, "
            f"left_mask_pub={self.left_mask_publish_count}, right_mask_pub={self.right_mask_publish_count}, "
            # f"json_pub={self.json_publish_count}, label_pub={self.label_publish_count}, "
            f"errors={self.error_count}"
        )
        
    def _load_model(self, model_path):
        if YOLOE is None:
            raise RuntimeError(
                "ultralytics.YOLOE is not available in the installed Ultralytics version. "
                "Please install a version that includes YOLOE support."
            )
        return YOLOE(model_path)

    def _resolve_model_path(self, model_path):
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

    def _configure_text_prompts(self, model):
        if hasattr(model, "set_classes"):
            model.set_classes(self.target_labels)
        else:
            self.get_logger().warn("Model does not expose set_classes(). Text prompt configuration skipped.")

    def _categorize_label(self, label):
        """Categorize a detected label into one of the prior categories.
        Returns: 'symmetric', 'asymmetric', 'other', or None if not found."""
        label_lower = str(label).lower().strip()
        
        for tag in symmetric_container_priors:
            if tag.lower() in label_lower:
                return 'symmetric'
        
        for tag in asymmetric_container_priors:
            if tag.lower() in label_lower:
                return 'asymmetric'
        
        for tag in other_priors:
            if tag.lower() in label_lower:
                return 'other'
        
        return None

    def _reinitialize_model_for_category(self, category):
        """Reinitialize both models with the appropriate prior set based on category."""
        if category == 'symmetric':
            new_labels = symmetric_container_priors
        elif category == 'asymmetric':
            new_labels = asymmetric_container_priors
        elif category == 'other':
            new_labels = other_priors
        else:
            return  # Unknown category, skip reinitialization
        
        self.target_labels = new_labels
        self.detected_object_category = category
        
        self.get_logger().info(
            f"Reinitializing YOLOE models with {category} container priors: {new_labels}"
        )
        
        # Update both models with the new label set
        self._configure_text_prompts(self.model_left)
        self._configure_text_prompts(self.model_right)
        
        self.model_reinitialized_based_on_label = True

    def _predict(self, model, frame):
        if self.use_tracker:
            return model.track(
                source=frame,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                tracker=self.tracker_config,
                verbose=False,
            )

        return model.predict(
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

    def _largest_target_index(self, result):
        if result.boxes is None or result.boxes.xyxy is None or result.boxes.cls is None:
            return None

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        best_idx = None
        best_area = -1.0
        for idx, (box, cls_id) in enumerate(zip(boxes_xyxy, class_ids)):
            x1, y1, x2, y2 = box
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            if area > best_area:
                best_area = area
                best_idx = idx

        return best_idx

    def _best_confidence_target(self, result, camera_side):
        if (
            result is None
            or result.boxes is None
            or result.boxes.cls is None
            or result.boxes.conf is None
        ):
            return None

        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        if len(class_ids) == 0 or len(confidences) == 0:
            return None

        best_item = None
        best_conf = -1.0
        for idx, (cls_id, conf) in enumerate(zip(class_ids, confidences)):
            label = self._class_name(result.names, int(cls_id))

            conf = float(conf)
            if conf > best_conf:
                best_conf = conf
                best_item = {
                    "camera": camera_side,
                    "index": idx,
                    "class_name": label,
                    "confidence": conf,
                }

        return best_item
    
    def _apply_roi_with_padding(self, frame, roi_spec, pad_value=0):
        """
        Apply a centered ROI by creating a padded frame: full image size with outer area filled with pad_value.
        ROI format: [center_x, center_y, width, height] where values are normalized [0.0-1.0] or absolute pixels.
        If only [width, height] is provided, the ROI is centered in the image.
        Returns: padded_frame (same size as input)
        """
        if not roi_spec:
            return frame.copy()

        h, w = frame.shape[:2]
        if len(roi_spec) >= 4:
            center_x, center_y, roi_w, roi_h = roi_spec[:4]
            if all(0 <= v <= 1 for v in roi_spec[:4]):
                center_x = int(center_x * w)
                center_y = int(center_y * h)
                roi_w = int(roi_w * w)
                roi_h = int(roi_h * h)
            else:
                center_x, center_y, roi_w, roi_h = int(center_x), int(center_y), int(roi_w), int(roi_h)
        elif len(roi_spec) >= 2:
            roi_w, roi_h = roi_spec[:2]
            if all(0 <= v <= 1 for v in roi_spec[:2]):
                roi_w = int(roi_w * w)
                roi_h = int(roi_h * h)
            else:
                roi_w, roi_h = int(roi_w), int(roi_h)
            center_x = w // 2
            center_y = h // 2
        else:
            return frame.copy()

        roi_w = max(1, min(int(roi_w), w))
        roi_h = max(1, min(int(roi_h), h))
        center_x = max(0, min(int(center_x), w - 1))
        center_y = max(0, min(int(center_y), h - 1))

        x1 = max(0, min(center_x - roi_w // 2, w - roi_w))
        y1 = max(0, min(center_y - roi_h // 2, h - roi_h))
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        # Create padded frame with neutral background.
        padded = np.full_like(frame, pad_value)
        padded[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        
        return padded

    def _normalize_track_id(self, track_id):
        if track_id is None:
            return None
        try:
            track_id_value = float(track_id)
        except (TypeError, ValueError):
            return None
        if np.isnan(track_id_value):
            return None
        return int(track_id_value)

    def _id_based_selection(self, result, camera_side):
        if (
            result is None
            or result.boxes is None
            or result.boxes.xyxy is None
            or result.boxes.conf is None
        ):
            return None

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None

        if len(boxes_xyxy) == 0:
            return None

        current_target_id = self.tracked_target_id.get(camera_side)
        previous_bbox = self.tracked_target_bbox.get(camera_side)

        if current_target_id is not None and track_ids is not None:
            for idx, track_id in enumerate(track_ids):
                normalized_track_id = self._normalize_track_id(track_id)
                if normalized_track_id == current_target_id:
                    self.tracked_target_bbox[camera_side] = boxes_xyxy[idx]
                    return idx

        if previous_bbox is not None:
            previous_bbox = np.asarray(previous_bbox, dtype=np.float32)
            prev_center = np.array(
                [
                    0.5 * (previous_bbox[0] + previous_bbox[2]),
                    0.5 * (previous_bbox[1] + previous_bbox[3]),
                ],
                dtype=np.float32,
            )

            best_idx = None
            best_distance = float("inf")
            for idx, box in enumerate(boxes_xyxy):
                box = np.asarray(box, dtype=np.float32)
                box_center = np.array(
                    [0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])],
                    dtype=np.float32,
                )
                distance = float(np.linalg.norm(box_center - prev_center))
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx

            if best_idx is not None:
                new_track_id = None
                if track_ids is not None and best_idx < len(track_ids):
                    new_track_id = self._normalize_track_id(track_ids[best_idx])
                self.tracked_target_id[camera_side] = new_track_id
                self.tracked_target_bbox[camera_side] = boxes_xyxy[best_idx]
                return best_idx

        if class_ids is None:
            return None

        best_idx = None
        best_conf = -1.0
        for idx, (cls_id, conf) in enumerate(zip(class_ids, confidences)):
            label = self._class_name(result.names, int(cls_id))
            conf = float(conf)
            if conf > best_conf:
                best_conf = conf
                best_idx = idx

        if best_idx is None:
            return None

        selected_track_id = None
        if track_ids is not None and best_idx < len(track_ids):
            selected_track_id = self._normalize_track_id(track_ids[best_idx])

        self.tracked_target_id[camera_side] = selected_track_id
        self.tracked_target_bbox[camera_side] = boxes_xyxy[best_idx]
        return best_idx

    def _select_target_index(self, result, camera_side):
        if self.tracking_strategy == "id":
            return self._id_based_selection(result, camera_side)
        elif self.tracking_strategy == "largest":
            return self._largest_target_index(result)
        elif self.tracking_strategy == "best_confidence":
            best_item = self._best_confidence_target(result, camera_side)
            return best_item["index"] if best_item is not None else None
        else:
            self.get_logger().warn(f"Unknown tracking strategy '{self.tracking_strategy}'. Defaulting to 'id' based selection.")
            return self._id_based_selection(result, camera_side)

    def _build_segmentation_mask(self, result, height, width, selected_index=None):
        mask_image = np.zeros((height, width), dtype=np.uint8)

        if (
            selected_index is None
            or result.masks is None
            or result.boxes is None
        ):
            return mask_image

        masks_data = result.masks.data
        class_ids = result.boxes.cls
        names = result.names

        if class_ids is None:
            return mask_image

        detection_count = min(len(masks_data), len(class_ids))
        if selected_index >= detection_count:
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

    def _build_bbox_entries(self, result, camera_side, selected_index=None):
        output_data = []

        if selected_index is None or result.boxes is None or result.boxes.xyxy is None:
            return output_data

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes_xyxy)

        if selected_index >= len(boxes_xyxy):
            return output_data

        box = boxes_xyxy[selected_index]
        cls_id = class_ids[selected_index]
        conf = confidences[selected_index]
        track_id = track_ids[selected_index]

        label = self._class_name(result.names, int(cls_id))

        x1, y1, x2, y2 = map(int, box)
        obj_dict = {
            "camera": camera_side,
            "class_name": label,
            "confidence": float(conf),
            "bbox": [x1, y1, x2, y2],
        }
        if track_id is not None:
            obj_dict["track_id"] = int(track_id)

        output_data.append(obj_dict)

        return output_data

    # def _publish_detected_label(self, label):
    #     if label is None:
    #         return False

    #     if self.label_publish_only_if_subscriber and self.label_pub.get_subscription_count() == 0:
    #         return False

    #     label_msg = String()
    #     label_msg.data = str(label)
    #     self.label_pub.publish(label_msg)
    #     self.label_publish_count += 1
    #     return True

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

    def _draw_detection_box(self, image_bgr, entries):
        if not entries:
            return

        entry = entries[0]
        x1, y1, x2, y2 = entry["bbox"]
        label = entry.get("class_name", "obj")
        conf = entry.get("confidence", 0.0)
        track_id = entry.get("track_id", None)

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 200, 255), 2)

        text = f"{label} {conf:.2f}"
        if track_id is not None:
            text += f" id={track_id}"
        cv2.putText(image_bgr, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    def _publish_visualization(self, frame_left, frame_right, left_mask, right_mask, left_entries, right_entries):
        if not self.visualization_available:
            return

        try:
            left_vis = frame_left.copy()
            right_vis = frame_right.copy()

            left_vis = self._overlay_mask(left_vis, left_mask)
            right_vis = self._overlay_mask(right_vis, right_mask)

            if left_entries is not None and right_entries is not None:
                self._draw_detection_box(left_vis, left_entries)
                self._draw_detection_box(right_vis, right_entries)

            cv2.putText(left_vis, "LEFT", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(right_vis, "RIGHT", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if left_vis.shape[0] != right_vis.shape[0]:
                target_h = min(left_vis.shape[0], right_vis.shape[0])
                left_w = int(left_vis.shape[1] * (target_h / left_vis.shape[0]))
                right_w = int(right_vis.shape[1] * (target_h / right_vis.shape[0]))
                left_vis = cv2.resize(left_vis, (left_w, target_h), interpolation=cv2.INTER_LINEAR)
                right_vis = cv2.resize(right_vis, (right_w, target_h), interpolation=cv2.INTER_LINEAR)

            panel = np.hstack([left_vis, right_vis])

            if self.visualization_scale > 0 and self.visualization_scale != 1.0:
                panel = cv2.resize(
                    panel,
                    None,
                    fx=self.visualization_scale,
                    fy=self.visualization_scale,
                    interpolation=cv2.INTER_LINEAR,
                )

            cv2.imshow(self.visualization_window_name, panel)
            cv2.waitKey(1)

        except Exception as exc:
            self.visualization_available = False
            self.get_logger().warn(
                f"Visualization disabled due to OpenCV display error: {exc}",
                throttle_duration_sec=5.0,
            )

    def callback_handover_detected(self, msg):
        self.get_logger().info(
            f"Handover status: {msg}"
        )
        if msg.data == True:
            self.is_handover_detected = True
            self.sub_handover_detected.destroy()
            self.json_pub.destroy()

    def image_callback(self, msg_left, msg_right):
        try:
            self.synced_pair_count += 1
            left_stamp = msg_left.header.stamp.sec + msg_left.header.stamp.nanosec * 1e-9
            right_stamp = msg_right.header.stamp.sec + msg_right.header.stamp.nanosec * 1e-9
            self.last_sync_delta_ms = round(abs(left_stamp - right_stamp) * 1000.0, 3)

            if self.synced_pair_count == 1 or self.synced_pair_count % 30 == 0:
                self.get_logger().debug(
                    f"Synced pair #{self.synced_pair_count} | dt={self.last_sync_delta_ms} ms "
                    f"(slop={self.sync_slop}s)"
                )

            frame_left = self.bridge.imgmsg_to_cv2(msg_left, desired_encoding="bgr8")
            frame_right = self.bridge.imgmsg_to_cv2(msg_right, desired_encoding="bgr8")

            # Apply ROI padding on first frame (before target_id is initialized).
            frame_left_for_pred = frame_left
            frame_right_for_pred = frame_right

            if self.tracked_target_id["left"] is None and self.left_crop_roi:
                frame_left_for_pred = self._apply_roi_with_padding(frame_left, self.left_crop_roi, pad_value=0)

            if self.tracked_target_id["right"] is None and self.right_crop_roi:
                frame_right_for_pred = self._apply_roi_with_padding(frame_right, self.right_crop_roi, pad_value=0)

            results_left = self._predict(self.model_left, frame_left_for_pred)
            results_right = self._predict(self.model_right, frame_right_for_pred)

            result_left = results_left[0] if results_left else None
            result_right = results_right[0] if results_right else None

            left_selected_idx = self._select_target_index(result_left, "left") if result_left is not None else None
            right_selected_idx = self._select_target_index(result_right, "right") if result_right is not None else None

            left_best = self._best_confidence_target(result_left, "left")
            right_best = self._best_confidence_target(result_right, "right")

            selected_detection = None
                
            if left_best is not None and right_best is not None:
                selected_detection = left_best if left_best["confidence"] >= right_best["confidence"] else right_best
            else:
                selected_detection = left_best if left_best is not None else right_best

            if not self.label_chosen:
                self.label_chosen = selected_detection["class_name"] if selected_detection is not None else None
            
                # Reinitialize model based on first successful detection
                if (not self.model_reinitialized_based_on_label and 
                    self.label_chosen is not None and 
                    selected_detection is not None):
                    category = self._categorize_label(self.label_chosen)
                    if category is not None:
                        self._reinitialize_model_for_category(category)

            # self._publish_detected_label(self.label_chosen) 

            if left_selected_idx is None and right_selected_idx is None:
                self.none_result_count += 1
                self.get_logger().warn(
                    f"No matching target detections on synced pair #{self.synced_pair_count}. Publishing zero masks.",
                    throttle_duration_sec=2.0,
                )
            elif left_selected_idx is not None and right_selected_idx is not None:
                self.both_result_count += 1
            elif left_selected_idx is not None:
                self.left_only_result_count += 1
                self.get_logger().warn(
                    f"Only LEFT produced result on pair #{self.synced_pair_count}. RIGHT mask will be zero.",
                    throttle_duration_sec=2.0,
                )
            else:
                self.right_only_result_count += 1
                self.get_logger().warn(
                    f"Only RIGHT produced result on pair #{self.synced_pair_count}. LEFT mask will be zero.",
                    throttle_duration_sec=2.0,
                )

            left_height, left_width = frame_left.shape[:2]
            if result_left is not None and left_selected_idx is not None:
                left_mask = self._build_segmentation_mask(result_left, left_height, left_width, left_selected_idx)
            else:
                left_mask = np.zeros((left_height, left_width), dtype=np.uint8)

            left_mask_msg = self.bridge.cv2_to_imgmsg(left_mask, encoding="mono8")
            left_mask_msg.header = msg_left.header
            self.left_mask_pub.publish(left_mask_msg)
            self.left_mask_publish_count += 1

            right_height, right_width = frame_right.shape[:2]
            if result_right is not None and right_selected_idx is not None:
                right_mask = self._build_segmentation_mask(result_right, right_height, right_width, right_selected_idx)
            else:
                right_mask = np.zeros((right_height, right_width), dtype=np.uint8)

            right_mask_msg = self.bridge.cv2_to_imgmsg(right_mask, encoding="mono8")
            right_mask_msg.header = msg_right.header
            self.right_mask_pub.publish(right_mask_msg)
            self.right_mask_publish_count += 1

            # bounding box thing
            left_entries = None
            right_entries = None
            combined_payload = None
            if not self.is_handover_detected:
                # self.get_logger().info(
                #     f"handover so far " + str(self.is_handover_detected)
                # )
                left_entries = (
                    self._build_bbox_entries(result_left, "left", left_selected_idx)
                    if result_left is not None
                    else []
                )
                right_entries = (
                    self._build_bbox_entries(result_right, "right", right_selected_idx)
                    if result_right is not None
                    else []
                )

                left_bbox = left_entries[0]["bbox"] if left_entries else None
                right_bbox = right_entries[0]["bbox"] if right_entries else None
                left_class = left_entries[0]["class_name"] if left_entries else None
                right_class = right_entries[0]["class_name"] if right_entries else None

                combined_payload = {
                    "left_bbox": left_bbox,
                    "right_bbox": right_bbox,
                    "left_class": left_class,
                    "right_class": right_class
                }

                json_msg = String()
                json_msg.data = json.dumps(combined_payload)
                self.json_pub.publish(json_msg)
                self.json_publish_count += 1

            if self.enable_visualization:
                self._publish_visualization(
                    frame_left,
                    frame_right,
                    left_mask,
                    right_mask,
                    left_entries,
                    right_entries,
                )

            if self.synced_pair_count % 30 == 0:
                self.get_logger().debug(
                    f"Pair #{self.synced_pair_count} published | "
                    #f"left_bbox={'set' if combined_payload['left_bbox'] is not None else 'null'}, "
                    #f"right_bbox={'set' if combined_payload['right_bbox'] is not None else 'null'}, "
                    f"left_mask_pub={self.left_mask_publish_count}, "
                    f"right_mask_pub={self.right_mask_publish_count}, "
                    # f"json_pub={self.json_publish_count}, label_pub={self.label_publish_count}"
                )

        except Exception as exc:
            self.error_count += 1
            self.get_logger().error(
                f"Error processing synchronized frames at pair #{self.synced_pair_count}: {exc}\n"
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
    rclpy.init(args=args)
    node = DoubleTrackerYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()