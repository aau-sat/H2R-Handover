#!/usr/bin/env python3

import math
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visual_properties.msg import Dimensions


class DimensionsTopicVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("dimensions_topic_visualizer")

        self.declare_parameter("topic", "/estimated_dimensions")
        self.declare_parameter("window_name", "Estimated Dimensions Mask")
        self.declare_parameter("print_period_s", 0.5)

        self.topic = self.get_parameter("topic").get_parameter_value().string_value
        self.window_name = self.get_parameter("window_name").get_parameter_value().string_value
        self.print_period_s = float(
            self.get_parameter("print_period_s").get_parameter_value().double_value
        )

        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_dims: Optional[Tuple[float, float, float, float, float, float]] = None
        self._latest_label: str = "unknown"
        self._latest_mask: Optional[np.ndarray] = None
        self._last_print_ns = 0

        self.sub = self.create_subscription(
            Dimensions,
            self.topic,
            self._dimensions_callback,
            qos_profile,
        )

        self.timer = self.create_timer(0.05, self._render)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info(f"Listening on topic: {self.topic}")

    def _dimensions_callback(self, msg: Dimensions) -> None:
        dims = (
            float(msg.height),
            float(msg.length),
            float(msg.width),
            float(msg.depth),
            float(msg.top_w),
            float(msg.bot_w),
        )

        label = msg.segmentation_label if msg.segmentation_label else "unknown"

        mask_img: Optional[np.ndarray] = None
        try:
            if msg.segmentation_mask.height > 0 and msg.segmentation_mask.width > 0:
                mask_img = self.bridge.imgmsg_to_cv2(msg.segmentation_mask, desired_encoding="mono8")
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode segmentation mask: {exc}")

        with self._lock:
            self._latest_dims = dims
            self._latest_label = label
            self._latest_mask = mask_img

    def _render(self) -> None:
        with self._lock:
            dims = self._latest_dims
            label = self._latest_label
            mask_img = None if self._latest_mask is None else self._latest_mask.copy()

        if dims is None:
            self._draw_waiting()
            return

        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._last_print_ns) >= int(self.print_period_s * 1e9):
            self._last_print_ns = now_ns
            self._print_dims(dims, label)

        self._draw_mask(mask_img, dims, label)

    def _draw_waiting(self) -> None:
        frame = np.zeros((240, 480, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Waiting for /estimated_dimensions...",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def _draw_mask(
        self,
        mask_img: Optional[np.ndarray],
        dims: Tuple[float, float, float, float, float, float],
        label: str,
    ) -> None:
        if mask_img is None:
            frame = np.zeros((240, 480, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Label: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "No segmentation mask in message",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
            return

        if mask_img.dtype != np.uint8:
            mask_img = mask_img.astype(np.uint8)

        if np.max(mask_img) <= 1:
            mask_img = mask_img * 255

        color_mask = cv2.applyColorMap(mask_img, cv2.COLORMAP_TURBO)

        h, w = color_mask.shape[:2]
        footer_h = 70
        canvas = np.zeros((h + footer_h, w, 3), dtype=np.uint8)
        canvas[:h] = color_mask

        cv2.putText(
            canvas,
            f"label={label}",
            (10, h + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        dim_text = (
            f"h={self._fmt(dims[0])}  l={self._fmt(dims[1])}  w={self._fmt(dims[2])}  "
            f"d={self._fmt(dims[3])}  top={self._fmt(dims[4])}  bot={self._fmt(dims[5])}"
        )
        cv2.putText(
            canvas,
            dim_text,
            (10, h + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 255, 120),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def _print_dims(self, dims: Tuple[float, float, float, float, float, float], label: str) -> None:
        self.get_logger().info(
            "Dimensions message | "
            f"label={label} "
            f"height={self._fmt(dims[0])}m "
            f"length={self._fmt(dims[1])}m "
            f"width={self._fmt(dims[2])}m "
            f"depth={self._fmt(dims[3])}m "
            f"top_w={self._fmt(dims[4])}m "
            f"bot_w={self._fmt(dims[5])}m"
        )

    def _fmt(self, value: float) -> str:
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.4f}"

    def destroy_node(self) -> bool:
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DimensionsTopicVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
