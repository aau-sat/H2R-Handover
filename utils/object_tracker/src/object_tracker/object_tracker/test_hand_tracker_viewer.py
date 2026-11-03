import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from hand_detection.msg import HandDetection
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class HandTrackerViewer(Node):
    def __init__(self):
        super().__init__("hand_tracker_viewer")

        self.declare_parameter("hand_topic", "/hand_tracker")
        self.declare_parameter("left_camera_topic", "/camera/left/color/image_raw")
        self.declare_parameter("right_camera_topic", "/camera/right/color/image_raw")
        self.declare_parameter("window_name", "Hand Tracker Viewer")
        self.declare_parameter("display_scale", 0.75)
        self.declare_parameter("refresh_hz", 15.0)

        self.hand_topic = str(self.get_parameter("hand_topic").value)
        self.left_camera_topic = str(self.get_parameter("left_camera_topic").value)
        self.right_camera_topic = str(self.get_parameter("right_camera_topic").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.refresh_hz = float(self.get_parameter("refresh_hz").value)

        sync_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.bridge = CvBridge()

        self.left_image = None
        self.right_image = None
        self.active_view = ""
        self.left_hands = []
        self.right_hands = []
        self.last_hand_msg_time = None

        self.create_subscription(Image, self.left_camera_topic, self._left_image_callback, sync_qos)
        self.create_subscription(Image, self.right_camera_topic, self._right_image_callback, sync_qos)
        self.create_subscription(HandDetection, self.hand_topic, self._hand_callback, 10)

        period = 1.0 / max(self.refresh_hz, 1.0)
        self.timer = self.create_timer(period, self._render)

        self.get_logger().info(
            f"Hand tracker viewer ready: hand_topic={self.hand_topic}, "
            f"left_camera_topic={self.left_camera_topic}, right_camera_topic={self.right_camera_topic}"
        )

    def _left_image_callback(self, msg: Image):
        self.left_image = self._to_bgr(msg)

    def _right_image_callback(self, msg: Image):
        self.right_image = self._to_bgr(msg)

    def _to_bgr(self, msg: Image):
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert image: {exc}")
            return None

    def _hand_callback(self, msg: HandDetection):
        self.active_view = msg.active_view
        self.left_hands = list(msg.left)
        self.right_hands = list(msg.right)
        self.last_hand_msg_time = time.time()

    @staticmethod
    def _draw_hand_centers(image, hands, color):
        for idx, hand in enumerate(hands):
            if not getattr(hand, "detected", False):
                continue

            center = list(getattr(hand, "active_hand_center", []))
            if len(center) < 2:
                continue

            cx, cy = int(center[0]), int(center[1])
            cv2.circle(image, (cx, cy), 8, color, 2, cv2.LINE_AA)
            cv2.putText(
                image,
                f"#{idx}",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _resize_to_height(image, target_height):
        if image is None:
            return None
        if image.shape[0] == target_height:
            return image
        scale = float(target_height) / float(image.shape[0])
        target_width = max(1, int(round(image.shape[1] * scale)))
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _fallback_canvas(title, width=640, height=480):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(canvas, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(canvas, "No camera frame yet", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2, cv2.LINE_AA)
        return canvas

    def _render(self):
        left = self.left_image.copy() if self.left_image is not None else self._fallback_canvas("LEFT")
        right = self.right_image.copy() if self.right_image is not None else self._fallback_canvas("RIGHT")

        left_count = len(self.left_hands)
        right_count = len(self.right_hands)

        self._draw_hand_centers(left, self.left_hands, (0, 255, 0))
        self._draw_hand_centers(right, self.right_hands, (0, 255, 0))

        cv2.putText(left, f"LEFT hands: {left_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, f"RIGHT hands: {right_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

        active_view_text = self.active_view if self.active_view else "none"
        now = time.time()
        if self.last_hand_msg_time is None:
            age_text = "n/a"
        else:
            age_text = f"{(now - self.last_hand_msg_time):.2f}s"

        cv2.putText(left, f"active_view: {active_view_text}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(right, f"active_view: {active_view_text}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(left, f"hand msg age: {age_text}", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 2, cv2.LINE_AA)
        cv2.putText(right, f"hand msg age: {age_text}", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 2, cv2.LINE_AA)

        target_h = min(left.shape[0], right.shape[0])
        left = self._resize_to_height(left, target_h)
        right = self._resize_to_height(right, target_h)

        separator = np.zeros((target_h, 16, 3), dtype=np.uint8)
        composed = np.hstack([left, separator, right])

        if self.display_scale > 0.0 and abs(self.display_scale - 1.0) > 1e-6:
            composed = cv2.resize(
                composed,
                (
                    max(1, int(round(composed.shape[1] * self.display_scale))),
                    max(1, int(round(composed.shape[0] * self.display_scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow(self.window_name, composed)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.get_logger().info("Close requested from OpenCV window.")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
