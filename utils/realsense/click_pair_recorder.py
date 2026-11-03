#!/usr/bin/env python3

import os
import threading
from datetime import datetime

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ClickPairRecorder(Node):
    def __init__(self):
        super().__init__('click_pair_recorder')

        self.left_topic = self.declare_parameter('left_topic', '/camera/left/color/image_raw').value
        self.right_topic = self.declare_parameter('right_topic', '/camera/right/color/image_raw').value
        self.output_dir = self.declare_parameter('output_dir', 'captured_data').value
        self.window_name = self.declare_parameter('window_name', 'Dual Camera Recorder').value

        os.makedirs(self.output_dir, exist_ok=True)

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_left = None
        self._latest_right = None
        self._pair_index = 0
        self._should_save = False
        self._running = True

        qos_depth = 10
        self.left_sub = message_filters.Subscriber(self, Image, self.left_topic)
        self.right_sub = message_filters.Subscriber(self, Image, self.right_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=20,
            slop=0.05,
        )
        self.ts.registerCallback(self._image_callback)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self.timer = self.create_timer(0.03, self._render)

        self.get_logger().info(f'Subscribed to:\n  left:  {self.left_topic}\n  right: {self.right_topic}')
        self.get_logger().info(f'Saving image pairs to: {os.path.abspath(self.output_dir)}')
        self.get_logger().info('Click inside the preview window to save one left/right pair. Press ESC to quit.')

    def _image_callback(self, left_msg, right_msg):
        try:
            left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert incoming images: {exc}')
            return

        with self._lock:
            self._latest_left = left_img
            self._latest_right = right_img

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self._lock:
                self._should_save = True

    def _render(self):
        with self._lock:
            left = None if self._latest_left is None else self._latest_left.copy()
            right = None if self._latest_right is None else self._latest_right.copy()
            should_save = self._should_save
            self._should_save = False

        if left is None or right is None:
            canvas = np.zeros((480, 960, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                'Waiting for both camera streams...',
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                'Click to save a pair once both images are available. ESC to exit.',
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(self.window_name, canvas)
            cv2.waitKey(1)
            return

        preview = self._compose_preview(left, right)
        cv2.imshow(self.window_name, preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self._running = False
            rclpy.shutdown()
            return

        if should_save:
            self._save_pair(left, right)

    def _compose_preview(self, left, right):
        left_view = self._annotate(left, 'LEFT')
        right_view = self._annotate(right, 'RIGHT')

        target_h = min(left_view.shape[0], right_view.shape[0])
        left_view = self._resize_to_height(left_view, target_h)
        right_view = self._resize_to_height(right_view, target_h)

        separator = np.zeros((target_h, 12, 3), dtype=np.uint8)
        separator[:] = (40, 40, 40)

        combined = np.hstack([left_view, separator, right_view])
        self._draw_help_text(combined)
        return combined

    def _annotate(self, image, label):
        annotated = image.copy()
        cv2.rectangle(annotated, (0, 0), (220, 48), (0, 0, 0), thickness=-1)
        cv2.putText(
            annotated,
            label,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _draw_help_text(self, canvas):
        cv2.rectangle(canvas, (0, canvas.shape[0] - 52), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), thickness=-1)
        cv2.putText(
            canvas,
            'Click anywhere in the window to save ONE new pair. ESC exits.',
            (18, canvas.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _resize_to_height(self, image, target_h):
        if image.shape[0] == target_h:
            return image
        scale = target_h / float(image.shape[0])
        target_w = max(1, int(round(image.shape[1] * scale)))
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _save_pair(self, left, right):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        left_path = os.path.join(self.output_dir, f'left_{self._pair_index:05d}_{stamp}.png')
        right_path = os.path.join(self.output_dir, f'right_{self._pair_index:05d}_{stamp}.png')

        left_ok = cv2.imwrite(left_path, left)
        right_ok = cv2.imwrite(right_path, right)

        if left_ok and right_ok:
            self.get_logger().info(
                f'Saved pair #{self._pair_index}:\n  {left_path}\n  {right_path}'
            )
            self._pair_index += 1
        else:
            self.get_logger().error('Failed to save one or both images.')


def main(args=None):
    rclpy.init(args=args)
    node = ClickPairRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
