#!/usr/bin/env python3

from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class TopicImageRecorder(Node):
    def __init__(self):
        super().__init__('topic_image_recorder')

        self.declare_parameter('image_topic', '/camera/left/color/image_raw')
        self.declare_parameter('save_dir', 'recorded_images')
        self.declare_parameter('fps', 2.0)
        self.declare_parameter('filename_prefix', 'frame')
        self.declare_parameter('save_as_jpg', False)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.save_dir = str(self.get_parameter('save_dir').value)
        self.fps = float(self.get_parameter('fps').value)
        self.filename_prefix = str(self.get_parameter('filename_prefix').value)
        self.save_as_jpg = bool(self.get_parameter('save_as_jpg').value)

        if self.fps <= 0.0:
            raise ValueError("Parameter 'fps' must be > 0")

        self.save_interval_s = 1.0 / self.fps
        self.last_save_time = None
        self.saved_count = 0

        self.bridge = CvBridge()
        self.output_path = Path(self.save_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)

        ext = 'jpg' if self.save_as_jpg else 'png'
        self.get_logger().info(
            f'Recording topic={self.image_topic} at {self.fps:.2f} FPS to {self.output_path} ({ext})'
        )

    def image_callback(self, msg: Image):
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_save_time is not None and (stamp_s - self.last_save_time) < self.save_interval_s:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert image: {exc}')
            return

        ext = 'jpg' if self.save_as_jpg else 'png'
        filename = f'{self.filename_prefix}_{self.saved_count:06d}.{ext}'
        file_path = self.output_path / filename

        if not cv2.imwrite(str(file_path), cv_img):
            self.get_logger().error(f'Failed to write image: {file_path}')
            return

        self.saved_count += 1
        self.last_save_time = stamp_s

        if self.saved_count == 1 or self.saved_count % 25 == 0:
            self.get_logger().info(f'Saved {self.saved_count} images. Last file: {file_path.name}')


def main(args=None):
    rclpy.init(args=args)
    node = TopicImageRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
