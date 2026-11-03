#!/usr/bin/env python3
"""
ROS2 node: subscribe to an image topic, run YOLOv26 segmentation, publish visualization and mask images.

Usage:
  ros2 run <pkg> ros_yolo_node.py  # or run directly with python3

Parameters (Node params / args can be edited in code):
  - input_topic: sensor_msgs/Image topic to subscribe
  - vis_topic: visualization output (bgr image with overlays)
  - mask_topic: combined mask output (bgr image)
  - weights: path to YOLO weights
  - conf: confidence threshold
  - device: CUDA device index or 'cpu'

Note: Requires `ultralytics`, `opencv-python`, `numpy`, and ROS2 Python packages (`rclpy`, `cv_bridge`).
"""

import sys
import time
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class YOLOSegNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_node')

        # --- Configuration ---
        self.declare_parameter('input_topic', '/camera/left/color/image_raw')
        self.declare_parameter('vis_topic', '/yolo_seg/vis_image')
        self.declare_parameter('mask_topic', '/yolo_seg/mask_image')
        self.declare_parameter('weights', 'models/hand_seg_finetuned_yolo_26n/weights/best.pt')
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('device', '0')

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.vis_topic = self.get_parameter('vis_topic').get_parameter_value().string_value
        self.mask_topic = self.get_parameter('mask_topic').get_parameter_value().string_value
        self.weights = self.get_parameter('weights').get_parameter_value().string_value
        self.conf = float(self.get_parameter('conf').get_parameter_value().double_value)
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.bridge = CvBridge()

        # Publishers
        self.pub_vis = self.create_publisher(Image, self.vis_topic, 1)
        self.pub_mask = self.create_publisher(Image, self.mask_topic, 1)

        # Subscriber
        self.sub = self.create_subscription(Image, self.input_topic, self.callback, 1)

        # Load model in a separate thread to avoid blocking ROS spin
        self.model = None
        t = threading.Thread(target=self._load_model, daemon=True)
        t.start()

        self.get_logger().info(f'YOLOSegNode started, subscribing to: {self.input_topic}')

    def _load_model(self):
        if YOLO is None:
            self.get_logger().error("ultralytics package not available. Install with 'pip install ultralytics'.")
            return
        try:
            self.get_logger().info(f'Loading model: {self.weights}')
            self.model = YOLO(self.weights)
            # warm-up
            self.get_logger().info('Model loaded')
        except Exception as e:
            self.get_logger().error(f'Failed loading model: {e}')

    def callback(self, msg: Image):
        if self.model is None:
            # still loading
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        # Run inference (synchronous)
        try:
            results = self.model(cv_img, conf=self.conf, device=self.device)
        except Exception as e:
            # fallback to predict API
            try:
                results = self.model.predict(source=cv_img, conf=self.conf)
            except Exception as e2:
                self.get_logger().error(f'Inference failed: {e} / {e2}')
                return

        if not results:
            return

        r = results[0]

        # Visualization image (plot includes masks/boxes)
        vis_img = None
        try:
            vis_img = r.plot()
        except Exception:
            # fallback to original image
            vis_img = cv_img

        try:
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding='bgr8')
            self.pub_vis.publish(vis_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish vis image: {e}')

        # Combined mask image (if available) — robust handling for torch/numpy masks
        try:
            masks = None
            if hasattr(r, 'masks') and r.masks is not None:
                masks = getattr(r.masks, 'data', None)

            # If masks is a torch tensor, convert to numpy
            try:
                import torch
                if masks is not None and hasattr(masks, 'cpu'):
                    masks = masks.cpu().numpy()
            except Exception:
                # torch may not be installed in this env; ignore
                pass

            if masks is not None:
                masks = np.asarray(masks)

                # Ensure boolean or 0/1 mask; combine instances
                combined = None
                if masks.ndim == 3:
                    # (N, H, W) -> logical OR across instances
                    try:
                        combined_bool = np.logical_or.reduce(masks, axis=0)
                    except Exception:
                        # fallback: any along axis
                        combined_bool = np.any(masks, axis=0)
                    combined = (combined_bool.astype(np.uint8) * 255)
                elif masks.ndim == 2:
                    combined = (masks.astype(np.uint8) * 255)

                # If combined mask size doesn't match image, resize
                if combined is not None:
                    h_img, w_img = cv_img.shape[:2]
                    if combined.shape[0] != h_img or combined.shape[1] != w_img:
                        combined = cv2.resize(combined, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

                    # Convert to BGR for visualization and publish
                    if combined.ndim == 2:
                        mask_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
                    else:
                        mask_bgr = combined

                    mask_msg = self.bridge.cv2_to_imgmsg(mask_bgr, encoding='bgr8')
                    self.pub_mask.publish(mask_msg)
            else:
                # No masks found in result; do not publish
                pass
        except Exception as e:
            self.get_logger().error(f'Failed to process/publish masks: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YOLOSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
