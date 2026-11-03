#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge
import cv2
import os

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        self.bridge = CvBridge()
        self.idx = 0
        
        # --- ZÄHLER STATT TIMER ---
        self.counter = 0
        self.save_every_nth = 10  # Bei 30 FPS Kamera-Input ergibt das 5 FPS Output
        # --------------------------

        self.save_path = "captured_data"
        os.makedirs(self.save_path, exist_ok=True)

        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/left/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/left/aligned_depth_to_color/image_raw')
        self.rgb_sub_right = message_filters.Subscriber(self, Image, '/camera/right/color/image_raw')
        self.depth_sub_right = message_filters.Subscriber(self, Image, '/camera/right/aligned_depth_to_color/image_raw')
        
        # Synchronizer mit etwas mehr "Slop", falls die Kamera-Hardware schwankt
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.rgb_sub_right, self.depth_sub_right], 20, slop=0.05)
        self.ts.registerCallback(self.callback)

        self.get_logger().info(f"Speichere jedes {self.save_every_nth}. Bildpaar...")

    def callback(self, rgb_msg, depth_msg, rgb_msg_right, depth_msg_right):
        # Erhöhe den Zähler bei jedem eintreffenden (synchronisierten) Nachrichtenpaar
        self.counter += 1
        print(f"Empfangenes Bildpaar #{self.counter}")

        # Nur speichern, wenn der Zähler durch 6 teilbar ist
        if self.counter % self.save_every_nth == 0:
            try:
                cv_rgb_left = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                cv_depth_left = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
                cv_rgb_right = self.bridge.imgmsg_to_cv2(rgb_msg_right, "bgr8")
                cv_depth_right = self.bridge.imgmsg_to_cv2(depth_msg_right, "passthrough")

                cv2.imwrite(os.path.join(self.save_path, f"rgb_left_{self.idx}.png"), cv_rgb_left)
                cv2.imwrite(os.path.join(self.save_path, f"depth_left_{self.idx}.png"), cv_depth_left)
                cv2.imwrite(os.path.join(self.save_path, f"rgb_right_{self.idx}.png"), cv_rgb_right)
                cv2.imwrite(os.path.join(self.save_path, f"depth_right_{self.idx}.png"), cv_depth_right)

                self.get_logger().info(f"Frame {self.counter} empfangen -> Bild {self.idx} gespeichert")
                self.idx += 1

            except Exception as e:
                self.get_logger().error(f"Fehler beim Speichern: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()