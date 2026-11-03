import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from rclpy.qos import QoSProfile

SAVE_DIR = "calib_images"
os.makedirs(SAVE_DIR, exist_ok=True)


class CalibrationImageSaver(Node):
    def __init__(self):
        super().__init__('calibration_image_saver')

        self.bridge = CvBridge()
        self.count = 0
        self.latest_frame = None

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            QoSProfile(depth=10)
        )

        self.get_logger().info("=== Calibration Image Capture ===")
        self.get_logger().info("SPACE or ENTER → save image")
        self.get_logger().info("ESC → exit")

        # Timer just for UI / key handling
        self.timer = self.create_timer(0.03, self.ui_loop)

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    def ui_loop(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        frame_save = frame.copy()

        cv2.putText(
            frame,
            f"Images captured: {self.count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Capture Calibration Images", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            self.get_logger().info("Exiting calibration capture")
            cv2.destroyAllWindows()
            rclpy.shutdown()

        if key == 32 or key == 13:  # SPACE or ENTER
            filename = os.path.join(SAVE_DIR, f"calib_{self.count:03d}.jpg")
            cv2.imwrite(filename, frame_save)
            self.get_logger().info(f"Saved {filename}")
            self.count += 1


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
