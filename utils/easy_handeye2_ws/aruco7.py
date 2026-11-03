import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile
from threading import Lock
from std_msgs.msg import String
from visualization_msgs.msg import Marker


class ArucoBoardDetector(Node):
    def __init__(self):
        super().__init__('aruco_board_detector')
        
        self.camera_name = self.declare_parameter('camera_name', 'left').get_parameter_value().string_value

        self.bridge = CvBridge()

        # --- BOARD CONFIGURATION ---
        self.markers_x = 3
        self.markers_y = 3
        self.marker_length = 0.0394      # meters
        self.marker_separation = 0.01

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.GridBoard_create(
            markersX=self.markers_x,
            markersY=self.markers_y,
            markerLength=self.marker_length,
            markerSeparation=self.marker_separation,
            dictionary=self.aruco_dict
        )
        self.parameters = cv2.aruco.DetectorParameters_create()
        # ---------------------------

        # --- PRECOMPUTE BOARD CENTER OFFSET (BOARD FRAME) ---
        board_width = self.markers_x * self.marker_length + (self.markers_x - 1) * self.marker_separation
        board_height = self.markers_y * self.marker_length + (self.markers_y - 1) * self.marker_separation

        self.board_center_offset = np.array([
            board_width / 2.0,
            board_height / 2.0,
            0.0
        ]).reshape((3, 1))
        # ---------------------------------------------------

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = f"{self.camera_name}_color_optical_frame"

        self.tf_broadcaster = TransformBroadcaster(self)
        self.transform_lock = Lock()
        self.latest_transform = None

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            f'/camera/{self.camera_name}/color/image_raw',
            self.image_callback,
            QoSProfile(depth=10)
        )
        self.matrix_sub = self.create_subscription(
            CameraInfo,
            f'/camera/{self.camera_name}/color/camera_info',
            self.camera_info_callback,
            QoSProfile(depth=10)
        )

        # Publishers
        #self.pose_pub = self.create_publisher(PoseStamped, '/aruco_board_pose', 10)
        self.timer = self.create_timer(0.03, self.publish_transform)
        
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_marker_pose', 10)
        self.marker_pub = self.create_publisher(String, '/aruco_marker_id', 10)
        self.marker_marker_pub = self.create_publisher(Marker, '/aruco_marker_visualization', 10)

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters
            )

            if ids is not None and len(ids) > 0:
                cv2.aruco.refineDetectedMarkers(
                    gray, self.board, corners, ids, rejected
                )

                valid, rvec, tvec = cv2.aruco.estimatePoseBoard(
                    corners,
                    ids,
                    self.board,
                    self.camera_matrix,
                    self.dist_coeffs,
                    None,
                    None
                )

                if valid > 0:
                    # Convert rotation
                    R, _ = cv2.Rodrigues(rvec)

                    # --- MOVE ORIGIN TO BOARD CENTER ---
                    center_tvec = tvec + R @ self.board_center_offset
                    # ----------------------------------

                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(
                        frame,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        center_tvec,
                        0.1
                    )

                    qx, qy, qz, qw = self.rotation_vector_to_quaternion(rvec)

                    with self.transform_lock:
                        self.latest_transform = {
                            'tvec': center_tvec.flatten(),
                            'quaternion': (qx, qy, qz, qw),
                            'timestamp': msg.header.stamp
                        }

                    # Publish PoseStamped (BOARD CENTER)
                    ps = PoseStamped()
                    ps.header = msg.header
                    ps.pose.position.x = center_tvec[0, 0]
                    ps.pose.position.y = center_tvec[1, 0]
                    ps.pose.position.z = center_tvec[2, 0]
                    ps.pose.orientation.x = qx
                    ps.pose.orientation.y = qy
                    ps.pose.orientation.z = qz
                    ps.pose.orientation.w = qw
                    self.pose_pub.publish(ps)

            cv2.imshow("Board Detection", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def rotation_vector_to_quaternion(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        q = np.empty((4,))
        t = np.trace(R)

        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[3] = 0.25 * s
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = (R[0, 2] - R[2, 0]) / s
            q[2] = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                q[3] = (R[2, 1] - R[1, 2]) / s
                q[0] = 0.25 * s
                q[1] = (R[0, 1] + R[1, 0]) / s
                q[2] = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                q[3] = (R[0, 2] - R[2, 0]) / s
                q[0] = (R[0, 1] + R[1, 0]) / s
                q[1] = 0.25 * s
                q[2] = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                q[3] = (R[1, 0] - R[0, 1]) / s
                q[0] = (R[0, 2] + R[2, 0]) / s
                q[1] = (R[1, 2] + R[2, 1]) / s
                q[2] = 0.25 * s

        return q

    def publish_transform(self):
        with self.transform_lock:
            if self.latest_transform:
                t = TransformStamped()
                t.header.stamp = self.latest_transform['timestamp']
                t.header.frame_id = self.camera_frame
                t.child_frame_id = "aruco_board_center"

                t.transform.translation.x = self.latest_transform['tvec'][0]
                t.transform.translation.y = self.latest_transform['tvec'][1]
                t.transform.translation.z = self.latest_transform['tvec'][2]

                qx, qy, qz, qw = self.latest_transform['quaternion']
                t.transform.rotation.x = qx
                t.transform.rotation.y = qy
                t.transform.rotation.z = qz
                t.transform.rotation.w = qw

                self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoBoardDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
