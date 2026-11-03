import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from functools import partial

class ArucoBoardDetector(Node):
    def __init__(self):
        super().__init__('aruco_board_detector')
        self.bridge = CvBridge()

        # --- BOARD CONFIGURATION ---
        self.markers_x = 3
        self.markers_y = 3
        self.marker_length = 0.037      
        self.marker_separation = 0.009
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        self.board = cv2.aruco.GridBoard_create(
            self.markers_x, self.markers_y, self.marker_length,
            self.marker_separation, self.aruco_dict
        )
        self.parameters = cv2.aruco.DetectorParameters_create()

        # --- PRECOMPUTE BOARD CENTER OFFSET ---
        # The origin is at the corner of marker ID 0. 
        # Center is half of the total width and height.
        total_width = (self.markers_x * self.marker_length) + ((self.markers_x - 1) * self.marker_separation)
        total_height = (self.markers_y * self.marker_length) + ((self.markers_y - 1) * self.marker_separation)

        # Offset in the BOARD'S local coordinate frame
        self.board_center_offset = np.array([
            total_width / 2.0,
            total_height / 2.0,
            0.0
        ])

        self.cameras = {
            'cam_left': {'matrix': None, 'dist': None, 'frame': 'cam_left_optical_frame'},
            'cam_right': {'matrix': None, 'dist': None, 'frame': 'cam_right_optical_frame'}
        }

        self.tf_broadcaster = TransformBroadcaster(self)
        self.publishers_map = {}
        
        for cam_id in self.cameras.keys():
            self.create_subscription(Image, f'/camera/{cam_id}/color/image_raw', 
                partial(self.image_callback, cam_id=cam_id), 10)
            self.create_subscription(CameraInfo, f'/camera/{cam_id}/color/camera_info', 
                partial(self.camera_info_callback, cam_id=cam_id), 10)
            self.publishers_map[cam_id] = {
                'pose': self.create_publisher(PoseStamped, f'/{cam_id}/aruco_marker_pose', 10),
                'marker': self.create_publisher(Marker, f'/{cam_id}/aruco_marker_visualization', 10)
            }

    def camera_info_callback(self, msg, cam_id):
        self.cameras[cam_id]['matrix'] = np.array(msg.k).reshape(3, 3)
        self.cameras[cam_id]['dist'] = np.array(msg.d)
        self.cameras[cam_id]['frame'] = msg.header.frame_id

    def image_callback(self, msg, cam_id):
        cam_data = self.cameras[cam_id]
        if cam_data['matrix'] is None: return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                    corners, ids, self.board, cam_data['matrix'], cam_data['dist'], None, None
                )
                
                if retval > 0:
                    # Transform tvec from corner to center
                    # center_tvec = tvec + R * offset_local
                    R, _ = cv2.Rodrigues(rvec)
                    center_tvec = tvec + (R @ self.board_center_offset.reshape(3, 1))

                    # 1. Draw axes at the NEW center
                    cv2.drawFrameAxes(frame, cam_data['matrix'], cam_data['dist'], rvec, center_tvec, 0.05)

                    # 2. Publish TF and Pose at center
                    qx, qy, qz, qw = self.rotation_vector_to_quaternion(rvec)
                    
                    t = TransformStamped()
                    t.header.stamp = msg.header.stamp
                    t.header.frame_id = cam_data['frame']
                    t.child_frame_id = f"{cam_id}_board_center"
                    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = center_tvec.flatten()
                    t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = qx, qy, qz, qw
                    self.tf_broadcaster.sendTransform(t)

                    ps = PoseStamped()
                    ps.header = msg.header
                    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = center_tvec.flatten()
                    ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = qx, qy, qz, qw
                    self.publishers_map[cam_id]['pose'].publish(ps)
                    self.publish_marker_viz(cam_id, ps)

            cv2.imshow(f"Detection - {cam_id}", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def publish_marker_viz(self, cam_id, pose_stamped):
        marker = Marker()
        marker.header = pose_stamped.header
        marker.ns = cam_id
        marker.id = 0
        marker.type = Marker.CUBE
        marker.pose = pose_stamped.pose
        # Scale reflects the physical board size
        marker.scale.x = (self.markers_x * self.marker_length) + ((self.markers_x - 1) * self.marker_separation)
        marker.scale.y = (self.markers_y * self.marker_length) + ((self.markers_y - 1) * self.marker_separation)
        marker.scale.z = 0.002
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = 0.6, 0.0, 1.0, 0.0
        self.publishers_map[cam_id]['marker'].publish(marker)

    def rotation_vector_to_quaternion(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        # Standard rotation matrix to quaternion conversion
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

def main(args=None):
    rclpy.init(args=args)
    node = ArucoBoardDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()