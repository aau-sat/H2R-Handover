# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import ast
from pupil_apriltags import Detector, Detection
from scipy.spatial.transform import Rotation as R
import os
import numpy as np
from Utils import to_homo
import json
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.msg as sensor_msgs
from realsense2_camera_msgs.msg import RGBD
import numpy as np
from PIL import Image
import cv2
import trimesh
import matplotlib.pyplot as plt
import socket
from rclpy.action import ActionClient
from mesh_generator.action import GenerateMesh

CAM_TAG_DEFAULT_DETECTION = Detection()
CAM_TAG_DEFAULT_DETECTION.tag_family = b'tagStandard41h12'
CAM_TAG_DEFAULT_DETECTION.tag_id = 0
CAM_TAG_DEFAULT_DETECTION.hamming = 0
CAM_TAG_DEFAULT_DETECTION.decision_margin = 0.0
CAM_TAG_DEFAULT_DETECTION.homography = np.array([[ 3.52548972e+01,  5.86007098e+00,  9.19357271e+02],
                                                  [-4.51931082e+00,  4.51964660e+01,  2.12557441e+02],
                                                  [-1.11048126e-02,  7.12507814e-03,  1.00000000e+00]])
CAM_TAG_DEFAULT_DETECTION.center = np.array([919.35727091, 212.55744096])
CAM_TAG_DEFAULT_DETECTION.corners = np.array([[874.0289917,  257.5776062],
                                                  [964.30993652, 254.24642944],
                                                  [966.36889648, 165.8653717],
                                                  [874.76098633, 171.19895935]])
CAM_TAG_DEFAULT_DETECTION.pose_R = np.array([[ 0.99489053,  0.04440697,  0.09066891],
                                                  [-0.03501397,  0.99409808, -0.10267929],
                                                  [-0.09469346,  0.09897998,  0.99057363]])
CAM_TAG_DEFAULT_DETECTION.pose_t = np.array([[ 0.64103056],
                                                  [-0.16901774],
                                                  [ 1.0445981]])
CAM_TAG_DEFAULT_DETECTION.pose_err = 0.0

# RGBD Message
# std_msgs/Header header
# sensor_msgs/CameraInfo rgb_camera_info
# sensor_msgs/CameraInfo depth_camera_info
# sensor_msgs/Image rgb
# sensor_msgs/Image depth

class PoseTracker(Node):
    def __init__(self, frame_buffer_size=1, intrinsics_path=None, zfar=np.inf):
        super().__init__('ros2_data_reader')
        self.zfar = zfar
        # realsense2_camera_msgs/msg/RGBD
        self.rgbd_sub = self.create_subscription(
            RGBD,
            '/camera/camera/rgbd',
            self.rgbd_callback,
            10)
        self.action_client = ActionClient(self, GenerateMesh, 'generate_mesh')
        # self.info_sub = self.create_subscription(sensor_msgs.CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.objects_detected = False
        self.color_file = None  # Placeholder for single color image
        self.depth_file = None  # Placeholder for single depth image
        self.start_mask = None   # Placeholder for mask image file paths
        self.K = np.loadtxt(intrinsics_path).reshape(3,3) if intrinsics_path is not None else np.eye(3)
        self.H = 480          # Placeholder for image height
        self.W = 640          # Placeholder for image width
        self.id_strs = []     # Placeholder for frame identifiers
        self.bridge = CvBridge()
        self.got_mask = False
        self.frame_buffer_size = frame_buffer_size
        self.mesh_file = None

        self.socket_path = "/tmp/sam3_socket"
    
    def request_start_mask(self):
        cv_image = self.color_file
        h, w, _ = cv_image.shape
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self.socket_path)

            # Send Header: H, W, x1, y1, x2, y2, label_len
            header = np.array([h, w], dtype=np.int32)
            client.sendall(header.tobytes())
            # client.sendall(label.encode())

            # Send Image
            client.sendall(cv_image.tobytes())

            # Receive Mask (size is H*W bytes for uint8)
            mask_data = b""
            while len(mask_data) < (h * w):
                mask_data += client.recv((h * w) - len(mask_data))
            
            mask = np.frombuffer(mask_data, dtype=np.uint8).reshape((h, w))
            
            # Now publish 'mask' to a new ROS2 topic
            self.get_logger().info("Mask received successfully")

        except Exception as e:
            self.get_logger().error(f"Failed to connect to SAM3 server: {e}")
        finally:
            client.close()

        self.start_mask = mask.astype(bool).astype(np.uint8)
        self.got_mask = True

        self.send_mesh_goal()

    # def camera_info_callback(self, msg):
    #     if self.K is None:  # Update cam_K only once to avoid redundant updates
    #         self.K = np.array(msg.k).reshape((3, 3))
    #         self.get_logger().info(f"Camera intrinsic matrix initialized: {self.K}")

    def rgbd_callback(self, msg):
        self.color_file = self.bridge.imgmsg_to_cv2(msg.rgb, "rgb8")
        self.depth_file = self.bridge.imgmsg_to_cv2(msg.depth, "32FC1") / 1e3
        if not self.got_mask:
            self.request_start_mask()

    # def convert_depth_image(self, ros_image):
    #     depth_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
    #     depth_array = np.array(depth_image, dtype=np.float32)
    #     return depth_array

    def get_mask(self):
        return self.start_mask

    def get_intrinsics(self):
        return self.K
    
    def get_color(self):
        return self.color_file
    
    def get_depth(self):
        return self.depth_file
    
    def get_mesh_file(self):
        return self.mesh_file
    
    def send_mesh_goal(self):
        goal_msg = GenerateMesh.Goal()
        goal_msg.rgb_image = self.bridge.cv2_to_imgmsg(self.color_file, "rgb8")
        goal_msg.mask_image = self.bridge.cv2_to_imgmsg(self.start_mask, "mono8")

        self.action_client.wait_for_server()

        self._send_goal_future = self.action_client.send_goal_async(goal_msg)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.status))
        self.get_logger().info('Resulting file: {0}'.format(result.mesh_path))
        self.mesh_file = result.mesh_path
    
    def publish_pose_cam_frame(self, pose_cam):
        # Publish the estimated pose as a ROS2 topic for visualization in RViz
        from geometry_msgs.msg import PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "camera_color_optical_frame"
        pose_msg.pose.position.x = pose_cam[0, 3]
        pose_msg.pose.position.y = pose_cam[1, 3]
        pose_msg.pose.position.z = pose_cam[2, 3]
        # Convert rotation matrix to quaternion
        rot = R.from_matrix(pose_cam[:3, :3])
        quat = rot.as_quat()  # Returns (x, y, z, w)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        if not hasattr(self, 'pose_publisher'):
            self.pose_publisher = self.create_publisher(PoseStamped, '/estimated_pose', 10)
        
        self.pose_publisher.publish(pose_msg)

def main():
    rclpy.init()

    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    # parser.add_argument('--mesh_file', type=str, default=f'demo_data/test3/mesh/center_mesh_demo_anchor.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--headless', action='store_true', help='do not show the visualization, good for running on a server')
    parser.add_argument('--mesh_scale', type=float, default=0.68)
    parser.add_argument('--intrinsics_path', type=str, default='demo_data/test3/cam_K.txt')
    parser.add_argument('--start_mask_path', type=str, default='demo_data/anchor_mask.png')
    parser.add_argument('--init_rot_guess', type=ast.literal_eval)

    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    tracker = PoseTracker(intrinsics_path=args.intrinsics_path, zfar=np.inf) # Transform into ROS2 node

    while rclpy.ok():
        rclpy.spin_once(tracker)

        mask = tracker.get_mask()
        color = tracker.get_color()
        depth = tracker.get_depth()
        mesh_file = tracker.get_mesh_file()
        if color is None or depth is None or mesh_file is None:
            continue
        else:
            break

    mesh = trimesh.load(mesh_file)
    mesh.apply_scale(args.mesh_scale)

    debug = args.debug
   
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    # I want to preserve the mesh axes for downstream transforms
    centroid = mesh.centroid
    # to_origin = np.eye(4)
    to_origin[:3, 3] = -centroid
    # extents = mesh.extents
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    bbox_homo = to_homo(np.stack([extents/2, extents/2, extents/2, extents/2], axis=0).reshape(4,3))
    for i in range(3):
        bbox_homo[i, i] *= -1

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    pose_initialized = False
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,  debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    # tracker = PoseTracker(intrinsics_path=args.intrinsics_path, zfar=np.inf) # Transform into ROS2 node
    stopped_tracking = False
    out_of_frame = False
    mesh = None

    while rclpy.ok():
        rclpy.spin_once(tracker)

        mask = tracker.get_mask()
        color = tracker.get_color()
        depth = tracker.get_depth()
        mesh_file = tracker.get_mesh_file()
        # if color is None or depth is None or mesh_file is None:
        #     continue

        H, W = color.shape[:2]
        
        depth[(depth < 0.1) | (depth >= np.inf)] = 0
        print('color shape:', color.shape, 'depth shape:', depth.shape)
            
        if not pose_initialized:
            pose = None
            mask = tracker.get_mask().astype(bool)
            print('mask shape:', mask.shape, 'mask sum:', mask.sum())
            pose = est.register(K=tracker.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter, init_rot_guess=args.init_rot_guess)
            pose_initialized = True
        else:
            print('color shape:', color.shape, 'depth shape:', depth.shape)
            print("@"*20 + "\nno masks\n" + "@"*20)
            pose = est.track_one(rgb=color, depth=depth, K=tracker.K, iteration=args.track_refine_iter)
        last_pose = pose.copy()
        transforms = {}            

        if debug>=1:
            vis = color.copy()
            if out_of_frame:
                continue
            print('pose:\n', pose)
            print(pose.dtype)
            print('center_pose:\n', (pose @ np.linalg.inv(to_origin)))
            center_pose = pose @ np.linalg.inv(to_origin)
            print(center_pose.dtype)
            # publish the estimated pose for rviz visualization
            tracker.publish_pose_cam_frame(center_pose)
            vis = draw_posed_3d_box(tracker.K, img=vis, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=tracker.K, thickness=2, transparency=0, is_input_rgb=True)
            if not args.headless:
                cv2.imshow('1', vis[...,::-1])
                cv2.waitKey(1)
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
