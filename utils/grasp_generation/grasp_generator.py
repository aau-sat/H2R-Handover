#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge
import socket
import os
import struct
from scipy.spatial.transform import Rotation
from realsense2_camera_msgs.msg import RGBD
import cv2
import sensor_msgs.msg as sensor_msgs
import json
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.time import Time
from rclpy.duration import Duration
import open3d as o3d
from utils.pc_utils import *
import message_filters
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from grasp_generation.action import GenerateGrasp

class GraspGenerator(Node):
    """
    ROS2 action server that generates grasp poses.
    
    Subscribes to:
        - /segmentation_mask (Image): Mask image (temporarily for single view, will be replaced by dual masks in the future)
        - /camera/{camera_name}/color/camera_info (sensor_msgs/CameraInfo): Camera intrinsic parameters
        - /camera/{camera_name}/aligned_depth_to_color/image_raw (Image): Depth images topic for each camera view
        """
        
    def __init__(self):
        super().__init__('grasp_generator_node')
        
        self.use_two_views = self.declare_parameter('use_two_views', True).get_parameter_value().bool_value
        self.camera_name_first = self.declare_parameter('camera_name_first', 'left').get_parameter_value().string_value
        self.camera_name_second = self.declare_parameter('camera_name_second', 'right').get_parameter_value().string_value
        self.default_object_label = self.declare_parameter('object_label', 'cup').get_parameter_value().string_value
        self.object_label_topic = self.declare_parameter('object_label_topic', '/detected_label').get_parameter_value().string_value
        self.object_label = self.default_object_label
        self.hand_tracker_topic = self.declare_parameter('hand_tracker_topic', '/hand_tracker').get_parameter_value().string_value
        self.handover_detected_topic = self.declare_parameter('handover_detected_topic', '/handover_detected').get_parameter_value().string_value

        self.first_depth_topic = f'/camera/{self.camera_name_first}/aligned_depth_to_color/image_raw'
        self.second_depth_topic = f'/camera/{self.camera_name_second}/aligned_depth_to_color/image_raw'
        self.first_color_topic = f'/camera/{self.camera_name_first}/color/image_raw'
        self.second_color_topic = f'/camera/{self.camera_name_second}/color/image_raw'
        self.first_camera_info_topic = f'/camera/{self.camera_name_first}/aligned_depth_to_color/camera_info'
        self.second_camera_info_topic = f'/camera/{self.camera_name_second}/aligned_depth_to_color/camera_info'

        self.target_frame = 'base'
        self.source_frame_first = f'{self.camera_name_first}_color_optical_frame'
        self.source_frame_second = f'{self.camera_name_second}_color_optical_frame'

        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Storage for latest images
        self.latest_depth_first = None
        self.latest_depth_second = None
        self.latest_color_first = None
        self.latest_color_second = None

        self.left_hand_visibility_score = 0
        self.right_hand_visibility_score = 0
        self.preferred_single_view_side = 'left'
        self.visibility_counting_enabled = False
        
        # Flag to prevent multiple simultaneous requests
        self.first_camera_initialized = False
        self.second_camera_initialized = False
        
        # create sensor qos profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
             depth=10
        )
        
        # Subscribers
        self.first_cam_info_sub = self.create_subscription(
            sensor_msgs.CameraInfo,
            self.first_camera_info_topic, 
            self.first_camera_info_callback, 
            qos_profile=qos_profile
        )
        self.second_cam_info_sub = self.create_subscription(
            sensor_msgs.CameraInfo,
            self.second_camera_info_topic,
            self.second_camera_info_callback,
            qos_profile=qos_profile
        )
        self.first_depth_sub = message_filters.Subscriber(self, Image, self.first_depth_topic, qos_profile=qos_profile)
        self.first_color_sub = message_filters.Subscriber(self, Image, self.first_color_topic, qos_profile=qos_profile)
        sync_topic_list_first = [self.first_depth_sub, self.first_color_sub]

        self.second_depth_sub = message_filters.Subscriber(self, Image, self.second_depth_topic, qos_profile=qos_profile)
        self.second_color_sub = message_filters.Subscriber(self, Image, self.second_color_topic, qos_profile=qos_profile)
        sync_topic_list_second = [self.second_depth_sub, self.second_color_sub]

        self.ts_first = message_filters.ApproximateTimeSynchronizer(
            sync_topic_list_first,
            queue_size=10,
            slop=0.05 
        )
        self.ts_first.registerCallback(self.sync_callback_first)

        # Keep both synchronized streams active even in single-view mode so the
        # hand-tracker-driven selector can choose the best camera side.
        self.ts_second = message_filters.ApproximateTimeSynchronizer(
            sync_topic_list_second,
            queue_size=10,
            slop=0.05 
        )
        self.ts_second.registerCallback(self.sync_callback_second)

        self.object_label_sub = self.create_subscription(
            String,
            self.object_label_topic,
            self.object_label_callback,
            10,
        )

        if not self.use_two_views:
            self.handover_detected_sub = self.create_subscription(
                Bool,
                self.handover_detected_topic,
                self.handover_detected_callback,
                qos_profile=qos_profile,
            )
            self.hand_tracker_sub = self.create_subscription(
                String,
                self.hand_tracker_topic,
                self.hand_tracker_callback,
                qos_profile=qos_profile
            )
        else:
            self.handover_detected_sub = None
            self.hand_tracker_sub = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_first = None
        self.tf_second = None

        self.grasp_socket_path = "/tmp/graspgen"
        self.sam_socket_path = "/tmp/sam3_socket"
        
        self.cb_group = ReentrantCallbackGroup()
        self.action_server = ActionServer(
            self,
            GenerateGrasp,
            'generate_grasp',
            execute_callback=self.execute_callback,
            callback_group=self.cb_group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

        self.get_logger().info(
            f'Grasp Generator Action Server initialized | '
            f'object_label_topic={self.object_label_topic}, fallback_label={self.default_object_label}'
        )

    def handover_detected_callback(self, msg):
        if self.use_two_views or not bool(msg.data):
            return

        self.visibility_counting_enabled = True
        if self.handover_detected_sub is not None:
            self.destroy_subscription(self.handover_detected_sub)
            self.handover_detected_sub = None
        self.get_logger().info('handover_detected=true received; starting hand visibility scoring and removing handover subscription.')

    def object_label_callback(self, msg):
        label = (msg.data or '').strip()
        if not label or label.lower() == 'none':
            return

        # if label != self.object_label:
        #     self.object_label = label
        #     self.get_logger().info(f"Updated object label from topic: {self.object_label}")
        self.object_label_sub.destroy()

    def hand_tracker_callback(self, msg):
        if self.use_two_views or not self.visibility_counting_enabled:
            return

        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f'Failed to decode hand tracker message: {exc}')
            return

        more_hand_visible = int(payload.get('more_hand_visible', 0))
        if more_hand_visible == 1:
            self.left_hand_visibility_score += 1
        elif more_hand_visible == 2:
            self.right_hand_visibility_score += 1
        else:
            self.left_hand_visibility_score += 1
            self.right_hand_visibility_score += 1

        self.preferred_single_view_side = (
            'left'
            if self.left_hand_visibility_score >= self.right_hand_visibility_score
            else 'right'
        )

    def _single_view_ready(self, side):
        if side == 'left':
            return (
                self.first_camera_initialized and
                self.latest_depth_first is not None and
                self.latest_color_first is not None and
                self.tf_first is not None
            )
        return (
            self.second_camera_initialized and
            self.latest_depth_second is not None and
            self.latest_color_second is not None and
            self.tf_second is not None
        )

    def _preferred_single_view(self):
        if self.preferred_single_view_side == 'right' and self._single_view_ready('right'):
            return 'right'
        if self.preferred_single_view_side == 'left' and self._single_view_ready('left'):
            return 'left'
        if self._single_view_ready('left'):
            return 'left'
        if self._single_view_ready('right'):
            return 'right'
        return None
    
    def first_camera_info_callback(self, msg):
        self.ffx = msg.k[0]
        self.ffy = msg.k[4]
        self.fcx = msg.k[2]
        self.fcy = msg.k[5]
        self.first_camera_initialized = True
        self.get_logger().info(f"Camera intrinsics received: fx={self.ffx}, fy={self.ffy}, cx={self.fcx}, cy={self.fcy}")
        # Unsubscribe after getting intrinsics
        self.destroy_subscription(self.first_cam_info_sub)

    def second_camera_info_callback(self, msg):
        self.sfx = msg.k[0]
        self.sfy = msg.k[4]
        self.scx = msg.k[2]
        self.scy = msg.k[5]
        self.second_camera_initialized = True
        self.get_logger().info(f"Camera intrinsics received: fx={self.sfx}, fy={self.sfy}, cx={self.scx}, cy={self.scy}")
        # Unsubscribe after getting intrinsics
        if self.second_cam_info_sub is not None:
            self.destroy_subscription(self.second_cam_info_sub)
            self.second_cam_info_sub = None

    def sync_callback_first(self, depth_first_msg, color_first_msg):
        if self.tf_first is None:
            try:
                self.tf_first = self.tf_buffer.lookup_transform(
                    target_frame=self.target_frame,
                    source_frame=self.source_frame_first,
                    time=Time(),
                    timeout=Duration(nanoseconds=100_000_000)
                )
            except Exception as ex:
                self.get_logger().info(
                    f'Could not find transform from {self.source_frame_first} to {self.target_frame}: {ex}'
                )
        
        if self.tf_first is not None and self.tf_second is not None:
            self.tf_listener.unregister()
             
        try:
            self.latest_depth_first = self.bridge.imgmsg_to_cv2(depth_first_msg, "32FC1") / 1e3
            self.latest_color_first = self.bridge.imgmsg_to_cv2(color_first_msg, "rgb8")
        except Exception as e:
            self.get_logger().error(f'Error converting depth images: {str(e)}')

    def sync_callback_second(self, depth_second_msg, color_second_msg):
        if self.tf_second is None:
            try:
                self.tf_second = self.tf_buffer.lookup_transform(
                    target_frame=self.target_frame,
                    source_frame=self.source_frame_second,
                    time=Time(),
                    timeout=Duration(nanoseconds=100_000_000)
                )
            except Exception as ex:
                self.get_logger().info(
                    f'Could not find transform from {self.source_frame_second} to {self.target_frame}: {ex}'
                )
        
        if self.tf_first is not None and self.tf_second is not None:
            self.tf_listener.unregister()
        
        try:
            self.latest_depth_second = self.bridge.imgmsg_to_cv2(depth_second_msg, "32FC1") / 1e3
            self.latest_color_second = self.bridge.imgmsg_to_cv2(color_second_msg, "rgb8")
        except Exception as e:
            self.get_logger().error(f'Error converting second depth image: {str(e)}')

    def _recv_exact(self, client, size):
        data = b""
        while len(data) < size:
            chunk = client.recv(size - len(data))
            if not chunk:
                raise ConnectionError('Connection closed while receiving data from segmentation service')
            data += chunk
        return data

    def request_segmentation_masks(self, color_image, label=None, bbox=None):
        if label is None:
            label = self.object_label

        h, w, _ = color_image.shape

        if bbox is None:
            x1, y1, x2, y2 = 0, 0, 0, 0
        else:
            x1, y1, x2, y2 = bbox

        label_bytes = label.encode()
        label_len = len(label_bytes)

        client = None
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self.sam_socket_path)

            # Send Header: H, W, x1, y1, x2, y2, label_len
            header = np.array([h, w, x1, y1, x2, y2, label_len], dtype=np.int32)
            client.sendall(header.tobytes())
            client.sendall(label_bytes)

            # Send Image
            client.sendall(color_image.tobytes())

            response = self._recv_exact(client, 8)
            object_ok = bool(np.frombuffer(response[:4], dtype=np.uint32)[0])
            hand_ok = bool(np.frombuffer(response[4:], dtype=np.uint32)[0])

            masks = []
            for _ in range(2):
                size_bytes = self._recv_exact(client, 4)
                payload_size = int(np.frombuffer(size_bytes, dtype=np.uint32)[0])
                mask_data = self._recv_exact(client, payload_size)
                masks.append(np.frombuffer(mask_data, dtype=np.uint8).reshape((h, w)))

            object_mask = masks[0] if len(masks) > 0 else None
            hand_mask = masks[1] if len(masks) > 1 else None

            self.get_logger().info(
                f'Segmentation masks received successfully: object_ok={object_ok}, hand_ok={hand_ok}'
            )

        except Exception as e:
            self.get_logger().error(f"Failed to connect to SAM3 server: {e}")
            object_ok = False
            hand_ok = False
            object_mask = None
            hand_mask = None
        finally:
            if client is not None:
                client.close()

        if object_mask is not None:
            object_mask = object_mask.astype(bool).astype(np.uint8)
        if hand_mask is not None:
            hand_mask = hand_mask.astype(bool).astype(np.uint8)

        # if object_mask is not None:
        #     cv2.imwrite('debug_sam3_request_object_mask.png', object_mask * 255)
        # if hand_mask is not None:
        #     cv2.imwrite('debug_sam3_request_hand_mask.png', hand_mask * 255)

        return {
            'object_ok': object_ok,
            'hand_ok': hand_ok,
            'object_mask': object_mask,
            'hand_mask': hand_mask,
        }

    def _transform_points_to_target(self, points, tf_msg):
        if points is None:
            return None

        points = np.asarray(points, dtype=np.float32)
        if len(points) == 0 or tf_msg is None:
            return points

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = Rotation.from_quat([
            tf_msg.transform.rotation.x,
            tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z,
            tf_msg.transform.rotation.w,
        ]).as_matrix()
        tf_matrix[:3, 3] = [
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z,
        ]
        point_cloud.transform(tf_matrix)
        return np.asarray(point_cloud.points)

    def _apply_dbscan(self, points, eps=0.02, min_points=10):
        """Apply DBSCAN clustering to filter out outliers from the point cloud."""
        if points is None or len(points) <= min_points: 
            return points
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        
        if len(labels) == 0:
            return points
        
        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            return points
        
        # Keep only the largest cluster
        counts = np.bincount(valid_labels)
        largest_cluster_label = np.argmax(counts)
        
        indices = np.where(labels == largest_cluster_label)[0]
        return points[indices]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points))
        if len(labels) == 0:
            return points

        largest_cluster_label = np.bincount(labels[labels >= 0]).argmax()
        inliers = labels == largest_cluster_label
        return points[inliers]
    
    def _build_view_point_clouds(self, depth_image, color_image, fx, fy, cx, cy, tf_msg, view_name):
        masks = self.request_segmentation_masks(color_image, label=self.object_label)
        object_mask = masks['object_mask']
        hand_mask = masks['hand_mask']

        if not masks['object_ok'] or object_mask is None:
            self.get_logger().warn(f'Object mask unavailable for {view_name} view; skipping grasp generation')
            return None

        depth_image = depth_image.astype(np.float32).reshape(depth_image.shape)

        if masks['hand_ok'] and hand_mask is not None:
            try:
                hand_pc, object_pc = depth_and_segmentation_to_point_clouds(
                    depth_image=depth_image,
                    segmentation_mask_obj=object_mask,
                    segmentation_mask_hand=hand_mask,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    target_object_id=1,
                )
                self.get_logger().info(
                    f'Generated point clouds ({view_name}): hand={hand_pc.shape}, object={object_pc.shape}'
                )
            except Exception as exc:
                self.get_logger().error(f'Failed to generate {view_name}-view point clouds: {str(exc)}')
                return None
        else:
            self.get_logger().warn(
                f'Hand mask unavailable for {view_name} view; creating empty scene point cloud'
            )
            try:
                _, object_pc = depth_and_segmentation_to_point_clouds_old(
                    depth_image=depth_image,
                    segmentation_mask=object_mask,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    target_object_id=1,
                    remove_object_from_scene=True,
                )
            except Exception as exc:
                self.get_logger().error(f'Failed to generate {view_name}-view object point cloud: {str(exc)}')
                return None
            hand_pc = np.empty((0, 3), dtype=np.float32)

        if hand_pc is not None and len(hand_pc) > 0:
            hand_pc = point_cloud_outlier_removal(torch.from_numpy(hand_pc).float(), threshold=0.005, K=10).cpu().numpy()
            
            # Apply DBSCAN to filter out outliers from the object point cloud.
            #if len(hand_pc) > 50:
            #    hand_pc = self._apply_dbscan(hand_pc)
        else:
            hand_pc = np.empty((0, 3), dtype=np.float32)

        if object_pc is None or len(object_pc) == 0:
            self.get_logger().warn(f'Object mask produced no points for {view_name} view')
            return None

        object_pc = point_cloud_outlier_removal(torch.from_numpy(object_pc).float()).cpu().numpy()
        
        # Apply DBSCAN to filter out outliers from the object point cloud.
        if len(object_pc) > 50:
            object_pc = self._apply_dbscan(object_pc)
            

        hand_pc = self._transform_points_to_target(hand_pc, tf_msg)
        object_pc = self._transform_points_to_target(object_pc, tf_msg)

        return hand_pc, object_pc

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        result = GenerateGrasp.Result()
        
        # at this point tf should be received
        self.tf_listener.unregister()

        first_view_ready = (
            self.first_camera_initialized and
            self.latest_depth_first is not None and
            self.latest_color_first is not None and
            self.tf_first is not None
        )
        second_view_ready = (
            self.second_camera_initialized and
            self.latest_depth_second is not None and
            self.latest_color_second is not None and
            self.tf_second is not None
        )

        if not self.use_two_views:
            preferred_view = self._preferred_single_view()
            if preferred_view == 'left':
                self.get_logger().info(
                    f'Single-view mode selecting left view based on hand visibility scores '
                    f'(left={self.left_hand_visibility_score}, right={self.right_hand_visibility_score})'
                )
                return await self.execute_callback_first_view(goal_handle, result)
            if preferred_view == 'right':
                self.get_logger().info(
                    f'Single-view mode selecting right view based on hand visibility scores '
                    f'(left={self.left_hand_visibility_score}, right={self.right_hand_visibility_score})'
                )
                return await self.execute_callback_second_view(goal_handle, result)

            self.get_logger().error('No single-view candidate is ready, aborting grasp generation')
            goal_handle.abort()
            result.status = False
            return result

        # Fallback behavior: if one view is unavailable, use the other view.
        if not first_view_ready and second_view_ready:
            self.get_logger().warn('First view unavailable, falling back to second view')
            return await self.execute_callback_second_view(goal_handle, result)
        if not second_view_ready and first_view_ready:
            self.get_logger().warn('Second view unavailable, falling back to first view')
            return await self.execute_callback_first_view(goal_handle, result)

        if not first_view_ready and not second_view_ready:
            self.get_logger().error('No complete view available (depth/color/tf), aborting grasp generation')
            goal_handle.abort()
            result.status = False
            return result
        
        view_candidates = []
        view_candidates.append((
            'first',
            self.latest_depth_first,
            self.latest_color_first,
            self.ffx, self.ffy, self.fcx, self.fcy,
            self.tf_first,
        ))
        view_candidates.append((
            'second',
            self.latest_depth_second,
            self.latest_color_second,
            self.sfx, self.sfy, self.scx, self.scy,
            self.tf_second,
        ))

        if not view_candidates:
            self.get_logger().error('No complete view available (depth/color/tf), aborting grasp generation')
            goal_handle.abort()
            result.status = False
            return result

        view_clouds = []
        for view_name, depth_image, color_image, fx, fy, cx, cy, tf_msg in view_candidates:
            clouds = self._build_view_point_clouds(
                depth_image=depth_image,
                color_image=color_image,
                fx=fx, fy=fy, cx=cx, cy=cy,
                tf_msg=tf_msg,
                view_name=view_name,
            )
            if clouds is not None:
                view_clouds.append(clouds)

        if not view_clouds:
            self.get_logger().error('No view produced usable object point cloud, aborting grasp generation')
            goal_handle.abort()
            result.status = False
            return result

        if len(view_clouds) == 1:
            hand_pc, object_pc = view_clouds[0]
        else:
            hand_pc = np.vstack([view_clouds[0][0], view_clouds[1][0]])
            object_pc = np.vstack([view_clouds[0][1], view_clouds[1][1]])
        
        try:
            # Validate point clouds before sending
            if object_pc is None or len(object_pc) == 0:
                self.get_logger().error(
                    f'Invalid object point cloud before sending: object={object_pc.shape if object_pc is not None else None}'
                )
                goal_handle.abort()
                result.status = False
                return result
            
            # Send grasp request via Unix socket
            grasp_matrices, grasp_labels = self.send_grasp_request(hand_pc, object_pc)
            
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.status = False
                return result
            
            if grasp_matrices is not None:
                # Convert to PoseArray and publish
                poses_msg = PoseArray()
                poses_msg.header.stamp = self.get_clock().now().to_msg()
                poses_msg.header.frame_id = self.target_frame
                for grasp_matrix in grasp_matrices:
                    pose_msg = self.matrix_to_pose(grasp_matrix)
                    poses_msg.poses.append(pose_msg)
                    # Log the result
                    pos = grasp_matrix[:3, 3]
                    self.get_logger().info(
                        f'Grasp pose chosen - Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]'
                    )
                result.grasp_goals = poses_msg
                result.grasp_labels = grasp_labels
                result.status = True
                goal_handle.succeed()
            else:
                result.status = False
                self.get_logger().warn('No valid grasp found')
                goal_handle.succeed()
                
        except Exception as e:
            self.get_logger().error(f'Error in grasp generation: {str(e)}')
            result.status = False
            goal_handle.abort()
            
        return result
    
    # assume that depth, color and tf are available at this point
    async def execute_callback_first_view(self, goal_handle, result):
        self.get_logger().info('Executing grasp generation using first view only')

        if self.latest_depth_first is None or self.latest_color_first is None or self.tf_first is None:
            self.get_logger().error('First view is not ready for grasp generation')
            goal_handle.abort()
            result.status = False
            return result

        try:
            view_clouds = self._build_view_point_clouds(
                depth_image=self.latest_depth_first,
                color_image=self.latest_color_first,
                fx=self.ffx, fy=self.ffy, cx=self.fcx, cy=self.fcy,
                tf_msg=self.tf_first,
                view_name='first',
            )
            if view_clouds is None:
                goal_handle.abort()
                result.status = False
                return result

            hand_pc, object_pc = view_clouds

            if object_pc is None or len(object_pc) == 0:
                self.get_logger().error(
                    f'Invalid object point cloud before sending: object={object_pc.shape if object_pc is not None else None}'
                )
                goal_handle.abort()
                result.status = False
                return result

            grasp_matrices, grasp_labels = self.send_grasp_request(hand_pc, object_pc)

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.status = False
                return result

            if grasp_matrices is not None:
                poses_msg = PoseArray()
                poses_msg.header.stamp = self.get_clock().now().to_msg()
                poses_msg.header.frame_id = self.target_frame
                for grasp_matrix in grasp_matrices:
                    pose_msg = self.matrix_to_pose(grasp_matrix)
                    poses_msg.poses.append(pose_msg)
                    pos = grasp_matrix[:3, 3]
                    self.get_logger().info(
                        f'Grasp pose chosen - Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]'
                    )
                result.grasp_goals = poses_msg
                result.grasp_labels = grasp_labels
                result.status = True
                goal_handle.succeed()
            else:
                result.status = False
                self.get_logger().warn('No valid grasp found from first view')
                goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f'Error in first-view grasp generation: {str(e)}')
            result.status = False
            goal_handle.abort()

        return result
    
    # assume that depth, color and tf are available at this point
    async def execute_callback_second_view(self, goal_handle, result):
        self.get_logger().info('Executing grasp generation using second view only')

        if self.latest_depth_second is None or self.latest_color_second is None or self.tf_second is None:
            self.get_logger().error('Second view is not ready for grasp generation')
            goal_handle.abort()
            result.status = False
            return result

        try:
            view_clouds = self._build_view_point_clouds(
                depth_image=self.latest_depth_second,
                color_image=self.latest_color_second,
                fx=self.sfx, fy=self.sfy, cx=self.scx, cy=self.scy,
                tf_msg=self.tf_second,
                view_name='second',
            )
            if view_clouds is None:
                goal_handle.abort()
                result.status = False
                return result

            hand_pc, object_pc = view_clouds
            
            if object_pc is None or len(object_pc) == 0:
                self.get_logger().error(
                    f'Invalid object point cloud before sending: object={object_pc.shape if object_pc is not None else None}'
                )
                goal_handle.abort()
                result.status = False
                return result

            grasp_matrices, grasp_labels = self.send_grasp_request(hand_pc, object_pc)

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.status = False
                return result

            if grasp_matrices is not None:
                poses_msg = PoseArray()
                poses_msg.header.stamp = self.get_clock().now().to_msg()
                poses_msg.header.frame_id = self.target_frame
                for grasp_matrix in grasp_matrices:
                    pose_msg = self.matrix_to_pose(grasp_matrix)
                    poses_msg.poses.append(pose_msg)
                    pos = grasp_matrix[:3, 3]
                    self.get_logger().info(
                        f'Grasp pose chosen - Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]'
                    )
                result.grasp_goals = poses_msg
                result.grasp_labels = grasp_labels
                result.status = True
                goal_handle.succeed()
            else:
                result.status = False
                self.get_logger().warn('No valid grasp found from second view')
                goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f'Error in second-view grasp generation: {str(e)}')
            result.status = False
            goal_handle.abort()

        return result
    
    def goal_callback(self, goal_request):
        """Accept new goals."""
        self.get_logger().info('Received new grasp generation request.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle cancellation requests."""
        self.get_logger().info('Received cancellation request.')
        return CancelResponse.ACCEPT
    
    def send_grasp_request(self, scene_pc, object_pc):
        """
        Send grasp generation request via Unix socket.
        
        Generates scene and object point clouds from depth and mask,
        then sends them to the grasp server.
        
        Args:
            depth_map: numpy array of float32 depth values (H, W)
            mask: numpy array of uint8 mask values (H, W)
            
        Returns:
            Tuple of (grasp_matrices, grasp_labels), or (None, None) if failed
        """
        if not os.path.exists(self.grasp_socket_path):
            self.get_logger().error(
                f'Socket {self.grasp_socket_path} not found. Is the grasp server running?'
            )
            return None
        
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
        try:
            client.connect(self.grasp_socket_path)
            
            # Protocol:
            # 1. Send Header (scene_pc_size, object_pc_size) - 2 int32 values
            # 2. Send scene_pc (Nx3 float32 data)
            # 3. Send object_pc (Mx3 float32 data)
            
            scene_size = scene_pc.shape[0]
            object_size = object_pc.shape[0]
            
            header = np.array([scene_size, object_size], dtype=np.int32)
            client.sendall(header.tobytes())
            
            # Send scene point cloud
            client.sendall(scene_pc.astype(np.float32).tobytes())
            
            # Send object point cloud
            client.sendall(object_pc.astype(np.float32).tobytes())
            
            # Receive Result (64 bytes for 4x4 matrix)
            # we receive number of grasps (int32) followed by that many 4x4 matrices (float32)
            response = b""
            while len(response) < 4:
                chunk = client.recv(4 - len(response))
                if not chunk:
                    break
                response += chunk
            if len(response) < 4:
                self.get_logger().error(
                    f'Failed to receive grasp count, received {len(response)} bytes'
                )
                return None
            
            grasp_count = int(np.frombuffer(response, dtype=np.int32)[0])
            self.get_logger().info(f'Grasp count received: {grasp_count}')
            
            if grasp_count <= 0:
                self.get_logger().warn('Server returned zero grasps')
                return None, None
            
            grasps = []
            for _ in range(grasp_count):
                response = b""
                while len(response) < 64:
                    chunk = client.recv(64 - len(response))
                    if not chunk:
                        break
                    response += chunk
            
                if len(response) == 64:
                    grasp_matrix = np.frombuffer(response, dtype=np.float32).reshape((4, 4))

                    # Check if the server returned a zero matrix (no valid grasp)
                    if np.all(grasp_matrix == 0):
                        self.get_logger().warn('Server returned no valid grasp (zero matrix)')
                        return None, None

                    self.get_logger().debug(f'\n{grasp_matrix}')
                    grasps.append(grasp_matrix)
                else:
                    self.get_logger().error(
                        f'Received {len(response)} bytes, expected 64'
                    )
                    return None, None

            grasp_labels = []
            for _ in range(grasp_count):
                label_bytes = b""
                while len(label_bytes) < 4:
                    chunk = client.recv(4 - len(label_bytes))
                    if not chunk:
                        break
                    label_bytes += chunk

                if len(label_bytes) == 4:
                    grasp_labels.append(int(np.frombuffer(label_bytes, dtype=np.int32)[0]))
                else:
                    self.get_logger().error(
                        f'Received {len(label_bytes)} bytes for grasp label, expected 4'
                    )
                    return None, None

            return grasps, grasp_labels  # Return grasp matrices and their labels
                
        except Exception as e:
            self.get_logger().error(f'Socket communication error: {str(e)}')
            # exit(1) # debug
            return None, None
        finally:
            client.close()
    
    def matrix_to_pose(self, matrix):
        """
        Convert a 4x4 transformation matrix to a PoseStamped message.
        
        Args:
            matrix: 4x4 numpy array representing the pose
            
        Returns:
            PoseStamped message
        """
        pose_msg = Pose()
        
        # Extract position (translation vector)
        pose_msg.position.x = float(matrix[0, 3])
        pose_msg.position.y = float(matrix[1, 3])
        pose_msg.position.z = float(matrix[2, 3])
        
        # Extract orientation (rotation matrix to quaternion)
        rotation_matrix = matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # Returns [x, y, z, w]
        
        pose_msg.orientation.x = float(quat[0])
        pose_msg.orientation.y = float(quat[1])
        pose_msg.orientation.z = float(quat[2])
        pose_msg.orientation.w = float(quat[3])
        
        return pose_msg


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenerator()
    
    executor = MultiThreadedExecutor()

    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

