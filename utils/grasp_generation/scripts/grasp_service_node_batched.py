#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from cv_bridge import CvBridge
import numpy as np
import socket
import os
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
from realsense2_camera_msgs.msg import RGBD
import cv2
import json
import sensor_msgs.msg as sensor_msgs
from typing import Tuple, Dict
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import open3d as o3d
import torch

# Taken from GraspGen
def depth2points(
    depth: np.array,
    fx: int,
    fy: int,
    cx: int,
    cy: int,
    xmap: np.array = None,
    ymap: np.array = None,
    rgb: np.array = None,
    seg: np.array = None,
    mask: np.arange = None,
) -> Dict:
    """Compute point cloud from a depth image.

    Args:
        depth (np.array): 2D depth image.
        fx (int): Focal length x.
        fy (int): Focal length y.
        cx (int): Camera principal point x.
        cy (int): Camera principal point y.
        xmap (np.array, optional): precomputed meshgrid map X. Defaults to None.
        ymap (np.array, optional): precomputed meshgrid map Y. Defaults to None.
        rgb (np.array, optional): RGB imgae for colored point cloud. Defaults to None.
        seg (np.array, optional): Segmentation image corresponding to the rgb image Defaults to None.
        mask (np.arange, optional): Binary mask. 1 to keep. Defaults to None.

    Returns:
        Dict: Output is a dictionary with fields `xyz`, `rgb`, and `index`,
        where `xyz` and `rgb` are Nx3 numpy arrays and `index` is a Nx1 array
        to indicate which point is valid.
    """
    """convert depth image to point cloud"""
    if rgb is not None:
        assert rgb.shape[0] == depth.shape[0] and rgb.shape[1] == depth.shape[1]
    if xmap is not None:
        assert xmap.shape[0] == depth.shape[0] and xmap.shape[1] == depth.shape[1]
    if ymap is not None:
        assert ymap.shape[0] == depth.shape[0] and ymap.shape[1] == depth.shape[1]

    im_height, im_width = depth.shape[0], depth.shape[1]

    if xmap is None or ymap is None:
        ww = np.linspace(0, im_width - 1, im_width)
        hh = np.linspace(0, im_height - 1, im_height)
        xmap, ymap = np.meshgrid(ww, hh)

    pt2 = depth
    pt0 = (xmap - cx) * pt2 / fx
    pt1 = (ymap - cy) * pt2 / fy

    mask_depth = np.ma.getmaskarray(np.ma.masked_greater(pt2, 0))
    if mask is None:
        mask = mask_depth
    else:
        mask_semantic = np.ma.getmaskarray(np.ma.masked_equal(mask, 1))
        mask = mask_depth * mask_semantic

    index = mask.flatten().nonzero()[0]

    pt2_valid = pt2.flatten()[:, np.newaxis].astype(np.float32)
    pt0_valid = pt0.flatten()[:, np.newaxis].astype(np.float32)
    pt1_valid = pt1.flatten()[:, np.newaxis].astype(np.float32)
    pc_xyz = np.concatenate((pt0_valid, pt1_valid, pt2_valid), axis=1)
    if rgb is not None:
        r = rgb[:, :, 0].flatten()[:, np.newaxis]
        g = rgb[:, :, 1].flatten()[:, np.newaxis]
        b = rgb[:, :, 2].flatten()[:, np.newaxis]
        pc_rgb = np.concatenate((r, g, b), axis=1)
    else:
        pc_rgb = None

    if seg is not None:
        pc_seg = seg.flatten()[:, np.newaxis]
    else:
        pc_seg = None

    return {"xyz": pc_xyz, "rgb": pc_rgb, "seg": pc_seg, "index": index}

# Taken from GraspGen
def depth_and_segmentation_to_point_clouds(
    depth_image: np.ndarray,
    segmentation_mask_obj: np.ndarray,
    segmentation_mask_hand: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_image: np.ndarray = None,
    target_object_id: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image and instance segmentation mask to scene and object point clouds.

    Args:
        depth_image: HxW depth image in meters
        segmentation_mask_obj: HxW instance segmentation mask for the target object with integer labels
        segmentation_mask_hand: HxW instance segmentation mask for the hand with integer labels
        fx, fy, cx, cy: Camera intrinsic parameters
        rgb_image: HxWx3 RGB image (optional, for colored point clouds)
        target_object_id: ID of the target object in the segmentation mask
        remove_object_from_scene: If True, removes object points from scene point cloud

    Returns:
        scene_pc: Nx3 point cloud of the hand in the scene (excluding object if remove_object_from_scene=True)
        object_pc: Mx3 point cloud of the target object only

    Raises:
        ValueError: If no target object found or multiple objects detected
    """
    # Check that segmentation mask contains the target object
    unique_ids_obj = np.unique(segmentation_mask_obj)
    unique_ids_hand = np.unique(segmentation_mask_hand)
    if target_object_id not in unique_ids_obj:
        raise ValueError(
            f"Target object ID {target_object_id} not found in segmentation mask. Available IDs: {unique_ids_obj}"
        )
    if target_object_id not in unique_ids_hand:
        raise ValueError(
            f"Target object ID {target_object_id} not found in hand segmentation mask. Available IDs: {unique_ids_hand}"
        )
    
    # Check that only background (0) and one object (target_object_id) are present
    non_background_ids_obj = unique_ids_obj[unique_ids_obj != 0]
    non_background_ids_hand = unique_ids_hand[unique_ids_hand != 0]
    if len(non_background_ids_obj) > 1:
        raise ValueError(
            f"Multiple objects detected in segmentation mask: {non_background_ids_obj}. Please ensure only one object is present."
        )
    if len(non_background_ids_hand) > 1:
        raise ValueError(
            f"Multiple objects detected in hand segmentation mask: {non_background_ids_hand}. Please ensure only one object is present."
        )

    # Convert depth image to point cloud
    pts_data_obj = depth2points(
        depth=depth_image,
        fx=int(fx),
        fy=int(fy),
        cx=int(cx),
        cy=int(cy),
        rgb=rgb_image,
        seg=segmentation_mask_obj,
    )
    pts_data_hand = depth2points(
        depth=depth_image,
        fx=int(fx),
        fy=int(fy),
        cx=int(cx),
        cy=int(cy),
        rgb=rgb_image,
        seg=segmentation_mask_hand,
    )

    # Object point cloud (all valid points)
    xyz = pts_data_obj["xyz"]
    seg = pts_data_obj["seg"]
    index = pts_data_obj["index"]

    # Filter valid points (non-zero depth)
    xyz_valid_obj = xyz[index]
    seg_valid_obj = seg[index] if seg is not None else None
    
    # Hand point cloud (all valid points)
    xyz = pts_data_hand["xyz"]
    seg = pts_data_hand["seg"]
    index = pts_data_hand["index"]
    xyz_valid_hand = xyz[index]
    seg_valid_hand = seg[index] if seg is not None else None

    # Object point cloud (only target object points)
    if seg_valid_obj is not None:
        object_mask = seg_valid_obj.flatten() == target_object_id
        object_pc = xyz_valid_obj[object_mask]
    else:
        raise ValueError("Segmentation data not available from depth2points")
    
    if seg_valid_hand is not None:
        hand_mask = seg_valid_hand.flatten() == target_object_id
        hand_pc = xyz_valid_hand[hand_mask]
    else:
        raise ValueError("Hand segmentation data not available from depth2points")

    if len(object_pc) == 0:
        raise ValueError(f"No points found for target object ID {target_object_id}")
    if len(hand_pc) == 0:
        raise ValueError(f"No points found for target object ID {target_object_id} in hand segmentation")

    print(f"Object point cloud: {len(object_pc)} points")
    print(f"Hand point cloud: {len(hand_pc)} points")
    
    # # export pointclouds for debugging
    # o3d_obj_pc = o3d.geometry.PointCloud()
    # o3d_obj_pc.points = o3d.utility.Vector3dVector(object_pc)
    # o3d.io.write_point_cloud("debug_object_pc.ply", o3d_obj_pc)
    # o3d_hand_pc = o3d.geometry.PointCloud()
    # o3d_hand_pc.points = o3d.utility.Vector3dVector(hand_pc)
    # o3d.io.write_point_cloud("debug_hand_pc.ply", o3d_hand_pc)
    # # merged point cloud
    # merged_pc = o3d.geometry.PointCloud()
    # merged_pc.points = o3d.utility.Vector3dVector(np.concatenate([object_pc, hand_pc], axis=0))
    # o3d.io.write_point_cloud("debug_merged_pc.ply", merged_pc)

    return hand_pc, object_pc

# Taken from GraspGen
def knn_points(X: torch.Tensor, K: int, norm: int):
    """
    Computes the K-nearest neighbors for each point in the point cloud X.

    Args:
        X: (N, 3) tensor representing the point cloud.
        K: Number of nearest neighbors.

    Returns:
        dists: (N, K) tensor containing squared Euclidean distances to the K nearest neighbors.
        idxs: (N, K) tensor containing indices of the K nearest neighbors.
    """
    N, _ = X.shape

    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(X, X, p=norm)  # (N, N)

    # Ignore self-distance (optional, but avoids trivial zero distance)
    self_mask = torch.eye(N, device=X.device, dtype=torch.bool)
    dist_matrix.masked_fill_(self_mask, float("inf"))  # Set self-distances to inf

    # Get the indices of the K-nearest neighbors
    dists, idxs = torch.topk(dist_matrix, K, dim=1, largest=False)

    return dists, idxs

# Taken from GraspGen
def point_cloud_outlier_removal(
    obj_pc: torch.Tensor, threshold: float = 0.009, K: int = 30
) -> torch.Tensor:
    """
    Remove outliers from a point cloud. K-nearest neighbors is used to compute the distance to the nearest neighbor for each point.
    If the distance is greater than a threshold, the point is considered an outlier and removed.

    RANSAC can also be used.

    Args:
        obj_pc (torch.Tensor or np.ndarray): (N, 3) tensor or array representing the point cloud.
        threshold (float): Distance threshold for outlier detection. Points with mean distance to K nearest neighbors greater than this threshold are removed.
        K (int): Number of nearest neighbors to consider for outlier detection.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing filtered and removed point clouds.
    """
    # Convert numpy array to torch tensor if needed
    if isinstance(obj_pc, np.ndarray):
        obj_pc = torch.from_numpy(obj_pc)

    obj_pc = obj_pc.float()
    obj_pc = obj_pc.unsqueeze(0)

    nn_dists, _ = knn_points(obj_pc[0], K=K, norm=1)

    mask = nn_dists.mean(1) < threshold
    filtered_pc = obj_pc[0, mask]
    filtered_pc = filtered_pc.view(-1, 3)

    return filtered_pc

class GraspGeneratorNode(Node):
    """
    ROS2 node that listens for handover detection and generates grasp poses.
    
    Subscribes to:
        - /handover_detected (Bool): Trigger for grasp generation
        - /detected_objects (String): JSON string of detected objects
    
    Publishes to:
        - /gripper_goal_pose (PoseStamped): Computed grasp pose
    """
        
    def __init__(self):
        super().__init__('grasp_generator_node')
        
        self.use_cam_perspective = self.declare_parameter('camera_perspective', False).get_parameter_value().bool_value
        self.camera_name = self.declare_parameter('camera_name', 'left').get_parameter_value().string_value
        
        self.target_frame = 'base'
        self.source_frame_right = f'{self.camera_name}_color_optical_frame'
            
        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Storage for latest images
        self.latest_color_right = None
        self.latest_depth_right = None
        self.right_detected_objects = []
        
        # Flag to prevent multiple simultaneous requests
        self.processing = False
        self.camera_initialized = False
        
        # Subscribers
        self.right_cam_info_sub = self.create_subscription(
            sensor_msgs.CameraInfo,
            f'/camera/{self.camera_name}/color/camera_info', 
            self.right_camera_info_callback, 
            10
        )
        self.stop_detector_sub = self.create_subscription(
            Bool,
            f'/handover_detected',
            self.stop_detector_callback,
            10
        )
        self.right_obj_detection_sub = self.create_subscription(
            String,
            '/detected_objects',
            self.right_obj_detection_callback,
            10)
        self.right_rgbd_sub = self.create_subscription(
            RGBD,
            f'/camera/{self.camera_name}/rgbd',
            self.right_rgbd_callback,
            10)
        # Publisher for gripper goal pose
        self.pose_pub = self.create_publisher(
            PoseArray,
            '/gripper_goal_pose',
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.sam_socket_path = "/tmp/sam3_socket"
        self.grasp_socket_path = "/tmp/graspgen"
        
        self.get_logger().info('Grasp Generator Node initialized')
        self.get_logger().info('Waiting for handover detection...')
    
    def right_camera_info_callback(self, msg):
        self.rfx = msg.k[0]
        self.rfy = msg.k[4]
        self.rcx = msg.k[2]
        self.rcy = msg.k[5]
        self.camera_initialized = True
        self.get_logger().info(f"Camera intrinsics received: fx={self.rfx}, fy={self.rfy}, cx={self.rcx}, cy={self.rcy}")
        # Unsubscribe after getting intrinsics
        self.destroy_subscription(self.right_cam_info_sub)

    def _get_most_confident_detection(self):
        detected_objects = self.right_detected_objects

        if len(detected_objects) == 0:
            return None

        most_confident = max(detected_objects, key=lambda x: x.get('conf', 0))
        return most_confident

    def right_obj_detection_callback(self, msg):
        # Here we would parse the message to extract detected objects
        try:
            detections = json.loads(msg.data)
            self.right_detected_objects = detections
            if len(detections) > 0:
                self.objects_detected = True
                # self.get_logger().info(f"Detected objects: {detections}")
            else:
                self.objects_detected = False
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to decode detection message: {e}")

    def right_rgbd_callback(self, msg):
        self.latest_color_right = self.bridge.imgmsg_to_cv2(msg.rgb, "rgb8")
        self.latest_depth_right = self.bridge.imgmsg_to_cv2(msg.depth, "32FC1") / 1e3

    def request_segmentation_mask(self, color_image, bbox, label):
        h, w, _ = color_image.shape
    
        label_len = len(label)
        if bbox is None:
            x1, y1, x2, y2 = 0, 0, 0, 0
        else:
            x1, y1, x2, y2 = bbox

        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self.sam_socket_path)

            # Send Header: H, W, x1, y1, x2, y2, label_len
            header = np.array([h, w, x1, y1, x2, y2, label_len], dtype=np.int32)
            client.sendall(header.tobytes())
            client.sendall(label.encode())

            # Send Image
            client.sendall(color_image.tobytes())

            # Receive Mask (size is H*W bytes for uint8)
            mask_data = b""
            while len(mask_data) < (h * w):
                mask_data += client.recv((h * w) - len(mask_data))
            
            mask = np.frombuffer(mask_data, dtype=np.uint8).reshape((h, w))
            
            self.get_logger().info("Mask received successfully")

        except Exception as e:
            self.get_logger().error(f"Failed to connect to SAM3 server: {e}")
        finally:
            client.close()

        mask = mask.astype(bool).astype(np.uint8)
        # debug: dump segmentation mask request image
        cv2.imwrite('debug_sam3_request_mask.png', mask * 255)
        return mask
    
    def stop_detector_callback(self, msg):
        """
        Triggered when stop is detected.
        Executes grasp generation pipeline.
        """
        if not msg.data:
            return  # Only process when stop is detected (True)
        
        if self.processing:
            self.get_logger().warn('Already processing a grasp request, ignoring...')
            return
        
        self.get_logger().info('Stop detected! Generating grasp...')

        latest_right_detection = self._get_most_confident_detection()
        if latest_right_detection is None:
            return
        if self.latest_color_right is None or self.latest_depth_right is None:
            return
        
        self.processing = True
        label = latest_right_detection['class']
        bbox_right = latest_right_detection['bbox']
        color_right = self.latest_color_right
        depth_right = self.latest_depth_right

        try:
            # Check if we have all required data
            if color_right is None or depth_right is None:
                self.get_logger().error('No right camera data available')
                return
            
            print('requesting segmentation mask from SAM3...')
            mask_right = self.request_segmentation_mask(color_right, bbox_right, label)
            print('segmentation mask received')
            
            if mask_right is None:
                self.get_logger().error('No mask image available')
                return
            
            # Validate dimensions
            if depth_right.shape != mask_right.shape:
                self.get_logger().error(
                    f'Dimension mismatch! Depth {depth_right.shape} vs Mask {mask_right.shape}'
                )
                return

            print('requesting hand segmentation mask from SAM3...')
            mask_hand_right = self.request_segmentation_mask(color_right, None, "hand")
            print('hand segmentation mask received')

            h, w = depth_right.shape
            depth_right = depth_right.astype(np.float32).reshape((h, w))
            
            try:
                hand_pc_right, object_pc_right = depth_and_segmentation_to_point_clouds(
                    depth_image=depth_right, 
                    segmentation_mask_obj=mask_right,
                    segmentation_mask_hand=mask_hand_right,
                    fx=self.rfx, fy=self.rfy, cx=self.rcx, cy=self.rcy, 
                    target_object_id=1, 
                )
                self.get_logger().info(f'Generated point clouds (right): hand={hand_pc_right.shape}, object={object_pc_right.shape}')
            except Exception as e:
                self.get_logger().error(f'Failed to generate point clouds: {str(e)}')
                return None
            
            object_pc_right_torch = torch.from_numpy(object_pc_right).float()
            object_pc_right = point_cloud_outlier_removal(object_pc_right_torch)
            object_pc_right = object_pc_right.cpu().numpy()
            
            hand_pc_right_torch = torch.from_numpy(hand_pc_right).float()
            hand_pc_right = point_cloud_outlier_removal(hand_pc_right_torch)
            hand_pc_right = hand_pc_right.cpu().numpy()
            
            try:                    
                t_right = self.tf_buffer.lookup_transform(
                    target_frame=self.target_frame,
                    source_frame=self.source_frame_right,
                    time=rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not find transform from {self.source_frame_right} to {self.target_frame}: {ex}')
                return
            
            # Transform scene point cloud
            pcd_hand_to_base = o3d.geometry.PointCloud()
            pcd_hand_to_base.points = o3d.utility.Vector3dVector(hand_pc_right)
            tf_matrix_right = np.eye(4)
            tf_matrix_right[:3, :3] = Rotation.from_quat([
                t_right.transform.rotation.x,
                t_right.transform.rotation.y,
                t_right.transform.rotation.z,
                t_right.transform.rotation.w
            ]).as_matrix()
            tf_matrix_right[:3, 3] = [t_right.transform.translation.x, t_right.transform.translation.y, t_right.transform.translation.z]
            pcd_hand_to_base.transform(tf_matrix_right)
            hand_pc_right = np.asarray(pcd_hand_to_base.points)
            
            # Transform object point cloud
            pcd_obj_to_base = o3d.geometry.PointCloud()
            pcd_obj_to_base.points = o3d.utility.Vector3dVector(object_pc_right)
            pcd_obj_to_base.transform(tf_matrix_right)
            object_pc_right = np.asarray(pcd_obj_to_base.points)
            
            # export pointclouds for debugging
            o3d_obj_pc = o3d.geometry.PointCloud()
            o3d_obj_pc.points = o3d.utility.Vector3dVector(object_pc_right)
            o3d.io.write_point_cloud("debug_object_pc.ply", o3d_obj_pc)
            o3d_hand_pc = o3d.geometry.PointCloud()
            o3d_hand_pc.points = o3d.utility.Vector3dVector(hand_pc_right)
            o3d.io.write_point_cloud("debug_hand_pc.ply", o3d_hand_pc)
            # merged point cloud
            merged_pc = o3d.geometry.PointCloud()
            merged_pc.points = o3d.utility.Vector3dVector(np.concatenate([object_pc_right, hand_pc_right], axis=0))
            o3d.io.write_point_cloud("debug_merged_pc.ply", merged_pc)
            
            # Send grasp request via Unix socket
            grasp_matrices = self.send_grasp_request(hand_pc_right, object_pc_right)
            
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
                        f'Grasp pose published - Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]'
                    )
                self.pose_pub.publish(poses_msg)

            else:
                self.get_logger().warn('No valid grasp found')
                
        except Exception as e:
            self.get_logger().error(f'Error in grasp generation: {str(e)}')
        finally:
            self.processing = False
            exit(0)  # for debug
    
    def send_grasp_request(self, scene_pc, object_pc):
        """
        Send grasp generation request via Unix socket.
        
        Generates scene and object point clouds from depth and mask,
        then sends them to the grasp server.
        
        Args:
            depth_map: numpy array of float32 depth values (H, W)
            mask: numpy array of uint8 mask values (H, W)
            
        Returns:
            4x4 numpy array representing the grasp pose, or None if failed
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
                return None
            
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
                        return None

                    self.get_logger().info('Received grasp matrix from server')
                    self.get_logger().debug(f'\n{grasp_matrix}')
                    grasps.append(grasp_matrix)
                else:
                    self.get_logger().error(
                        f'Received {len(response)} bytes, expected 64'
                    )
                    return None
            return grasps  # Return list of grasp matrices
                
        except Exception as e:
            self.get_logger().error(f'Socket communication error: {str(e)}')
            return None
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
    node = GraspGeneratorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
