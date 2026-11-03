#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import socket
import os
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
from realsense2_camera_msgs.msg import RGBD
import sensor_msgs.msg as sensor_msgs
import cv2
import json

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
        
        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Storage for latest images
        self.latest_color = None
        self.latest_depth = None
        self.latest_rgb = None
        self.latest_mask = None
        self.detected_objects = []
        
        # Flag to prevent multiple simultaneous requests
        self.processing = False
        # Subscribers
        self.cam_info_sub = self.create_subscription(
            sensor_msgs.CameraInfo,
            '/camera/left/color/camera_info', 
            self.camera_info_callback, 
            10
        )
        self.stop_detector_sub = self.create_subscription(
            Bool,
            '/handover_detected',
            self.stop_detector_callback,
            10
        )
        self.obj_detection_sub = self.create_subscription(
            String,
            '/detected_objects',
            self.obj_detection_callback,
            10)
        self.rgbd_sub = self.create_subscription(
            RGBD,
            '/camera/left/rgbd',
            self.rgbd_callback,
            10)
        # Publisher for gripper goal pose
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/gripper_goal_pose',
            10
        )

        self.bridge = CvBridge()
        self.sam_socket_path = "/tmp/sam3_socket"
        self.grasp_socket_path = "/tmp/graspgen"
        
        self.get_logger().info('Grasp Generator Node initialized')
        self.get_logger().info('Waiting for handover detection...')
    
    def camera_info_callback(self, msg):
        self.lfx = msg.k[0]
        self.lfy = msg.k[4]
        self.lcx = msg.k[2]
        self.lcy = msg.k[5]
        self.camera_initialized = True
        self.get_logger().info(f"Camera intrinsics received: fx={self.lfx}, fy={self.lfy}, cx={self.lcx}, cy={self.lcy}")
        # Unsubscribe after getting intrinsics
        self.destroy_subscription(self.cam_info_sub)
    
    def _get_most_confident_detection(self):
        if len(self.detected_objects) == 0:
            return None
        
        most_confident = max(self.detected_objects, key=lambda x: x.get('conf', 0))
        return most_confident

    def obj_detection_callback(self, msg):
        # Here we would parse the message to extract detected objects
        try:
            detections = json.loads(msg.data)
            self.detected_objects = detections
            if len(detections) > 0:
                self.objects_detected = True
                # self.get_logger().info(f"Detected objects: {detections}")
            else:
                self.objects_detected = False
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to decode detection message: {e}")

    def rgbd_callback(self, msg):
        self.latest_color = self.bridge.imgmsg_to_cv2(msg.rgb, "rgb8")
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg.depth, "32FC1") / 1e3

    def request_segmentation_mask(self, color_image, bbox, label):
        h, w, _ = color_image.shape

        label_len = len(label)
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

        self.latest_mask = mask.astype(bool).astype(np.uint8)
        # debug: dump segmentation mask request image
        cv2.imwrite('debug_sam3_request_mask.png', self.latest_mask * 255)
    
    def stop_detector_callback(self, msg):
        """
        Triggered when stop signal is detected.
        Executes grasp generation pipeline.
        """
        if not msg.data:
            return  # Only process when stop signal is detected (True)
        
        if self.processing:
            self.get_logger().warn('Already processing a grasp request, ignoring...')
            return
        
        self.get_logger().info('Stop detector triggered! Generating grasp...')

        latest_detection = self._get_most_confident_detection()
        if latest_detection is None:
            return
        if self.latest_color is None or self.latest_depth is None:
            return
        
        self.processing = True
        label = latest_detection['class']
        bbox = latest_detection['bbox']
        color = self.latest_color
        depth = self.latest_depth
        
        try:
            # Check if we have all required data
            if color is None:
                self.get_logger().error('No color or depth image available')
                return
            self.request_segmentation_mask(color, bbox, label)
            mask = self.latest_mask
            # print mask
            print(mask)
            if mask is None:
                self.get_logger().error('No mask image available')
                return
            
            # Validate dimensions
            if depth.shape != mask.shape:
                self.get_logger().error(
                    f'Dimension mismatch! Depth {depth.shape} vs Mask {mask.shape}'
                )
                return
            
            # Send grasp request via Unix socket
            grasp_matrix = self.send_grasp_request(depth, mask)
            
            if grasp_matrix is not None:
                # Convert to PoseStamped and publish
                pose_msg = self.matrix_to_pose_stamped(grasp_matrix)
                self.pose_pub.publish(pose_msg)
                
                # Log the result
                pos = grasp_matrix[:3, 3]
                self.get_logger().info(
                    f'Grasp pose published - Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]'
                )
            else:
                self.get_logger().warn('No valid grasp found')
                
        except Exception as e:
            self.get_logger().error(f'Error in grasp generation: {str(e)}')
        finally:
            self.processing = False
            exit(0) # debug
    
    def send_grasp_request(self, depth_map, mask):
        """
        Send grasp generation request via Unix socket.
        
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
        
        h, w = depth_map.shape
        depth_dtype_size = depth_map.itemsize  # 4 bytes for float32
        
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
        try:
            client.connect(self.grasp_socket_path)
            
            # 1. Send Header (H, W, depth_item_size) - 12 bytes
            header = np.array([h, w, depth_dtype_size], dtype=np.int32)
            client.sendall(header.tobytes())
            
            # 2. Send Depth Data (float32 raw bytes)
            client.sendall(depth_map.tobytes())
            
            # 3. Send Mask Data (uint8 raw bytes)
            client.sendall(mask.tobytes())
            
            # 4. Receive Result (64 bytes for 4x4 matrix)
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
                return grasp_matrix
            else:
                self.get_logger().error(
                    f'Received {len(response)} bytes, expected 64'
                )
                return None
                
        except Exception as e:
            self.get_logger().error(f'Socket communication error: {str(e)}')
            return None
        finally:
            client.close()
    
    def matrix_to_pose_stamped(self, matrix):
        """
        Convert a 4x4 transformation matrix to a PoseStamped message.
        
        Args:
            matrix: 4x4 numpy array representing the pose
            
        Returns:
            PoseStamped message
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'camera_color_optical_frame'
        
        # Extract position (translation vector)
        pose_msg.pose.position.x = float(matrix[0, 3])
        pose_msg.pose.position.y = float(matrix[1, 3])
        pose_msg.pose.position.z = float(matrix[2, 3])
        
        # Extract orientation (rotation matrix to quaternion)
        rotation_matrix = matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # Returns [x, y, z, w]
        
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        
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
