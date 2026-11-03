# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import argparse
import ast
from scipy.spatial.transform import Rotation as R
import os
import numpy as np
import json

from std_msgs.msg import Bool
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
from std_msgs.msg import String
import time

# RGBD Message
# std_msgs/Header header
# sensor_msgs/CameraInfo rgb_camera_info
# sensor_msgs/CameraInfo depth_camera_info
# sensor_msgs/Image rgb
# sensor_msgs/Image depth

class GraspingCoordinator(Node):
    def __init__(self, frame_buffer_size=1):
        super().__init__('grasping_coordinator')
        self.obj_detection_sub = self.create_subscription(
            String,
            '/detected_objects',
            self.obj_detection_callback,
            10)
        # realsense2_camera_msgs/msg/RGBD
        self.rgbd_sub = self.create_subscription(
            RGBD,
            '/camera/camera/rgbd',
            self.rgbd_callback,
            10)
        # subscribe to the trigger
        self.trigger_sub = self.create_subscription(
            Bool,
            '/object_ready_for_grasp',
            self.trigger_callback,
            10)
        self.objects_detected = False
        self.color_file = None  # Placeholder for single color image
        self.depth_file = None  # Placeholder for single depth image
        self.mask = None   # Placeholder for mask image file paths
        self.bridge = CvBridge()
        self.got_mask = False
        self.frame_buffer_size = frame_buffer_size
        self.mesh_file = None
        self.detected_objects = []
        self.detection_start_request = False
        self.object_stopped = False

        self.sam3_socket_path = "/tmp/sam3_socket"
        self.graspgen_socket_path = "/tmp/graspgen"

        self.get_logger().info("GraspingCoordinator Node started.")


    
    def prepare_for_tracking(self):
        if not self.got_mask:
            self.prepare_tracking_assets() 

    def any_objects_detected(self):
        return self.objects_detected

    def _get_most_confident_detection(self):
        if not self.detected_objects:
            return None
        
        most_confident = max(self.detected_objects, key=lambda x: x.get('conf', 0))
        return most_confident


    def request_segmentation_mask(self):
        self.got_mask = False
        try: 
            cv_image = self.color_file
            h, w, _ = cv_image.shape
        except Exception as e:
            self.get_logger().error(f"Failed to get color image for segmentation request: {e}")
            return

        detection = self._get_most_confident_detection()
        label = detection['class']
        label_len = len(label)
        x1, y1, x2, y2 = detection['bbox']

        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self.sam3_socket_path)

            # Send Header: H, W, x1, y1, x2, y2, label_len
            header = np.array([h, w, x1, y1, x2, y2, label_len], dtype=np.int32)
            client.sendall(header.tobytes())
            client.sendall(label.encode())

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

        self.mask = mask.astype(bool).astype(np.uint8)
        self.got_mask = True
        # debug: dump segmentation mask request image
        cv2.imwrite('sam3_request_mask.png', self.mask * 255)



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
        self.color_file = self.bridge.imgmsg_to_cv2(msg.rgb, "rgb8")
        self.depth_file = self.bridge.imgmsg_to_cv2(msg.depth, "32FC1") / 1e3

    def trigger_callback(self, msg):
        if msg.data:
            self.object_stopped = True
            self.get_logger().info("Received trigger: Object is ready for grasping.")

    def get_mask_status(self):
        return self.got_mask

    def get_mask(self):
        return self.mask

    def get_intrinsics(self):
        return self.K
    
    def get_color(self):
        return self.color_file
    
    def get_depth(self):
        return self.depth_file
    
    def get_mesh_file(self):
        return self.mesh_file

    
def main():
    rclpy.init()

    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    #parser.add_argument('--mesh_file', type=str, default=None)

    args = parser.parse_args()

    coordinator = GraspingCoordinator()
    while True:
        # first observe if object is stopped and ready for grasping
        while rclpy.ok():
            rclpy.spin_once(coordinator)
            if coordinator.object_stopped:
                print("Object ready for grasping, requested segmentation mask...")
                break


        # observe object detections and request segmentation mask from SAM3 server when detections are available
        while rclpy.ok():
            rclpy.spin_once(coordinator)
            if coordinator.any_objects_detected():
                
                coordinator.request_segmentation_mask()
                print("Object detected, proceeding segmenting...")
                break
        
        
    
        
        coordinator.object_stopped = False  # Reset for next round of detection

        # coordinator.destroy_node()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()
