# Computes the transform between two cameras based on their transforms to a common base frame.
# Execute easy_handeye2 calibration first to get the calibration files for both cameras, 
# then run this node to compute and publish the cam-to-cam transform as a static TF.

import rclpy
from rclpy.node import Node
import sys
from geometry_msgs.msg import TransformStamped
import argparse

import os
import math
import yaml
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import numpy as np

class CamToCamTransformConfig:
    def __init__(self, calib_file):
        try:
            calib = yaml.safe_load(open(calib_file, 'r'))
        except IOError as e:
            print ('Failed reading calibration parameters from file %s' % calib_file)
            sys.exit(-1)
        self.tf = TransformStamped()
        self.tf.header.frame_id = calib['frame_id']
        self.tf.child_frame_id = calib['child_frame_id']
        self.tf.transform.translation.x = calib['translation']['x']
        self.tf.transform.translation.y = calib['translation']['y']
        self.tf.transform.translation.z = calib['translation']['z']
        self.tf.transform.rotation.x = calib['rotation']['x']
        self.tf.transform.rotation.y = calib['rotation']['y']
        self.tf.transform.rotation.z = calib['rotation']['z']
        self.tf.transform.rotation.w = calib['rotation']['w']

class CamToBaseTransformConfig:
    def __init__(self, calib_file, refined_second=False):
        try:
            calib = yaml.safe_load(open(calib_file, 'r'))
        except IOError as e:
            print ('Failed reading calibration parameters from file %s' % calib_file)
            sys.exit(-1)
        self.tf = TransformStamped()
        if refined_second:
            self.tf.header.frame_id = calib['frame_id']
            self.tf.child_frame_id = calib['child_frame_id']
            self.tf.transform.translation.x = calib['translation']['x']
            self.tf.transform.translation.y = calib['translation']['y']
            self.tf.transform.translation.z = calib['translation']['z']
            self.tf.transform.rotation.x = calib['rotation']['x']
            self.tf.transform.rotation.y = calib['rotation']['y']
            self.tf.transform.rotation.z = calib['rotation']['z']
            self.tf.transform.rotation.w = calib['rotation']['w']
        else:
            self.tf.header.frame_id = calib['parameters']['robot_base_frame']
            self.tf.child_frame_id = calib['parameters']['tracking_base_frame']
            self.tf.transform.translation.x = calib['transform']['translation']['x']
            self.tf.transform.translation.y = calib['transform']['translation']['y']
            self.tf.transform.translation.z = calib['transform']['translation']['z']
            self.tf.transform.rotation.x = calib['transform']['rotation']['x']
            self.tf.transform.rotation.y = calib['transform']['rotation']['y']
            self.tf.transform.rotation.z = calib['transform']['rotation']['z']
            self.tf.transform.rotation.w = calib['transform']['rotation']['w']

class CamToCamTransformBroadcaster(Node):
    def __init__(self, cam_to_cam_cfg: CamToCamTransformConfig):
        super().__init__('cam_transform_broadcaster')
        self.broadcaster = StaticTransformBroadcaster(self)
        timer_period = 1.0  # seconds
        self.cam_to_cam_cfg = cam_to_cam_cfg
        self.timer = self.create_timer(timer_period, self.publish_transform)
                
    def publish_transform(self):
        transform = self.cam_to_cam_cfg.tf
        transform.header.stamp = self.get_clock().now().to_msg()
        self.broadcaster.sendTransform(transform)

class CamToBaseTransformBroadcaster(Node):
    def __init__(self, first_cam_cfg: CamToBaseTransformConfig, second_cam_cfg: CamToBaseTransformConfig):
        super().__init__('cam_transform_broadcaster')
        self.broadcaster = StaticTransformBroadcaster(self)
        timer_period = 0.05  # seconds
        self.first_cam_cfg = first_cam_cfg
        self.second_cam_cfg = second_cam_cfg
        self.timer = self.create_timer(timer_period, self.publish_transform)
                
    def publish_transform(self):
        transform_left = self.first_cam_cfg.tf
        transform_right = self.second_cam_cfg.tf
        stamp = self.get_clock().now().to_msg()
        transform_left.header.stamp = stamp
        transform_right.header.stamp = stamp
        self.broadcaster.sendTransform([transform_left, transform_right])

parser = argparse.ArgumentParser(description='Set the transform between two cameras based on their transforms to a common base frame.')
parser.add_argument('--tf_type', type=str, default='cam-to-base', help='Type of TF to publish: cam-to-cam or cam-to-base. Default: cam-to-base')
parser.add_argument('--first_to_base_calibration', type=str, default='/home/student-team/.ros2/easy_handeye2/calibrations/calibration_right_camera.calib', 
                    help='Calibration file for the first camera (for cam-to-base TF)')
parser.add_argument('--second_to_base_calibration', type=str, default='/home/student-team/.ros2/easy_handeye2/calibrations/calibration_left_camera.calib', 
                    help='Calibration file for the second camera (for cam-to-base TF)')
parser.add_argument('--cam_to_cam_calibration', type=str, default='cam_to_cam.yaml', help='Calibration file for the first camera (for cam-to-cam TF)')
parser.add_argument('--refined_second_to_base', action='store_true', default=False, 
                    help='Use refined calibration for the second camera (for cam-to-base TF)')
parser.add_argument('--refined_second_to_base_calibration', type=str, default='refined_cam_to_base5.yaml', 
                    help='Refined calibration file for the second camera (for cam-to-base TF, optional)')

if __name__ == '__main__':
    args = parser.parse_args()
    
    rclpy.init()
    
    tf_type = args.tf_type
    
    if tf_type == 'cam-to-base':
        first_to_base_calib = args.first_to_base_calibration
        if args.refined_second_to_base:
            second_to_base_calib = args.refined_second_to_base_calibration
        else:
            second_to_base_calib = args.second_to_base_calibration
        first_cam_cfg = CamToBaseTransformConfig(first_to_base_calib)
        second_cam_cfg = CamToBaseTransformConfig(second_to_base_calib, refined_second=args.refined_second_to_base)
        broadcaster = CamToBaseTransformBroadcaster(first_cam_cfg, second_cam_cfg)

    else:
        print('Mode: cam-to-cam.')
        cam_to_cam_calib = args.cam_to_cam_calibration
    
        cam_to_cam_tf = CamToCamTransformConfig(cam_to_cam_calib)
        broadcaster = CamToCamTransformBroadcaster(cam_to_cam_tf)
        
    rclpy.spin(broadcaster)
