#!/usr/bin/env python3
'''
Node subscribes to /estimated_pose and /estimated_pose_segmentation, compares the two poses, and logs the position error and axis errors.
'''

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np

class CompareTracking(Node):
    def __init__(self):
        super().__init__('compare_tracking_node')
        self.bridge = CvBridge()
        self.camera_info = None


        self.pose_foundationpose_sub = message_filters.Subscriber(self, PoseStamped, '/estimated_pose')
        self.pose_segmented_sub = message_filters.Subscriber(self, PoseStamped, '/estimated_pose_segmenation')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.pose_foundationpose_sub, self.pose_segmented_sub],
            queue_size=10,
            slop=0.1 
        )
        self.ts.registerCallback(self.compare_poses)
        self.get_logger().info('Ready. Subscribed to /estimated_pose and /estimated_pose_segmentation...')
        
    def compare_poses(self, pose_foundationpose_msg, pose_segmented_msg):
        try:
            pos_foundation = np.array([
                pose_foundationpose_msg.pose.position.x,
                pose_foundationpose_msg.pose.position.y,
                pose_foundationpose_msg.pose.position.z
            ])
            
            pos_segmented = np.array([
                pose_segmented_msg.pose.position.x,
                pose_segmented_msg.pose.position.y,
                pose_segmented_msg.pose.position.z
            ])

            # Calculate the Euclidean distance (position error) and individual axis errors
            position_error = np.linalg.norm(pos_foundation - pos_segmented)
            error_vector = pos_foundation - pos_segmented

            self.get_logger().info(f'Position error: {position_error:.4f} meters')
            self.get_logger().info(f'Error X: {error_vector[0]:.4f}, Y: {error_vector[1]:.4f}, Z: {error_vector[2]:.4f}')
        except Exception as e:
            self.get_logger().error(f'Error processing pose messages: {e}')

def main(args=None):
    rclpy.init(args=args)
    compare_tracking_node = CompareTracking()
    
    try:
        rclpy.spin(compare_tracking_node)
    except KeyboardInterrupt:
        pass
    finally:
        compare_tracking_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()