#!/usr/bin/env python3

"""
Node subscribes to /segmentation_mask and /camera/camera/aligned_depth_to_color/image_raw, 
estimates the pose of the segmented object, and publishes it asynchronously to /estimated_pose.
Follows ROS2 best practices for Linux Ubuntu.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
import threading


class PoseEstimator(Node):
    """
    ROS2 Node for estimating and publishing the 3D pose of a segmented object.
    
    Uses asynchronous publishing via a timer to decouple camera frame rates 
    from pose publishing rates. Background thread continuously updates the 
    Kalman Filter state to prevent stale data.
    """

    def __init__(self):
        """
        Initializes the PoseEstimator node, sets up publishers, subscribers, 
        timers, and the Kalman Filter.
        """
        super().__init__('pose_estimator_node')
        self.bridge = CvBridge()
        self.camera_info = None
        
        # Thread safety for asynchronous publishing
        self.state_lock = threading.Lock()
        
        # Tracking Variables
        self.last_u = None
        self.last_v = None
        self.current_frame_id = "camera_link"
        
        # Kalman Filter Setup
        self.last_time = None
        self.last_time_timer = None
        
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)

        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
        
        
        
        
        self.frames_lost = 0
        self.max_coast_frames = 12
        
        # Reinit and Interpolation
        self.reinit_cache = {}
        self.interpolation_active = False
        self.interpolation_frames = []
        self.interpolation_frame_idx = 0
        self.last_kf_position = np.array([0.0, 0.0, 0.0])
        self.current_published_pos = np.array([0.0, 0.0, 0.0])
        
        # Parameters (Type-safe extraction for ROS2 Foxy/Humble/Iron)
        self.declare_parameter('mask_topic', '/segmentation/left')
        self.declare_parameter('depth_topic', '/camera/left/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/left/aligned_depth_to_color/camera_info')
        self.declare_parameter('segmentation_reinit_topic', '/segmentation_mask_reinit')
        self.declare_parameter('pose_topic', '/estimated_pose')
        self.declare_parameter('interpolation_num_points', 10)
        self.declare_parameter('publish_rate_hz', 30.0)
        
        self.mask_topic = self.get_parameter('mask_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.segmentation_reinit_topic = self.get_parameter('segmentation_reinit_topic').get_parameter_value().string_value
        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.interpolation_num_points = self.get_parameter('interpolation_num_points').get_parameter_value().integer_value
        self.publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        
        # Flags
        self.camera_info_received = False
        self.mask_received = False
        self.depth_received = False
        self.is_initialized = False

        # Publishers & Subscribers
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.info_callback, 30)

        # QoS for sensor data on Linux Ubuntu
        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.mask_sub = message_filters.Subscriber(self, Image, self.mask_topic, qos_profile=qos_profile)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.mask_sub, self.depth_sub], queue_size=30, slop=0.01)
        self.ts.registerCallback(self.image_callback)
        
        self.reinit_sub = self.create_subscription(Header, self.segmentation_reinit_topic, self.reinit_callback, 10)
        
        # Setup high-frequency publishing timer
        timer_period = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(timer_period, self.timer_publish_callback)


    def info_callback(self, msg: CameraInfo) -> None:
        """
        Callback for camera intrinsics.

        Args:
            msg (CameraInfo): Camera information message containing intrinsics matrix.
        """
        self.camera_info = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}
        if not self.camera_info_received:
            self.camera_info_received = True
            self.destroy_subscription(self.info_sub)


    def reinit_callback(self, msg: Header) -> None:
        """
        Callback to cache re-initialization events.

        Args:
            msg (Header): Header message containing timestamp of the reinit event.
        """
        timestamp_key = (msg.stamp.sec, msg.stamp.nanosec)
        self.reinit_cache[timestamp_key] = msg


    def compute_smooth_interpolation(self, start_pos: np.ndarray, end_pos: np.ndarray, num_points: int) -> list:
        """
        Computes C2-continuous smooth interpolation using cubic splines.

        Args:
            start_pos (np.ndarray): Starting 3D position [x, y, z].
            end_pos (np.ndarray): Ending 3D position [x, y, z].
            num_points (int): Number of points to generate.

        Returns:
            list: List of interpolated np.ndarray positions.
        """
        if num_points < 2:
            num_points = 2
        t = np.linspace(0, 1, num_points)
        x_spline = CubicSpline([0, 1], [start_pos[0], end_pos[0]], bc_type='natural')
        y_spline = CubicSpline([0, 1], [start_pos[1], end_pos[1]], bc_type='natural')
        z_spline = CubicSpline([0, 1], [start_pos[2], end_pos[2]], bc_type='natural')
        return [np.array([x_spline(t_val), y_spline(t_val), z_spline(t_val)]) for t_val in t]


    def start_interpolation(self, start_pos: np.ndarray, end_pos: np.ndarray) -> None:
        """
        Initiates the interpolation sequence.

        Args:
            start_pos (np.ndarray): The starting coordinates.
            end_pos (np.ndarray): The target coordinates.
        """
        self.interpolation_frames = self.compute_smooth_interpolation(start_pos, end_pos, self.interpolation_num_points)
        self.interpolation_frame_idx = 0
        self.interpolation_active = True


    def reset_interpolation(self) -> None:
        """
        Resets the interpolation state machine to inactive.
        """
        self.interpolation_active = False
        self.interpolation_frames = []
        self.interpolation_frame_idx = 0


    def timer_publish_callback(self) -> None:
        """
        Publishes the pose continuously using Kalman filter predictions.

        Calculates the time delta since the last timer trigger, advances the 
        Kalman filter state using the physical velocity model, and publishes 
        the predicted pose. This ensures a smooth output rate even if 
        camera frames drop on Linux Ubuntu.
        """
        current_time_sec = self.get_clock().now().nanoseconds / 1e9

        with self.state_lock:
            if not self.is_initialized:
                self.last_time_timer = current_time_sec
                return

            dt = current_time_sec - self.last_time_timer
            self.last_time_timer = current_time_sec

            self.kf.transitionMatrix[0, 3] = dt
            self.kf.transitionMatrix[1, 4] = dt
            self.kf.transitionMatrix[2, 5] = dt

            # Predict internal state continuously
            prediction = self.kf.predict()
            publish_pos = np.array([float(prediction[0, 0]), float(prediction[1, 0]), float(prediction[2, 0])])

            # Override with interpolation if active
            if self.interpolation_active and self.interpolation_frames:
                if self.interpolation_frame_idx < len(self.interpolation_frames):
                    publish_pos = self.interpolation_frames[self.interpolation_frame_idx]
                    self.interpolation_frame_idx += 1
                else:
                    self.reset_interpolation()
            
            # Coasting behavior
            if self.frames_lost > 0 and not self.interpolation_active:
                self.kf.statePost[3:6, 0] *= 0.95

            self.current_published_pos = publish_pos

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.current_frame_id

            pose_msg.pose.position.x = publish_pos[0]
            pose_msg.pose.position.y = publish_pos[1]
            pose_msg.pose.position.z = publish_pos[2]
            pose_msg.pose.orientation.w = 1.0

            self.pose_pub.publish(pose_msg)


    def image_callback(self, mask_msg: Image, depth_msg: Image) -> None:
        """
        Synchronized callback for mask and depth images to correct the KF.

        Processes incoming frames to extract the object's measurement and 
        corrects the Kalman filter state in the background. Does not publish.

        Args:
            mask_msg (Image): The segmentation mask.
            depth_msg (Image): The aligned depth image.
        """
        if self.camera_info is None:
            return

        try:
            timestamp_key = (mask_msg.header.stamp.sec, mask_msg.header.stamp.nanosec)
            has_reinit = timestamp_key in self.reinit_cache
            
            if has_reinit:
                del self.reinit_cache[timestamp_key] # Consume event to prevent replays
            
            # Unconditional cache cleanup to prevent indefinite leaking
            if len(self.reinit_cache) > 100:
                oldest_keys = sorted(self.reinit_cache.keys())[:-50]
                for k in oldest_keys: 
                    del self.reinit_cache[k]

            cv_mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='passthrough')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            binary_mask = cv_mask.astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_measurement = False
            x_meters, y_meters, z_meters, u, v = 0.0, 0.0, 0.0, 0, 0

            if contours:
                ref_u = self.last_u if self.last_u is not None else binary_mask.shape[1] // 2
                ref_v = self.last_v if self.last_v is not None else binary_mask.shape[0] // 2
                
                closest_contour, min_dist, best_M = None, float('inf'), None
                for contour in contours:
                    if cv2.contourArea(contour) < 50: continue
                    M = cv2.moments(contour)
                    if M["m00"] == 0: continue
                    cu, cv = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    dist = (cu - ref_u)**2 + (cv - ref_v)**2
                    if dist < min_dist:
                        min_dist, closest_contour, best_M = dist, contour, M

                # Safe explicit checking for best_M
                if closest_contour is not None and best_M is not None:
                    u, v = int(best_M["m10"] / best_M["m00"]), int(best_M["m01"] / best_M["m00"])
                    clean_mask = np.zeros_like(binary_mask)
                    cv2.drawContours(clean_mask, [closest_contour], -1, 255, thickness=cv2.FILLED)
                    inner_mask = cv2.erode(clean_mask, np.ones((5, 5), np.uint8), iterations=2)
                    valid_depth_mask = (inner_mask > 0) & (cv_depth > 0) & np.isfinite(cv_depth)
                    segmented_depths = cv_depth[valid_depth_mask]

                    if len(segmented_depths) > 0:
                        z_meters = float(np.median(segmented_depths)) / 1000.0 
                        x_meters = ((u - self.camera_info['cx']) * z_meters) / self.camera_info['fx']
                        y_meters = ((v - self.camera_info['cy']) * z_meters) / self.camera_info['fy']
                        valid_measurement = True
                        self.last_u, self.last_v = u, v

            with self.state_lock:
                self.current_frame_id = depth_msg.header.frame_id
                
                if not self.is_initialized:
                    if not valid_measurement: return
                    self.kf.statePost = np.array([[x_meters], [y_meters], [z_meters], [0], [0], [0]], np.float32)
                    self.kf.statePre = self.kf.statePost
                    self.last_kf_position = np.array([x_meters, y_meters, z_meters])
                    self.current_published_pos = self.last_kf_position.copy()
                    self.is_initialized = True
                    self.frames_lost = 0
                else:
                    if valid_measurement:
                        measurement = np.array([[x_meters], [y_meters], [z_meters]], np.float32)
                        estimated_state = self.kf.correct(measurement)
                        smoothed_pos = np.array([float(estimated_state[0, 0]), float(estimated_state[1, 0]), float(estimated_state[2, 0])])
                        
                        if has_reinit:
                            # Start interpolation from the visual position to the new corrected truth
                            self.start_interpolation(self.current_published_pos, smoothed_pos)
                            
                        # Continually update background truth, ignoring interpolation state
                        self.last_kf_position = smoothed_pos
                        self.frames_lost = 0
                    else:
                        self.frames_lost += 1
                        if self.frames_lost > self.max_coast_frames:
                            self.is_initialized = False
                            self.reset_interpolation()

            # Visualization
            depth_vis = cv2.normalize(cv_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            if valid_measurement:
                cv2.circle(depth_vis, (u, v), 5, (0, 255, 0), -1) 
            cv2.imshow("Tracked Object Centroid", depth_vis)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")


def main(args=None):
    """Entry point for the ROS2 node."""
    rclpy.init(args=args)
    node = PoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()