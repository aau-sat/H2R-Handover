#!/usr/bin/env python3

'''
Node subscribes to /segmentation_mask and /camera/camera/aligned_depth_to_color/image_raw, estimates the pose of the segmented object, and publishes it to /estimated_pose_segmentation.
'''

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        return np.eye(3, dtype=float)

    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=float)


def rot_to_quat(R: np.ndarray):
    tr = float(np.trace(R))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return qx / n, qy / n, qz / n, qw / n


def transform_to_matrix(t) -> np.ndarray:
    tx = t.translation.x
    ty = t.translation.y
    tz = t.translation.z

    qx = t.rotation.x
    qy = t.rotation.y
    qz = t.rotation.z
    qw = t.rotation.w

    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = quat_to_rot(qx, qy, qz, qw)
    T[0:3, 3] = [tx, ty, tz]
    return T


def pose_to_matrix(p: PoseStamped) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = quat_to_rot(
        p.pose.orientation.x,
        p.pose.orientation.y,
        p.pose.orientation.z,
        p.pose.orientation.w,
    )
    T[0:3, 3] = [
        p.pose.position.x,
        p.pose.position.y,
        p.pose.position.z,
    ]
    return T


def matrix_to_pose_stamped(T: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    qx, qy, qz, qw = rot_to_quat(T[0:3, 0:3])
    msg.pose.position.x = float(T[0, 3])
    msg.pose.position.y = float(T[1, 3])
    msg.pose.position.z = float(T[2, 3])
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg

class PoseEstimator(Node):
    def __init__(self):
        super().__init__('pose_estimator_node')
        self.bridge = CvBridge()
        self.camera_info = None
        
        # Tracking Variables
        self.last_u = None
        self.last_v = None
        
        #Kalman Filter Setup
        self.last_time = None
        
        self.kf = cv2.KalmanFilter(6, 3) # 6 state variables (x, y, z, vx, vy, vz), 3 measurements (x, y, z)
        
        # Measurement Matrix (H): Maps the state to the measurement
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)

        self.kf.transitionMatrix = np.eye(6, dtype=np.float32) #Transition Matrix (F)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2 #Measurement Noise Covariance (R)
        
        self.frames_lost = 0
        self.max_coast_frames = 12 # At 12 FPS, this is 1 second of coasting
        
        self.declare_parameter('camera_name', 'left')
        self.camera_name = self.get_parameter('camera_name').value
        self.declare_parameter('mask_topic', f'/segmentation/{self.camera_name}')
        self.declare_parameter('depth_topic', f'/camera/{self.camera_name}/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', f'/camera/{self.camera_name}/aligned_depth_to_color/camera_info')
        self.declare_parameter('pose_topic', '/estimated_pose')
        self.declare_parameter('target_frame', '')
        self.declare_parameter('tf_static_topic', '/tf_static')
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('debug_status_period_s', 2.0)
        
        # Get parameter values
        self.mask_topic = str(self.get_parameter('mask_topic').value or f'/segmentation/{self.camera_name}')
        self.depth_topic = str(
            self.get_parameter('depth_topic').value
            or f'/camera/{self.camera_name}/aligned_depth_to_color/image_raw'
        )
        self.camera_info_topic = str(
            self.get_parameter('camera_info_topic').value
            or f'/camera/{self.camera_name}/aligned_depth_to_color/camera_info'
        )
        self.pose_topic = str(self.get_parameter('pose_topic').value or '/estimated_pose')
        self.target_frame = str(self.get_parameter('target_frame').value or 'base')
        self.tf_static_topic = str(self.get_parameter('tf_static_topic').value or '/tf_static')
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value or 10)
        self.sync_slop = float(self.get_parameter('sync_slop').value or 0.05)
        self.debug_status_period_s = float(self.get_parameter('debug_status_period_s').value or 2.0)
        
        # Track if first data has been received
        self.camera_info_received = False
        self.mask_received = False
        self.depth_received = False

        # Runtime diagnostics
        self.sync_count = 0
        self.pose_pub_count = 0
        self.init_wait_count = 0
        self.last_sync_count = 0
        self.last_pose_pub_count = 0
        self.static_tf_map = {}
        self.missing_tf_warn_count = 0

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)

        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.info_callback, 10)

        # tf_qos = QoSProfile(
        #     depth=10,
        #     durability=DurabilityPolicy.TRANSIENT_LOCAL,
        #     history=HistoryPolicy.KEEP_LAST,
        #     reliability=ReliabilityPolicy.RELIABLE
        # )
        self.tf_static_sub = self.create_subscription(
            TFMessage,
            self.tf_static_topic,
            self.tf_static_callback,
            10
            # tf_qos
        )

        # create sensor qos profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
             depth=10
        )

        self.mask_sub = message_filters.Subscriber(self, Image, self.mask_topic, qos_profile=qos_profile)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.mask_sub, self.depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop
        )
        self.ts.registerCallback(self.image_callback)
        # self.debug_timer = self.create_timer(self.debug_status_period_s, self.debug_status_callback)
        self.get_logger().info(
            f'Ready with topics: mask={self.mask_topic}, depth={self.depth_topic}, '
            f'camera_info={self.camera_info_topic}, pose_out={self.pose_topic}'
        )
        self.get_logger().info(
            f'Pose publisher active on topic: {self.pose_topic}'
        )
        self.get_logger().info(
            f'Synchronizer config: queue_size={self.sync_queue_size}, slop={self.sync_slop:.3f}s, '
            f'debug_status_period={self.debug_status_period_s:.1f}s'
        )
        if self.target_frame:
            self.get_logger().info(
                f'Transform enabled: source=<depth_msg.frame_id> -> target={self.target_frame}, '
                f'tf_topic={self.tf_static_topic}'
            )
        else:
            self.get_logger().info('Transform disabled: publishing raw camera-frame pose')

    def tf_static_callback(self, msg):
        for t in msg.transforms:
            parent = t.header.frame_id
            child = t.child_frame_id
            self.static_tf_map[(parent, child)] = transform_to_matrix(t.transform)

    def get_transform_target_from_source(self, target_frame, source_frame):
        if (target_frame, source_frame) in self.static_tf_map:
            return self.static_tf_map[(target_frame, source_frame)]

        if (source_frame, target_frame) in self.static_tf_map:
            return np.linalg.inv(self.static_tf_map[(source_frame, target_frame)])

        return None

    def debug_status_callback(self):
        sync_delta = self.sync_count - self.last_sync_count
        pub_delta = self.pose_pub_count - self.last_pose_pub_count
        self.last_sync_count = self.sync_count
        self.last_pose_pub_count = self.pose_pub_count

        self.get_logger().info(
            f'Status: sync={self.sync_count} (+{sync_delta}), pub={self.pose_pub_count} (+{pub_delta}), '
            f'init_wait={self.init_wait_count}, frames_lost={self.frames_lost}'
        )

        if self.sync_count > 0 and sync_delta == 0:
            self.get_logger().warn(
                'No synchronized mask-depth pairs in last period. '
                'Try larger sync_slop (e.g. 0.03-0.08) or check timestamp alignment.',
                throttle_duration_sec=5.0
            )

    def info_callback(self, msg):
        self.camera_info = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5]
        }
        if not self.camera_info_received:
            self.get_logger().info(f'Camera info received from {self.camera_info_topic}')
            self.camera_info_received = True
            self.destroy_subscription(self.info_sub) # Unsubscribe after getting intrinsics

    def image_callback(self, mask_msg, depth_msg):
        self.sync_count += 1
        if self.sync_count == 1 or self.sync_count % 60 == 0:
            dt_sync = (
                abs(mask_msg.header.stamp.sec - depth_msg.header.stamp.sec)
                + abs(mask_msg.header.stamp.nanosec - depth_msg.header.stamp.nanosec) * 1e-9
            )
            self.get_logger().info(
                f'Sync callback: count={self.sync_count}, dt(mask-depth)~{dt_sync:.6f}s, '
                f'mask={mask_msg.header.stamp.sec}.{mask_msg.header.stamp.nanosec:09d}, '
                f'depth={depth_msg.header.stamp.sec}.{depth_msg.header.stamp.nanosec:09d}'
            )

        if not self.mask_received or not self.depth_received:
            self.get_logger().info('Synchronized mask and depth data received')
            self.mask_received = True
            self.depth_received = True
        
        if self.camera_info is None:
            self.get_logger().warn('Waiting for camera intrinsics...', throttle_duration_sec=2.0)
            return

        try:
            cv_mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='passthrough')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            binary_mask = cv_mask.astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Variables to hold extracted measurement
            valid_measurement = False
            x_meters, y_meters, z_meters = 0.0, 0.0, 0.0
            u, v = 0, 0

            # Try to extract a valid measurement from the image
            if contours:
                if self.last_u is None or self.last_v is None:
                    ref_u = binary_mask.shape[1] // 2
                    ref_v = binary_mask.shape[0] // 2
                else:
                    ref_u = self.last_u
                    ref_v = self.last_v

                closest_contour = None
                min_dist = float('inf')
                best_M = None

                for contour in contours:
                    if cv2.contourArea(contour) < 50:
                        continue
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cu = int(M["m10"] / M["m00"])
                    cv = int(M["m01"] / M["m00"])
                    dist = (cu - ref_u)**2 + (cv - ref_v)**2
                    if dist < min_dist:
                        min_dist = dist
                        closest_contour = contour
                        best_M = M

                if closest_contour is not None and best_M is not None:
                    u = int(best_M["m10"] / best_M["m00"])
                    v = int(best_M["m01"] / best_M["m00"])

                    clean_mask = np.zeros_like(binary_mask)
                    cv2.drawContours(clean_mask, [closest_contour], -1, 255, thickness=cv2.FILLED)

                    kernel = np.ones((5, 5), np.uint8)
                    inner_mask = cv2.erode(clean_mask, kernel, iterations=2)

                    valid_depth_mask = (inner_mask > 0) & (cv_depth > 0) & np.isfinite(cv_depth)
                    segmented_depths = cv_depth[valid_depth_mask]

                    if len(segmented_depths) > 0:
                        z_meters = float(np.median(segmented_depths)) / 1000.0 
                        
                        fx, fy = self.camera_info['fx'], self.camera_info['fy']
                        cx, cy = self.camera_info['cx'], self.camera_info['cy']

                        x_meters = ((u - cx) * z_meters) / fx
                        y_meters = ((v - cy) * z_meters) / fy
                        
                        valid_measurement = True
                        self.last_u = u
                        self.last_v = v

            # Kalman Filter Process
            current_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
            
            # If KF is completely uninitialized, we MUST wait for a valid measurement to start
            if self.last_time is None:
                if not valid_measurement:
                    self.init_wait_count += 1
                    if self.init_wait_count == 1 or self.init_wait_count % 30 == 0:
                        self.get_logger().info(
                            f'Waiting for first valid measurement: contours={len(contours)}, '
                            f'last_centroid=({self.last_u},{self.last_v})'
                        )
                    return # Can't initialize without knowing where the object is
                
                self.last_time = current_time
                self.kf.statePost = np.array([[x_meters], [y_meters], [z_meters], [0], [0], [0]], np.float32)
                self.kf.statePre = self.kf.statePost
                smoothed_x, smoothed_y, smoothed_z = x_meters, y_meters, z_meters
            
            # KF is already running
            else:
                dt = current_time - self.last_time
                self.last_time = current_time
                
                # Update Transition Matrix with current dt
                self.kf.transitionMatrix[0, 3] = dt
                self.kf.transitionMatrix[1, 4] = dt
                self.kf.transitionMatrix[2, 5] = dt
                
                # Predict where the object is
                prediction = self.kf.predict()
                
                # Check for extreme outliers (teleportation)
                if valid_measurement:
                    meas_pos = np.array([x_meters, y_meters, z_meters])
                    pred_pos = np.array([prediction[0, 0], prediction[1, 0], prediction[2, 0]])
                    if np.linalg.norm(pred_pos - meas_pos) > 0.5:
                        self.get_logger().warn('Measurement outlier detected. Ignoring camera data.')
                        valid_measurement = False

                if valid_measurement:
                    # Update with real data
                    measurement = np.array([[x_meters], [y_meters], [z_meters]], np.float32)
                    estimated_state = self.kf.correct(measurement)
                    
                    smoothed_x = float(estimated_state[0, 0])
                    smoothed_y = float(estimated_state[1, 0])
                    smoothed_z = float(estimated_state[2, 0])
                    self.frames_lost = 0
                else:
                    # Missing or rejected data: Apply Coasting and Damping
                    self.frames_lost += 1
                    
                    if self.frames_lost > self.max_coast_frames:
                        self.get_logger().warn('Object lost for too long. Resetting tracker.')
                        self.last_time = None 
                        self.frames_lost = 0
                        self.last_u = None
                        self.last_v = None
                        return # Stop publishing until found again
                    
                    # Damping factor: reduce velocity by 20% every frame we coast
                    self.kf.statePost[3, 0] *= 0.8
                    self.kf.statePost[4, 0] *= 0.8
                    self.kf.statePost[5, 0] *= 0.8
                    
                    smoothed_x = float(self.kf.statePost[0, 0])
                    smoothed_y = float(self.kf.statePost[1, 0])
                    smoothed_z = float(self.kf.statePost[2, 0])

            pose_msg = PoseStamped()
            pose_msg.header.stamp = depth_msg.header.stamp
            pose_msg.header.frame_id = depth_msg.header.frame_id

            pose_msg.pose.position.x = smoothed_x
            pose_msg.pose.position.y = smoothed_y
            pose_msg.pose.position.z = smoothed_z

            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            pose_msg.pose.orientation.w = 1.0

            if self.target_frame:
                source_frame = pose_msg.header.frame_id
                T_target_source = self.get_transform_target_from_source(self.target_frame, source_frame)
                if T_target_source is None:
                    self.missing_tf_warn_count += 1
                    if self.missing_tf_warn_count == 1 or self.missing_tf_warn_count % 30 == 0:
                        self.get_logger().warn(
                            f'Missing static transform for {self.target_frame} <-> {source_frame}. '
                            f'Waiting for TF on {self.tf_static_topic}.'
                        )
                    return

                T_pose_source = pose_to_matrix(pose_msg)
                T_pose_target = T_target_source @ T_pose_source
                pose_msg = matrix_to_pose_stamped(
                    T_pose_target,
                    self.target_frame,
                    pose_msg.header.stamp
                )

            self.pose_pub.publish(pose_msg)
            self.pose_pub_count += 1
            self.get_logger().debug(
                f'Published pose to {self.pose_topic}: '
                f'x={smoothed_x:.4f}, y={smoothed_y:.4f}, z={smoothed_z:.4f}, '
                f'frame={pose_msg.header.frame_id}, '
                f'stamp={pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec:09d}, '
                f'valid_measurement={valid_measurement}, centroid_u={u}, centroid_v={v}'
            )

            # # Visualization 
            # depth_vis = cv2.normalize(cv_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            # if valid_measurement:
            #     cv2.circle(depth_vis, (u, v), 5, (0, 255, 0), -1) 
            # cv2.imshow("Tracked Object Centroid", depth_vis)
            # cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
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
    
