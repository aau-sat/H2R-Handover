#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import socket
import numpy as np
import open3d as o3d
from scipy import stats
from visual_properties.msg import Dimensions
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import time

class ObjectDimensionCalculator:
    def __init__(self, intrinsics):
        self.intr = intrinsics

    def get_object_points(self, depth_meters, cv_mask):
        """Deproject object mask to 3D points and remove outliers."""
        h, w = depth_meters.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        obj_px = cv_mask > 0
        z = depth_meters[obj_px]
        u_obj = u[obj_px]
        v_obj = v[obj_px]

        valid = z > 0.1
        z, u_obj, v_obj = z[valid], u_obj[valid], v_obj[valid]
        if len(z) == 0:
            return None

        x = (u_obj - self.intr['cx']) * z / self.intr['fx']
        y = (v_obj - self.intr['cy']) * z / self.intr['fy']
        points = np.stack((x, y, z), axis=-1)
        if len(points) < 50:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_clean = pcd.select_by_index(ind)
        return np.asarray(pcd_clean.points)

    def get_dimensions(self, depth_meters, cv_mask):
        """Returns axis-aligned dimensions (X, Y, Z) in meters."""
        pts = self.get_object_points(depth_meters, cv_mask)
        if pts is None:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        aabb = pcd.get_axis_aligned_bounding_box()
        return aabb.get_max_bound() - aabb.get_min_bound()

    def get_height(self, depth_meters, cv_mask, T_cam_to_base=None):
        """Returns physical object height in meters."""
        pts = self.get_object_points(depth_meters, cv_mask)
        if pts is None:
            return None

        if T_cam_to_base is not None:
            ones = np.ones((len(pts), 1), dtype=pts.dtype)
            pts_h = np.hstack([pts, ones])
            pts_robot = (T_cam_to_base @ pts_h.T).T[:, :3]
            z_vals = pts_robot[:, 2]
        else:
            z_vals = pts[:, 1]

        z_lo = np.percentile(z_vals, 1.0)
        z_hi = np.percentile(z_vals, 99.0)
        return float(z_hi - z_lo)

    def get_width_bb(self, depth_meters, cv_mask,
                     top_frac=0.20, bot_frac=0.10,
                     outlier_std=2.0, min_pts=5):
        """Estimate top and bottom widths from mask bands in metric units."""
        ys, xs = np.where(cv_mask > 0)
        if len(xs) < min_pts:
            return None

        row_min, row_max = int(ys.min()), int(ys.max())
        bb_height = row_max - row_min
        if bb_height < 4:
            return None

        top_row_hi = row_min + max(1, int(np.round(bb_height * top_frac)))
        bot_row_lo = row_max - max(1, int(np.round(bb_height * bot_frac)))

        def _band_width_m(row_lo, row_hi):
            in_band = (ys >= row_lo) & (ys <= row_hi)
            u_band = xs[in_band]
            v_band = ys[in_band]

            if len(u_band) < min_pts:
                return None

            u_mean, u_std = np.mean(u_band), np.std(u_band)
            if u_std > 0:
                keep = np.abs(u_band - u_mean) <= outlier_std * u_std
                u_clean = u_band[keep]
                v_clean = v_band[keep]
            else:
                u_clean = u_band
                v_clean = v_band

            if len(u_clean) < min_pts:
                return None

            u_left = int(u_clean.min())
            u_right = int(u_clean.max())
            z_band = depth_meters[v_clean, u_clean]
            z_valid = z_band[z_band > 0.1]
            if len(z_valid) == 0:
                return None

            z_avg = float(np.median(z_valid))
            x_left = (u_left - self.intr['cx']) * z_avg / self.intr['fx']
            x_right = (u_right - self.intr['cx']) * z_avg / self.intr['fx']
            return float(abs(x_right - x_left))

        w_top = _band_width_m(row_min, top_row_hi)
        w_bot = _band_width_m(bot_row_lo, row_max)
        return {'top_m': w_top, 'bot_m': w_bot}

    def robust_mean(self, values):
        if not values:
            return None
        if len(values) < 3:
            return float(np.mean(values))
        return self._filter_kde_mode(values)

    def _filter_kde_mode(self, values):
        data = np.array(values)
        try:
            kernel = stats.gaussian_kde(data)
            x_test = np.linspace(min(data), max(data), 200)
            y_test = kernel(x_test)
            return float(x_test[np.argmax(y_test)])
        except Exception:
            return float(np.median(data))

    def consolidate_all(self, estimates):
        results = {}
        for key in ['height', 'length', 'width', 'depth', 'top_w', 'bot_w']:
            results[key] = self.robust_mean(estimates[key])
        return results


class DimensionsEstimator(Node):
    def __init__(self):
        super().__init__('dimensions_estimator')
        self.camera_name = self.declare_parameter('camera_name', 'left').get_parameter_value().string_value
        self.num_frames = self.declare_parameter('num_frames', 18).get_parameter_value().integer_value # was 11
        self.publish_mode = self.declare_parameter('publish_mode', 'continuous').get_parameter_value().string_value
        self.publish_period_s = self.declare_parameter('publish_period_s', 0.5).get_parameter_value().double_value
        self.object_label = self.declare_parameter('object_label', 'cup').get_parameter_value().string_value
        self.sam_socket_path = self.declare_parameter('sam_socket_path', '/tmp/sam3_batched_socket').get_parameter_value().string_value
        self.depth_topic = f'/camera/{self.camera_name}/aligned_depth_to_color/image_raw'
        self.color_topic = f'/camera/{self.camera_name}/color/image_raw'
        self.camera_info_topic = f'/camera/{self.camera_name}/aligned_depth_to_color/camera_info'
        # self.use_camera_to_base = self.declare_parameter('use_camera_to_base', False).get_parameter_value().bool_value
        # default_t = [-0.9126294122484981, 0.007181141011057546, 0.3065324849864441]
        # default_q = [-0.31239835432412144, 0.7321115479914871, -0.5496956278095257, 0.2534850415314091]
        # self.cam_to_base_translation = list(self.declare_parameter('cam_to_base_translation', default_t).get_parameter_value().double_array_value)
        # self.cam_to_base_quaternion_xyzw = list(self.declare_parameter('cam_to_base_quaternion_xyzw', default_q).get_parameter_value().double_array_value)

        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Storage for latest images
        self.latest_depth = []
        self.latest_color = []
        self.latest_mask = []
        self.calculator = None
        self.measurement_done = False
        self.estimated_dimensions = None
        self.republish_timer = None
        # self.T_cam_to_base = self._build_cam_to_base_transform()

        self.publish_mode = str(self.publish_mode).strip().lower()
        if self.publish_mode not in ('continuous', 'exit'):
            self.get_logger().warn(
                f"Unknown publish_mode '{self.publish_mode}'. Falling back to 'continuous'."
            )
            self.publish_mode = 'continuous'
        
        # Flag to prevent multiple simultaneous requests
        self.camera_initialized = False
        
        # Subscribers
        self.cam_info_sub = self.create_subscription(
            sensor_msgs.CameraInfo,
            self.camera_info_topic, 
            self.camera_info_callback, 
            10
        )
        
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=input_qos)
        self.color_sub = message_filters.Subscriber(self, Image, self.color_topic, qos_profile=input_qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.color_sub],
            queue_size=30,
            slop=0.01,
        )
        self.ts.registerCallback(self.sync_callback)
        
        # Publisher for dimensions estimation results
        self.dimensions_pub = self.create_publisher(Dimensions, '/estimated_dimensions', 10)
        self.get_logger().info('Dimensions Estimator Node has been started.')

    # def _build_cam_to_base_transform(self):
    #     if len(self.cam_to_base_translation) != 3 or len(self.cam_to_base_quaternion_xyzw) != 4:
    #         self.get_logger().warn('Invalid camera-to-base transform parameters; height will use camera frame.')
    #         return None

    #     try:
    #         rot = Rotation.from_quat(np.array(self.cam_to_base_quaternion_xyzw, dtype=np.float64)).as_matrix()
    #         T = np.eye(4, dtype=np.float64)
    #         T[:3, :3] = rot
    #         T[:3, 3] = np.array(self.cam_to_base_translation, dtype=np.float64)
    #         return T
    #     except Exception as exc:
    #         self.get_logger().warn(f'Failed to parse camera-to-base transform: {exc}')
    #         return None

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        intrinsics = {
            'fx': float(self.fx),
            'fy': float(self.fy),
            'cx': float(self.cx),
            'cy': float(self.cy),
            'width': int(msg.width),
            'height': int(msg.height),
        }
        self.calculator = ObjectDimensionCalculator(intrinsics)
        self.camera_initialized = True
        self.get_logger().info(f"Camera intrinsics received: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        # Unsubscribe after getting intrinsics
        self.destroy_subscription(self.cam_info_sub)

    def _recv_exact(self, client, size):
        data = b""
        while len(data) < size:
            chunk = client.recv(size - len(data))
            if not chunk:
                raise ConnectionError('Connection closed while receiving data from SAM3 service')
            data += chunk
        return data

    def request_batched_masks(self, color_frames):
        if len(color_frames) == 0:
            return None

        h, w, c = color_frames[0].shape
        if c != 3:
            self.get_logger().error(f'Invalid color channels in first frame: expected 3, got {c}')
            return None

        for idx, color in enumerate(color_frames):
            if color.shape != (h, w, 3):
                self.get_logger().error(
                    f'Inconsistent color shape at index {idx}: got {color.shape}, expected {(h, w, 3)}'
                )
                return None

        label_bytes = self.object_label.encode()
        header = np.array([len(color_frames), h, w, len(label_bytes)], dtype=np.int32)

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(self.sam_socket_path)
            client.sendall(header.tobytes())
            client.sendall(label_bytes)
            client.sendall(b"".join(frame.tobytes() for frame in color_frames))

            count_bytes = self._recv_exact(client, 4)
            mask_count = int(np.frombuffer(count_bytes, dtype=np.int32)[0])
            if mask_count != len(color_frames):
                self.get_logger().error(
                    f'Batched SAM3 response count mismatch: expected {len(color_frames)}, got {mask_count}'
                )
                return None

            payload = self._recv_exact(client, mask_count * h * w)
            masks = []
            for i in range(mask_count):
                start = i * h * w
                end = start + h * w
                mask = np.frombuffer(payload[start:end], dtype=np.uint8).reshape((h, w))
                masks.append((mask > 0).astype(np.uint8))

            self.get_logger().info(f'Received {len(masks)} masks from batched SAM3 service.')
            return masks
        except Exception as exc:
            self.get_logger().error(f'Failed batched SAM3 request: {exc}')
            return None
        finally:
            client.close()

    def sync_callback(self, depth_msg, color_msg):
        if self.measurement_done:
            return

        if not self.camera_initialized:
            self.get_logger().warn('Camera intrinsics not yet received, skipping sync callback')
            return
       
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) / 1000.0
            else:
                depth_m = depth.astype(np.float32)

            self.latest_depth.append(depth_m)
            self.latest_color.append(np.ascontiguousarray(color, dtype=np.uint8))
        except Exception as e:
            self.get_logger().error(f'Error converting synchronized images: {str(e)}')
            return
        
        if len(self.latest_depth) >= self.num_frames:
            if self.depth_sub is not None and getattr(self.depth_sub, 'sub', None) is not None:
                self.destroy_subscription(self.depth_sub.sub)
                self.depth_sub = None
            if self.color_sub is not None and getattr(self.color_sub, 'sub', None) is not None:
                self.destroy_subscription(self.color_sub.sub)
                self.color_sub = None

            start_time = time.time()
            masks = self.request_batched_masks(self.latest_color)
            if masks is None or len(masks) != len(self.latest_depth):
                self.get_logger().error('Could not retrieve a valid batched mask set; aborting estimation.')
                return
            self.latest_mask = masks
            if len(self.latest_mask) > 0:
                print(f"Duration for SAM3 batched request: {time.time() - start_time:.2f} seconds")
                cv2.imwrite('debug_latest_mask.png', self.latest_mask[-1] * 255)
            self.estimate_dimensions_and_publish()

    def _republish_estimate(self):
        if self.estimated_dimensions is None:
            return
        self.dimensions_pub.publish(self.estimated_dimensions)

    def estimate_dimensions_and_publish(self):
        if self.calculator is None:
            self.get_logger().error('Calculator is not initialized; camera intrinsics are missing.')
            return

        estimates = {
            'height': [],
            'length': [],
            'width': [],
            'depth': [],
            'top_w': [],
            'bot_w': [],
        }

        for depth_m, mask in zip(self.latest_depth, self.latest_mask):
            obj_pts = self.calculator.get_object_points(depth_m, mask)
            if obj_pts is None:
                continue

            # T = self.T_cam_to_base if self.use_camera_to_base else None
            h_val = self.calculator.get_height(depth_m, mask, T_cam_to_base=None)
            dims = self.calculator.get_dimensions(depth_m, mask)
            prof = self.calculator.get_width_bb(depth_m, mask)

            if h_val is not None:
                estimates['height'].append(h_val)
            if dims is not None:
                estimates['length'].append(float(dims[0]))
                estimates['width'].append(float(dims[1]))
                estimates['depth'].append(float(dims[2]))
            if prof is not None:
                if prof.get('top_m') is not None:
                    estimates['top_w'].append(prof['top_m'])
                if prof.get('bot_m') is not None:
                    estimates['bot_w'].append(prof['bot_m'])

        final_results = self.calculator.consolidate_all(estimates)

        def _safe_val(value):
            return float(value) if value is not None else float('nan')

        self.estimated_dimensions = Dimensions()
        self.estimated_dimensions.height = _safe_val(final_results.get('height'))
        self.estimated_dimensions.length = _safe_val(final_results.get('length'))
        self.estimated_dimensions.width = _safe_val(final_results.get('width'))
        self.estimated_dimensions.depth = _safe_val(final_results.get('depth'))
        self.estimated_dimensions.top_w = _safe_val(final_results.get('top_w'))
        self.estimated_dimensions.bot_w = _safe_val(final_results.get('bot_w'))

        self.dimensions_pub.publish(self.estimated_dimensions)
        self.measurement_done = True

        self.get_logger().info(
            'Published estimated dimensions from '
            f'{len(self.latest_depth)} valid frames: '
            f"height={self.estimated_dimensions.height:.4f}, "
            f"length={self.estimated_dimensions.length:.4f}, "
            f"width={self.estimated_dimensions.width:.4f}, "
            f"depth={self.estimated_dimensions.depth:.4f}, "
            f"top_w={self.estimated_dimensions.top_w:.4f}, "
            f"bot_w={self.estimated_dimensions.bot_w:.4f}"
        )

        if self.publish_mode == 'exit':
            self.get_logger().info('publish_mode=exit: shutting down after first publish.')
            rclpy.shutdown()
            return
        
        if self.depth_sub is not None and getattr(self.depth_sub, 'sub', None) is not None:
            try:
                self.destroy_subscription(self.depth_sub.sub)
            except Exception:
                pass
            self.depth_sub = None

        if self.color_sub is not None and getattr(self.color_sub, 'sub', None) is not None:
            try:
                self.destroy_subscription(self.color_sub.sub)
            except Exception:
                pass
            self.color_sub = None

        if self.republish_timer is None:
            self.republish_timer = self.create_timer(self.publish_period_s, self._republish_estimate)
            self.get_logger().info(
                f'publish_mode=continuous: republishing every {self.publish_period_s:.2f}s.'
            )

def main(args=None):
    rclpy.init(args=args)
    node = DimensionsEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()