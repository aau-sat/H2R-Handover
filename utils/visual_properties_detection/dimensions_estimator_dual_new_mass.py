#!/usr/bin/env python3

from cProfile import label
import json

from matplotlib import image

import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import message_filters
import socket
import numpy as np
import open3d as o3d
from scipy import stats
from scipy.spatial.transform import Rotation
from visual_properties.msg import Dimensions
from segmented_detection.msg import SegmentedObject
from tf2_msgs.msg import TFMessage
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import time
import json
import threading
import base64
import io
import sys
from pathlib import Path
import trimesh

try:
	from utils.vlm.client import infer_answer_from_base64
except ModuleNotFoundError:
	sys.path.append(str(Path(__file__).resolve().parents[1]))
	from vlm.client import infer_answer_from_base64

try:
	from utils.config import object_priors, symmetric_container_priors, asymmetric_container_priors, other_priors, vlm_prompt, left_color_camera_topic_suffix, right_color_camera_topic_suffix
except ModuleNotFoundError:
	sys.path.append(str(Path(__file__).resolve().parents[2]))
	from utils.config import object_priors, symmetric_container_priors, asymmetric_container_priors, other_priors, vlm_prompt, left_color_camera_topic_suffix, right_color_camera_topic_suffix

# ---------------------------------------------------------------------------
# ObjectDimensionCalculator
# ---------------------------------------------------------------------------

class ObjectDimensionCalculator:
	def __init__(self, intrinsics):
		self.intr = intrinsics
		# Precompute pixel coordinate grids to avoid allocation overhead on every frame
		self.u, self.v = np.meshgrid(
			np.arange(intrinsics['width']),
			np.arange(intrinsics['height'])
		)

	# ------------------------------------------------------------------
	# Internal helper: deproject + mask-filter + outlier removal
	# ------------------------------------------------------------------
	def get_object_points(self, depth_meters, cv_mask):
		"""Deproject object mask to 3D points and remove outliers."""
		obj_px = cv_mask > 0
		z = depth_meters[obj_px]
		u_obj = self.u[obj_px]
		v_obj = self.v[obj_px]

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
		pcd = pcd.voxel_down_sample(voxel_size=0.003)
		_, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
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

	def get_height(self, depth_meters, cv_mask, T_cam_to_base=None, return_details=False):
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

		z_lo, z_hi = np.percentile(z_vals, [1.0, 99.0])
		height = float(z_hi - z_lo)
		return (height, z_lo, z_hi) if return_details else height

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


	# ------------------------------------------------------------------
	# Statistics helpers
	# ------------------------------------------------------------------
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
		for key in ['height', 'length', 'width', 'depth', 'top_w', 'bot_w', 'volume']:
			if key in estimates:
				results[key] = self.robust_mean(estimates[key])
		return results

	def apply_dbscan(self, points, eps=0.02, min_points=10):
		if points is None or len(points) < min_points:
			return points
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points)
		labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
		if len(labels) == 0:
			return points
		valid_labels = labels[labels >= 0]
		if len(valid_labels) == 0:
			return points
		counts = np.bincount(valid_labels)
		largest_cluster_idx = np.argmax(counts)
		indices = np.where(labels == largest_cluster_idx)[0]
		return points[indices]

	def get_fused_box_dimensions(self, left_pts, right_pts, min_pts=50, logger=None):
		if left_pts is None and right_pts is None:
			return None
			
		if left_pts is not None and right_pts is not None:
			fused_pts = np.vstack([left_pts, right_pts])
		elif left_pts is not None:
			fused_pts = left_pts
		else:
			fused_pts = right_pts
			
		if len(fused_pts) < min_pts:
			return None
			
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(fused_pts)
		_, ind = pcd.remove_statistical_outlier(nb_neighbors=12, std_ratio=1.5)
		clean_fused = np.asarray(pcd.select_by_index(ind).points)
		
		if len(clean_fused) < min_pts:
			return None
			
		pcd_clean = o3d.geometry.PointCloud()
		pcd_clean.points = o3d.utility.Vector3dVector(clean_fused)
		try:
			obb = pcd_clean.get_minimal_oriented_bounding_box()
		except AttributeError:
			obb = pcd_clean.get_oriented_bounding_box()
			
		extents = np.asarray(obb.extent)
		R = np.asarray(obb.R)
		
		# Vertical axis in robot base frame is Z [0, 0, 1]
		z_alignment = np.abs(R[2, :])
		h_idx = int(np.argmax(z_alignment))
		others = [i for i in range(3) if i != h_idx]
		
		# Longest horizontal axis defines the orientation yaw
		major_h_idx = others[int(np.argmax(extents[others]))]
		h_axis_robot = R[:, major_h_idx]
		h_axis_xy = np.array([h_axis_robot[0], h_axis_robot[1]], dtype=float)
		norm_xy = float(np.linalg.norm(h_axis_xy))
		if norm_xy > 1e-8:
			h_axis_xy /= norm_xy
			yaw_deg = float(np.degrees(np.arctan2(h_axis_xy[1], h_axis_xy[0])))
			yaw_deg = float(((yaw_deg + 90.0) % 180.0) - 90.0)
		else:
			yaw_deg = 0.0
			
		res = {
			'width_m':  float(extents[others[0]]),
			'depth_m':  float(extents[others[1]]),
			'height_m': float(extents[h_idx]),
			'yaw_deg_robot': yaw_deg,
		}
		if logger is not None:
			logger.info(f'[box] Fused robot-frame OBB extents: {extents}, yaw: {yaw_deg:.2f}')
		return res
# ---------------------------------------------------------------------------
# ROS Node
# ---------------------------------------------------------------------------

class DimensionsEstimator(Node):
	def __init__(self):
		super().__init__('dimensions_estimator')

		# --- Parameters ---
		self.num_frames = self.declare_parameter('num_frames', 10).get_parameter_value().integer_value
		self.publish_mode = self.declare_parameter('publish_mode', 'continuous').get_parameter_value().string_value
		self.publish_period_s = self.declare_parameter('publish_period_s', 0.5).get_parameter_value().double_value
		self.object_label = None
		self.sam_socket_path = self.declare_parameter('sam_socket_path', '/tmp/sam3_socket').get_parameter_value().string_value
		self.declare_parameter('label_topic', '/detected_label')
		self.declare_parameter('bbox_topic', '/object_detections')

		self.prior_labels = object_priors
		self.vlm_prompt = vlm_prompt
		self.vlm_label = None

		self.sam_retry = 0
		self._label_wait_logged = False
		self.first_image_base64 = None	
		self.bbox = None

		self.camera_side = self.declare_parameter('camera_side', 'left').get_parameter_value().string_value
		if self.camera_side not in ['left', 'right']:
			self.get_logger().warn(f"Invalid camera_side '{self.camera_side}', defaulting to 'left'")
			self.camera_side = 'left'

		self.other_side = 'right' if self.camera_side == 'left' else 'left'

		# --- Topic names ---
		self.depth_topic = self.declare_parameter(
			'depth_topic', f'/camera/{self.camera_side}/aligned_depth_to_color/image_raw'
		).get_parameter_value().string_value
		self.color_topic = self.declare_parameter(
			'color_topic', f'/camera/{self.camera_side}/color/{left_color_camera_topic_suffix if self.camera_side == "left" else right_color_camera_topic_suffix}'
		).get_parameter_value().string_value
		self.cam_info_topic = self.declare_parameter(
			'cam_info_topic', f'/camera/{self.camera_side}/aligned_depth_to_color/camera_info'
		).get_parameter_value().string_value

		depth_topic_sec = f'/camera/{self.other_side}/aligned_depth_to_color/image_raw'
		color_topic_sec = f'/camera/{self.other_side}/color/{right_color_camera_topic_suffix if self.camera_side == "left" else left_color_camera_topic_suffix}'
		cam_info_topic_sec = f'/camera/{self.other_side}/aligned_depth_to_color/camera_info'

		if self.camera_side == 'left':
			self.depth_topic_left = self.depth_topic
			self.color_topic_left = self.color_topic
			self.cam_info_topic_left = self.cam_info_topic
			
			self.depth_topic_right = depth_topic_sec
			self.color_topic_right = color_topic_sec
			self.cam_info_topic_right = cam_info_topic_sec
		else:
			self.depth_topic_right = self.depth_topic
			self.color_topic_right = self.color_topic
			self.cam_info_topic_right = self.cam_info_topic
			
			self.depth_topic_left = depth_topic_sec
			self.color_topic_left = color_topic_sec
			self.cam_info_topic_left = cam_info_topic_sec

		# --- State ---
		self.bridge = CvBridge()
		self.measurement_done = False
		self.estimated_dimensions = None
		self.vlm_mass_estimate = None
		self.republish_timer = None

		# Camera buffers
		self._cam = {
			'left': {
				'depth': [],
				'color': None,
				'mask': None,
				'calc': None,
				'initialized': False,
				'mask_thread': None,
				'vlm_thread': None,
				'frame_id': None,
				'tf': None
			},
			'right': {
				'depth': [],
				'color': None,
				'mask': None,
				'calc': None,
				'initialized': False,
				'mask_thread': None,
				'vlm_thread': None,
				'frame_id': None,
				'tf': None
			}
		}

		self.publish_mode = str(self.publish_mode).strip().lower()
		if self.publish_mode not in ('continuous', 'exit'):
			self.get_logger().warn(
				f"Unknown publish_mode '{self.publish_mode}'. Falling back to 'continuous'."
			)
			self.publish_mode = 'continuous'

		# --- QoS ---
		input_qos = QoSProfile(
			history=HistoryPolicy.KEEP_LAST,
			depth=10,
			reliability=ReliabilityPolicy.BEST_EFFORT,
			durability=DurabilityPolicy.VOLATILE,
		)

		# --- Static Transforms Buffer ---
		self.static_transforms = {}
		self.tf_static_sub = self.create_subscription(
			TFMessage,
			'/tf_static',
			self.tf_static_callback,
			QoSProfile(
				depth=100,
				durability=DurabilityPolicy.TRANSIENT_LOCAL,
				reliability=ReliabilityPolicy.RELIABLE,
				history=HistoryPolicy.KEEP_LAST
			)
		)

		# --- Camera-info subscribers (one-shot) ---
		self._cam_info_sub_left = self.create_subscription(
			sensor_msgs.CameraInfo,
			self.cam_info_topic_left,
			lambda msg: self._camera_info_callback(msg, 'left'),
			10,
		)
		self._cam_info_sub_right = self.create_subscription(
			sensor_msgs.CameraInfo,
			self.cam_info_topic_right,
			lambda msg: self._camera_info_callback(msg, 'right'),
			10,
		)

		# topics from yoloe node
		self.label_topic = self.get_parameter('label_topic').get_parameter_value().string_value
		self.bbox_topic = self.get_parameter('bbox_topic').get_parameter_value().string_value

		# subscription for yoloe node topics
		self.sub_label = self.create_subscription(
			String,
			self.label_topic,
			self.label_topic_callback,
			10
		)
  
		self.sub_bbox = self.create_subscription(
			String,
			self.bbox_topic,
			self.bbox_topic_callback,
			10
		)

		# --- Synchronized depth+color subscribers ---
		self._depth_sub_left = message_filters.Subscriber(
			self, Image, self.depth_topic_left, qos_profile=input_qos)
		self._color_sub_left = message_filters.Subscriber(
			self, Image, self.color_topic_left, qos_profile=input_qos)
		self._ts_left = message_filters.ApproximateTimeSynchronizer(
			[self._depth_sub_left, self._color_sub_left],
			queue_size=30, slop=0.01,
		)
		self._ts_left.registerCallback(lambda d, c: self._sync_callback(d, c, 'left'))

		self._depth_sub_right = message_filters.Subscriber(
			self, Image, self.depth_topic_right, qos_profile=input_qos)
		self._color_sub_right = message_filters.Subscriber(
			self, Image, self.color_topic_right, qos_profile=input_qos)
		self._ts_right = message_filters.ApproximateTimeSynchronizer(
			[self._depth_sub_right, self._color_sub_right],
			queue_size=30, slop=0.01,
		)
		self._ts_right.registerCallback(lambda d, c: self._sync_callback(d, c, 'right'))

		# --- Publisher ---
		self.dimensions_pub = self.create_publisher(Dimensions, '/estimated_dimensions', 10)
		self.segmented_pub = self.create_publisher(SegmentedObject, '/segmented_object', 10)
		self.get_logger().info(f'Dimensions Estimator Node has been started (dual view mode, primary side: {self.camera_side}).')

	def tf_static_callback(self, msg):
		for t in msg.transforms:
			child = t.child_frame_id.lstrip('/')
			self.static_transforms[child] = t
			self.get_logger().info(f"Buffered static transform for child frame: {child}")
			
			for side in ['left', 'right']:
				frame = self._cam[side].get('frame_id')
				if frame and frame.lstrip('/') == child:
					self._cam[side]['tf'] = t
					self.get_logger().info(f"Resolved static transform for {side} ({child})")

	def _camera_info_callback(self, msg, side):
		if self._cam[side]['initialized']:
			return
		intrinsics = {
			'fx': float(msg.k[0]),
			'fy': float(msg.k[4]),
			'cx': float(msg.k[2]),
			'cy': float(msg.k[5]),
			'width': int(msg.width),
			'height': int(msg.height),
		}
		self._cam[side]['calc'] = ObjectDimensionCalculator(intrinsics)
		self._cam[side]['frame_id'] = msg.header.frame_id
		self._cam[side]['initialized'] = True
		self.get_logger().info(
			f"{side.upper()} Camera intrinsics received: "
			f"fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
			f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}, "
			f"frame_id={msg.header.frame_id}"
		)
		
		# Resolve transform if already buffered
		child_norm = msg.header.frame_id.lstrip('/')
		if child_norm in self.static_transforms:
			self._cam[side]['tf'] = self.static_transforms[child_norm]
			self.get_logger().info(f"Resolved buffered transform for {side} ({child_norm})")

		if side == 'left':
			self.destroy_subscription(self._cam_info_sub_left)
		else:
			self.destroy_subscription(self._cam_info_sub_right)

	def _recv_exact(self, client, size):
		data = b""
		while len(data) < size:
			chunk = client.recv(size - len(data))
			if not chunk:
				raise ConnectionError('Connection closed while receiving data from SAM3 service')
			data += chunk
		return data

	def request_single_mask(self, color_image, label, bbox=None):
		if color_image is None:
			return None, False

		h, w, c = color_image.shape
		if c != 3:
			self.get_logger().error(f'Invalid color channels: expected 3, got {c}')
			return None, False

		label_bytes = f"single:{label}".encode()
  
		if bbox is not None:
			x1, y1, x2, y2 = bbox
		else:
			x1, y1, x2, y2 = 0, 0, 0, 0
      
		header = np.array([h, w, x1, y1, x2, y2, len(label_bytes)], dtype=np.int32)

		client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		try:
			client.connect(self.sam_socket_path)
			client.sendall(header.tobytes())
			client.sendall(label_bytes)
			client.sendall(np.ascontiguousarray(color_image, dtype=np.uint8).tobytes())

			object_ok_bytes = self._recv_exact(client, 4)
			object_ok = bool(np.frombuffer(object_ok_bytes, dtype=np.uint32)[0])

			size_bytes = self._recv_exact(client, 4)
			payload_size = int(np.frombuffer(size_bytes, dtype=np.uint32)[0])
			mask_data = self._recv_exact(client, payload_size)
			object_mask = np.frombuffer(mask_data, dtype=np.uint8).reshape((h, w))
			object_mask = (object_mask > 0).astype(np.uint8)

			self.get_logger().info(f'SAM3 mask received: object_ok={object_ok}')
			return object_mask, object_ok
		except Exception as exc:
			self.get_logger().error(f'SAM3 request failed: {exc}')
			return None, False
		finally:
			client.close()

	def _notify_grasp_generator(self, side):
		segmented_msg = SegmentedObject()
		segmented_msg.view = side
		segmented_msg.segmentation_label = self.object_label
		cam = self._cam[side]
		try:
			mask_image = (cam['mask'] * 255).astype(np.uint8)
			segmented_msg.segmentation_mask = self.bridge.cv2_to_imgmsg(mask_image, encoding='mono8')
		except Exception as e:
			self.get_logger().error(f'Failed to convert mask to ROS Image for {side}: {e}')
			empty_mask = np.zeros((cam['mask'].shape[0], cam['mask'].shape[1]), dtype=np.uint8)
			segmented_msg.segmentation_mask = self.bridge.cv2_to_imgmsg(empty_mask, encoding='mono8')
   
		self.segmented_pub.publish(segmented_msg)

	def _sync_callback(self, depth_msg, color_msg, side):
		if self.measurement_done:
			return

		cam = self._cam[side]
		if not cam['initialized']:
			return

		try:
			depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
			if cam['color'] is None:
				cam['color'] = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
			depth_m = depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 \
				else depth.astype(np.float32)
		except Exception as e:
			self.get_logger().error(f'Image conversion error on {side}: {e}')
			return

		cam['depth'].append(depth_m)
		color = cam['color']

		if side == self.camera_side and self.first_image_base64 is None:
			color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
			self.first_image_base64 = cv2.imencode('.png', color_bgr)[1].tobytes()
			self.first_image_base64 = base64.b64encode(self.first_image_base64).decode('ascii')
   
		if side == self.camera_side and cam['vlm_thread'] is None:
			self.get_logger().info(f'Spawning background thread for VLM label inference on {side}...')
			cam['vlm_thread'] = threading.Thread(
				target=self._infer_vlm_label_async,
				args=(self.first_image_base64,),
			)
			cam['vlm_thread'].start()
   
		if self.object_label is None:
			if not self._label_wait_logged:
				self.get_logger().info('Waiting for object label from YOLOE...')
				self._label_wait_logged = True
			return
		self._label_wait_logged = False

		if cam['mask'] is None and cam['mask_thread'] is None:
			color_copy = color.copy()
			label_copy = self.object_label
			bbox_side = None
			if self.bbox is not None:
				try:
					bbox_dict = json.loads(self.bbox)
					bbox_side = bbox_dict.get(f'{side}_bbox')
				except Exception:
					pass
			print(f'Initiating background SAM3 mask request for {side} with label: {label_copy}')
			cam['mask_thread'] = threading.Thread(
				target=self._fetch_mask_async,
				args=(side, color_copy, label_copy, bbox_side),
			)
			cam['mask_thread'].start()

		if len(cam['depth']) >= self.num_frames:
			self._destroy_cam_subscribers(side)
			
			if cam['mask_thread'] is not None and cam['mask_thread'].is_alive():
				self.get_logger().info(f'Waiting for background SAM3 mask generation on {side}...')
				cam['mask_thread'].join()

			old_label = self.object_label
			if cam['mask'] is None: 
				primary_cam = self._cam[self.camera_side]
				if primary_cam.get('vlm_thread') is not None and primary_cam['vlm_thread'].is_alive():
					primary_cam['vlm_thread'].join()
				if self.vlm_label is not None:
					self.get_logger().info(f'Attempting SAM3 mask request for {side} with VLM label: {self.vlm_label}')
					bbox_side = None
					if self.bbox is not None:
						try:
							bbox_dict = json.loads(self.bbox)
							bbox_side = bbox_dict.get(f'{side}_bbox')
						except Exception:
							pass
					cam['mask'], successfull = self.request_single_mask(color, self.vlm_label, bbox_side)
					if successfull:
						self.object_label = self.vlm_label
						self.get_logger().info(f'SAM3 mask obtained successfully for {side} with VLM label.')
					else:
						self.get_logger().warning(f'SAM3 mask request for {side} with VLM label failed.')
						cam['mask'] = None
						
						prior_reinitialized = False
						for prior in symmetric_container_priors:
							if prior in self.vlm_label:
								self.prior_labels = symmetric_container_priors
								prior_reinitialized = True
								break
						if not prior_reinitialized:
							for prior in asymmetric_container_priors:
								if prior in self.vlm_label:
									self.prior_labels = asymmetric_container_priors
									prior_reinitialized = True
									break
						if not prior_reinitialized:
							for prior in other_priors:
								if prior in self.vlm_label:
									self.prior_labels = other_priors
						if old_label in self.prior_labels:
							self.prior_labels.remove(old_label)
      
			while cam['mask'] is None and self.sam_retry < len(self.prior_labels):
				successfull = False
				prior_label = self.prior_labels[self.sam_retry]
				self.get_logger().warning(
					f'{side} mask failed. Retrying with prior label #{self.sam_retry}: {prior_label}'
				)
				bbox_side = None
				if self.bbox is not None:
					try:
						bbox_dict = json.loads(self.bbox)
						bbox_side = bbox_dict.get(f'{side}_bbox')
					except Exception:
						pass
				cam['mask'], successfull = self.request_single_mask(color, prior_label, bbox_side)
				if successfull:
					self.object_label = prior_label
				else:
					self.get_logger().warning(f'SAM3 mask request for {side} with prior label {prior_label} failed.')
					cam['mask'] = None
				
				if successfull:
					self.get_logger().info(f'{side} mask retrieved on attempt {self.sam_retry + 1} with label: {self.object_label}')
					break
				
				if not successfull:
					self.sam_retry += 1

			if self.object_label is None or cam['mask'] is None:
				self.get_logger().error(
					f'Could not retrieve object mask for {side} after {self.sam_retry} attempts. Aborting.')
				return

			self._check_and_estimate()

	def _infer_vlm_label_async(self, image_base64):
		self.get_logger().info('Starting VLM label inference in the background...')
		try:
			label = infer_answer_from_base64(image_base64, self.vlm_prompt)
			self.vlm_label = label.strip().lower() if label is not None else None
			self.get_logger().info(f'Background VLM label inference completed: {self.vlm_label}')
		except Exception as e:
			self.get_logger().error(f'Background VLM label inference failed: {e}')
			self.vlm_label = None

	def _infer_vlm_mass(self):
		"""Synchronous VLM query to estimate object mass from the first image.
		Stores textual response in `self.vlm_mass_estimate` and logs the result.
		"""
		if self.first_image_base64 is None:
			self.get_logger().warning('No image available for VLM mass inference.')
			self.vlm_mass_estimate = None
			return
		self.get_logger().info('Requesting VLM mass estimate (synchronous)...')
		mass_prompt = (
			self.vlm_prompt + '\nPlease estimate the mass of the main object in the image in kilograms. '
			"Provide a concise answer (e.g. '0.5 kg' or '0.4-0.6 kg')."
		)
		try:
			answer = infer_answer_from_base64(self.first_image_base64, mass_prompt)
			self.vlm_mass_estimate = answer.strip() if answer is not None else None
			self.get_logger().info(f'VLM mass estimate: {self.vlm_mass_estimate}')
		except Exception as e:
			self.get_logger().error(f'VLM mass inference failed: {e}')
			self.vlm_mass_estimate = None

	def _fetch_mask_async(self, side, color_image, label_snapshot, bbox=None):
		self.get_logger().info(f'Requesting SAM3 mask in the background for {side}...')
		mask, successfull = self.request_single_mask(color_image, label_snapshot, bbox)
		if successfull:
			self._cam[side]['mask'] = mask
			self.get_logger().info(f'Background SAM3 mask for {side} received successfully.')
		else:
			self._cam[side]['mask'] = None
			self.get_logger().error(f'Background SAM3 mask request for {side} failed.')

	def _destroy_cam_subscribers(self, side):
		if side == 'left':
			subs = [self._depth_sub_left, self._color_sub_left]
			self._depth_sub_left = None
			self._color_sub_left = None
		else:
			subs = [self._depth_sub_right, self._color_sub_right]
			self._depth_sub_right = None
			self._color_sub_right = None
			
		for sub in subs:
			if sub and getattr(sub, 'sub', None):
				try:
					self.destroy_subscription(sub.sub)
				except Exception:
					pass
			
	def label_topic_callback(self, msg):
		self.label_only = msg.data if msg.data is not None else "cup"
		self.object_label = self.label_only
		self._label_wait_logged = False
  
	def bbox_topic_callback(self, msg):
		self.get_logger().info(f"Received bbox update: {msg.data}")
		if msg.data is None:
			return
		self.bbox = msg.data
		try:
			self.destroy_subscription(self.sub_bbox)
		except Exception:
			pass
	
	def _check_and_estimate(self):
		left_ready = (len(self._cam['left']['depth']) >= self.num_frames and 
					  self._cam['left']['mask'] is not None)
		right_ready = (len(self._cam['right']['depth']) >= self.num_frames and 
					   self._cam['right']['mask'] is not None)

		if left_ready and right_ready and not self.measurement_done:
			# Check if static transforms are resolved
			tf_ready = (self._cam['left']['tf'] is not None and 
						self._cam['right']['tf'] is not None)
			if not tf_ready:
				if not hasattr(self, '_tf_wait_logged') or not self._tf_wait_logged:
					self.get_logger().info("Waiting for static transforms from /tf_static...")
					self._tf_wait_logged = True
				return
			self._tf_wait_logged = False

			self._notify_grasp_generator('left')
			self._notify_grasp_generator('right')
			# Ask VLM for a mass estimate (synchronous, simple direct query)
			try:
				self._infer_vlm_mass()
			except Exception as e:
				self.get_logger().error(f'VLM mass inference failed: {e}')
			self.estimate_dimensions_and_publish()
			self.report_data()

	def _average_depth(self, depth_list):
		if not depth_list:
			return None
		depth_arr = np.stack(depth_list, axis=0).astype(np.float32)
		depth_arr[depth_arr == 0] = np.nan
		with np.errstate(all='ignore'):
			avg_d = np.nan_to_num(np.nanmean(depth_arr, axis=0))
		return avg_d

	def _transform_points_to_base(self, points, tf_msg):
		if points is None or len(points) == 0 or tf_msg is None:
			return points
		T = np.eye(4)
		T[:3, :3] = Rotation.from_quat([
			tf_msg.transform.rotation.x,
			tf_msg.transform.rotation.y,
			tf_msg.transform.rotation.z,
			tf_msg.transform.rotation.w,
		]).as_matrix()
		T[:3, 3] = [
			tf_msg.transform.translation.x,
			tf_msg.transform.translation.y,
			tf_msg.transform.translation.z,
		]
		ones = np.ones((len(points), 1), dtype=points.dtype)
		points_h = np.hstack([points, ones])
		points_base = (T @ points_h.T).T[:, :3]
		return points_base

	def estimate_dimensions_and_publish(self):
		self.measurement_done = True
		self.start_time_dim = self.get_clock().now()
		
		label = str(self.object_label).lower()
		is_symmetric = any(tag in label for tag in symmetric_container_priors + other_priors)
		
		consolidated = {'yaw_deg_camera': 0.0}

		if not is_symmetric:
			self.get_logger().info(f"Using BOX estimation strategy (fused OBB in robot frame) for label '{self.object_label}'")
			
			left_depths = self._cam['left']['depth']
			left_mask = self._cam['left']['mask']
			left_avg_d = self._average_depth(left_depths)
			
			right_depths = self._cam['right']['depth']
			right_mask = self._cam['right']['mask']
			right_avg_d = self._average_depth(right_depths)
			
			left_pts = self._cam['left']['calc'].get_object_points(left_avg_d, left_mask) if left_mask is not None else None
			right_pts = self._cam['right']['calc'].get_object_points(right_avg_d, right_mask) if right_mask is not None else None
			
			if left_pts is not None and len(left_pts) > 50:
				left_pts = self._cam['left']['calc'].apply_dbscan(left_pts)
			if right_pts is not None and len(right_pts) > 50:
				right_pts = self._cam['right']['calc'].apply_dbscan(right_pts)
				
			left_pts_base = self._transform_points_to_base(left_pts, self._cam['left']['tf'])
			right_pts_base = self._transform_points_to_base(right_pts, self._cam['right']['tf'])
			
			# Call OBB estimation from ObjectDimensionCalculator class
			box_res = self._cam[self.camera_side]['calc'].get_fused_box_dimensions(
				left_pts_base, right_pts_base, logger=self.get_logger()
			)
			
			if box_res:
				consolidated = {
					'width':  box_res['width_m'],
					'depth':  box_res['depth_m'],
					'height': box_res['height_m'],
					'length': max(box_res['width_m'], box_res['depth_m']),
					'top_w':  box_res['width_m'],
					'bot_w':  box_res['width_m'],
					'yaw_deg_camera': box_res['yaw_deg_robot'],
				}
			else:
				self.get_logger().warn("Dual-view OBB estimation failed, falling back to symmetric method")
				is_symmetric = True

		if is_symmetric:
			self.get_logger().info(f"Using SYMMETRIC estimation strategy (averaged dual view) for label '{self.object_label}'")
			left_avg_d = self._average_depth(self._cam['left']['depth'])
			left_mask = self._cam['left']['mask']
			left_calc = self._cam['left']['calc']
			
			left_h = left_calc.get_height(left_avg_d, left_mask) if left_mask is not None else None
			left_dims = left_calc.get_dimensions(left_avg_d, left_mask) if left_mask is not None else None
			left_prof = left_calc.get_width_bb(left_avg_d, left_mask) if left_mask is not None else None
			
			right_avg_d = self._average_depth(self._cam['right']['depth'])
			right_mask = self._cam['right']['mask']
			right_calc = self._cam['right']['calc']
			
			right_h = right_calc.get_height(right_avg_d, right_mask) if right_mask is not None else None
			right_dims = right_calc.get_dimensions(right_avg_d, right_mask) if right_mask is not None else None
			right_prof = right_calc.get_width_bb(right_avg_d, right_mask) if right_mask is not None else None
			
			h_vals = [h for h in [left_h, right_h] if h is not None]
			h_final = float(np.mean(h_vals)) if h_vals else 0.0
			
			l_vals = [float(dims[0]) for dims in [left_dims, right_dims] if dims is not None]
			w_vals = [float(dims[1]) for dims in [left_dims, right_dims] if dims is not None]
			d_vals = [float(dims[2]) for dims in [left_dims, right_dims] if dims is not None]
			
			l_final = float(np.mean(l_vals)) if l_vals else 0.0
			w_final = float(np.mean(w_vals)) if w_vals else 0.0
			d_final = float(np.mean(d_vals)) if d_vals else 0.0
			
			top_w_vals = []
			bot_w_vals = []
			if left_prof:
				if left_prof.get('top_m') is not None:
					top_w_vals.append(left_prof['top_m'])
				if left_prof.get('bot_m') is not None:
					bot_w_vals.append(left_prof['bot_m'])
			if right_prof:
				if right_prof.get('top_m') is not None:
					top_w_vals.append(right_prof['top_m'])
				if right_prof.get('bot_m') is not None:
					bot_w_vals.append(right_prof['bot_m'])
					
			top_w_final = float(np.mean(top_w_vals)) if top_w_vals else 0.0
			bot_w_final = float(np.mean(bot_w_vals)) if bot_w_vals else 0.0
			
			consolidated = {
				'height': h_final,
				'length': l_final,
				'width': w_final,
				'depth': d_final,
				'top_w': top_w_final,
				'bot_w': bot_w_final,
				'yaw_deg_camera': 0.0
			}

		if consolidated is None:
			self.get_logger().warn('Could not consolidate dimension estimates; skipping publish.')
			return

		self.end_time_dim = self.get_clock().now()
		
		def _safe(v):
			return float(v) if v is not None else float('nan')

		self.estimated_dimensions = Dimensions()
		self.estimated_dimensions.height  = _safe(consolidated.get('height'))
		self.estimated_dimensions.top_w   = _safe(consolidated.get('top_w'))
		self.estimated_dimensions.bot_w   = _safe(consolidated.get('bot_w'))
		self.estimated_dimensions.yaw_deg_camera = _safe(consolidated.get('yaw_deg_camera', 0.0))

		self.dimensions_pub.publish(self.estimated_dimensions)

		if self.publish_mode == 'exit':
			self.get_logger().info('publish_mode=exit: shutting down.')
			rclpy.shutdown()
			return

		if self.republish_timer is None:
			self.republish_timer = self.create_timer(
				self.publish_period_s, self._republish_estimate)
			self.get_logger().info(
				f'publish_mode=continuous: republishing every {self.publish_period_s:.2f} s.'
			)

	def _republish_estimate(self):
		if self.estimated_dimensions is not None:
			self.dimensions_pub.publish(self.estimated_dimensions)

	def report_data(self):
		dim_mass_data = {
			"width_top_est_mm_vision": self.estimated_dimensions.top_w * 1000.0,
			"width_bottom_est_mm_vision": self.estimated_dimensions.bot_w * 1000.0,
			"height_est_mm_vision": self.estimated_dimensions.height * 1000.0,
			"yaw_deg_camera_vision": self.estimated_dimensions.yaw_deg_camera,
			"geometry_est_timepoint": (self.end_time_dim - self.start_time_dim).nanoseconds / 1e9
		}
		self.get_logger().get_child("SUBMISSION").info(f"{json.dumps(dim_mass_data)}")

		self.get_logger().info(
			f'Published estimated dimensions | '
			f"height={self.estimated_dimensions.height:.4f} m, "
			f"top_w={self.estimated_dimensions.top_w:.4f} m, "
			f"bot_w={self.estimated_dimensions.bot_w:.4f} m, "
			f"yaw_deg_camera={self.estimated_dimensions.yaw_deg_camera:.2f}"
		)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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