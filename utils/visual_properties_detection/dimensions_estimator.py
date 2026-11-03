#!/usr/bin/env python3

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
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import time
import json
from ultralytics import YOLO
from collections import Counter

# ---------------------------------------------------------------------------
# ObjectDimensionCalculator
# ---------------------------------------------------------------------------

class ObjectDimensionCalculator:
	def __init__(self, intrinsics):
		self.intr = intrinsics

	# ------------------------------------------------------------------
	# Internal helper: deproject + mask-filter + outlier removal
	# ------------------------------------------------------------------
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
	# Volume estimation via PCA-aligned slicing
	# ------------------------------------------------------------------
	def get_pca_volume(self, depth_meters, cv_mask, slice_height_m=0.004,
					   height_limit=None, outlier_std=2.0):
		"""Volume estimation using PCA alignment and cylindrical slicing."""
		pts = self.get_object_points(depth_meters, cv_mask)
		if pts is None or len(pts) < 100:
			return None

		pts_mean = np.mean(pts, axis=0)
		_, _, Vh = np.linalg.svd(pts - pts_mean, full_matrices=False)
		pc1 = Vh[0]
		if pc1[1] > 0:
			Vh[0] = -Vh[0]

		pts_pca = (pts - pts_mean) @ Vh.T
		h_vals = pts_pca[:, 0]
		h_min, h_max = np.percentile(h_vals, [1, 99])
		if h_max <= h_min:
			return None

		h_bins = np.arange(h_min, h_max, slice_height_m)
		total_vol, slice_data, acc_h = 0.0, [], 0.0

		for i in range(len(h_bins) - 1):
			h_lo, h_hi = h_bins[i], h_bins[i + 1]
			thick = abs(h_hi - h_lo)
			if height_limit and (acc_h + thick) > height_limit:
				break

			mask = (h_vals >= h_lo) & (h_vals < h_hi)
			pts_slice = pts_pca[mask]
			if len(pts_slice) < 5:
				continue

			def _span(data):
				m, s = np.mean(data), np.std(data)
				clean = data[np.abs(data - m) <= outlier_std * s] if s > 0 else data
				return np.max(clean) - np.min(clean) if len(clean) >= 5 else None

			s2, s3 = _span(pts_slice[:, 1]), _span(pts_slice[:, 2])
			if s2 is None or s3 is None:
				continue

			diam = max(s2, s3)
			total_vol += np.pi * (diam / 2.0) ** 2 * thick
			acc_h += thick
			slice_data.append({'rel_h': acc_h, 'diameter': diam})

		return total_vol, slice_data

	def get_cylindrical_volume(self, depth_meters, cv_mask,
							   slice_height_m=0.005, height_limit=None):
		"""Alias for get_pca_volume with slightly coarser slices."""
		return self.get_pca_volume(depth_meters, cv_mask, slice_height_m, height_limit)

	# ------------------------------------------------------------------
	# Multi-camera consensus volume
	# ------------------------------------------------------------------
	def calculate_consensus_volume(self, slices_l, slices_r, slice_height_m=0.005):
		"""Merge per-camera slice lists and compute consensus volume."""
		if not slices_l and not slices_r:
			return 0.0, []
		bins = {}
		for s in slices_l + slices_r:
			idx = int(round(s['rel_h'] / slice_height_m))
			bins.setdefault(idx, []).append(s['diameter'])

		merged, total_vol = [], 0.0
		for idx in sorted(bins.keys()):
			diam = np.mean(bins[idx])
			total_vol += np.pi * (diam / 2.0) ** 2 * slice_height_m
			merged.append({'rel_h': idx * slice_height_m, 'diameter': diam})
		return total_vol, merged

	# ------------------------------------------------------------------
	# Mass estimation
	# ------------------------------------------------------------------
	def estimate_mass(self, volume_ml, label, filled=False):
		"""Estimate mass in grams based on volume, label, and fill state."""
		l_low = label.lower()
		m_cont = 0.0
		if 'cup' in l_low:
			# Quadratic fit: (200→3g), (500→11g), (600→8g)
			m_cont = -0.00014167 * (volume_ml ** 2) + 0.125833 * volume_ml - 16.5
			m_cont = max(0.1, m_cont)
		elif 'wine glass' in l_low:
			m_cont = 75.0 + (volume_ml * (61.0 / 354.0))
			m_cont = max(1.0, m_cont)

		rice_mass = (volume_ml * 0.9) * 0.9 if filled else 0.0
		if filled:
			m_cont = m_cont + rice_mass
		return m_cont

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


# ---------------------------------------------------------------------------
# Helper: build 4×4 camera-to-base transform from translation + quaternion
# ---------------------------------------------------------------------------

def build_cam_to_base(translation, quaternion_xyzw):
	"""Build a 4×4 homogeneous transform matrix (camera → base)."""
	r = Rotation.from_quat(np.array(quaternion_xyzw)).as_matrix()
	T = np.eye(4)
	T[:3, :3] = r
	T[:3, 3] = np.array(translation)
	return T


# ---------------------------------------------------------------------------
# ROS Node
# ---------------------------------------------------------------------------

class DimensionsEstimator(Node):
	def __init__(self):
		super().__init__('dimensions_estimator')

		# --- Parameters ---
		self.num_frames = self.declare_parameter('num_frames', 10).get_parameter_value().integer_value
		self.num_filling_frames = self.declare_parameter('num_filling_frames', 5).get_parameter_value().integer_value
		self.publish_mode = self.declare_parameter('publish_mode', 'continuous').get_parameter_value().string_value
		self.publish_period_s = self.declare_parameter('publish_period_s', 0.5).get_parameter_value().double_value
		#self.object_label = self.declare_parameter('object_label', 'single:cup').get_parameter_value().string_value
		self.sam_socket_path = self.declare_parameter('sam_socket_path', '/tmp/sam3_socket').get_parameter_value().string_value
		#self.is_filled = self.declare_parameter('is_filled', True).get_parameter_value().bool_value
		self.declare_parameter('label_topic', '/detected_label')
		self.declare_parameter('bbox_topic', '/object_detections')
		self.object_label = 'single:cup'  
		self.crop_margin_filling = self.declare_parameter('crop_margin_filling', 5).get_parameter_value().integer_value #px

		
		# --- Topic names (left camera) ---
		self.depth_topic_left  = self.declare_parameter(
			'depth_topic_left',  '/camera/left/aligned_depth_to_color/image_raw'
		).get_parameter_value().string_value
		self.color_topic_left  = self.declare_parameter(
			'color_topic_left',  '/camera/left/color/image_raw'
		).get_parameter_value().string_value
		self.cam_info_topic_left = self.declare_parameter(
			'cam_info_topic_left', '/camera/left/aligned_depth_to_color/camera_info'
		).get_parameter_value().string_value

		# --- Topic names (right camera) ---
		self.depth_topic_right  = self.declare_parameter(
			'depth_topic_right',  '/camera/right/aligned_depth_to_color/image_raw'
		).get_parameter_value().string_value
		self.color_topic_right  = self.declare_parameter(
			'color_topic_right',  '/camera/right/color/image_raw'
		).get_parameter_value().string_value
		self.cam_info_topic_right = self.declare_parameter(
			'cam_info_topic_right', '/camera/right/aligned_depth_to_color/camera_info'
		).get_parameter_value().string_value

		# --- State ---
		self.bridge = CvBridge()
		self.measurement_done = False
		self.estimated_dimensions = None
		self.republish_timer = None

		# Per-camera buffers  {side: {'depth': [], 'mask': None, 'calc': None, 'initialized': False}}
		self._cam = {
			'left':  {'depth': [], 'color': [], 'mask': None, 'calc': None, 'initialized': False},
			'right': {'depth': [], 'color': [], 'mask': None, 'calc': None, 'initialized': False},
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

		# --- Camera-info subscribers (one-shot, destroyed after first message) ---
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

		# --- Synchronized depth+color subscribers (left) ---
		self._depth_sub_left = message_filters.Subscriber(
			self, Image, self.depth_topic_left, qos_profile=input_qos)
		self._color_sub_left = message_filters.Subscriber(
			self, Image, self.color_topic_left, qos_profile=input_qos)
		self._ts_left = message_filters.ApproximateTimeSynchronizer(
			[self._depth_sub_left, self._color_sub_left],
			queue_size=30, slop=0.01,
		)
		self._ts_left.registerCallback(lambda d, c: self._sync_callback(d, c, 'left'))

		# --- Synchronized depth+color subscribers (right) ---
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
		self.get_logger().info('Dimensions Estimator Node has been started (dual-camera mode).')

		# filling level estimation (yolo)
		self.bbox = None
		self.model = YOLO('filling-level-predictor-3.pt')

	# ------------------------------------------------------------------
	# Camera-info callback (one per side)
	# ------------------------------------------------------------------
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
		self._cam[side]['initialized'] = True
		self.get_logger().info(
			f"[{side}] Camera intrinsics received: "
			f"fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
			f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}"
		)
		# Unsubscribe once we have the info
		if side == 'left':
			self.destroy_subscription(self._cam_info_sub_left)
		else:
			self.destroy_subscription(self._cam_info_sub_right)

	# ------------------------------------------------------------------
	# SAM3 mask request
	# ------------------------------------------------------------------
	def _recv_exact(self, client, size):
		data = b""
		while len(data) < size:
			chunk = client.recv(size - len(data))
			if not chunk:
				raise ConnectionError('Connection closed while receiving data from SAM3 service')
			data += chunk
		return data

	def request_single_mask(self, color_image):
		if color_image is None:
			return None

		h, w, c = color_image.shape
		if c != 3:
			self.get_logger().error(f'Invalid color channels: expected 3, got {c}')
			return None

		label_bytes = self.object_label.encode()
		header = np.array([h, w, 0, 0, 0, 0, len(label_bytes)], dtype=np.int32)

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

			self.get_logger().info(
				f'SAM3 mask received: object_ok={object_ok}')
			return object_mask
		except Exception as exc:
			self.get_logger().error(f'SAM3 request failed: {exc}')
			return None
		finally:
			client.close()

	# ------------------------------------------------------------------
	# Sync callback (shared by both cameras)
	# ------------------------------------------------------------------
	def _sync_callback(self, depth_msg, color_msg, side):
		if self.measurement_done:
			return

		cam = self._cam[side]
		if not cam['initialized']:
			self.get_logger().warn(
				f'[{side}] Camera intrinsics not yet received, skipping frame.')
			return

		try:
			depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
			color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
			depth_m = depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 \
				else depth.astype(np.float32)
		except Exception as e:
			self.get_logger().error(f'[{side}] Image conversion error: {e}')
			return

		cam['depth'].append(depth_m)
		cam['color'].append(color_msg)

		# Request mask from SAM3 on first frame
		if cam['mask'] is None:
			cam['mask'] = self.request_single_mask(color)

		if len(cam['depth']) >= self.num_frames:
			# Retry mask if first attempt failed
			if cam['mask'] is None:
				cam['mask'] = self.request_single_mask(color)

			if cam['mask'] is None:
				self.get_logger().error(
					f'[{side}] Could not retrieve object mask; aborting this camera.')
				return

			# Save debug mask
			# cv2.imwrite(f'debug_mask_{side}.png', cam['mask'] * 255)

			# Tear down this camera's subscribers
			self._destroy_cam_subscribers(side)

			# Check if *both* cameras are ready to compute final estimate
			self._check_and_estimate()

	def _destroy_cam_subscribers(self, side):
		if side == 'left':
			for sub in [self._depth_sub_left, self._color_sub_left]:
				if sub and getattr(sub, 'sub', None):
					try:
						self.destroy_subscription(sub.sub)
					except Exception:
						pass
			self._depth_sub_left = None
			self._color_sub_left = None
		else:
			for sub in [self._depth_sub_right, self._color_sub_right]:
				if sub and getattr(sub, 'sub', None):
					try:
						self.destroy_subscription(sub.sub)
					except Exception:
						pass
			self._depth_sub_right = None
			self._color_sub_right = None
			
	# callbacks for yoloe node topics
	def label_topic_callback(self, msg):
		self.get_logger().info(f"Received label update: {msg.data}")
		self.label_only = msg.data if msg.data is not None else "cup"
		self.object_label = f"single:{self.label_only}"
		
		try:
			self.destroy_subscription(self.sub_label)
		except Exception:
			pass
	def bbox_topic_callback(self, msg):
		self.get_logger().info(f"Received bbox update: {msg.data}")
		if msg.data is not None:
			self.bbox = msg.data
			try:
				self.destroy_subscription(self.sub_bbox)
			except Exception:
				pass
	
	# ------------------------------------------------------------------
	# Trigger estimation only when both cameras have enough frames
	# ------------------------------------------------------------------
	def _check_and_estimate(self):
		left_ready  = (len(self._cam['left']['depth'])  >= self.num_frames and len(self._cam['left']['color']) >= self.num_frames
					   and self._cam['left']['mask']  is not None)
		right_ready = (len(self._cam['right']['depth']) >= self.num_frames and len(self._cam['right']['color']) >= self.num_frames
					   and self._cam['right']['mask'] is not None)

		if left_ready and right_ready and not self.measurement_done:
			self.estimate_dimensions_and_publish()
			self.estimate_filling_level()
			self.estimate_mass()
			self.report_data()

	# Estimates the mass
	def estimate_mass(self):

		self.final_mass = 0
		if self.label_only is not None and self.vol_ml and self.is_filled:

			fusion_calc = self._cam['left']['calc']
			self.final_mass = fusion_calc.estimate_mass(self.vol_ml, self.label_only, filled=self.is_filled)


		self.end_time_mass = self.get_clock().now()


	# Estimates filling level
	def estimate_filling_level(self):
		self.measurement_done = True
		# log start time
		self.start_time_filling = self.get_clock().now()

		# open up bounding boxes
		left_bbox = None
		right_bbox = None
		if self.bbox is not None:
			bbox_dict = json.loads(self.bbox)
			left_bbox = bbox_dict['left_bbox']
			right_bbox = bbox_dict['right_bbox']
		else: 
			self.get_logger().warn('No bounding box received for filling level estimation; skipping.')
			return
		
		# estimate binary filling level: 0 -> False, 50%, 90% -> 1
		evals = []
		for i in range(self.num_filling_frames):
			latest_image_l = self._cam['left']['color'][i]
			latest_image_r = self._cam['right']['color'][i]

			cv_img_l = self.bridge.imgmsg_to_cv2(latest_image_l, desired_encoding='bgr8')
			cv_img_r = self.bridge.imgmsg_to_cv2(latest_image_r, desired_encoding='bgr8')

			if (left_bbox is not None and right_bbox is not None):
				margin = self.crop_margin_filling
				left_bbox = [max(0, left_bbox[0]-margin), max(0, left_bbox[1]-margin), min(cv_img_l.shape[1], left_bbox[2]+margin), min(cv_img_l.shape[0], left_bbox[3]+margin)]
				right_bbox = [max(0, right_bbox[0]-margin), max(0, right_bbox[1]-margin), min(cv_img_r.shape[1], right_bbox[2]+margin),	 min(cv_img_r.shape[0], right_bbox[3]+margin)]
				crop_l = cv_img_l[int(left_bbox[1]):int(left_bbox[3]), int(left_bbox[0]):int(left_bbox[2])]
				crop_r = cv_img_r[int(right_bbox[1]):int(right_bbox[3]), int(right_bbox[0]):int(right_bbox[2])]

				# debug
				# cv2.imshow("Debug", crop_l)
				# cv2.waitKey(1)

				# inference
				result_l = self.model(crop_l)[0]
				result_r = self.model(crop_r)[0]
				
				class_l = result_l.probs.top1
				class_r = result_r.probs.top1

				evals.append(class_l)
				evals.append(class_r)
		
		# finding out what was the more common
		# Count the occurrences of each class
		vote_counts = Counter(evals)
		# most_common(1) returns a list like: [('Car', 3)]
		# We add [0][0] to extract just the label 'Car'
		final_class = vote_counts.most_common(1)[0][0]

		# Classes in custom yolo model: 0 -> empty, 1 -> 90% filled, 2 -> 50% filled
		classes_map = {0: 'empty', 1: '90% filled', 2: '50% filled'}
		self.is_filled = final_class == 1 or final_class == 2 #50%, 90%

		# log end time
		self.end_time_filling = self.get_clock().now()

		self.get_logger().info(f'[Filling Level] Inferred class: {final_class} ({classes_map[final_class]})')

	# ------------------------------------------------------------------
	# Core estimation & publishing: dimensions
	# ------------------------------------------------------------------
	def estimate_dimensions_and_publish(self):
		self.measurement_done = True # to prevent possible second call

		self.start_time_dim = self.get_clock().now()
		camera_slices  = {}   # side -> list of slice dicts
		camera_metrics = {}   # side -> consolidated dict
		#label = self.object_label.split(':')[-1]  # e.g. 'single:cup' -> 'cup'

		for side in ('left', 'right'):
			cam = self._cam[side]
			calc = cam['calc']
			if calc is None:
				self.get_logger().warn(f'[{side}] No calculator; skipping.')
				continue

			mask = cam['mask']
			depths = cam['depth']
			if not depths or mask is None:
				continue

			estimates = {'height': [], 'top_w': [], 'bot_w': [], 'volume': [],
						 'length': [], 'width': [], 'depth': []}
			depth_stack, mask_stack = [], []

			for depth_m in depths:
				h_val = calc.get_height(depth_m, mask, T_cam_to_base=None)
				dims  = calc.get_dimensions(depth_m, mask)
				prof  = calc.get_width_bb(depth_m, mask)
				v_res = calc.get_cylindrical_volume(depth_m, mask, height_limit=h_val)

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
				if v_res is not None:
					estimates['volume'].append(v_res[0])

				depth_stack.append(depth_m)
				mask_stack.append(mask)

			consolidated = calc.consolidate_all(estimates)
			# self.end_time_dim = self.get_clock().now()
			camera_metrics[side] = consolidated

			# Build averaged depth + majority-vote mask for consensus slicing
			depth_arr = np.stack(depth_stack, axis=0).astype(np.float32)
			depth_arr[depth_arr == 0] = np.nan
			with np.errstate(all='ignore'):
				avg_d = np.nan_to_num(np.nanmean(depth_arr, axis=0))
			mask_arr = np.stack(mask_stack, axis=0)
			avg_m = (np.sum(mask_arr > 0, axis=0) > (len(depth_stack) / 2)).astype(np.uint8) * 255

			v_res_avg = calc.get_cylindrical_volume(
				avg_d, avg_m, height_limit=consolidated.get('height'))
			if v_res_avg:
				camera_slices[side] = v_res_avg[1]

		self.end_time_dim = self.get_clock().now()
		self.start_time_mass = self.get_clock().now()
		# --- Dual-camera volume fusion ---
		vol_ml = None
		mass_empty = None
		mass_filled = None
		avg_h = avg_t = avg_b = None
		if 'left' in camera_slices and 'right' in camera_slices:
			fusion_calc = self._cam['left']['calc']
			f_vol, _ = fusion_calc.calculate_consensus_volume(
				camera_slices['left'], camera_slices['right'])

			lm = camera_metrics.get('left', {})
			rm = camera_metrics.get('right', {})

			h_l = lm.get('height') or 0.0
			h_r = rm.get('height') or 0.0
			avg_h = (h_l + h_r) / 2.0 if (h_l and h_r) else (h_l or h_r)

			t_l = lm.get('top_w') or 0.0
			t_r = rm.get('top_w') or 0.0
			avg_t = (t_l + t_r) / 2.0 if (t_l and t_r) else (t_l or t_r)

			b_l = lm.get('bot_w') or 0.0
			b_r = rm.get('bot_w') or 0.0
			avg_b = (b_l + b_r) / 2.0 if (b_l and b_r) else (b_l or b_r)

			r1, r2 = avg_t / 2.0, avg_b / 2.0
			v_frust = (1.0 / 3.0) * np.pi * avg_h * (r1 ** 2 + r2 ** 2 + r1 * r2) \
				if avg_h else 0.0

			self.vol_ml = ((v_frust + f_vol) / 2.0) * 1e6 # calculate volume for mass estimation later
			#final_mass = fusion_calc.estimate_mass(vol_ml, label, filled=self.is_filled)
			
		elif len(camera_metrics) == 1:
			# Fall back to single-camera estimate
			side = next(iter(camera_metrics))
			calc = self._cam[side]['calc']
			cm = camera_metrics[side]
			avg_h = cm.get('height') or 0.0
			if side in camera_slices:
				single_vol = sum(
					np.pi * (s['diameter'] / 2.0) ** 2 * 0.005
					for s in camera_slices[side]
				)
				self.vol_ml = single_vol * 1e6 # calculate volume for mass estimation later
				#final_mass = calc.estimate_mass(vol_ml, label, filled=self.is_filled)

		# self.get_logger().info(
		# 		f'[Fusion] volume={vol_ml:.1f} mL, '
		# 		f'mass={final_mass:.1f} g'
		# 	)

		
		
		# --- Build and publish Dimensions message ---
		def _safe(v):
			return float(v) if v is not None else float('nan')

		# Pick representative metrics from either camera
		lm = camera_metrics.get('left', {})
		rm = camera_metrics.get('right', {})

		def _avg_metric(key):
			l_v, r_v = lm.get(key), rm.get(key)
			if l_v is not None and r_v is not None:
				return (l_v + r_v) / 2.0
			return l_v or r_v

		# Publishing dimensions
		self.estimated_dimensions = Dimensions()
		self.estimated_dimensions.height  = _safe(avg_h if avg_h else _avg_metric('height'))
		self.estimated_dimensions.length  = _safe(_avg_metric('length'))
		self.estimated_dimensions.width   = _safe(_avg_metric('width'))
		self.estimated_dimensions.depth   = _safe(_avg_metric('depth'))
		self.estimated_dimensions.top_w   = _safe(avg_t if avg_t else _avg_metric('top_w'))
		self.estimated_dimensions.bot_w   = _safe(avg_b if avg_b else _avg_metric('bot_w'))

		# Attach volume / mass if the Dimensions message has those fields
		# if hasattr(self.estimated_dimensions, 'volume_ml'):
		# 	self.estimated_dimensions.volume_ml   = _safe(vol_ml)
		# if hasattr(self.estimated_dimensions, 'mass'):
		# 	self.estimated_dimensions.mass_empty  = _safe(final_mass)

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
				# Report things for submission
		dim_mass_data = {
			"width_top_est_mm_vision": self.estimated_dimensions.top_w * 1000.0,
			"width_bottom_est_mm_vision": self.estimated_dimensions.bot_w * 1000.0,
			"height_est_mm_vision": self.estimated_dimensions.height * 1000.0,
			"geometry_est_timepoint": (self.end_time_dim - self.start_time_dim).nanoseconds / 1e9,
			"mass_full_est_g_vision": self.final_mass,
			"mass_full_est_vision_timepoint": (self.end_time_mass - self.start_time_mass).nanoseconds / 1e9,
			"fill_level_est_percent_vision": 0.9 if self.is_filled else 0.0,
			"fill_level_vision_timepoint": (self.end_time_filling - self.start_time_filling).nanoseconds / 1e9
		}
		# Log with prefix for the extraction script
		self.get_logger().get_child("SUBMISSION").info(f"{json.dumps(dim_mass_data)}")

		self.get_logger().info(
			f'Published estimated dimensions | '
			f"height={self.estimated_dimensions.height:.4f} m, "
			f"top_w={self.estimated_dimensions.top_w:.4f} m, "
			f"bot_w={self.estimated_dimensions.bot_w:.4f} m, "
			f"volume={self.vol_ml:.1f} mL, "
			f"mass={self.final_mass:.1f} g"
			if self.vol_ml is not None
			else f'Published estimated dimensions (no volume computed).'
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