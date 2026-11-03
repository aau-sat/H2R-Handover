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
	# Box estimation via trimesh Oriented Bounding Box
	# ------------------------------------------------------------------
	def get_box_dimensions(self, depth_frames_masks, min_pts=100, glb_out=None):
		"""Aggregate cleaned point clouds from multiple frames and fit a
		trimesh OBB to recover width, depth, and height in metres.

		Args:
			depth_frames_masks: list of (depth_meters, cv_mask) tuples.
			min_pts: minimum points required after outlier removal.
			glb_out: optional file path (str). If given, saves the filtered
			         point cloud + OBB as a GLB scene to that path.

		Returns:
			dict with 'width_m', 'depth_m', 'height_m', and 'yaw_deg_camera',
			or None on failure.
		"""
		all_pts = []
		for d_m, m_raw in depth_frames_masks:
			pts = self.get_object_points(d_m, m_raw)  # already outlier-cleaned
			if pts is not None:
				all_pts.append(pts)

		if not all_pts:
			return None

		agg = np.vstack(all_pts)

		# Second outlier pass on the aggregated multi-frame cloud
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(agg)
		_, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
		clean = np.asarray(pcd.select_by_index(ind).points)

		if len(clean) < min_pts:
			return None

		# --- Step 1: RANSAC plane detection → axis-aligned reconstruction
		# (avoids the diagonal-OBB problem with L-shaped partial clouds)
		result = _fit_planes_and_reconstruct(clean)

		if result is not None:
			reconstructed, obb = result
			print('[box] RANSAC plane reconstruction OK')
		else:
			# Fallback: OBB-based (may be diagonal for 2-face views)
			print('[box] RANSAC failed — falling back to OBB reconstruction')
			reconstructed, obb = _reconstruct_box_surface_obb_fallback(clean)

		# Diagnostic: Check raw point ranges
		p_min, p_max = np.min(clean, axis=0), np.max(clean, axis=0)
		print(f'[box] Aggregated cloud AABB (m): {p_max - p_min}')

		# Use oriented_bounds to get the true dimensions from the reconstructed mesh.
		# This is more robust than obb.extents if the transform contains scaling.
		to_origin, extents = trimesh.bounds.oriented_bounds(reconstructed)
		
		# Identify the vertical axis (most aligned with camera-Y/gravity)
		# to_origin maps world->local, so its rows are the local axes in world frame.
		# We want the axis in OBB-space that aligns with [0,1,0] in camera-space.
		R_local_to_world = np.linalg.inv(to_origin)[:3, :3]
		y_alignment = np.abs(R_local_to_world[1, :]) # Row 1 is Y-components
		h_idx = np.argmax(y_alignment)
		
		# Other two are horizontal (Width/Depth)
		others = [i for i in range(3) if i != h_idx]

		# Use the longer horizontal box axis to define stable yaw.
		major_h_idx = others[int(np.argmax(extents[others]))]
		h_axis_world = R_local_to_world[:, major_h_idx]
		h_axis_xz = np.array([h_axis_world[0], 0.0, h_axis_world[2]], dtype=float)
		norm_xz = float(np.linalg.norm(h_axis_xz))
		if norm_xz > 1e-8:
			h_axis_xz /= norm_xz
			yaw_deg = float(np.degrees(np.arctan2(h_axis_xz[2], h_axis_xz[0])))
			# Canonicalize for box 180-degree symmetry.
			yaw_deg = float(((yaw_deg + 90.0) % 180.0) - 90.0)
		else:
			yaw_deg = 0.0
		
		res = {
			'width_m':  float(extents[others[0]]),
			'depth_m':  float(extents[others[1]]),
			'height_m': float(extents[h_idx]),
			'yaw_deg_camera': yaw_deg,
		}
		print(f'[box] Reconstructed size (oriented_bounds): {extents}')
		print(f'[box] result dict: {res}')
  
		if glb_out is not None:
			try:
				scene = trimesh.Scene()
				scene.add_geometry(trimesh.PointCloud(clean), node_name='clean_cloud')
				scene.add_geometry(obb, node_name='fitted_obb')
				scene.export(glb_out)
				print(f'[box] GLB scene with cleaned cloud and OBB saved to: {glb_out}')
			except Exception as e:
				print(f'[box] Failed to save GLB output: {e}')


		return res


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
# Box surface reconstruction helper
# ---------------------------------------------------------------------------

def _fit_planes_and_reconstruct(clean_pts, distance_threshold=0.008,
                                n_per_face=40, min_inliers=40):
    """Detect the 2 visible box-face planes via RANSAC and reconstruct all 6 faces.

    This avoids the diagonal-OBB problem that occurs when an L-shaped partial
    cloud (2 perpendicular faces) is given directly to an OBB fitter: the OBB
    principal axis aligns with the corner diagonal instead of the face normals.

    Algorithm
    ---------
    1. RANSAC plane fit on full cloud  → dominant face normal n1  (usually front)
    2. Remove inliers, RANSAC again   → second  face normal n2  (usually side)
    3. n3 = cross(n1, n2) → re-orthogonalise all 3 axes
    4. Project all observed points onto the 3 axes to get exact per-axis extents
    5. Uniformly sample n_per_face points on each of the 6 faces
    6. Build a trimesh Box from the resulting frame + extents

    Returns
    -------
    (reconstructed_pts, obb) or None if RANSAC fails / faces not perpendicular.
    """
    normals = []
    pts_rem = clean_pts.copy()

    for _ in range(2):
        if len(pts_rem) < 50:
            break
        pcd_r = o3d.geometry.PointCloud()
        pcd_r.points = o3d.utility.Vector3dVector(pts_rem)
        try:
            plane_model, inliers = pcd_r.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=200,
            )
        except Exception:
            break
        if len(inliers) < min_inliers:
            break
        n = np.array(plane_model[:3], dtype=float)
        n /= np.linalg.norm(n)
        normals.append(n)
        # Remove inliers so the next iteration targets the other face
        mask = np.ones(len(pts_rem), dtype=bool)
        mask[inliers] = False
        pts_rem = pts_rem[mask]

    if len(normals) < 2:
        return None  # only 1 (or 0) planes found

    n1, n2 = normals[0], normals[1]
    # The two face normals of a box must be perpendicular
    if abs(np.dot(n1, n2)) > 0.3:   # >~17° off-perpendicular → unreliable
        return None

    # Build exact orthonormal frame from the two detected normals
    n3 = np.cross(n1, n2);  n3 /= np.linalg.norm(n3)
    n2 = np.cross(n3, n1);  n2 /= np.linalg.norm(n2)   # re-orthogonalise
    R  = np.column_stack([n1, n2, n3])   # 3×3, columns = world axes of box

    # Project all observed points into the box frame (R is orthogonal → R.T = inv)
    pts_local = clean_pts @ R          # (N, 3)
    lo = pts_local.min(axis=0)         # tight extents along each box axis
    hi = pts_local.max(axis=0)
    print(f'[box] Local projected bounds: lo={lo}, hi={hi}, range={hi-lo}')

    # --- Generate uniform samples on all 6 faces -------------------------
    rng = np.random.default_rng(42)

    def _face(fixed_ax, fixed_val, ax1, ax2):
        pts = np.zeros((n_per_face, 3))
        pts[:, fixed_ax] = fixed_val
        pts[:, ax1]      = rng.uniform(lo[ax1], hi[ax1], n_per_face)
        pts[:, ax2]      = rng.uniform(lo[ax2], hi[ax2], n_per_face)
        return pts

    local_faces = np.vstack([
        _face(0, lo[0], 1, 2), _face(0, hi[0], 1, 2),  # ±n1 faces
        _face(1, lo[1], 0, 2), _face(1, hi[1], 0, 2),  # ±n2 faces
        _face(2, lo[2], 0, 1), _face(2, hi[2], 0, 1),  # ±n3 faces
    ])
    reconstructed = local_faces @ R.T   # back to world frame

    # Build trimesh Box using the exact frame
    extents_vec  = hi - lo
    print(f'[box] Plane-fit extents vector: {extents_vec}')
    center_world = ((lo + hi) / 2.0) @ R.T
    T4 = np.eye(4)
    T4[:3, :3] = R
    T4[:3,  3] = center_world
    obb = trimesh.primitives.Box(extents=extents_vec, transform=T4)

    return reconstructed, obb


def _reconstruct_box_surface_obb_fallback(clean_pts, n_per_face=40):
    """OBB-based fallback reconstruction (used when RANSAC plane detection fails).

    NOTE: This will give a diagonal result when the partial cloud is L-shaped
    (2 perpendicular visible faces).  Only use as a last resort.
    """
    tm = trimesh.PointCloud(clean_pts)
    obb_init = tm.bounding_box_oriented
    T      = obb_init.transform
    T_inv  = np.linalg.inv(T)
    pts_h     = np.hstack([clean_pts, np.ones((len(clean_pts), 1))])
    pts_local = (T_inv @ pts_h.T).T[:, :3]
    lo = pts_local.min(axis=0)
    hi = pts_local.max(axis=0)
    rng = np.random.default_rng(42)

    def _face(fixed_ax, fixed_val, ax1, ax2):
        pts = np.zeros((n_per_face, 3))
        pts[:, fixed_ax] = fixed_val
        pts[:, ax1]      = rng.uniform(lo[ax1], hi[ax1], n_per_face)
        pts[:, ax2]      = rng.uniform(lo[ax2], hi[ax2], n_per_face)
        return pts

    local_faces = np.vstack([
        _face(0, lo[0], 1, 2), _face(0, hi[0], 1, 2),
        _face(1, lo[1], 0, 2), _face(1, hi[1], 0, 2),
        _face(2, lo[2], 0, 1), _face(2, hi[2], 0, 1),
    ])
    rec_h = np.hstack([local_faces, np.ones((len(local_faces), 1))])
    reconstructed = (T @ rec_h.T).T[:, :3]

    tm_full = trimesh.PointCloud(reconstructed)
    obb = tm_full.bounding_box_oriented
    return reconstructed, obb


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
		
		self.vol_ml = None
  
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

		# --- Topic names ---
		self.depth_topic  = self.declare_parameter(
			'depth_topic',  f'/camera/{self.camera_side}/aligned_depth_to_color/image_raw'
		).get_parameter_value().string_value
		self.color_topic  = self.declare_parameter(
			'color_topic',  f'/camera/{self.camera_side}/color/{left_color_camera_topic_suffix if self.camera_side == "left" else right_color_camera_topic_suffix}'
		).get_parameter_value().string_value
		self.cam_info_topic = self.declare_parameter(
			'cam_info_topic', f'/camera/{self.camera_side}/aligned_depth_to_color/camera_info'
		).get_parameter_value().string_value

		# --- State ---
		self.bridge = CvBridge()
		self.measurement_done = False
		self.estimated_dimensions = None
		self.republish_timer = None

		# Camera buffer
		self._cam = {'depth': [], 'color': None, 'mask': None, 'calc': None, 'initialized': False, 'mask_thread': None, 'vlm_thread': None}

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

		# --- Camera-info subscriber (one-shot, destroyed after first message) ---
		self._cam_info_sub = self.create_subscription(
			sensor_msgs.CameraInfo,
			self.cam_info_topic,
			self._camera_info_callback,
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
		self._depth_sub = message_filters.Subscriber(
			self, Image, self.depth_topic, qos_profile=input_qos)
		self._color_sub = message_filters.Subscriber(
			self, Image, self.color_topic, qos_profile=input_qos)
		self._ts = message_filters.ApproximateTimeSynchronizer(
			[self._depth_sub, self._color_sub],
			queue_size=30, slop=0.01,
		)
		self._ts.registerCallback(self._sync_callback)

		# --- Publisher ---
		self.dimensions_pub = self.create_publisher(Dimensions, '/estimated_dimensions', 10)
		self.segmented_pub = self.create_publisher(SegmentedObject, '/segmented_object', 10)
		self.get_logger().info(f'Dimensions Estimator Node has been started ({self.camera_side} camera mode).')

	# ------------------------------------------------------------------
	# Camera-info callback (one per side)
	# ------------------------------------------------------------------
	def _camera_info_callback(self, msg):
		if self._cam['initialized']:
			return
		intrinsics = {
			'fx': float(msg.k[0]),
			'fy': float(msg.k[4]),
			'cx': float(msg.k[2]),
			'cy': float(msg.k[5]),
			'width': int(msg.width),
			'height': int(msg.height),
		}
		self._cam['calc'] = ObjectDimensionCalculator(intrinsics)
		self._cam['initialized'] = True
		self.get_logger().info(
			f"Camera intrinsics received: "
			f"fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
			f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}"
		)
		# Unsubscribe once we have the info
		self.destroy_subscription(self._cam_info_sub)

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

	def request_single_mask(self, color_image, label, bbox=None):
		if color_image is None:
			return None, False

		h, w, c = color_image.shape
		if c != 3:
			self.get_logger().error(f'Invalid color channels: expected 3, got {c}')
			return None, False

		# Use current sam_retry value to select label
		
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

			self.get_logger().info(
				f'SAM3 mask received: object_ok={object_ok}')
			return object_mask, object_ok
		except Exception as exc:
			self.get_logger().error(f'SAM3 request failed: {exc}')
			return None, False
		finally:
			client.close()

	def _notify_grasp_generator(self):
		# Publish segmented object message for grasp generator
		segmented_msg = SegmentedObject()
     	# Add segmentation label and mask
		segmented_msg.view = self.camera_side
		segmented_msg.segmentation_label = self.object_label
		cam = self._cam
		try:
			mask_image = (cam['mask'] * 255).astype(np.uint8)  # Ensure mask is 0-255
			segmented_msg.segmentation_mask = self.bridge.cv2_to_imgmsg(mask_image, encoding='mono8')
		except Exception as e:
			self.get_logger().error(f'Failed to convert mask to ROS Image: {e}')
			empty_mask = np.zeros((cam['mask'].shape[0], cam['mask'].shape[1]), dtype=np.uint8)
			segmented_msg.segmentation_mask = self.bridge.cv2_to_imgmsg(empty_mask, encoding='mono8')
   
		self.segmented_pub.publish(segmented_msg)
	# ------------------------------------------------------------------
	# Sync callback (shared by both cameras)
	# ------------------------------------------------------------------
	def _sync_callback(self, depth_msg, color_msg):
		if self.measurement_done:
			return

		cam = self._cam
		if not cam['initialized']:
			self.get_logger().warn(
				f'Camera intrinsics not yet received, skipping frame.')
			return

		try:
			depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
			if cam['color'] is None:
				cam['color'] = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
			depth_m = depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 \
				else depth.astype(np.float32)
		except Exception as e:
			self.get_logger().error(f'Image conversion error: {e}')
			return

		cam['depth'].append(depth_m)
		color = cam['color']

		# Store the first received color frame
		if self.first_image_base64 is None:
			color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV encoding
			self.first_image_base64 = cv2.imencode('.png', color_bgr)[1].tobytes()
			self.first_image_base64 = base64.b64encode(self.first_image_base64).decode('ascii')
   
		if cam['vlm_thread'] is None:
			# spawn a VLM inference thread immediately (while we wait for YOLOE label) so it can run in parallel and hopefully be ready by the time we exhaust the YOLO/prior labels
			self.get_logger().info('Spawning background thread for VLM label inference...')
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
			bbox_copy = self.bbox
			print(f'Initiating background SAM3 mask request with label: {label_copy}')
			cam['mask_thread'] = threading.Thread(
				target=self._fetch_mask_async,
				args=(color_copy, label_copy, bbox_copy),
			)
			cam['mask_thread'].start()

		if len(cam['depth']) >= self.num_frames:
			# Wait for SAM3 mask if thread is running
			if cam['mask_thread'] is not None and cam['mask_thread'].is_alive():
				self.get_logger().info('Waiting for background SAM3 mask generation...')
				cam['mask_thread'].join()

			old_label = self.object_label
			# VLM fallback if no mask obtained yet
			if cam['mask'] is None: 
				# remove failed label from prior_labels to avoid retrying it again
				if cam.get('vlm_thread') is not None and cam['vlm_thread'].is_alive():
					cam['vlm_thread'].join()
				if self.vlm_label is not None:
					self.get_logger().info(f'Attempting SAM3 mask request with VLM label: {self.vlm_label}')
					cam['mask'], successfull = self.request_single_mask(color, self.vlm_label, self.bbox)
					if successfull:
						self.object_label = self.vlm_label
						self.get_logger().info('SAM3 mask obtained successfully with VLM label.')
					else:
						self.get_logger().warning('SAM3 mask request with VLM label failed.')
						cam['mask'] = None  # Ensure mask is None for retry logic
						# narrow down the prior list based on VLM label
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
      
			# Retry mask with prior labels if initial attempt(s) failed
			while cam['mask'] is None and self.sam_retry < len(self.prior_labels):
				successfull = False
				# if self.sam_retry == 0:
				# 	# First attempt: use YOLO label again
				# 	self.get_logger().info(f'Requesting mask with YOLO label: {self.object_label}')
				# 	cam['mask'], successfull = self.request_single_mask(color, self.object_label)
				# 	if not successfull:
				# 		self.get_logger().warning('Initial SAM3 mask request with YOLO label failed.')
				# 		cam['mask'] = None  # Ensure mask is None for retry logic

				print(f'Attempt {self.sam_retry}: SAM3 mask request failed with current label. Trying prior label...')
				# Fallback attempts: use prior labels
				prior_label = self.prior_labels[self.sam_retry]
				self.get_logger().warning(
					f'YOLO label failed. Retrying with prior label #{self.sam_retry}: {prior_label}'
				)
				cam['mask'], successfull = self.request_single_mask(color, prior_label)
				if successfull:
					self.object_label = prior_label
				else:
					self.get_logger().warning(f'SAM3 mask request with prior label {prior_label} failed.')
					cam['mask'] = None  # Ensure mask is None for next retry

				# elif self.sam_retry > len(self.prior_labels):
				# 	self.get_logger().warning('All prior labels exhausted. Retrying with VLM label inference.')
				# 	label = infer_answer_from_base64(self.first_image_base64, self.vlm_prompt)
				# 	self.get_logger().info(f'VLM inferred label: {label}. Requesting SAM3 mask with this label...')
				# 	cam['mask'], successfull = self.request_single_mask(color, label)
				# 	self.object_label = label if successfull else None
				if successfull:
					self.get_logger().info(f'Mask retrieved on attempt {self.sam_retry + 1} with label: {self.object_label}')
					break
				
				if not successfull:
					self.sam_retry += 1

			# Check if any mask was obtained
			if self.object_label is None or cam['mask'] is None:
				self.get_logger().error(
					f'Could not retrieve object mask after {self.sam_retry} attempts. Aborting.')
				return

			# Tear down this camera's subscribers
			self._destroy_cam_subscribers()
			self._notify_grasp_generator()

			# Check if ready to compute final estimate
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

	def _fetch_mask_async(self, color_image, label_snapshot, bbox=None):
		self.get_logger().info('Requesting SAM3 mask in the background...')
		print(f'Background thread: requesting SAM3 mask with label="{label_snapshot}" and bbox={bbox}')
		mask, successfull = self.request_single_mask(color_image, label_snapshot, bbox)
		if successfull:
			self._cam['mask'] = mask
			self.get_logger().info('Background SAM3 mask received successfully.')
		else:
			self._cam['mask'] = None
			self.get_logger().error('Background SAM3 mask request failed.')

	def _destroy_cam_subscribers(self):
		for sub in [self._depth_sub, self._color_sub]:
			if sub and getattr(sub, 'sub', None):
				try:
					self.destroy_subscription(sub.sub)
				except Exception:
					pass
		self._depth_sub = None
		self._color_sub = None
			
	# callbacks for yoloe node topics
	def label_topic_callback(self, msg):
		#self.get_logger().info(f"Received label update: {msg.data}")
		self.label_only = msg.data if msg.data is not None else "cup"
		self.object_label = self.label_only
		self._label_wait_logged = False
		# Keep the subscription alive so later YOLOE updates can replace the label.
  
	def bbox_topic_callback(self, msg):
		self.get_logger().info(f"Received bbox update: {msg.data}")
		if msg.data is None:
			return

		try:
			bbox_payload = json.loads(msg.data)
			self.bbox = bbox_payload.get('left_bbox')
			# left_class = bbox_payload.get('left_class')
			if self.bbox is None:
				self.get_logger().warn('bbox payload did not contain left_bbox; ignoring update.')
			# else:
			# 	self.get_logger().info(f"Using left_bbox={self.bbox} (left_class={left_class})")
		except Exception as exc:
			self.get_logger().error(f'Failed to parse bbox payload: {exc}')
			return

		try:
			self.destroy_subscription(self.sub_bbox)
		except Exception:
			pass
	
	# ------------------------------------------------------------------
	# Trigger estimation only when both cameras have enough frames
	# ------------------------------------------------------------------
	def _check_and_estimate(self):
		ready = (len(self._cam['depth']) >= self.num_frames and 
				 self._cam['mask'] is not None)

		if ready and not self.measurement_done:
			self.estimate_dimensions_and_publish()
			self.report_data()

	# ------------------------------------------------------------------
	# Core estimation & publishing: dimensions
	# ------------------------------------------------------------------
	def estimate_dimensions_and_publish(self):
		self.measurement_done = True # to prevent possible second call

		self.start_time_dim = self.get_clock().now()
		
		cam = self._cam
		calc = cam['calc']
		if calc is None:
			self.get_logger().warn('No calculator; skipping.')
			return

		mask = cam['mask']
		depths = cam['depth']
		if not depths or mask is None:
			return

		consolidated = {'yaw_deg_camera': 0.0}

		# --- Strategy Selection ---
		label = str(self.object_label).lower()
		is_symmetric = any(tag in label for tag in symmetric_container_priors + other_priors)
		
		if not is_symmetric:
			self.get_logger().info(f"Using BOX estimation strategy for label '{self.object_label}'")
			# Prepare input for get_box_dimensions
			frames_masks = [(d, mask) for d in depths]
			
			# Optional GLB output for debugging
			# glb_path = f"box_reconstruction_{self.camera_side}.glb"
			box_res = calc.get_box_dimensions(frames_masks)
			
			if box_res:
				consolidated = {
					'width':  box_res['width_m'],
					'depth':  box_res['depth_m'],
					'height': box_res['height_m'],
					'length': max(box_res['width_m'], box_res['depth_m']),
					'top_w':  box_res['width_m'],
					'bot_w':  box_res['width_m'],
					'yaw_deg_camera': box_res.get('yaw_deg_camera', 0.0),
				}
			else:
				self.get_logger().warn("Box estimation failed, falling back to basic method")
				is_symmetric = True  # Fallback to symmetric method if box estimation fails

		if is_symmetric:
			estimates = {'height': [], 'top_w': [], 'bot_w': [], 'volume': [],
						 'length': [], 'width': [], 'depth': []}
			depth_stack, mask_stack = [], []

			for depth_m in depths:
				h_val = calc.get_height(depth_m, mask, T_cam_to_base=None)
				dims  = calc.get_dimensions(depth_m, mask)
				prof  = calc.get_width_bb(depth_m, mask)

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
				
				depth_stack.append(depth_m)
				mask_stack.append(mask)

			consolidated = calc.consolidate_all(estimates)
			consolidated['yaw_deg_camera'] = 0.0

		if consolidated is None:
			self.get_logger().warn('Could not consolidate dimension estimates; skipping publish.')
			return

		self.end_time_dim = self.get_clock().now()
		
		# --- Build and publish Dimensions message ---
		def _safe(v):
			return float(v) if v is not None else float('nan')

		# Publishing dimensions
		self.estimated_dimensions = Dimensions()
		self.estimated_dimensions.height  = _safe(consolidated.get('height'))
		self.estimated_dimensions.top_w   = _safe(consolidated.get('top_w'))
		self.estimated_dimensions.bot_w   = _safe(consolidated.get('bot_w'))
		self.estimated_dimensions.yaw_deg_camera = _safe(consolidated.get('yaw_deg_camera', 0.0))

		# Add segmentation label and mask
		# self.estimated_dimensions.segmentation_label = self.object_label if self.object_label else "unknown"
		# try:
		# 	mask_image = (cam['mask'] * 255).astype(np.uint8)  # Ensure mask is 0-255
		# 	self.estimated_dimensions.segmentation_mask = self.bridge.cv2_to_imgmsg(mask_image, encoding='mono8')
		# except Exception as e:
		# 	self.get_logger().error(f'Failed to convert mask to ROS Image: {e}')
		# 	empty_mask = np.zeros((cam['mask'].shape[0], cam['mask'].shape[1]), dtype=np.uint8)
		# 	self.estimated_dimensions.segmentation_mask = self.bridge.cv2_to_imgmsg(empty_mask, encoding='mono8')

		# Attach volume if the Dimensions message has that field
		# if hasattr(self.estimated_dimensions, 'volume_ml'):
		# 	self.estimated_dimensions.volume_ml   = _safe(self.vol_ml)
		# print(f'Publishing estimated dimensions with label: {self.estimated_dimensions.segmentation_label}')
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
			"yaw_deg_camera_vision": self.estimated_dimensions.yaw_deg_camera,
			"geometry_est_timepoint": (self.end_time_dim - self.start_time_dim).nanoseconds / 1e9
		}
		# Log with prefix for the extraction script
		self.get_logger().get_child("SUBMISSION").info(f"{json.dumps(dim_mass_data)}")

		if self.vol_ml is not None:
			self.get_logger().info(
				f'Published estimated dimensions | '
				f"height={self.estimated_dimensions.height:.4f} m, "
				f"top_w={self.estimated_dimensions.top_w:.4f} m, "
				f"bot_w={self.estimated_dimensions.bot_w:.4f} m, "
				f"yaw_deg_camera={self.estimated_dimensions.yaw_deg_camera:.2f}, "
				f"volume={self.vol_ml:.1f} mL"
			)
		else:
			self.get_logger().info(
				f'Published estimated dimensions | '
				f"height={self.estimated_dimensions.height:.4f} m, "
				f"top_w={self.estimated_dimensions.top_w:.4f} m, "
				f"bot_w={self.estimated_dimensions.bot_w:.4f} m, "
				f"yaw_deg_camera={self.estimated_dimensions.yaw_deg_camera:.2f}, "
				f"volume=not_computed"
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