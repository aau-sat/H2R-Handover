import socket
import os
import json
import torch
import numpy as np
import trimesh
import argparse
import time
import open3d as o3d
import meshcat.geometry as g
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import (
    point_cloud_outlier_removal,
    filter_colliding_grasps,
)
from grasp_gen.robot import get_gripper_info
import trimesh.transformations as tra

# Optional: For visualization
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    make_frame,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)

# Configuration Constants
SOCKET_PATH = "/tmp/graspgen"
REQUEST_PREGENERATE = 1
REQUEST_HAND_FILTER = 2
REQUEST_REGENERATE = 3

def voxel_downsample(points, voxel_size=0.004):
    """ 
    Reduces point cloud density for significantly faster collision checking.
    """
    if len(points) == 0: return points
    # Round coordinates to the nearest voxel grid cell
    coords = (points / voxel_size).astype(int)
    # Keep only one point per voxel cell
    _, indices = np.unique(coords, axis=0, return_index=True)
    return points[indices]


def _safe_normalize(vec, fallback=None):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        if fallback is None:
            return None
        vec = np.asarray(fallback, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return None
    return vec / norm


def _axis_to_basis(axis):
    axis = _safe_normalize(axis, fallback=[0.0, 0.0, 1.0])
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    x_axis = _safe_normalize(np.cross(reference, axis), fallback=[1.0, 0.0, 0.0])
    if x_axis is None:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    y_axis = _safe_normalize(np.cross(axis, x_axis), fallback=[0.0, 1.0, 0.0])
    if y_axis is None:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return x_axis, y_axis, axis


def _sample_frustum_surface_points(axis, center, height, top_w, bot_w, n_height=50, n_theta=50):
    axis = _safe_normalize(axis, fallback=[0.0, 0.0, 1.0])
    if axis is None:
        return np.zeros((0, 3), dtype=np.float32)

    x_axis, y_axis, z_axis = _axis_to_basis(axis)
    center = np.asarray(center, dtype=np.float32)
    height = float(max(height, 1e-4))
    top_r = float(max(top_w, 1e-4)) / 2.0
    bot_r = float(max(bot_w, 1e-4)) / 2.0

    z_samples = np.linspace(-0.5 * height, 0.5 * height, n_height)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    points = []
    for z in z_samples:
        alpha = (z + 0.5 * height) / height
        radius = (1.0 - alpha) * bot_r + alpha * top_r
        ring = (
            center[None, :]
            + z_axis[None, :] * z
            + x_axis[None, :] * (radius * cos_theta)[:, None]
            + y_axis[None, :] * (radius * sin_theta)[:, None]
        )
        points.append(ring)

    points.append(center[None, :])
    return np.vstack(points).astype(np.float32)


def _sample_rounded_cup_surface_points(
    axis,
    center,
    height,
    cup_w,
    bottom_round_ratio=0.35,
    n_height=60,
    n_theta=50,
):
    axis = _safe_normalize(axis, fallback=[0.0, 0.0, 1.0])
    if axis is None:
        return np.zeros((0, 3), dtype=np.float32)

    x_axis, y_axis, z_axis = _axis_to_basis(axis)
    center = np.asarray(center, dtype=np.float32)
    height = float(max(height, 1e-4))
    radius = float(max(cup_w, 1e-4)) / 2.0

    # Lower part is rounded (hemisphere-like), upper part stays cylindrical.
    round_h = float(np.clip(bottom_round_ratio, 0.1, 0.65)) * height
    round_h = float(min(max(round_h, 1e-4), height - 1e-4))

    z_samples = np.linspace(-0.5 * height, 0.5 * height, n_height)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    z_round_end = -0.5 * height + round_h
    points = []
    for z in z_samples:
        if z <= z_round_end:
            s = (z - (-0.5 * height)) / round_h
            s = float(np.clip(s, 0.0, 1.0))
            ring_r = radius * np.sqrt(max(0.0, 1.0 - (1.0 - s) ** 2))
        else:
            ring_r = radius

        ring = (
            center[None, :]
            + z_axis[None, :] * z
            + x_axis[None, :] * (ring_r * cos_theta)[:, None]
            + y_axis[None, :] * (ring_r * sin_theta)[:, None]
        )
        points.append(ring)

    points.append(center[None, :])
    return np.vstack(points).astype(np.float32)


def _points_inside_frustum_mask(points, axis, center, height, top_w, bot_w, radial_margin=0.004, z_margin=0.004, strict=False):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)

    axis = _safe_normalize(axis, fallback=[0.0, 0.0, 1.0])
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis, y_axis, z_axis = _axis_to_basis(axis)
    center = np.asarray(center, dtype=np.float32)

    height = float(max(height, 1e-4))
    top_r = float(max(top_w, 1e-4)) / 2.0
    bot_r = float(max(bot_w, 1e-4)) / 2.0

    rel = points - center[None, :]
    z_local = rel @ z_axis
    x_local = rel @ x_axis
    y_local = rel @ y_axis

    half_h = 0.5 * height
    if strict:
        z_mask = (z_local > -half_h) & (z_local < half_h)
    else:
        z_mask = (z_local >= (-half_h - z_margin)) & (z_local <= (half_h + z_margin))

    alpha = (z_local + half_h) / height
    alpha = np.clip(alpha, 0.0, 1.0)
    radius = (1.0 - alpha) * bot_r + alpha * top_r
    radial = np.sqrt(x_local * x_local + y_local * y_local)
    if strict:
        radial_mask = radial < radius
    else:
        radial_mask = radial <= (radius + radial_margin)

    return z_mask & radial_mask


def _filter_points_inside_frustum(points, axis, center, height, top_w, bot_w, radial_margin=0.004, z_margin=0.004):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points
    mask = _points_inside_frustum_mask(
        points=points,
        axis=axis,
        center=center,
        height=height,
        top_w=top_w,
        bot_w=bot_w,
        radial_margin=radial_margin,
        z_margin=z_margin,
        strict=False,
    )
    return points[mask]


def _remove_points_inside_frustums(points, frustums, strict=True):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0 or not frustums:
        return points

    keep_mask = np.ones(len(points), dtype=bool)
    for fr in frustums:
        inside = _points_inside_frustum_mask(
            points=points,
            axis=fr['axis'],
            center=fr['center'],
            height=fr['height'],
            top_w=fr['top_w'],
            bot_w=fr['bot_w'],
            strict=strict,
        )
        keep_mask &= ~inside
    return points[keep_mask]


def _remove_points_inside_centered_object_walls(
    hand_points,
    object_points,
    wall_epsilon=0.003,
    num_bins=48,
    min_points_per_bin=8,
):
    hand_points = np.asarray(hand_points, dtype=np.float32)
    object_points = np.asarray(object_points, dtype=np.float32)
    if len(hand_points) == 0 or len(object_points) == 0:
        return hand_points

    z_obj = object_points[:, 2]
    z_min = float(np.percentile(z_obj, 1.0))
    z_max = float(np.percentile(z_obj, 99.0))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min + 1e-4:
        return hand_points

    bins = np.linspace(z_min, z_max, max(int(num_bins), 8) + 1)
    obj_bin_ids = np.clip(np.digitize(z_obj, bins) - 1, 0, len(bins) - 2)

    bin_centers = []
    xy_centers = []
    wall_radii = []
    for bin_idx in range(len(bins) - 1):
        mask = obj_bin_ids == bin_idx
        if np.count_nonzero(mask) < min_points_per_bin:
            continue

        pts = object_points[mask]
        xy_center = np.median(pts[:, :2], axis=0)
        radial = np.linalg.norm(pts[:, :2] - xy_center[None, :], axis=1)
        radius = float(np.percentile(radial, 90.0))
        if not np.isfinite(radius) or radius <= 1e-4:
            continue

        bin_centers.append(0.5 * (bins[bin_idx] + bins[bin_idx + 1]))
        xy_centers.append(xy_center)
        wall_radii.append(radius)

    if len(wall_radii) < 2:
        return hand_points

    bin_centers = np.asarray(bin_centers, dtype=np.float32)
    xy_centers = np.asarray(xy_centers, dtype=np.float32)
    wall_radii = np.asarray(wall_radii, dtype=np.float32)

    if len(wall_radii) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        padded_radii = np.pad(wall_radii, (2, 2), mode='edge')
        wall_radii = np.convolve(padded_radii, kernel, mode='valid')

    z_hand = hand_points[:, 2]
    z_inside = (z_hand >= bin_centers[0]) & (z_hand <= bin_centers[-1])
    if not np.any(z_inside):
        return hand_points

    interp_radius = np.interp(z_hand, bin_centers, wall_radii)
    interp_x = np.interp(z_hand, bin_centers, xy_centers[:, 0])
    interp_y = np.interp(z_hand, bin_centers, xy_centers[:, 1])
    radial_hand = np.linalg.norm(
        hand_points[:, :2] - np.stack([interp_x, interp_y], axis=1),
        axis=1,
    )

    inside_object = z_inside & (radial_hand <= (interp_radius + float(max(wall_epsilon, 0.0))))
    return hand_points[~inside_object]


def _estimate_wine_glass_cup_band(points, axis, center, height, num_bins=32):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return None

    axis = _safe_normalize(axis, fallback=[0.0, 0.0, 1.0])
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis, y_axis, z_axis = _axis_to_basis(axis)
    center = np.asarray(center, dtype=np.float32)

    rel = points - center[None, :]
    z_local = rel @ z_axis
    radial = np.sqrt((rel @ x_axis) ** 2 + (rel @ y_axis) ** 2)

    z_min = float(np.min(z_local))
    z_max = float(np.max(z_local))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min + 1e-4:
        return None

    bins = np.linspace(z_min, z_max, max(int(num_bins), 8) + 1)
    bin_ids = np.clip(np.digitize(z_local, bins) - 1, 0, len(bins) - 2)

    bin_centers = []
    bin_radii = []
    for bin_idx in range(len(bins) - 1):
        mask = bin_ids == bin_idx
        if np.count_nonzero(mask) < 8:
            continue
        bin_centers.append(0.5 * (bins[bin_idx] + bins[bin_idx + 1]))
        bin_radii.append(float(np.median(radial[mask])))

    if len(bin_radii) < 4:
        return None

    radii = np.asarray(bin_radii, dtype=np.float32)
    bin_centers = np.asarray(bin_centers, dtype=np.float32)
    if len(radii) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        padded = np.pad(radii, (2, 2), mode='edge')
        radii_smooth = np.convolve(padded, kernel, mode='valid')
    else:
        radii_smooth = radii

    profile_low = float(np.percentile(radii_smooth, 20.0))
    profile_high = float(np.percentile(radii_smooth, 85.0))
    if not np.isfinite(profile_low) or not np.isfinite(profile_high) or profile_high <= 1e-4:
        print(f"Invalid radius profile: profile_low={profile_low}, profile_high={profile_high}")
        return None

    # A wine/glass cup is the sustained wide part of the vertical radius profile.
    # Do not assume it flares upward: some glasses have bot_w > top_w, and the foot
    # can also make the lower quartile wider than the rim side.
    contrast = profile_high - profile_low
    if contrast <= max(0.002, 0.08 * profile_high):
        cup_start = float(bin_centers[0])
        cup_end = float(bin_centers[-1])
        cup_radius = float(np.median(radii_smooth))
        return {
            'cup_start': cup_start,
            'cup_end': cup_end,
            'cup_height': cup_end - cup_start,
            'cup_radius': cup_radius,
            'cup_bot_w': max(2.0 * cup_radius, 1e-4),
        }

    threshold = profile_low + 0.45 * contrast
    above = np.where(radii_smooth >= threshold)[0]
    if len(above) == 0:
        return None

    runs = []
    run_start = int(above[0])
    prev = int(above[0])
    for idx in above[1:]:
        idx = int(idx)
        if idx != prev + 1:
            runs.append((run_start, prev))
            run_start = idx
        prev = idx
    runs.append((run_start, prev))

    # Prefer wide runs that reach the upper part of the object. This rejects a
    # wide table-contact foot while still accepting regular tumblers or cups.
    z_span = max(float(bin_centers[-1] - bin_centers[0]), 1e-4)
    upper_cut = float(bin_centers[0] + 0.45 * z_span)
    candidate_runs = [run for run in runs if float(bin_centers[run[1]]) >= upper_cut]
    if not candidate_runs:
        candidate_runs = runs

    def _run_score(run):
        start, end = run
        length = end - start + 1
        mean_radius = float(np.mean(radii_smooth[start:end + 1]))
        end_height = (float(bin_centers[end]) - float(bin_centers[0])) / z_span
        return (end_height, length, mean_radius)

    cup_start_idx, cup_end_idx = max(candidate_runs, key=_run_score)
    cup_start = float(bin_centers[cup_start_idx])
    cup_end = float(bin_centers[cup_end_idx])
    if cup_end <= cup_start + 1e-4:
        return None

    cup_radius = float(np.percentile(radii_smooth[cup_start_idx:cup_end_idx + 1], 85.0))
    return {
        'cup_start': cup_start,
        'cup_end': cup_end,
        'cup_height': cup_end - cup_start,
        'cup_radius': cup_radius,
        'cup_bot_w': max(2.0 * cup_radius, 1e-4),
    }


def _split_points_by_axis_band(points, axis, center, band_start, band_end, z_margin=0.004):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty

    axis = _safe_normalize(axis, fallback=[0.0, 0.0, 1.0])
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis, y_axis, z_axis = _axis_to_basis(axis)
    center = np.asarray(center, dtype=np.float32)

    rel = points - center[None, :]
    z_local = rel @ z_axis
    outside_mask = (z_local < (band_start - z_margin)) | (z_local > (band_end + z_margin))
    inside_mask = ~outside_mask
    return points[outside_mask], points[inside_mask]


def _build_object_proxy(object_pc, completion_meta, return_frustums=False):
    if completion_meta is None or not completion_meta.get('reconstruct', False):
        empty = np.zeros((0, 3), dtype=np.float32)
        return (empty, []) if return_frustums else empty

    object_pc = np.asarray(object_pc, dtype=np.float32)
    if len(object_pc) == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return (empty, []) if return_frustums else empty

    frustums = []

    axis_prior = completion_meta.get('axis_world')
    if axis_prior is not None:
        axis_prior = _safe_normalize(axis_prior, fallback=[0.0, 0.0, 1.0])
    center_prior = completion_meta.get('center_world')
    if center_prior is not None:
        center_prior = np.asarray(center_prior, dtype=np.float32)
        if center_prior.shape != (3,) or not np.all(np.isfinite(center_prior)):
            center_prior = None

    # Always use metadata prior when available; otherwise fall back to world-up.
    axis = axis_prior if axis_prior is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    center = center_prior if center_prior is not None else object_pc.mean(axis=0)
    object_label = str(completion_meta.get('object_label', '') or '').strip().lower()

    height = completion_meta.get('height')
    top_w = completion_meta.get('top_w')
    bot_w = completion_meta.get('bot_w')

    if height is None or not np.isfinite(height) or height <= 0:
        axis_proj = (object_pc - center[None]) @ axis
        height = float(np.percentile(axis_proj, 99.0) - np.percentile(axis_proj, 1.0))
        if height <= 1e-4:
            height = float(np.linalg.norm(object_pc.max(axis=0) - object_pc.min(axis=0)))

    if top_w is None or not np.isfinite(top_w) or top_w <= 0:
        top_w = float(np.linalg.norm(object_pc.max(axis=0) - object_pc.min(axis=0))) * 0.6
    if bot_w is None or not np.isfinite(bot_w) or bot_w <= 0:
        bot_w = top_w
    
    print(f"Proxy estimation for '{object_label}': axis={axis}, center={center}, height={height:.3f}, top_w={top_w:.3f}, bot_w={bot_w:.3f}")

    if 'glass' in object_label:
        cup_band = _estimate_wine_glass_cup_band(
            points=object_pc,
            axis=axis,
            center=center,
            height=height,
        )
        if cup_band is not None:
            cup_start = cup_band['cup_start']
            cup_end = cup_band['cup_end']
            cup_height_raw = cup_band['cup_height']
            # Shift start upward to avoid stem-cup transition artifacts.
            cup_start = cup_start + 0.12 * cup_height_raw
            cup_height = max(cup_end - cup_start, 1e-4)
            cup_width = max(float(top_w), float(cup_band.get('cup_bot_w', 0.0)))
            stem_pc, cup_pc = _split_points_by_axis_band(
                points=object_pc,
                axis=axis,
                center=center,
                band_start=cup_start,
                band_end=cup_end,
            )
            cup_center = center + axis * (0.5 * (cup_start + cup_end))
            cup_proxy = _sample_rounded_cup_surface_points(
                axis=axis,
                center=cup_center,
                height=cup_height,
                cup_w=cup_width,
                bottom_round_ratio=0.4,
            )
            frustums.append({
                'axis': axis,
                'center': cup_center,
                'height': cup_height,
                'top_w': cup_width,
                'bot_w': cup_width,
            })
            cup_inlier = _filter_points_inside_frustum(
                points=cup_pc,
                axis=axis,
                center=cup_center,
                height=cup_height,
                top_w=cup_width,
                bot_w=cup_width,
            )

            parts = []
            if len(stem_pc) > 0:
                parts.append(stem_pc)
            if len(cup_inlier) > 0:
                parts.append(cup_inlier)
            if len(cup_proxy) > 0:
                parts.append(cup_proxy)

            if len(parts) == 0:
                empty = np.zeros((0, 3), dtype=np.float32)
                return (empty, frustums) if return_frustums else empty
            merged = np.vstack(parts) if len(parts) > 1 else parts[0]
            down = voxel_downsample(merged, voxel_size=0.0035)
            return (down, frustums) if return_frustums else down

    proxy = _sample_frustum_surface_points(
        axis=axis,
        center=center,
        height=height,
        top_w=top_w,
        bot_w=bot_w,
    )
    frustums.append({
        'axis': axis,
        'center': center,
        'height': height,
        'top_w': top_w,
        'bot_w': bot_w,
    })

    object_pc_inlier = _filter_points_inside_frustum(
        points=object_pc,
        axis=axis,
        center=center,
        height=height,
        top_w=top_w,
        bot_w=bot_w,
    )

    if len(proxy) == 0 and len(object_pc_inlier) == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return (empty, frustums) if return_frustums else empty

    if len(proxy) == 0:
        merged = object_pc_inlier
    elif len(object_pc_inlier) == 0:
        merged = proxy
    else:
        merged = np.vstack([object_pc_inlier, proxy])

    down = voxel_downsample(merged, voxel_size=0.0035)
    return (down, frustums) if return_frustums else down

def recv_all(conn, n):
    """ Read as long data is received """
    data = bytearray()
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data


def _send_int32(conn, value):
    conn.sendall(np.array([int(value)], dtype=np.int32).tobytes())


def _send_grasp_response(conn, grasps, labels, widths=None):
    grasps = [] if grasps is None else list(grasps)
    labels = [] if labels is None else list(labels)

    if len(grasps) == 0:
        _send_int32(conn, 0)
        print("[Width] Sent 0 grasp widths (empty response)")
        return

    _send_int32(conn, len(grasps))
    for grasp in grasps:
        conn.sendall(np.asarray(grasp, dtype=np.float32).tobytes())
    if len(labels) < len(grasps):
        labels.extend([0] * (len(grasps) - len(labels)))
    for label in labels[:len(grasps)]:
        _send_int32(conn, label)
        
    # Send widths: one float32 per grasp (max object width in metres)
    if widths is None:
        widths = np.zeros(len(grasps), dtype=np.float32)
    widths = np.asarray(widths, dtype=np.float32)
    if len(widths) < len(grasps):
        widths = np.concatenate([widths, np.zeros(len(grasps) - len(widths), dtype=np.float32)])
    widths_to_send = widths[:len(grasps)]
    conn.sendall(widths_to_send.tobytes())

    preview_count = min(8, len(widths_to_send))
    preview_values = ", ".join(f"{float(w):.4f}" for w in widths_to_send[:preview_count])
    preview_suffix = "..." if len(widths_to_send) > preview_count else ""
    print(f"[Width] Sent {len(widths_to_send)} grasp widths (m): [{preview_values}{preview_suffix}]")
    print(
        f"[Width] Stats: min={float(np.min(widths_to_send)):.4f}m "
        f"max={float(np.max(widths_to_send)):.4f}m "
        f"mean={float(np.mean(widths_to_send)):.4f}m"
    )

def filter_by_spherical_constraints(grasp_poses, azimuth_range=(0, 360), elevation_range=(-90, 90)):
    """ Filters grasps based on camera-space approach angles. """
    vecs = grasp_poses[:, :3, 2] # Approach vector (Z-axis of gripper)
    mags = np.linalg.norm(vecs, axis=1, keepdims=True)
    vn = vecs / (mags + 1e-8)
    
    phi_deg = np.degrees(np.arctan2(vn[:, 0], -vn[:, 1])) % 360
    theta_deg = np.degrees(np.arcsin(np.clip(vn[:, 2], -1.0, 1.0)))
    
    a_min, a_max = azimuth_range
    if a_min <= a_max:
        phi_mask = (phi_deg >= a_min) & (phi_deg <= a_max)
    else:
        phi_mask = (phi_deg >= a_min) | (phi_deg <= a_max)
        
    e_min, e_max = elevation_range
    theta_mask = (theta_deg >= e_min) & (theta_deg <= e_max)
    
    return phi_mask & theta_mask


def apply_spherical_filter_adaptive_phi(
    grasp_poses, scores, azimuth_range, elevation_range,
    phi_step=5.0, max_expansion=180.0
):
    """
    Applies the spherical constraint filter but softens *only* the phi (azimuth)
    range iteratively when no grasp survives the strict constraint.

    Strategy:
      1. Try the original phi range (and the fixed theta range).
      2. If no grasp survives, widen phi symmetrically by `phi_step` degrees on
         both sides and retry.  Repeat until at least one grasp is found or the
         expansion reaches `max_expansion` (i.e. the full circle).
      3. Once a non-empty set is found, perform **one additional expansion step**
         so we collect more candidates.  This lets us pick the most confident
         grasp from a slightly wider pool rather than being forced to take the
         single lucky survivor of the first successful iteration.
      4. Return the filtered poses/scores together with a flag indicating whether
         the constraints were softened.
    """
    orig_a_min, orig_a_max = azimuth_range
    orig_half_width = (orig_a_max - orig_a_min) / 2.0  # half-width of original window
    orig_center = (orig_a_min + orig_a_max) / 2.0      # centre of the original window

    expansion = 0.0          # how many degrees we have widened on each side
    softened = False
    found_on_prev = False     # True once we found grasps and are doing the final step

    while True:
        a_min = orig_center - (orig_half_width + expansion)
        a_max = orig_center + (orig_half_width + expansion)

        # Clamp to [0, 360) and handle wrap-around
        a_min_mod = a_min % 360
        a_max_mod = a_max % 360

        mask = filter_by_spherical_constraints(
            grasp_poses,
            azimuth_range=(a_min_mod, a_max_mod),
            elevation_range=elevation_range,
        )
        candidates = grasp_poses[mask]

        if len(candidates) > 0:
            if expansion == 0.0:
                # Strict constraints already gave us grasps — return immediately,
                # no need for an extra iteration.
                return mask, False
            if found_on_prev:
                # This is the bonus "one last step" after softening — return.
                total_expansion = expansion - phi_step  # expansion at first success
                print(
                    f"[AdaptivePhi] Constraints softened by ±{total_expansion:.1f}° on phi. "
                    f"Returning {len(candidates)} candidate(s) from final window "
                    f"[{a_min_mod:.1f}°, {a_max_mod:.1f}°]."
                )
                return mask, True
            else:
                # First softened iteration that found grasps — do one more step.
                found_on_prev = True
                softened = True
        else:
            if found_on_prev:
                # Previous step found grasps but this wider step somehow found
                # none (edge case with wrap-around arithmetic) — use prev result.
                # Re-run previous window to get those grasps back.
                prev_a_min = orig_center - (orig_half_width + expansion - phi_step)
                prev_a_max = orig_center + (orig_half_width + expansion - phi_step)
                prev_mask = filter_by_spherical_constraints(
                    grasp_poses,
                    azimuth_range=(prev_a_min % 360, prev_a_max % 360),
                    elevation_range=elevation_range,
                )
                return prev_mask, softened

        # Expand phi window for next iteration
        expansion += phi_step

        if expansion > max_expansion:
            # Full circle searched — return whatever theta filter keeps
            mask_theta_only = filter_by_spherical_constraints(
                grasp_poses,
                azimuth_range=(0, 360),
                elevation_range=elevation_range,
            )
            print(
                "[AdaptivePhi] Max phi expansion reached. "
                f"Returning {mask_theta_only.sum()} grasp(s) with theta-only filter."
            )
            return mask_theta_only, True

def filter_by_finger_orientation(grasp_poses, max_tilt_deg=20):
    """ Ensures fingers are horizontal (left-right) relative to camera Y. """
    finger_axis = grasp_poses[:, :3, 0] # X-axis of gripper
    mags = np.linalg.norm(finger_axis, axis=1, keepdims=True)
    finger_axis_n = finger_axis / (mags + 1e-8)
    
    z_component = np.abs(finger_axis_n[:, 2])
    threshold = np.sin(np.radians(max_tilt_deg))
    return z_component < threshold


def filter_by_gripper_z_rotation(grasp_poses, max_roll_deg=20):
    """Filter grasps by roll about local gripper Z.

    Roll is measured with the gripper Y axis against world-up, both projected
    into the plane orthogonal to local gripper Z. This matches the convention
    where gripper Z points toward the object and gripper Y should stay upright.

    Symmetry handling: 180-degree flips are treated as equivalent (0 == 180),
    so +30 and -330 are naturally identical under this metric.
    """
    if len(grasp_poses) == 0:
        return np.zeros((0,), dtype=bool)

    max_roll_deg = float(max_roll_deg)
    if max_roll_deg < 0:
        max_roll_deg = -max_roll_deg

    y_axis = grasp_poses[:, :3, 1]
    z_axis = grasp_poses[:, :3, 2]

    y_norm = np.linalg.norm(y_axis, axis=1, keepdims=True)
    z_norm = np.linalg.norm(z_axis, axis=1, keepdims=True)
    y_axis_n = y_axis / (y_norm + 1e-8)
    z_axis_n = z_axis / (z_norm + 1e-8)

    world_up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(grasp_poses), 1))
    alt_refs = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(grasp_poses), 1))

    ref_proj = world_up - np.sum(world_up * z_axis_n, axis=1, keepdims=True) * z_axis_n
    ref_norm = np.linalg.norm(ref_proj, axis=1, keepdims=True)
    use_alt = ref_norm[:, 0] < 1e-6

    if np.any(use_alt):
        alt_proj = alt_refs - np.sum(alt_refs * z_axis_n, axis=1, keepdims=True) * z_axis_n
        ref_proj[use_alt] = alt_proj[use_alt]
        ref_norm = np.linalg.norm(ref_proj, axis=1, keepdims=True)

    valid_ref = ref_norm[:, 0] > 1e-6
    ref_proj_n = np.zeros_like(ref_proj)
    ref_proj_n[valid_ref] = ref_proj[valid_ref] / ref_norm[valid_ref]

    y_proj = y_axis_n - np.sum(y_axis_n * z_axis_n, axis=1, keepdims=True) * z_axis_n
    y_proj_norm = np.linalg.norm(y_proj, axis=1, keepdims=True)
    valid_y = y_proj_norm[:, 0] > 1e-6
    y_proj_n = np.zeros_like(y_proj)
    y_proj_n[valid_y] = y_proj[valid_y] / y_proj_norm[valid_y]

    cos_angle = np.sum(y_proj_n * ref_proj_n, axis=1)
    cos_angle = np.clip(np.abs(cos_angle), 0.0, 1.0)
    roll_deg = np.degrees(np.arccos(cos_angle))

    return valid_ref & valid_y & (roll_deg <= max_roll_deg)


def flatten_grasp_orientations(grasp_poses: np.ndarray) -> np.ndarray:
    """
    Rotate each grasp about its own approach axis (gripper Z) so that the
    finger-spread axis (gripper X) becomes perfectly horizontal in the world
    frame (zero world-Z component).  The approach direction is preserved.

    This is useful when the robot must approach from a fixed direction but the
    gripper fingers should always be level (no roll).

    Args:
        grasp_poses: Kx4x4 array of grasp poses.

    Returns:
        Kx4x4 array with the same translation but re-oriented rotation.
    """
    world_z = np.array([0.0, 0.0, 1.0])
    result = grasp_poses.copy()

    for i, pose in enumerate(grasp_poses):
        R = pose[:3, :3]          # current rotation
        z = R[:, 2]               # approach direction — keep this fixed

        # Project old finger-X onto the world-horizontal plane
        x_old = R[:, 0]
        x_flat = x_old - np.dot(x_old, world_z) * world_z

        # If x is already pointing straight up/down (degenerate), skip
        if np.linalg.norm(x_flat) < 1e-6:
            continue

        # Make x perpendicular to the approach z as well
        x_flat = x_flat - np.dot(x_flat, z) * z
        norm = np.linalg.norm(x_flat)
        if norm < 1e-6:
            continue
        x_flat /= norm

        # Recompute y to keep a right-handed frame
        y_new = np.cross(z, x_flat)
        y_norm = np.linalg.norm(y_new)
        if y_norm < 1e-6:
            continue
        y_new /= y_norm

        # Assemble new rotation (columns: x, y, z unchanged)
        R_new = np.stack([x_flat, y_new, z], axis=1)   # (3, 3)
        result[i, :3, :3] = R_new

    return result


def _make_grasp_parallel_to_ground(pose: np.ndarray, pivot_offset: float = 0.15) -> np.ndarray:
    """
    Force a grasp orientation to be level with the ground:
    - approach axis (gripper Z) is projected to the world XY plane
    - finger axis (gripper X) is also horizontal and orthogonal to Z

    The grasp is rotated around the contact point located `pivot_offset`
    meters along the local gripper Z axis, so the touch point stays fixed.
    """
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    result = np.array(pose, dtype=np.float32, copy=True)
    R = result[:3, :3]
    t_old = result[:3, 3].copy()

    z_old = R[:, 2]
    z_new = z_old - np.dot(z_old, world_up) * world_up
    z_norm = np.linalg.norm(z_new)
    if z_norm < 1e-6:
        # Degenerate case: preserve original orientation.
        return result
    z_new = z_new / z_norm

    x_new = np.cross(world_up, z_new)
    x_norm = np.linalg.norm(x_new)
    if x_norm < 1e-6:
        return result
    x_new = x_new / x_norm

    y_new = np.cross(z_new, x_new)
    y_norm = np.linalg.norm(y_new)
    if y_norm < 1e-6:
        return result
    y_new = y_new / y_norm

    pivot_local = np.array([0.0, 0.0, float(pivot_offset)], dtype=np.float32)
    touch_world = t_old + R @ pivot_local

    result[:3, :3] = np.stack([x_new, y_new, z_new], axis=1)
    result[:3, 3] = touch_world - result[:3, :3] @ pivot_local
    return result

def _count_hand_points_in_finger_gap(
    hand_pc: np.ndarray,
    grasp_poses: np.ndarray,
    finger_half_width: float = 0.068,
    finger_height: float = 0.01,
    finger_tip_depth: float = 0.085,
    fingertip_z_offset: float = 0.11,
) -> np.ndarray:
    """
    For each grasp pose, count how many hand point-cloud points fall inside
    the finger-gap bounding box (in the gripper's local frame).

    The bounding box is defined in gripper-local coordinates as:
        X: [-finger_half_width, +finger_half_width]  (finger spread)
        Y: [-finger_height,     +finger_height]      (finger height)
        Z: [fingertip_z_offset - finger_tip_depth,
            fingertip_z_offset + finger_tip_depth]   (depth into object)

    Returns:
        overlap_counts: (K,) int array with the count per grasp.
    """
    if len(hand_pc) == 0 or len(grasp_poses) == 0:
        return np.zeros(len(grasp_poses), dtype=np.int32)

    hand_pc_h = np.hstack([hand_pc, np.ones((len(hand_pc), 1), dtype=np.float32)])  # (N, 4)
    overlap_counts = np.zeros(len(grasp_poses), dtype=np.int32)

    z_lo = fingertip_z_offset - finger_tip_depth
    z_hi = fingertip_z_offset + finger_tip_depth

    for i, pose in enumerate(grasp_poses):
        # Transform hand points into grasp-local frame
        try:
            T_inv = np.linalg.inv(pose)
        except np.linalg.LinAlgError:
            continue
        local = (T_inv @ hand_pc_h.T).T[:, :3]  # (N, 3)
        inside = (
            (local[:, 0] >= -finger_half_width) & (local[:, 0] <= finger_half_width) &
            (local[:, 1] >= -finger_height)      & (local[:, 1] <= finger_height) &
            (local[:, 2] >= z_lo)                & (local[:, 2] <= z_hi)
        )
        overlap_counts[i] = int(np.sum(inside))

    return overlap_counts


def filter_grasps_finger_gap_raycasting(
    scene_pc: np.ndarray,
    grasp_poses: np.ndarray,
    # Robotiq 2F-140 geometry (from gripper config):
    #   contact_points[*].location[Z] = 0.195  → fingertip_z_offset
    #   contact_points[*].location[X] = ±0.068 → finger_half_width (max aperture)
    #   closing_regions.extents = [0.12, 0.02, 0.06]
    #     half Y = 0.01 → finger_height
    #     half Z = 0.03 → finger_tip_depth
    finger_half_width: float = 0.068,
    finger_height: float = 0.01,
    finger_tip_depth: float = 0.085,
    fingertip_z_offset: float = 0.195,
    ray_grid_y: int = 20,
    ray_grid_z: int = 60,
    voxel_size: float = 0.004,
    return_debug_rays: bool = False,
    return_widths: bool = True,
):
    """
    Filter grasps where scene points are located between the gripper fingers
    using Open3D ray casting.

    A grid of rays is cast from the face of one finger tip to the other,
    covering the entire fingertip rectangular area (Y x Z plane at
    ±finger_half_width along the gripper X-axis).  If ANY ray hits the
    voxelised scene mesh before reaching the opposite finger, scene geometry
    is inside the finger gap and the grasp is rejected.

    Gripper convention (GraspGen):
        X-axis  — finger spread direction (ray travel direction)
        Y-axis  — finger height direction
        Z-axis  — approach / depth direction (into object)

    Args:
        scene_pc:           Nx3 scene point cloud (world frame, same as grasps)
        grasp_poses:        Kx4x4 homogeneous grasp poses
        finger_half_width:  Half the gap between the two finger faces (m).
                            Rays travel from –X to +X over 2*finger_half_width.
        finger_height:      Half-height of the fingertip pad along the gripper
                            Y-axis (m).  Rays are sampled in [−finger_height,
                            +finger_height].
        finger_tip_depth:   Half-depth of the fingertip pad along the gripper
                            Z-axis (m).  Rays are sampled in [−finger_tip_depth,
                            +finger_tip_depth].
        fingertip_z_offset: Offset along the gripper Z-axis (approach direction)
                            from the grasp origin to the centre of the fingertip
                            face (m).  GraspGen places the grasp origin at the
                            palm / wrist; a positive value shifts the ray plane
                            forward toward the fingertip contact surface.
        ray_grid_y:         Number of ray samples along the Y-axis of the finger pad.
        ray_grid_z:         Number of ray samples along the Z-axis of the finger pad.
        voxel_size:         Side length of the small boxes used to voxelise the
                            scene point cloud.  Smaller = more accurate, slower.

     Returns:
        If return_debug_rays is False and return_widths is False:
            mask: K-length boolean array, True if the finger gap is clear.
        If return_debug_rays is True:
            (mask, debug) where debug stores clipped ray segments from cast results.
        If return_widths is True:
            An additional grasp_widths (K,) float32 array is appended to the return
            tuple containing the maximum object width seen by any ray per grasp.
            Width is computed via dual-direction raycasting: width = gap - (t_fwd + t_rev).
    """

    if len(scene_pc) == 0 or len(grasp_poses) == 0:
        empty_mask = np.ones(len(grasp_poses), dtype=bool)
        empty_widths = np.zeros(len(grasp_poses), dtype=np.float32)
        if return_debug_rays:
            result = (empty_mask, {
                "origins": np.zeros((0, 3), dtype=np.float32),
                "endpoints_clipped": np.zeros((0, 3), dtype=np.float32),
                "hit_mask": np.zeros((0,), dtype=bool),
                "max_dist": 2.0 * finger_half_width,
                "n_per": 0,
                "gap_clear": empty_mask,
                "voxel_centers": np.zeros((0, 3), dtype=np.float32),
            })
            return result + (empty_widths,) if return_widths else result
        return (empty_mask, empty_widths) if return_widths else empty_mask


    # ------------------------------------------------------------------
    # 1. Build RaycastingScene: voxelise scene PC → one small box per voxel.
    # ------------------------------------------------------------------
    coords = np.round(scene_pc / voxel_size).astype(int)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    voxel_centers = scene_pc[unique_idx]

    print(f"[FingerGap] Building RaycastingScene from {len(voxel_centers)} voxels "
          f"(voxel_size={voxel_size}m, grid={ray_grid_y}x{ray_grid_z}, "
          f"z_offset={fingertip_z_offset:.3f}m) ...")

    unit_box = o3d.geometry.TriangleMesh.create_box(
        width=voxel_size, height=voxel_size, depth=voxel_size
    )
    unit_box.translate(-np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2]))
    unit_verts = np.asarray(unit_box.vertices)   # (8, 3)
    unit_tris  = np.asarray(unit_box.triangles)  # (12, 3)

    all_vertices, all_triangles, vert_offset = [], [], 0
    for center in voxel_centers:
        all_vertices.append(unit_verts + center)
        all_triangles.append(unit_tris + vert_offset)
        vert_offset += len(unit_verts)

    merged_mesh = o3d.geometry.TriangleMesh()
    merged_mesh.vertices  = o3d.utility.Vector3dVector(np.vstack(all_vertices))
    merged_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(all_triangles))

    t_mesh   = o3d.t.geometry.TriangleMesh.from_legacy(merged_mesh)
    rc_scene = o3d.t.geometry.RaycastingScene()
    rc_scene.add_triangles(t_mesh)

    # ------------------------------------------------------------------
    # 2. Build ray grid.
    #
    #  For each grasp we create ray_grid_y × ray_grid_z rays.  Each ray
    #  starts on the face of the –X finger and shoots along the +X direction.
    #
    #  Local offsets on the finger face plane (Y-Z):
    #    y ∈ linspace(−finger_height,    +finger_height,    ray_grid_y)
    #    z ∈ linspace(−finger_tip_depth, +finger_tip_depth, ray_grid_z)
    #
    #  The entire plane is additionally shifted by fingertip_z_offset along
    #  the gripper Z-axis so the rays originate at the physical fingertip
    #  face rather than at the palm / grasp-origin depth.
    # ------------------------------------------------------------------
    ys = np.linspace(-finger_height,    finger_height,    ray_grid_y)
    zs = np.linspace(0, finger_tip_depth, ray_grid_z)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")          # (ray_grid_y, ray_grid_z)
    local_offsets = np.stack(
        [np.zeros_like(yy), yy, zz], axis=-1
    ).reshape(-1, 3)                                      # (n_rays_per_grasp, 3)
    n_per = len(local_offsets)                            # ray_grid_y * ray_grid_z

    K = len(grasp_poses)
    all_rays = np.zeros((K * n_per, 6), dtype=np.float32)
    max_dist = 2.0 * finger_half_width                    # ray must travel this far to be clear

    for i, pose in enumerate(grasp_poses):
        R = pose[:3, :3]                    # (3, 3)
        t = pose[:3,  3]                    # (3,)
        x_axis = R[:, 0]                    # gripper X  – ray direction
        y_axis = R[:, 1]                    # gripper Y  – finger height
        z_axis = R[:, 2]                    # gripper Z  – approach depth

        # World-space offsets from grasp origin for each grid point
        world_offsets = (local_offsets[:, 1:2] * y_axis[None, :]   # Y contribution
                       + local_offsets[:, 2:3] * z_axis[None, :])  # Z contribution

        # Ray origins: –X finger face, shifted forward to the physical fingertip.
        # fingertip_z_offset moves the whole plane along the approach axis (Z)
        # so it sits at the actual finger contact surface, not the palm origin.
        fingertip_center = t + z_axis * fingertip_z_offset
        origins = (fingertip_center - x_axis * finger_half_width) + world_offsets  # (n_per, 3)

        row = i * n_per
        all_rays[row:row + n_per, :3] = origins.astype(np.float32)
        all_rays[row:row + n_per, 3:] = x_axis.astype(np.float32)  # shared direction

    # ------------------------------------------------------------------
    # 3. Batch-cast all rays at once (forward: -X → +X).
    # ------------------------------------------------------------------
    rays_tensor = o3d.core.Tensor(all_rays, dtype=o3d.core.Dtype.Float32)
    ans = rc_scene.cast_rays(rays_tensor)
    t_hit_flat = ans["t_hit"].numpy()                    # (K*n_per,)
    t_hit = t_hit_flat.reshape(K, n_per)                 # (K, n_per)

    # A grasp is clear only when ALL rays pass through without hitting anything
    # within the finger gap distance.
    finite_hit = np.isfinite(t_hit)
    blocking_hit = finite_hit & (t_hit < max_dist)
    gap_clear = ~np.any(blocking_hit, axis=1)            # (K,)

    num_blocked = int(np.sum(~gap_clear))
    print(f"[FingerGap] {num_blocked}/{K} grasps blocked by scene geometry "
          f"inside finger gap (rays: {n_per} per grasp).")
    
    # ------------------------------------------------------------------
    # 4. Dual-direction raycasting to measure object width per grasp.
    #
    #  Reverse rays travel from the +X finger face back toward the -X face.
    #  For each ray pair:
    #      width = max_dist - (t_fwd + t_rev)
    #  clipped to [0, max_dist].  The per-grasp width is the maximum
    #  width observed across the whole ray grid.
    # ------------------------------------------------------------------
    grasp_widths = np.zeros(K, dtype=np.float32)
    if return_widths:
        all_rays_rev = all_rays.copy()
        for i, pose in enumerate(grasp_poses):
            R = pose[:3, :3]
            t = pose[:3,  3]
            x_axis = R[:, 0]
            y_axis = R[:, 1]
            z_axis = R[:, 2]
            world_offsets = (local_offsets[:, 1:2] * y_axis[None, :]
                           + local_offsets[:, 2:3] * z_axis[None, :])
            fingertip_center = t + z_axis * fingertip_z_offset
            origins_rev = (fingertip_center + x_axis * finger_half_width) + world_offsets
            row = i * n_per
            all_rays_rev[row:row + n_per, :3] = origins_rev.astype(np.float32)
            all_rays_rev[row:row + n_per, 3:] = (-x_axis).astype(np.float32)

        rays_rev_tensor = o3d.core.Tensor(all_rays_rev, dtype=o3d.core.Dtype.Float32)
        ans_rev = rc_scene.cast_rays(rays_rev_tensor)
        t_rev_flat = ans_rev["t_hit"].numpy()           # (K*n_per,)
        t_rev = t_rev_flat.reshape(K, n_per)            # (K, n_per)

        t_fwd_clipped = np.where(np.isfinite(t_hit), t_hit, max_dist)
        t_rev_clipped = np.where(np.isfinite(t_rev), t_rev, max_dist)
        ray_widths = np.clip(max_dist - (t_fwd_clipped + t_rev_clipped), 0.0, max_dist)
        grasp_widths = ray_widths.max(axis=1).astype(np.float32)
        print(f"[FingerGap] Width estimation: min={grasp_widths.min():.4f}m "
              f"max={grasp_widths.max():.4f}m mean={grasp_widths.mean():.4f}m")


    if return_debug_rays:
        origins = all_rays[:, :3].astype(np.float32)
        dirs = all_rays[:, 3:].astype(np.float32)
        t_vis = np.minimum(t_hit_flat, max_dist)
        t_vis[~np.isfinite(t_vis)] = max_dist
        endpoints_clipped = origins + dirs * t_vis[:, None]
        debug_dict = {
            "origins": origins,
            "endpoints_clipped": endpoints_clipped.astype(np.float32),
            "hit_mask": np.isfinite(t_hit_flat) & (t_hit_flat < max_dist),
            "max_dist": max_dist,
            "n_per": n_per,
            "gap_clear": gap_clear,
            "voxel_centers": voxel_centers.astype(np.float32),
        }
        result = (gap_clear, debug_dict)
        return result + (grasp_widths,) if return_widths else result

    return (gap_clear, grasp_widths) if return_widths else gap_clear


def compute_grasp_widths_finger_gap_raycasting(
    object_pc: np.ndarray,
    grasp_poses: np.ndarray,
    finger_half_width: float = 0.068,
    finger_height: float = 0.01,
    finger_tip_depth: float = 0.085,
    fingertip_z_offset: float = 0.195,
    ray_grid_y: int = 20,
    ray_grid_z: int = 60,
    voxel_size: float = 0.004,
):
    """
    Estimate grasp widths from the object point cloud using the same fingertip
    ray grid as filter_grasps_finger_gap_raycasting, without doing collision
    filtering again.
    """
    if object_pc is None or len(object_pc) == 0 or len(grasp_poses) == 0:
        return np.zeros(len(grasp_poses), dtype=np.float32)

    coords = np.round(object_pc / voxel_size).astype(int)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    voxel_centers = object_pc[unique_idx]

    print(f"[FingerGap] Building width RaycastingScene from {len(voxel_centers)} voxels "
          f"(voxel_size={voxel_size}m, grid={ray_grid_y}x{ray_grid_z}, "
          f"z_offset={fingertip_z_offset:.3f}m) ...")

    unit_box = o3d.geometry.TriangleMesh.create_box(
        width=voxel_size, height=voxel_size, depth=voxel_size
    )
    unit_box.translate(-np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2]))
    unit_verts = np.asarray(unit_box.vertices)
    unit_tris = np.asarray(unit_box.triangles)

    all_vertices, all_triangles, vert_offset = [], [], 0
    for center in voxel_centers:
        all_vertices.append(unit_verts + center)
        all_triangles.append(unit_tris + vert_offset)
        vert_offset += len(unit_verts)

    merged_mesh = o3d.geometry.TriangleMesh()
    merged_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(all_vertices))
    merged_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(all_triangles))

    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(merged_mesh)
    rc_scene = o3d.t.geometry.RaycastingScene()
    rc_scene.add_triangles(t_mesh)

    ys = np.linspace(-finger_height, finger_height, ray_grid_y)
    zs = np.linspace(0, finger_tip_depth, ray_grid_z)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    local_offsets = np.stack([np.zeros_like(yy), yy, zz], axis=-1).reshape(-1, 3)
    n_per = len(local_offsets)

    K = len(grasp_poses)
    max_dist = 2.0 * finger_half_width
    all_rays = np.zeros((K * n_per, 6), dtype=np.float32)
    all_rays_rev = np.zeros((K * n_per, 6), dtype=np.float32)

    for i, pose in enumerate(grasp_poses):
        R = pose[:3, :3]
        t = pose[:3, 3]
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]
        world_offsets = (
            local_offsets[:, 1:2] * y_axis[None, :]
            + local_offsets[:, 2:3] * z_axis[None, :]
        )
        fingertip_center = t + z_axis * fingertip_z_offset
        row = i * n_per

        origins = (fingertip_center - x_axis * finger_half_width) + world_offsets
        all_rays[row:row + n_per, :3] = origins.astype(np.float32)
        all_rays[row:row + n_per, 3:] = x_axis.astype(np.float32)

        origins_rev = (fingertip_center + x_axis * finger_half_width) + world_offsets
        all_rays_rev[row:row + n_per, :3] = origins_rev.astype(np.float32)
        all_rays_rev[row:row + n_per, 3:] = (-x_axis).astype(np.float32)

    rays_tensor = o3d.core.Tensor(all_rays, dtype=o3d.core.Dtype.Float32)
    rays_rev_tensor = o3d.core.Tensor(all_rays_rev, dtype=o3d.core.Dtype.Float32)
    t_fwd = rc_scene.cast_rays(rays_tensor)["t_hit"].numpy().reshape(K, n_per)
    t_rev = rc_scene.cast_rays(rays_rev_tensor)["t_hit"].numpy().reshape(K, n_per)

    t_fwd_clipped = np.where(np.isfinite(t_fwd), t_fwd, max_dist)
    t_rev_clipped = np.where(np.isfinite(t_rev), t_rev, max_dist)
    ray_widths = np.clip(max_dist - (t_fwd_clipped + t_rev_clipped), 0.0, max_dist)
    ray_widths = np.where(ray_widths == max_dist, -1.0, ray_widths)
    grasp_widths = ray_widths.max(axis=1).astype(np.float32)
    print(f"[FingerGap] Width estimation for filtered grasps: min={grasp_widths.min():.4f}m "
          f"max={grasp_widths.max():.4f}m mean={grasp_widths.mean():.4f}m")
    return grasp_widths


def compute_finger_gap_ray_origins(
    grasp_poses: np.ndarray,
    finger_half_width: float = 0.068,
    finger_height: float = 0.01,
    finger_tip_depth: float = 0.085,
    fingertip_z_offset: float = 3,
    ray_grid_y: int = 20,
    ray_grid_z: int = 60,
) -> tuple:
    """
    Compute the ray origin and endpoint grids used by filter_grasps_finger_gap_raycasting.
    Returns (origins, endpoints) each shaped (K * ray_grid_y * ray_grid_z, 3) so they can
    be visualised as two point clouds — origins on the –X finger face (green),
    endpoints on the +X finger face (red).

    Useful for visually verifying that fingertip_z_offset places the ray plane
    at the correct physical location on the gripper.
    """
    ys = np.linspace(-finger_height,    finger_height,    ray_grid_y)
    zs = np.linspace(0, finger_tip_depth, ray_grid_z)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    local_offsets = np.stack(
        [np.zeros_like(yy), yy, zz], axis=-1
    ).reshape(-1, 3)  # (n_per, 3)

    all_origins, all_endpoints = [], []
    for pose in grasp_poses:
        R = pose[:3, :3]
        t = pose[:3,  3]
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        world_offsets = (local_offsets[:, 1:2] * y_axis[None, :]
                       + local_offsets[:, 2:3] * z_axis[None, :])

        fingertip_center = t + z_axis * fingertip_z_offset
        origins   = (fingertip_center - x_axis * finger_half_width) + world_offsets
        endpoints = (fingertip_center + x_axis * finger_half_width) + world_offsets

        all_origins.append(origins)
        all_endpoints.append(endpoints)

    return np.vstack(all_origins), np.vstack(all_endpoints)


def _rgb_to_hex(color):
    return (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])


def visualize_finger_gap_rays(
    vis,
    name: str,
    origins: np.ndarray,
    endpoints: np.ndarray,
    color=(255, 80, 0),
    max_rays: int = 12000,
):
    """
    Visualize ray segments in MeshCat.

    To keep the UI responsive when many grasps are evaluated, we render up to
    `max_rays` uniformly sampled segments.
    """
    if vis is None or len(origins) == 0:
        return

    n_rays = len(origins)
    if n_rays > max_rays:
        idx = np.linspace(0, n_rays - 1, max_rays, dtype=int)
        origins = origins[idx]
        endpoints = endpoints[idx]
        print(f"[FingerGapViz] Showing {max_rays}/{n_rays} rays (uniform subsample).")
    else:
        print(f"[FingerGapViz] Showing all {n_rays} rays.")

    material = g.MeshPhongMaterial(color=_rgb_to_hex(color))
    for i, (start, end) in enumerate(zip(origins, endpoints)):
        pts = np.stack([start, end], axis=1).astype(float)  # (3, 2)
        vis[f"{name}/ray_{i:05d}"].set_object(g.Line(g.PointsGeometry(pts), material))


def _visualize_finger_gap_debug(vis, ray_debug, voxel_size):
    if vis is None or ray_debug is None:
        return

    visualize_pointcloud(
        vis,
        "finger_gap_rays/rc_scene_voxels",
        ray_debug["voxel_centers"],
        [80, 200, 255],
        size=voxel_size,
    )

    invalid_idxs = np.where(~ray_debug["gap_clear"])[0]
    n_per = int(ray_debug["n_per"])

    if len(invalid_idxs) == 0 or n_per <= 0:
        return

    origins_chunks = []
    ends_chunks = []

    for k in invalid_idxs:
        start = int(k) * n_per
        end = start + n_per
        hit_mask_k = ray_debug["hit_mask"][start:end]
        if np.any(hit_mask_k):
            origins_chunks.append(ray_debug["origins"][start:end][hit_mask_k])
            ends_chunks.append(ray_debug["endpoints_clipped"][start:end][hit_mask_k])

    if len(origins_chunks) == 0:
        return

    ray_origins_vis = np.vstack(origins_chunks)
    ray_ends_vis = np.vstack(ends_chunks)
    print(
        f"[FingerGapViz] invalid grasps={len(invalid_idxs)}: visualizing "
        f"{len(ray_origins_vis)} hit rays total."
    )
    visualize_pointcloud(
        vis, "finger_gap_rays/left_face",
        ray_origins_vis, [0, 220, 80], size=0.003
    )
    visualize_pointcloud(
        vis, "finger_gap_rays/right_face",
        ray_ends_vis, [220, 40, 0], size=0.003
    )
    visualize_finger_gap_rays(
        vis,
        "finger_gap_rays/segments",
        ray_origins_vis,
        ray_ends_vis,
        color=(255, 120, 0),
        max_rays=1200,
    )


def save_for_3d_viewer(scene_pc, object_pc, best_poses, gripper_mesh, filename="grasp_result.glb"):
    """
    Saves a debug scene:
    - Gray: Background scene (downsampled)
    - Blue: Segmented object (what the model sees)
    - Red/Gold: Top 3 grasps
    """
    geometries = []

    # 1. Background Scene (Light Gray & Sparse)
    if scene_pc is not None and len(scene_pc) > 0:
        pcd_scene = trimesh.points.PointCloud(vertices=scene_pc[::10])
        pcd_scene.colors = [200, 200, 200, 100] 
        geometries.append(pcd_scene)

    # 2. Segmented Object (Solid Blue)
    if object_pc is not None and len(object_pc) > 0:
        pcd_obj = trimesh.points.PointCloud(vertices=object_pc)
        pcd_obj.colors = [0, 100, 255, 255] 
        geometries.append(pcd_obj)

    # 3. Add Top 3 Gripper Meshes
    colors = [[255, 0, 0, 255], [255, 140, 0, 255], [255, 215, 0, 255]]

    if best_poses is not None and len(best_poses) > 0:
        for i, pose in enumerate(best_poses[:1]):
            gripper_vis = gripper_mesh.copy()
            gripper_vis.apply_transform(pose)
            gripper_vis.visual.face_colors = colors[i]
            geometries.append(gripper_vis)

    if geometries:
        scene = trimesh.Scene(geometries)
        scene.export(filename)
        print(f"--- [VIZ] Debug visualization saved to {filename} ---")

def visualize_grasps(viz, 
                    scene_pc, 
                    scene_pc_colors, 
                    object_pc, 
                    object_pc_colors, 
                    grasps,
                    top_grasp,
                    grasps_colors,
                    gripper_mesh, 
                    gripper_name="default_gripper",
                    voxel_size=0.002):

    # Clear old visualization markers
    if viz is not None:
        viz["scene_pc"].delete()
        # viz["object_pc"].delete()
        viz["collision_free_grasps"].delete()
        viz["top_grasp"].delete()
        viz["finger_gap_rays"].delete()

    # Visualize scene point cloud - use RGB colors if available, otherwise use gray
    if scene_pc is not None:
        if scene_pc_colors is not None:
            visualize_pointcloud(
                viz, "scene_pc", scene_pc, scene_pc_colors, size=voxel_size
            )
        else:
            visualize_pointcloud(
                viz, "scene_pc", scene_pc, [128, 128, 128], size=voxel_size
            )

    if object_pc is not None:
        # Visualize object point cloud - use RGB colors if available, otherwise use green
        if object_pc_colors is not None:
            visualize_pointcloud(
                viz, "object_pc", object_pc, object_pc_colors, size=0.0025
            )
        else:
            visualize_pointcloud(
                viz, "object_pc", object_pc, [0, 255, 0], size=0.0025
            )

    # Visualize collision-free grasps
    for i, (grasp, score) in enumerate(
        zip(grasps, grasps_colors)
    ):
        visualize_grasp(
            viz,
            f"collision_free_grasps/{i:03d}/grasp",
            grasp,
            color=score,
            gripper_name=gripper_name,
            linewidth=0.8,
            )

def process_point_cloud(pc, grasps):
    """Process point cloud and grasps by centering them."""
    # Center point cloud and grasps
    T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, T_subtract_pc_mean)
    grasps_centered = np.array(
        [T_subtract_pc_mean @ np.array(g) for g in grasps.tolist()]
    )
    return pc_centered, grasps_centered, T_subtract_pc_mean

def run_server(args):
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_info = get_gripper_info(grasp_cfg.data.gripper_name)
    grasp_sampler = GraspGenSampler(grasp_cfg)

    ray_viz_enabled = bool(args.visualize and args.visualize_finger_gap_rays)

    # Initialize visualization if requested
    vis = None
    if bool(args.visualize):
        vis = create_visualizer()

    cached_session = {
        'raw_grasps': None,
        'raw_scores': None,
        'grasps_oriented_orig': None,
        'grasps_oriented_shifted': None,
        'scores': None,
        'stage1_labels': None,
        'T_center': None,
        'object_pc_centered': None,
        'hand_pc_centered': None,
        'top_grasp': None,
        'top_label': None,
        'final_grasps': None,
        'final_labels': None,
        'stage1_complete': False,
        'stage2_complete': False,
        'is_symmetric': False,
    }

    def _send_final_grasps(grasp_matrices, grasp_labels, grasp_widths=None):
        _send_grasp_response(conn, grasp_matrices, grasp_labels, widths=grasp_widths)


    def _cache_stage1_results(**kwargs):
        cached_session.update(kwargs)
        cached_session['stage1_complete'] = True
        cached_session['stage2_complete'] = False

    def _cache_stage2_results(final_grasps, final_labels):
        cached_session['final_grasps'] = final_grasps
        cached_session['final_labels'] = final_labels
        cached_session['stage2_complete'] = True

    def _rotation_about_center(rotation_deg):
        return tra.rotation_matrix(np.deg2rad(float(rotation_deg)), [0.0, 0.0, 1.0])

    def _apply_stage1_geometry_filters(grasps_centered, scores):
        if grasps_centered is None or len(grasps_centered) == 0:
            return (
                np.zeros((0, 4, 4), dtype=np.float32),
                np.zeros((0, 4, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        grasps_work = np.asarray(grasps_centered, dtype=np.float32)
        scores_work = np.asarray(scores, dtype=np.float32)

        if args.orient and len(grasps_work) > 0:
            grasps_work = flatten_grasp_orientations(grasps_work)
            print(f"[Stage1] Flattened {len(grasps_work)} grasp finger orientations to horizontal.")

        grasps_shifted = grasps_work @ tra.translation_matrix([0, 0, 0.02])

        s_mask, phi_was_softened = apply_spherical_filter_adaptive_phi(
            grasps_shifted,
            scores_work,
            azimuth_range=tuple(args.phi_range),
            elevation_range=tuple(args.theta_range),
            phi_step=2.0,
        )

        grasps_work = grasps_work[s_mask]
        grasps_shifted = grasps_shifted[s_mask]
        scores_work = scores_work[s_mask]

        if not phi_was_softened:
            print(f"[Filter] Spherical: {len(grasps_shifted)} grasps kept with strict phi range.")
        else:
            print(f"[Filter] Spherical (phi softened): {len(grasps_shifted)} grasps kept.")

        if args.orient and len(grasps_shifted) > 0:
            z_mask = filter_by_gripper_z_rotation(
                grasps_shifted,
                max_roll_deg=args.max_z_rotation_deg,
            )
            grasps_work = grasps_work[z_mask]
            grasps_shifted = grasps_shifted[z_mask]
            scores_work = scores_work[z_mask]

        return grasps_work, grasps_shifted, scores_work

    def _run_stage2_style_selection(
        pipeline_stage,
        hand_pc_centered,
        object_pc_centered,
        grasps_oriented_orig,
        grasps_oriented_shifted,
        scores_oriented,
        T_center,
        is_box
    ):
        if grasps_oriented_shifted is None or grasps_oriented_orig is None or scores_oriented is None:
            print(f'[{pipeline_stage}] Cached grasp set is incomplete; returning empty response.')
            _send_grasp_response(conn, [], [])
            return False

        if object_pc_centered is None or len(object_pc_centered) == 0:
            print(f'[{pipeline_stage}] Missing cached object_pc_centered; returning empty response.')
            _send_grasp_response(conn, [], [])
            return False

        if hand_pc_centered is None:
            hand_pc_centered = np.zeros((0, 3), dtype=np.float32)

        start_gap_filter_time = time.time()
        finger_half_width = float(getattr(gripper_info, 'finger_half_width', 0.1))

        # gap_result = filter_grasps_finger_gap_raycasting(
        #     scene_pc=hand_pc_centered,
        #     grasp_poses=grasps_oriented_shifted,
        #     finger_half_width=finger_half_width,
        #     fingertip_z_offset=0.11,
        #     voxel_size=args.voxel_size,
        #     return_debug_rays=ray_viz_enabled,
        #     return_widths=False,
        # )
        # if isinstance(gap_result, tuple):
        #     gap_mask = np.asarray(gap_result[0], dtype=bool)
        #     ray_debug = gap_result[1] if len(gap_result) > 1 else None
        # else:
        #     gap_mask = np.asarray(gap_result, dtype=bool)
        #     ray_debug = None
        # print(f'[{pipeline_stage}] Finger-gap filter completed in {time.time() - start_gap_filter_time:.2f}s')

        final_grasps_centered = grasps_oriented_orig#[gap_mask]
        final_grasps_shifted = grasps_oriented_shifted#[gap_mask]
        print(f'[{pipeline_stage}] Len final grasps after finger gap filtering: {len(final_grasps_centered)}')
        final_scores = scores_oriented#[gap_mask]
        
        # sort by score
        # top_idx = np.argsort(final_scores)[::-1]
        # final_grasps_centered = final_grasps_centered[top_idx]
        # final_grasps_shifted = final_grasps_shifted[top_idx]
        # final_scores = final_scores[top_idx]
        
        top_len = len(final_grasps_centered)
        if top_len > args.max_top_k:
            # Label-weighted selection: Top (2) → 0.45, Mid (1) → 0.45, Bottom (0) → 0.10
            _T_inv_tmp = tra.inverse_matrix(T_center)
            _g_robots_tmp = _T_inv_tmp[None] @ final_grasps_centered  # (N, 4, 4)
            _f_zs_tmp = _g_robots_tmp[:, 2, 3] + _g_robots_tmp[:, 2, 2] * 0.15
            _labels_tmp = np.where(_f_zs_tmp > t2, 2, np.where(_f_zs_tmp > t1, 1, 0))
            
            if is_box:
                _label_weights = np.array([0.10, 0.70, 0.20])  # for boxes, prioritize top grasps more
            else:
                _label_weights = np.array([0.10, 0.45, 0.45])  # for non-boxes, keep more balanced
            
            _probs_tmp = _label_weights[_labels_tmp].astype(np.float64)
            _probs_tmp /= _probs_tmp.sum()
            random_indices = np.random.choice(top_len, size=args.max_top_k, replace=False, p=_probs_tmp)
            final_grasps_centered = final_grasps_centered[random_indices]
            final_grasps_shifted = final_grasps_shifted[random_indices]
            final_scores = final_scores[random_indices]
            _sel = _labels_tmp[random_indices]
            print(f'[{pipeline_stage}] Label-weighted selection: Top={np.sum(_sel==2)}, Mid={np.sum(_sel==1)}, Bottom={np.sum(_sel==0)}')

        final_widths = compute_grasp_widths_finger_gap_raycasting(
            object_pc=object_pc_centered,
            grasp_poses=final_grasps_shifted,
            finger_half_width=finger_half_width,
            fingertip_z_offset=0.11,
            voxel_size=args.voxel_size,
        )

        finger_offset = 0.15
        T_inv_center = tra.inverse_matrix(T_center)
        grasp_labels = []
        grasp_colors = []
        adjusted_grasps = []

        start_label_time = time.time()
        for g_centered in final_grasps_centered:
            g_robot = T_inv_center @ g_centered
            f_z = (g_robot[:3, 3] + g_robot[:3, 2] * finger_offset)[2]

            adjusted_grasp = g_centered
            if f_z > t2:
                grasp_labels.append(2)
                grasp_colors.append([0, 80, 255])
            elif f_z > t1:
                grasp_labels.append(1)
                grasp_colors.append([0, 255, 100])
                if is_wine_glass:
                    adjusted_grasp = _make_grasp_parallel_to_ground(
                        g_centered,
                        pivot_offset=finger_offset,
                    )
            else:
                grasp_labels.append(0)
                grasp_colors.append([255, 230, 0])
                if is_wine_glass:
                    adjusted_grasp = _make_grasp_parallel_to_ground(
                        g_centered,
                        pivot_offset=finger_offset,
                    )
            adjusted_grasps.append(adjusted_grasp)

        final_grasps_centered = np.asarray(adjusted_grasps, dtype=np.float32)

        top_poses = final_grasps_centered
        top_labels = grasp_labels
        top_colors = grasp_colors
        top_widths = final_widths.astype(np.float32)

        print(f'[{pipeline_stage}] Labeling completed in {time.time() - start_label_time:.2f}s.')

        if args.visualize:
            visualize_grasps(
                viz=vis,
                scene_pc=None,#hand_pc_centered,
                scene_pc_colors=None,
                object_pc=object_pc_centered,
                object_pc_colors=None,
                grasps=top_poses,
                top_grasp=top_poses[0] if len(top_poses) > 0 else None,
                grasps_colors=top_colors,
                gripper_mesh=gripper_info.visual_mesh,
                gripper_name=grasp_cfg.data.gripper_name,
                voxel_size=args.voxel_size,
            )

        if ray_viz_enabled and ray_debug is not None:
            _visualize_finger_gap_debug(vis, ray_debug, args.voxel_size)

        top_poses_world = np.asarray([T_inv_center @ pose for pose in top_poses], dtype=np.float32)
        _send_final_grasps(top_poses_world, top_labels, grasp_widths=top_widths)
        return

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # server.settimeout(1.0)
    if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)
    server.bind(SOCKET_PATH)
    server.listen(5)
    
    is_wine_glass = False
    t1, t2 = 0, 0

    print(f"GraspGen Server (Optimized) | Voxel: {args.voxel_size}m | Margin: {args.collision_margin}m")

    try:
        while True:
            # try:
            conn, _ = server.accept()
            # except socket.timeout: continue

            with conn:
                start_time = time.time()
                
                # Receive request type and payload sizes
                header = recv_all(conn, 16)  # 4 int32 values: request_type, scene_size, object_size, meta_size
                if not header: continue
                request_type, scene_size, object_size, meta_size = np.frombuffer(header, dtype=np.int32)
                print(
                    f"\nReceived request: request_type={request_type}, scene_size={scene_size}, object_size={object_size}, meta_size={meta_size}"
                )

                is_stage1_pregeneration = (request_type == REQUEST_PREGENERATE)
                is_stage2_hand_filtering = (request_type == REQUEST_HAND_FILTER)
                is_stage3_regeneration = (request_type == REQUEST_REGENERATE)

                if is_stage1_pregeneration:
                    pipeline_stage = "Stage 1 (Object Pregeneration)"
                elif is_stage2_hand_filtering:
                    pipeline_stage = "Stage 2 (Hand Filtering)"
                elif is_stage3_regeneration:
                    pipeline_stage = "Stage 3 (Regeneration)"
                else:
                    pipeline_stage = "Unknown"

                print(f"[Pipeline] Processing {pipeline_stage}")
                
                scene_pc_filtered = np.zeros((0, 3), dtype=np.float32)
                if scene_size > 0:
                    scene_data = recv_all(conn, scene_size * 3 * 4)
                    if not scene_data:
                        continue
                    scene_pc = np.frombuffer(scene_data, dtype=np.float32).reshape((scene_size, 3))
                    scene_pc_filtered = scene_pc
                
                object_pc = np.zeros((0, 3), dtype=np.float32)
                if object_size > 0:
                    object_data = recv_all(conn, object_size * 3 * 4)
                    if not object_data:
                        continue
                    object_pc = np.frombuffer(object_data, dtype=np.float32).reshape((object_size, 3))

                completion_meta = {}
                if meta_size > 0:
                    meta_data = recv_all(conn, meta_size)
                    if meta_data:
                        try:
                            completion_meta = json.loads(meta_data.decode('utf-8'))
                        except Exception as exc:
                            print(f"[Reconstruct] Failed to parse completion metadata: {exc}")
                            completion_meta = {}

                if is_stage2_hand_filtering:
                    if not cached_session['stage1_complete']:
                        print('[Stage2] Stage 1 cache is not ready; returning empty response.')
                        _send_grasp_response(conn, [], [])
                        continue

                    hand_pc = scene_pc_filtered
                    # if len(hand_pc) > 0:
                    #     hand_pc = voxel_downsample(hand_pc, voxel_size=args.voxel_size)
                    #     hand_pc = hand_pc[np.linalg.norm(hand_pc, axis=1) < 0.25]
                    #     print(f'[Stage2] Received hand PC: {len(hand_pc)} points after downsampling')

                    is_symmetric = cached_session.get('is_symmetric', False)
                    T_center = cached_session['T_center']
                    if T_center is None:
                        print('[Stage2] Missing cached T_center; returning empty response.')
                        _send_grasp_response(conn, [], [])
                        continue

                    hand_pc_centered = tra.transform_points(hand_pc, T_center) if len(hand_pc) > 0 else hand_pc
                    # hand_pc_centered = voxel_downsample(hand_pc_centered, voxel_size=args.voxel_size)
                    print(f'[Stage2] Hand PC centered and downsampled: {len(hand_pc_centered)} points')

                    grasps_oriented_shifted = cached_session['grasps_oriented_shifted']
                    grasps_oriented_orig = cached_session['grasps_oriented_orig']
                    scores_oriented = cached_session['scores']

                    if grasps_oriented_shifted is None or grasps_oriented_orig is None or scores_oriented is None:
                        print('[Stage2] Cached grasp set is incomplete; returning empty response.')
                        _send_grasp_response(conn, [], [])
                        continue
                    
                    # TODO
                    # top_grasp_id = np.sort(scores_oriented)[0]
                    # print(f'[Stage2] top grasp id: {top_grasp_id:.4f}')
                    # top_grasp = grasps_oriented_orig[top_grasp_id]
                    # print(f'[Stage2] top grasp (original):\n{top_grasp}')

                    object_pc_centered = cached_session.get('object_pc_centered')
                    if object_pc_centered is not None and len(hand_pc_centered) > 0:
                        before_trim = len(hand_pc_centered)
                        hand_pc_centered = _remove_points_inside_centered_object_walls(
                            hand_points=hand_pc_centered,
                            object_points=object_pc_centered,
                            wall_epsilon=args.hand_object_trim_epsilon,
                        )
                        removed_trim = before_trim - len(hand_pc_centered)
                        print(
                            f'[Stage2] Trimmed {removed_trim}/{before_trim} hand points inside centered object walls '
                            f'(epsilon={args.hand_object_trim_epsilon:.4f}m)'
                        )

                    start_gap_filter_time = time.time()
                    finger_half_width = float(getattr(gripper_info, 'finger_half_width', 0.1))

                    gap_result = filter_grasps_finger_gap_raycasting(
                        scene_pc=hand_pc_centered,
                        grasp_poses=grasps_oriented_shifted,
                        finger_half_width=finger_half_width,
                        fingertip_z_offset=0.11,
                        voxel_size=args.voxel_size,
                        return_debug_rays=ray_viz_enabled,
                        return_widths=False,
                    )
                    if isinstance(gap_result, tuple):
                        gap_mask = np.asarray(gap_result[0], dtype=bool)
                        ray_debug = gap_result[1] if len(gap_result) > 1 else None
                    else:
                        gap_mask = np.asarray(gap_result, dtype=bool)
                        ray_debug = None
                    print('Stage2 finger-gap filter completed in {:.2f}s'.format(time.time() - start_gap_filter_time))

                    final_grasps_centered = grasps_oriented_orig[gap_mask]
                    final_grasps_shifted = grasps_oriented_shifted[gap_mask]
                    print('Len final grasps after finger gap filtering:', len(final_grasps_centered))
                    final_scores = scores_oriented[gap_mask]

                    # ------------------------------------------------------------------
                    # Fallback: if the hand filter removed ALL grasps, pick the
                    # max_top_k grasps that overlap the hand point cloud the least.
                    # ------------------------------------------------------------------
                    if len(final_grasps_centered) < 3:
                        print(
                            '[Stage2] No grasps survived hand filtering — '
                            'falling back to least-overlap selection.'
                        )
                        finger_half_width_fb = float(getattr(gripper_info, 'finger_half_width', 0.1))
                        overlap_counts = _count_hand_points_in_finger_gap(
                            hand_pc=hand_pc_centered,
                            grasp_poses=grasps_oriented_shifted,
                            finger_half_width=finger_half_width_fb,
                            fingertip_z_offset=0.11,
                        )
                        # Sort ascending by overlap count, take best max_top_k
                        fallback_order = np.argsort(overlap_counts)
                        fallback_k = min(args.max_top_k, len(grasps_oriented_orig))
                        fallback_idx = fallback_order[:fallback_k]
                        final_grasps_centered = grasps_oriented_orig[fallback_idx]
                        final_grasps_shifted = grasps_oriented_shifted[fallback_idx]
                        final_scores = scores_oriented[fallback_idx]
                        if fallback_k > 0:
                            print(
                                f'[Stage2] Fallback selected {len(final_grasps_centered)} grasps '
                                f'(min overlap={overlap_counts[fallback_order[0]]} pts, '
                                f'max overlap={overlap_counts[fallback_order[fallback_k-1]]} pts).'
                            )
                        else:
                            print('[Stage2] Fallback: no pre-filter grasps available either; returning empty.')

                    if object_pc_centered is None or len(object_pc_centered) == 0:
                        print('[Stage2] Missing cached object_pc_centered; returning empty response.')
                        _send_grasp_response(conn, [], [])
                        continue
                    
                    # sort by score
                    # NOTE: score sort removed — the label-weighted random selection
                    # that follows would discard this ordering anyway.
                    # top_idx = np.argsort(final_scores)[::-1]
                    # final_grasps_centered = final_grasps_centered[top_idx]
                    # final_grasps_shifted = final_grasps_shifted[top_idx]
                    # final_scores = final_scores[top_idx]

                    # limit to max_top_k using label-weighted sampling
                    top_len = len(final_grasps_centered)
                    if top_len > args.max_top_k:
                        # Compute labels for all candidates before down-selecting so we can
                        # build a probability distribution that favours Top > Mid > Bottom.
                        # Weights: Top (2) → 0.45, Mid (1) → 0.45, Bottom (0) → 0.10
                        
                        _T_inv_tmp = tra.inverse_matrix(T_center)
                        _g_robots_tmp = _T_inv_tmp[None] @ final_grasps_centered  # (N, 4, 4)
                        _f_zs_tmp = _g_robots_tmp[:, 2, 3] + _g_robots_tmp[:, 2, 2] * 0.15
                        _labels_tmp = np.where(_f_zs_tmp > t2, 2, np.where(_f_zs_tmp > t1, 1, 0))
                        
                        if not is_symmetric:
                            _label_weights = np.array([0.10, 0.70, 0.20])  # for boxes, prioritize top grasps more
                        else:
                            _label_weights = np.array([0.10, 0.45, 0.45])  # for non-boxes, keep more balanced
                        
                        _probs_tmp = _label_weights[_labels_tmp].astype(np.float64)
                        _probs_tmp /= _probs_tmp.sum()
                        random_indices = np.random.choice(top_len, size=args.max_top_k, replace=False, p=_probs_tmp)
                        final_grasps_centered = final_grasps_centered[random_indices]
                        final_grasps_shifted = final_grasps_shifted[random_indices]
                        final_scores = final_scores[random_indices]
                        _sel = _labels_tmp[random_indices]
                        # print(f'[Stage2] Label-weighted selection: Top={np.sum(_sel==2)}, Mid={np.sum(_sel==1)}, Bottom={np.sum(_sel==0)}')

                    final_widths = compute_grasp_widths_finger_gap_raycasting(
                        object_pc=object_pc_centered,
                        grasp_poses=final_grasps_shifted,
                        finger_half_width=finger_half_width,
                        fingertip_z_offset=0.11,
                        voxel_size=args.voxel_size,
                    )

                    # print(final_widths)
                    
                    # TODO
                    # if len(final_grasps_centered) == 0:
                    #     # set to top grasp before filtering to ensure we return something
                    #     final_grasps_centered = np.array([top_grasp])
                    #     final_scores = np.array([scores_oriented[top_grasp_id]])
                    
                    # Finger center offset for wrist poses
                    finger_offset = 0.15

                    # Calculate Labels for all valid grasps (to use for visualization)
                    T_inv_center = tra.inverse_matrix(T_center)
                    grasp_labels = []
                    grasp_colors = []
                    adjusted_grasps = []

                    start_label_time = time.time()
                    for g_centered in final_grasps_centered:
                        # 1. Transform grasp back to robot frame temporarily
                        g_robot = T_inv_center @ g_centered
                        # 2. Get finger center Z coordinate
                        f_z = (g_robot[:3, 3] + g_robot[:3, 2] * finger_offset)[2]

                        adjusted_grasp = g_centered

                        # 3. Categorize and color
                        if f_z > t2:
                            grasp_labels.append(2)
                            grasp_colors.append([0, 80, 255])   # Blue (Top)
                        elif f_z > t1:
                            grasp_labels.append(1)
                            grasp_colors.append([0, 255, 100])  # Green (Mid)
                            if is_wine_glass:
                                adjusted_grasp = _make_grasp_parallel_to_ground(
                                    g_centered,
                                    pivot_offset=finger_offset,
                                )
                        else:
                            grasp_labels.append(0)
                            grasp_colors.append([255, 230, 0])  # Yellow (Bottom)
                            if is_wine_glass:
                                adjusted_grasp = _make_grasp_parallel_to_ground(
                                    g_centered,
                                    pivot_offset=finger_offset,
                                )
                        adjusted_grasps.append(adjusted_grasp)

                    final_grasps_centered = np.asarray(adjusted_grasps, dtype=np.float32)

                    top_poses = final_grasps_centered
                    top_labels = grasp_labels
                    top_colors = grasp_colors
                    top_widths = final_widths.astype(np.float32)
                    
                    print('Stage 2 labeling completed in {:.2f}s.'.format(time.time() - start_label_time))
                
                    # Stage 2: Visualize filtered grasps on hand PC (now in scene_pc_centered)
                    if args.visualize:
                        visualize_grasps(
                            viz=vis,
                            scene_pc=hand_pc_centered,  # Hand PC
                            scene_pc_colors=None,
                            object_pc=None,  # No object PC in Stage 2
                            object_pc_colors=None,
                            grasps=top_poses,
                            top_grasp=top_poses[0] if len(top_poses) > 0 else None,
                            grasps_colors=top_colors,
                            gripper_mesh=gripper_info.visual_mesh,
                            gripper_name=grasp_cfg.data.gripper_name,
                            voxel_size=args.voxel_size,
                        )

                    if ray_viz_enabled and ray_debug is not None:
                        _visualize_finger_gap_debug(vis, ray_debug, args.voxel_size)
                    
                    T_inv_center = tra.inverse_matrix(T_center)
                    top_poses_world = np.asarray([T_inv_center @ pose for pose in top_poses], dtype=np.float32)

                    cached_session['hand_pc_centered'] = np.asarray(hand_pc_centered, dtype=np.float32)
                    _cache_stage2_results(top_poses_world, top_labels)
                    _send_final_grasps(top_poses_world, top_labels, grasp_widths=top_widths)

                    continue

                if is_stage3_regeneration:
                    if not cached_session['stage1_complete']:
                        print('[Stage3] Stage 1 cache is not ready; returning empty response.')
                        _send_grasp_response(conn, [], [])
                        continue

                    raw_grasps = cached_session.get('raw_grasps')
                    raw_scores = cached_session.get('raw_scores')
                    T_center = cached_session.get('T_center')
                    is_symmetric = cached_session.get('is_symmetric', False)
                    object_pc_centered = cached_session.get('object_pc_centered')
                    hand_pc_centered = cached_session.get('hand_pc_centered')

                    if raw_grasps is None or raw_scores is None or T_center is None:
                        print('[Stage3] Missing cached raw grasps, raw scores, or T_center; returning empty response.')
                        _send_grasp_response(conn, [], [])
                        continue

                    rotation_deg = float(completion_meta.get('rotation_deg', 0.0) or 0.0)
                    print(f'[Stage3] Replaying grasp pipeline with rotation_deg={rotation_deg:.2f} deg.')
                    rotation_tf = _rotation_about_center(rotation_deg)

                    raw_grasps = np.asarray(raw_grasps, dtype=np.float32)
                    grasps_centered = np.asarray([T_center @ grasp for grasp in raw_grasps], dtype=np.float32)
                    grasps_centered = np.asarray([rotation_tf @ grasp for grasp in grasps_centered], dtype=np.float32)

                    if object_pc_centered is not None and len(object_pc_centered) > 0:
                        object_pc_centered = tra.transform_points(np.asarray(object_pc_centered, dtype=np.float32), rotation_tf)
                    if hand_pc_centered is not None and len(hand_pc_centered) > 0:
                        hand_pc_centered = tra.transform_points(np.asarray(hand_pc_centered, dtype=np.float32), rotation_tf)

                    grasps_oriented_orig, grasps_oriented_shifted, scores_oriented = _apply_stage1_geometry_filters(
                        grasps_centered,
                        raw_scores,
                    )

                    _run_stage2_style_selection(
                        'Stage3',
                        hand_pc_centered,
                        object_pc_centered,
                        grasps_oriented_orig,
                        grasps_oriented_shifted,
                        scores_oriented,
                        T_center,
                        not is_symmetric,
                    )

                    continue

                # if is_get_results:
                #     if cached_session['stage2_complete'] and cached_session['final_grasps'] is not None:
                #         _send_final_grasps(cached_session['final_grasps'], cached_session.get('final_labels') or [])
                #     else:
                #         _send_grasp_response(conn, [], [])
                #     continue

                request_object_label = str(completion_meta.get('object_label', '') or '').strip().lower()
                reconstruct = args.reconstruct and bool(completion_meta.get('reconstruct', False))
                is_symmetric = bool(completion_meta.get('reconstruct', False))
                print('request_object_label: ', request_object_label)
                if 'glass' in request_object_label:
                    is_wine_glass = True
                else:
                    is_wine_glass = False
                
                object_pc_augmented = None
                if reconstruct:
                    object_pc_augmented, reconstruct_frustums = _build_object_proxy(
                        object_pc,
                        completion_meta,
                        return_frustums=True,
                    )
                    if len(object_pc_augmented) > 0:
                        pc_filtered_np = object_pc_augmented
                    else:
                        pc_filtered_np = object_pc

                    # Remove points strictly inside virtual transparent-object volume
                    # to avoid finger-depth collapse artifacts polluting collision checks.
                    scene_pc_filtered = _remove_points_inside_frustums(
                        scene_pc_filtered,
                        reconstruct_frustums,
                        strict=True,
                    )
                else:
                    pc_filtered_np = object_pc

                # Filter object points
                # pc_filtered, _ = point_cloud_outlier_removal(torch.from_numpy(object_pc))
                # pc_filtered_np = pc_filtered.numpy()

                # Calculate center of mass as average of min/max coordinates
                min_coords = np.min(pc_filtered_np, axis=0)
                max_coords = np.max(pc_filtered_np, axis=0)
                # center_of_mass = (min_coords + max_coords) / 2.0
                # print(f"Center of Mass after outlier removal (XYZ): {center_of_mass}")

                # Calculate vertical thirds for labeling (Top/Mid/Bottom)
                obj_z_min = min_coords[2]
                obj_z_max = max_coords[2]
                obj_height = obj_z_max - obj_z_min
                estimated_height = float(completion_meta.get('height', 0.0) or obj_height)
                if obj_height < (estimated_height * 0.6):  # If point cloud is very short, trust metadata more
                    obj_height = estimated_height # fallback to metadata height if point cloud is too short
                t1 = obj_z_min + obj_height / 3.0  # Threshold between bottom/mid
                t2 = obj_z_min + 2 * obj_height / 3.0  # Threshold between mid/top
                print(f"Object Height (Z) Range: {obj_z_min:.3f} to {obj_z_max:.3f}. Thirds: <{t1:.3f}, <{t2:.3f}")

                # Speed up collision check: Combine and downsample environment
                env_pc_downsampled = voxel_downsample(scene_pc_filtered, voxel_size=args.voxel_size)

                start_inference_time = time.time()
                # Run Inference
                # Note: GraspGen usually expects input centered at origin or in a specific range.
                # However, the node now sends points in the robot 'base' frame.
                # We perform centering locally to help the network, then transform back.
                # grasps, conf = grasp_sampler.run_inference(
                #     pc_filtered_np, grasp_threshold=0.7, num_grasps=args.num_grasps
                # )
                grasps, conf = GraspGenSampler.run_inference(
                    pc_filtered_np, grasp_sampler, grasp_threshold=0.6, num_grasps=args.num_grasps
                )
                if reconstruct:
                    pc_filtered_np = object_pc
                print('Inference completed in {:.2f}s.'.format(time.time() - start_inference_time))
                grasps_np, scores_np = grasps.cpu().numpy(), conf.cpu().numpy()
                # Ensure grasps have correct homogeneous coordinate
                grasps_np[:, 3, 3] = 1
                raw_grasps = grasps_np.copy()
                raw_scores = scores_np.copy()
                
                # centering
                # We center based on the object to keep numeric stability high for the next filters
                pc_filtered_np, grasps_np, T_center = (
                    process_point_cloud(
                        pc_filtered_np, grasps_np
                    )
                )
                env_pc_downsampled_centered = tra.transform_points(env_pc_downsampled, T_center) # for collision checking
                
                # crop environment pointcloud to a reasonable radius around the object to speed up collision checking
                env_pc_downsampled_centered = env_pc_downsampled_centered[np.linalg.norm(env_pc_downsampled_centered, axis=1) < 0.25] # 0.5m radius cutoff

                # Parse ranges (args already give us floats via nargs=2)
                phi_range = tuple(args.phi_range)
                theta_range = tuple(args.theta_range)

                #figure out cylindrical coordinates 
                
                # print('Center of mass: ', center_of_mass)
                # if center_of_mass[0] >= 0.15: # (m) object is on the left side 
                #     phi_range = (0, phi_range[1])
                # elif center_of_mass[0] <= -0.15: # object is on the right side
                #     phi_range = (phi_range[0], 0)
                # else:
                #     phi_range = (phi_range[0]+45.0, phi_range[1]/2) # object is in the middle
                # #else object is in the middle, do nothing
                
                # Optional: flatten all grasps to zero finger tilt before filtering
                if args.orient and len(grasps_np) > 0:
                    grasps_np = flatten_grasp_orientations(grasps_np)
                    print(f"[Orient] Flattened {len(grasps_np)} grasp finger orientations to horizontal.")

                # Apply Z-offset (pushed-in version) for filtering.
                # We will filter on grasps_shifted, but eventually return the original grasps_np.
                T_grasp_vertical_offset = tra.translation_matrix([0, 0, 0.02]) # 2cm forward shift
                grasps_shifted = grasps_np @ T_grasp_vertical_offset

                start_filter_time = time.time()
                # Filter 1: Spherical Constraints (adaptive phi softening) on the test poses
                s_mask, phi_was_softened = apply_spherical_filter_adaptive_phi(
                   grasps_shifted, scores_np,
                   azimuth_range=phi_range,
                   elevation_range=theta_range,
                   phi_step=2.0,
                )
                # Keep both synchronized
                grasps_np = grasps_np[s_mask]
                grasps_shifted = grasps_shifted[s_mask]
                scores_np = scores_np[s_mask]

                if not phi_was_softened:
                    print(f"[Filter] Spherical: {len(grasps_shifted)} grasps kept with strict phi range.")
                else:
                    print(f"[Filter] Spherical (phi softened): {len(grasps_shifted)} grasps kept.")

                # Filter 2: Finger orientation and Z-axis roll (on test poses, though orientation is same)
                grasps_oriented_orig = []
                scores_oriented = []
                if args.orient and len(grasps_shifted) > 0:
                    z_mask = filter_by_gripper_z_rotation(
                        grasps_shifted,
                        max_roll_deg=args.max_z_rotation_deg,
                    )
                    orient_mask = z_mask
                    grasps_oriented_orig = grasps_np[orient_mask]
                    grasps_oriented_shifted = grasps_shifted[orient_mask]
                    scores_oriented = scores_np[orient_mask]
                    print('Completed intermediate filtering in {:.2f}s.'.format(time.time() - start_filter_time))
                else:
                    grasps_oriented_orig = grasps_np
                    grasps_oriented_shifted = grasps_shifted
                    scores_oriented = scores_np

                # Prepare valid grasps for output (both stages now have valid_grasps_orig set)
                print(f"[{pipeline_stage}] {len(grasps_oriented_orig)} valid grasps after geometric filters.")

                if args.visualize and len(object_pc) > 0:
                    if reconstruct:
                        # center the augmented cloud for visualization to match the grasps, which are centered
                        object_pc_augmented = tra.transform_points(object_pc_augmented, T_center)
                    grasp_colors = get_color_from_score(scores_oriented, use_255_scale=True)
                    visualize_grasps(
                        viz=vis,
                        scene_pc=None,
                        scene_pc_colors=None,
                        object_pc=pc_filtered_np if not reconstruct else object_pc_augmented,
                        object_pc_colors=None,
                        grasps=grasps_oriented_orig,
                        top_grasp=None,
                        grasps_colors=grasp_colors,
                        gripper_mesh=gripper_info.visual_mesh,
                        gripper_name=grasp_cfg.data.gripper_name,
                        voxel_size=args.voxel_size,
                    )

                if len(grasps_oriented_orig) > 0:
                    # stage1_labels = np.asarray(grasp_labels, dtype=np.int32)
                    # stage1_top_grasp = top_poses[0] if len(top_poses) > 0 else None
                    # stage1_top_label = int(top_labels[0]) if len(top_labels) > 0 else None
                    _cache_stage1_results(
                        raw_grasps=np.asarray(raw_grasps, dtype=np.float32),
                        raw_scores=np.asarray(raw_scores, dtype=np.float32),
                        grasps_oriented_orig=np.asarray(grasps_oriented_orig, dtype=np.float32),
                        grasps_oriented_shifted=np.asarray(grasps_oriented_shifted, dtype=np.float32),
                        scores=np.asarray(scores_oriented, dtype=np.float32),
                        stage1_labels=None,
                        T_center=T_center,
                        object_pc_centered=np.asarray(pc_filtered_np, dtype=np.float32),
                        top_grasp=None, #np.asarray(stage1_top_grasp, dtype=np.float32) if stage1_top_grasp is not None else None,
                        top_label=None,#stage1_top_label,
                        final_grasps=None,
                        final_labels=None,
                        is_symmetric=is_symmetric,
                    )
                    _send_int32(conn, 1)
                    continue
                            
                # Return empty if nothing found
                print(f"[{pipeline_stage}] No valid grasps found.")
                _cache_stage1_results(
                    raw_grasps=raw_grasps,
                    raw_scores=np.asarray(scores_np, dtype=np.float32),
                    grasps_oriented_orig=np.zeros((0, 4, 4), dtype=np.float32),
                    grasps_oriented_shifted=np.zeros((0, 4, 4), dtype=np.float32),
                    scores=np.zeros((0,), dtype=np.float32),
                    stage1_labels=np.zeros((0,), dtype=np.int32),
                    T_center=T_center if 'T_center' in locals() else None,
                    object_pc_centered=np.asarray(pc_filtered_np, dtype=np.float32) if 'pc_filtered_np' in locals() else None,
                    top_grasp=None,
                    top_label=None,
                    final_grasps=None,
                    final_labels=None,
                    is_symmetric=is_symmetric,
                )
                _send_int32(conn, 1)
                
                if args.visualize and len(grasps_np) > 0:
                    grasps = grasps_np
                    grasp_colors = get_color_from_score(scores_np, use_255_scale=True)
                    visualize_grasps(
                        viz=vis,
                        scene_pc=env_pc_downsampled_centered,
                        scene_pc_colors=None,
                        object_pc=pc_filtered_np if not reconstruct else object_pc_augmented,
                        object_pc_colors=None,
                        grasps=grasps,
                        top_grasp=None,
                        grasps_colors=grasp_colors,
                        gripper_mesh=gripper_info.visual_mesh,
                        gripper_name=grasp_cfg.data.gripper_name
                    )
                    

    except KeyboardInterrupt:
        print("\nServer shutting down.")
    except Exception as e:
        print(f"Critical error in server loop: {e}")
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gripper_config", type=str, required=True)
    parser.add_argument("--phi_range", type=float, nargs=2, default=[270, 90],
                        metavar=("PHI_MIN", "PHI_MAX"),
                        help="Azimuth range in degrees (base frame), e.g. --phi_range 0 180")
    parser.add_argument("--theta_range", type=float, nargs=2, default=[-50, 3], 
                        metavar=("THETA_MIN", "THETA_MAX"),
                        help="Elevation range in degrees (base frame), e.g. --theta_range -30 0")
    parser.add_argument("--num_grasps", type=int, default=2000)
    parser.add_argument("--voxel_size", type=float, default=0.004, help="Voxel size (m) for collision speed")
    parser.add_argument("--collision_margin", type=float, default=0.001, help="Padding around fingers")
    parser.add_argument(
        "--hand_object_trim_epsilon",
        type=float,
        default=0.0001,
        help="Meters of tolerance around the estimated object wall when removing impossible hand points inside the object.",
    )
    parser.add_argument("--orient", action="store_true",
                        help="Filter grasps by finger orientation and roll around gripper Z-axis (requires --max_z_rotation_deg)")
    parser.add_argument("--max_z_rotation_deg", type=float, default=20.0,
                        help="Max allowed roll around gripper Z-axis in degrees (0/180 treated as equivalent)")
    parser.add_argument("--reconstruct", action="store_true",
                        help="Enable fusing a synthetic proxy cloud from prior dimensions and axis metadata. Activates for relevant objects only.")
    parser.add_argument("--max_top_k", type=int, default=50, help="Number of top grasps to send back to the robot")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--visualize_finger_gap_rays",
        action="store_true",
        help="Show the ray-casting debug voxels and hit rays used by the finger-gap filter.",
    )
    
    args = parser.parse_args()
    run_server(args)
