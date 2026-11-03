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
    if len(radii) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        radii_smooth = np.convolve(radii, kernel, mode='same')
    else:
        radii_smooth = radii

    lower_band = np.median(radii_smooth[: max(1, len(radii_smooth) // 4)])
    upper_band = np.median(radii_smooth[-max(1, len(radii_smooth) // 4):])
    if not np.isfinite(lower_band) or not np.isfinite(upper_band) or upper_band <= lower_band:
        return None

    threshold = lower_band + 0.45 * (upper_band - lower_band)
    above = np.where(radii_smooth >= threshold)[0]
    if len(above) == 0:
        return None

    cup_start_idx = int(above[0])
    cup_end_idx = int(above[-1])
    cup_start = float(bin_centers[cup_start_idx])
    cup_end = float(bin_centers[cup_end_idx])
    if cup_end <= cup_start + 1e-4:
        return None

    cup_radius = float(radii_smooth[cup_start_idx])
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


def _build_transparent_proxy(object_pc, completion_meta, return_frustums=False):
    if completion_meta is None or not completion_meta.get('transparent', False):
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

    if object_label == 'wine glass':
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
            cup_width = top_w
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
        If return_debug_rays is False:
            mask: K-length boolean array, True if the finger gap is clear.
        If return_debug_rays is True:
            (mask, debug) where debug stores clipped ray segments from cast results.
    """

    if len(scene_pc) == 0 or len(grasp_poses) == 0:
        empty_mask = np.ones(len(grasp_poses), dtype=bool)
        if return_debug_rays:
            return empty_mask, {
                "origins": np.zeros((0, 3), dtype=np.float32),
                "endpoints_clipped": np.zeros((0, 3), dtype=np.float32),
                "hit_mask": np.zeros((0,), dtype=bool),
                "max_dist": 2.0 * finger_half_width,
                "n_per": 0,
                "gap_clear": empty_mask,
                "voxel_centers": np.zeros((0, 3), dtype=np.float32),
            }
        return empty_mask

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
    # 3. Batch-cast all rays at once and evaluate per-grasp.
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

    if return_debug_rays:
        origins = all_rays[:, :3].astype(np.float32)
        dirs = all_rays[:, 3:].astype(np.float32)
        t_vis = np.minimum(t_hit_flat, max_dist)
        t_vis[~np.isfinite(t_vis)] = max_dist
        endpoints_clipped = origins + dirs * t_vis[:, None]
        return gap_clear, {
            "origins": origins,
            "endpoints_clipped": endpoints_clipped.astype(np.float32),
            "hit_mask": np.isfinite(t_hit_flat) & (t_hit_flat < max_dist),
            "max_dist": max_dist,
            "n_per": n_per,
            "gap_clear": gap_clear,
            "voxel_centers": voxel_centers.astype(np.float32),
        }

    return gap_clear


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
        viz["object_pc"].delete()
        viz["collision_free_grasps"].delete()
        viz["top_grasp"].delete()
        viz["finger_gap_rays"].delete()

    # Visualize scene point cloud - use RGB colors if available, otherwise use gray
    if scene_pc_colors is not None:
        visualize_pointcloud(
            viz, "scene_pc", scene_pc, scene_pc_colors, size=voxel_size
        )
    else:
        visualize_pointcloud(
            viz, "scene_pc", scene_pc, [128, 128, 128], size=voxel_size
        )

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
            # if i < 2:   
            #     visualize_mesh(
            #         viz,
            #         f"collision_free_grasps/grasp_{i:03d}",
            #         gripper_mesh,
            #         color=[0, 40, 150],
            #         transform=grasp,
            #     )
        # if top_grasp is not None:
        #     # Highlight top grasp
        #     visualize_grasp(
        #         viz,
        #         "top_grasp",
        #         top_grasp,
        #         color=[255, 0, 0],
        #         gripper_name=gripper_name,
        #         linewidth=2.0,
        #     )

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

    # Initialize visualization if requested
    if args.visualize:
        vis = create_visualizer()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # server.settimeout(1.0)
    if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)
    server.bind(SOCKET_PATH)
    server.listen(5)

    print(f"GraspGen Server (Optimized) | Voxel: {args.voxel_size}m | Margin: {args.collision_margin}m")

    try:
        while True:
            # try:
            conn, _ = server.accept()
            # except socket.timeout: continue

            with conn:
                start_time = time.time()
                
                # Receive scene and object point clouds
                header = recv_all(conn, 12)  # 3 int32 values: scene_size, object_size, meta_size
                if not header: continue
                scene_size, object_size, meta_size = np.frombuffer(header, dtype=np.int32)
                print(
                    f"\nReceived request: scene_size={scene_size}, object_size={object_size}, meta_size={meta_size}"
                )
                
                # Receive scene point cloud (scene_size x 3 float32)
                scene_data = recv_all(conn, scene_size * 3 * 4)
                if not scene_data: continue
                scene_pc = np.frombuffer(scene_data, dtype=np.float32).reshape((scene_size, 3))
                # scene_pc_filtered, _ = point_cloud_outlier_removal(torch.from_numpy(scene_pc))
                # scene_pc_filtered = scene_pc_filtered.numpy()
                scene_pc_filtered = scene_pc # filtering is done on the sender side
                
                # Receive object point cloud (object_size x 3 float32)
                object_data = recv_all(conn, object_size * 3 * 4)
                if not object_data: continue
                object_pc = np.frombuffer(object_data, dtype=np.float32).reshape((object_size, 3))

                completion_meta = {}
                if meta_size > 0:
                    meta_data = recv_all(conn, meta_size)
                    if meta_data:
                        try:
                            completion_meta = json.loads(meta_data.decode('utf-8'))
                        except Exception as exc:
                            print(f"[Transparent] Failed to parse completion metadata: {exc}")
                            completion_meta = {}

                request_object_label = str(completion_meta.get('object_label', '') or '').strip().lower()
                is_wine_glass = request_object_label == 'wine glass'

                if args.transparent:
                    object_pc_augmented, transparent_frustums = _build_transparent_proxy(
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
                        transparent_frustums,
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
                center_of_mass = (min_coords + max_coords) / 2.0
                print(f"Center of Mass after outlier removal (XYZ): {center_of_mass}")

                # Calculate vertical thirds for labeling (Top/Mid/Bottom)
                obj_z_min = min_coords[2]
                obj_z_max = max_coords[2]
                obj_height = obj_z_max - obj_z_min
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
                    pc_filtered_np, grasp_sampler, grasp_threshold=0.7, num_grasps=args.num_grasps
                )
                print('Inference completed in {:.2f}s.'.format(time.time() - start_inference_time))
                grasps_np, scores_np = grasps.cpu().numpy(), conf.cpu().numpy()
                # Ensure grasps have correct homogeneous coordinate
                grasps_np[:, 3, 3] = 1 
                
                # centering
                # We center based on the object to keep numeric stability high for the next filters
                pc_filtered_np, grasps_np, T_center = (
                    process_point_cloud(
                        pc_filtered_np, grasps_np
                    )
                )
                scene_pc_centered = tra.transform_points(scene_pc_filtered, T_center) # for visualization
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
                # We will filter on grasps_test, but eventually return the original grasps_np.
                T_grasp_vertical_offset = tra.translation_matrix([0, 0, 0.02]) # 2cm forward shift
                grasps_test = grasps_np @ T_grasp_vertical_offset

                start_filter_time = time.time()
                # Filter 1: Spherical Constraints (adaptive phi softening) on the test poses
                s_mask, phi_was_softened = apply_spherical_filter_adaptive_phi(
                   grasps_test, scores_np,
                   azimuth_range=phi_range,
                   elevation_range=theta_range,
                   phi_step=2.0,
                )
                # Keep both synchronized
                grasps_np = grasps_np[s_mask]
                grasps_test = grasps_test[s_mask]
                scores_np = scores_np[s_mask]

                if not phi_was_softened:
                    print(f"[Filter] Spherical: {len(grasps_test)} grasps kept with strict phi range.")
                else:
                    print(f"[Filter] Spherical (phi softened): {len(grasps_test)} grasps kept.")

                # Filter 2: Finger orientation and Z-axis roll (on test poses, though orientation is same)
                grasps_oriented_orig = []
                grasps_oriented_test = []
                scores_oriented = []
                if args.orient and len(grasps_test) > 0:
                    # f_mask = filter_by_finger_orientation(grasps_test, max_tilt_deg=15)
                    z_mask = filter_by_gripper_z_rotation(
                        grasps_test,
                        max_roll_deg=args.max_z_rotation_deg,
                    )
                    # orient_mask = f_mask & z_mask
                    orient_mask = z_mask
                    grasps_oriented_orig = grasps_np[orient_mask]
                    grasps_oriented_test = grasps_test[orient_mask]
                    scores_oriented = scores_np[orient_mask]
                    print('Completed intermediate filtering in {:.2f}s.'.format(time.time() - start_filter_time))
                else:
                    grasps_oriented_orig = grasps_np
                    grasps_oriented_test = grasps_test
                    scores_oriented = scores_np
                
                valid_grasps_orig = []
                valid_grasps_test = []
                valid_scores = []
                valid_grasps = np.zeros((0, 4, 4), dtype=np.float32)
                
                valid_grasps_debug = None
                valid_scores_debug = None
                
                # Filter 3: Collision Check (Optimized with downsampled PC)
                if len(grasps_oriented_test) > 0:
                    # start_collision_filter_time = time.time()
                    # if len(env_pc_downsampled_centered) == 0 or args.collision_margin == 0:
                    #     print("[Filter] Collision: No scene points in vicinity, skipping collision check.")
                    #     c_mask = np.ones(len(grasps_oriented_test), dtype=bool)
                    # else:
                    #     c_mask = filter_colliding_grasps(
                    #         scene_pc=env_pc_downsampled_centered,
                    #         grasp_poses=grasps_oriented_test,
                    #         gripper_collision_mesh=gripper_info.collision_mesh,
                    #         collision_threshold=args.collision_margin
                    #     )
                    # print('Filtered collisions in {:.2f}s'.format(time.time() - start_collision_filter_time))
                    valid_grasps_orig = grasps_oriented_orig #grasps_oriented_orig[c_mask]
                    valid_grasps_test = grasps_oriented_test #grasps_oriented_test[c_mask]
                    valid_scores = scores_oriented #scores_oriented[c_mask]

                    # Filter 4: Finger-gap ray casting on test poses
                    if len(valid_grasps_test) > 0:
                        start_gap_filter_time = time.time()
                        _ray_hw = (gripper_info.finger_half_width
                                   if hasattr(gripper_info, "finger_half_width") else 0.068)
                        _ray_zo = 0.11 # 11cm fits to our gripper 

                        valid_grasps_debug = valid_grasps_orig.copy() # for visualization before gap filter
                        valid_scores_debug = valid_scores.copy()
                        gap_result = filter_grasps_finger_gap_raycasting(
                            scene_pc=env_pc_downsampled_centered,
                            grasp_poses=valid_grasps_test,
                            finger_half_width=_ray_hw,
                            fingertip_z_offset=_ray_zo,
                            voxel_size=args.voxel_size,
                            return_debug_rays=args.visualize,
                        )
                        if isinstance(gap_result, tuple):
                            gap_mask, ray_debug = gap_result
                        else:
                            gap_mask = gap_result
                            ray_debug = None

                        valid_grasps = valid_grasps_orig[gap_mask]
                        valid_scores = valid_scores[gap_mask]

                        print('Finger-gap filter in {:.2f}s: {} grasps kept'.format(
                            time.time() - start_gap_filter_time, len(valid_grasps)))

                        if args.visualize and ray_debug is not None:
                            # Visualize the exact voxelized scene used by rc_scene.
                            visualize_pointcloud(
                                vis,
                                "finger_gap_rays/rc_scene_voxels",
                                ray_debug["voxel_centers"],
                                [80, 200, 255],
                                size=args.voxel_size,
                            )

                            # Visualize only rays that actually hit geometry for
                            # all invalid grasps in this batch.
                            invalid_idxs = np.where(~ray_debug["gap_clear"])[0]
                            n_per = int(ray_debug["n_per"])

                            if len(invalid_idxs) > 0 and n_per > 0:
                                origins_chunks = []
                                ends_chunks = []

                                for k in invalid_idxs:
                                    start = int(k) * n_per
                                    end = start + n_per
                                    hit_mask_k = ray_debug["hit_mask"][start:end]
                                    if np.any(hit_mask_k):
                                        origins_chunks.append(ray_debug["origins"][start:end][hit_mask_k])
                                        ends_chunks.append(ray_debug["endpoints_clipped"][start:end][hit_mask_k])

                                if len(origins_chunks) > 0:
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

                    if len(valid_grasps) > 0:
                        # Finger center offset for wrist poses (0.11)
                        finger_offset = 0.15 #0.11
                        
                        # Calculate Labels for all valid grasps (to use for visualization)
                        T_inv_center = tra.inverse_matrix(T_center)
                        grasp_labels = []
                        grasp_colors = []
                        adjusted_grasps = []
                        
                        for g_centered in valid_grasps:
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

                        valid_grasps = np.asarray(adjusted_grasps, dtype=np.float32)

                        top_idx = np.argsort(valid_scores)[::-1]
                        top_poses = valid_grasps[top_idx]
                        
                        # Labels for the sorted top_poses (for sending later)
                        top_labels = [grasp_labels[i] for i in top_idx]

                        elapsed = time.time() - start_time
                        print(f"Success: {len(valid_grasps)} grasps found in {elapsed:.2f}s. Top Score: {valid_scores[top_idx[0]]:.3f}")

                        if args.visualize:
                            # Visualization in centered frame
                            save_for_3d_viewer(
                                scene_pc=scene_pc_centered,
                                object_pc=pc_filtered_np,
                                best_poses=top_poses,
                                gripper_mesh=gripper_info.collision_mesh
                            )
                            # grasp_colors was calculated above based on height
                            visualize_grasps(
                                viz=vis,
                                scene_pc=scene_pc_centered,
                                scene_pc_colors=None,
                                object_pc=pc_filtered_np,
                                object_pc_colors=None,
                                grasps=valid_grasps,
                                top_grasp=top_poses[0],
                                grasps_colors=grasp_colors,
                                gripper_mesh=gripper_info.visual_mesh,
                                gripper_name=grasp_cfg.data.gripper_name,
                                voxel_size=args.voxel_size,
                            )


                        # first send number of valid grasps as int32
                        num_valid = len(valid_grasps)
                        conn.sendall(num_valid.to_bytes(4, byteorder='little'))
                        # then send each grasp pose as 16 float32 values (4x4 matrix) followed by the int32 label
                        for i, pose in enumerate(top_poses):
                            pose_original = tra.inverse_matrix(T_center) @ pose
                            # Send Pose
                            conn.sendall(pose_original.astype(np.float32).tobytes())
                           
                        for i, pose in enumerate(top_poses):
                            label = top_labels[i]
                            # Send Label (4 bytes, little endian)
                            conn.sendall(int(label).to_bytes(4, byteorder='little', signed=True))
                        continue
                            
                # Return empty if nothing found
                print("No valid grasps found.")
                # return 0 numbers of grasps
                conn.sendall((0).to_bytes(4, byteorder='little'))
                
                if args.visualize and len(grasps_np) > 0:
                    if valid_grasps_debug is not None:
                        grasps = valid_grasps_debug
                        scores = valid_scores_debug
                    else:
                        grasps = grasps_np
                        scores = scores_np
                    grasp_colors = get_color_from_score(scores_np, use_255_scale=True)
                    # visualize_grasps(
                    #     viz=vis,
                    #     scene_pc=env_pc_downsampled_centered,
                    #     scene_pc_colors=None,
                    #     object_pc=pc_filtered_np,
                    #     object_pc_colors=None,
                    #     grasps=grasps,
                    #     top_grasp=None,
                    #     grasps_colors=grasp_colors,
                    #     gripper_mesh=gripper_info.visual_mesh,
                    #     gripper_name=grasp_cfg.data.gripper_name
                    # )
                    

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
    parser.add_argument("--theta_range", type=float, nargs=2, default=[-50, 5], 
                        metavar=("THETA_MIN", "THETA_MAX"),
                        help="Elevation range in degrees (base frame), e.g. --theta_range -30 0")
    parser.add_argument("--num_grasps", type=int, default=300)
    parser.add_argument("--voxel_size", type=float, default=0.004, help="Voxel size (m) for collision speed")
    parser.add_argument("--collision_margin", type=float, default=0.001, help="Padding around fingers")
    parser.add_argument("--orient", action="store_true",
                        help="Filter grasps by finger orientation and roll around gripper Z-axis (requires --max_z_rotation_deg)")
    parser.add_argument("--max_z_rotation_deg", type=float, default=20.0,
                        help="Max allowed roll around gripper Z-axis in degrees (0/180 treated as equivalent)")
    parser.add_argument("--transparent", action="store_true",
                        help="Fuse a synthetic proxy cloud from prior dimensions and axis metadata")
    parser.add_argument("--visualize", action="store_true")
    
    args = parser.parse_args()
    run_server(args)