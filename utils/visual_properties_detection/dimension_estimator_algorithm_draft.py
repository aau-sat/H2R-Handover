# This file contains the algorithm to estimate height, width at top and width at bottom 
# based on pointcloud and segmentation mask. 
# It estimates the properties of different depth and mask pairs of the object
# and then averages over all estimates to get a final estimate.
# For width estimation, it uses pointcloud in cameras frame
# For height estimation it uses transformed pointcloud in robot frame

# In this file currently the intrinsics, transforms and paths for depth mask pairs are hardcoded.
# It iterates (start_idx, end_idx) over the given scene
# It needs to be changed into a node in the future


import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy import stats
import matplotlib.pyplot as plt

# --- HARDCODED INTRINSICS ---
# Since depth and mask are already aligned, we only need one set of intrinsics.
# Use whichever intrinsics match the resolution of your depth/mask images.
DEPTH_INTRINSICS = {
    'fx': 609.5201416015625, 'fy': 610.1154174804688,
    'cx': 326.4013366699219, 'cy': 242.26622009277344,
    'width': 640, 'height': 480
}

# --- HAND-EYE CALIBRATION (eye-on-base) ---
# Transform from camera_color_optical_frame → robot base frame
# Obtained from easy_handeye2 calibration: my_eob_calib.calib
_T_translation = np.array([-0.9126294122484981,
                             0.007181141011057546,
                             0.3065324849864441])
_T_quaternion = np.array([-0.31239835432412144,   # x
                              0.7321115479914871,    # y
                             -0.5496956278095257,    # z
                              0.2534850415314091])   # w

# Build 4x4 homogeneous matrix (camera → base)
_R = Rotation.from_quat(_T_quaternion).as_matrix()  # (3,3)
CAMERA_TO_BASE = np.eye(4)
CAMERA_TO_BASE[:3, :3] = _R
CAMERA_TO_BASE[:3, 3] = _T_translation



class ObjectDimensionCalculator:
    def __init__(self):
        self.intr = DEPTH_INTRINSICS

    # ------------------------------------------------------------------
    # Internal helper: deproject + mask-filter + outlier removal
    # ------------------------------------------------------------------
    def get_object_points(self, depth_meters, cv_mask):
        """Deproject the depth image into 3-D, keep only pixels where the
        mask is non-zero, and remove statistical outliers.

        Because depth and mask are already aligned (same pixel grid), no
        camera-to-camera transform is needed – we simply apply the mask
        directly in pixel space.

        Returns:
            np.ndarray of shape (N, 3) in metres, or None if fewer than
            50 points survive.
        """
        h, w = depth_meters.shape

        # 1. Build pixel grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # 2. Apply mask in pixel space
        obj_px = cv_mask > 0                        # (H, W) bool
        z = depth_meters[obj_px]                    # depth at masked pixels
        u_obj = u[obj_px]
        v_obj = v[obj_px]

        # 3. Discard invalid / too-close depth readings
        valid = z > 0.1                             # > 10 cm
        z, u_obj, v_obj = z[valid], u_obj[valid], v_obj[valid]

        if len(z) == 0:
            return None

        # 4. Deproject to 3-D using depth intrinsics
        x = (u_obj - self.intr['cx']) * z / self.intr['fx']
        y = (v_obj - self.intr['cy']) * z / self.intr['fy']
        points = np.stack((x, y, z), axis=-1)       # (N, 3)

        if len(points) < 50:
            return None

        # 5. Statistical outlier removal
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_clean = pcd.select_by_index(ind)

        return np.asarray(pcd_clean.points)

    # ------------------------------------------------------------------
    # Overall bounding-box dimensions
    # ------------------------------------------------------------------
    def get_dimensions(self, depth_meters, cv_mask):
        """Returns overall dimensions (X, Y, Z) in metres."""
        pts = self.get_object_points(depth_meters, cv_mask)
        if pts is None:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        aabb = pcd.get_axis_aligned_bounding_box()
        dims = aabb.get_max_bound() - aabb.get_min_bound()
        return dims  # [X, Y, Z]

    def get_height(self, depth_meters, cv_mask, T_cam_to_base=None, return_details=False):
        """Returns the physical height of the object in metres.

        If return_details is True, returns (height, z_lo, z_hi).
        """
        pts = self.get_object_points(depth_meters, cv_mask)
        if pts is None:
            return None

        if T_cam_to_base is not None:
            # Homogeneous coordinates: (N,4)
            ones = np.ones((len(pts), 1), dtype=pts.dtype)
            pts_h = np.hstack([pts, ones])
            # Transform to robot base frame
            pts_robot = (T_cam_to_base @ pts_h.T).T[:, :3]
            z_vals = pts_robot[:, 2]
        else:
            # Legacy: camera Y span
            z_vals = pts[:, 1]

        # Use percentile trimming instead of hard min/max to reject outlier
        # spikes (e.g. table pixels leaking through the mask edge, depth noise).
        # We use 5th–95th percentile to clearly visualize outliers.
        pct_lo, pct_hi = 1.0, 99.0
        z_lo = np.percentile(z_vals, pct_lo)
        z_hi = np.percentile(z_vals, pct_hi)
        print(f"  [Height debug] Z range: {np.min(z_vals)*100:.1f}–{np.max(z_vals)*100:.1f} cm  "
              f"| Trimmed ({pct_lo}–{pct_hi}%): {z_lo*100:.1f}–{z_hi*100:.1f} cm  "
              f"→ height = {(z_hi - z_lo)*100:.2f} cm")
        
        height = float(z_hi - z_lo)
        if return_details:
            return height, z_lo, z_hi
        return height

    # ------------------------------------------------------------------
    # Simple bounding-box width estimation (camera / pixel frame)
    # ------------------------------------------------------------------
    def get_width_bb(self, depth_meters, cv_mask,
                     top_frac=0.20, bot_frac=0.10,
                     outlier_std=2.0, min_pts=5):
        """Estimate object width at the top and bottom purely in the
        camera (pixel) frame – no PCA, no 3-D transforms.

        Algorithm
        ---------
        1. Compute the axis-aligned bounding box of the (non-zero) mask.
        2. *Top band*  : uppermost ``top_frac`` fraction of the BB rows.
           *Bottom band*: lowermost ``bot_frac`` fraction of the BB rows.
        3. For each band, collect every masked pixel column (u) in that
           row range.  Apply simple σ-rejection and take the span
           [min_u … max_u] of the survivors.
        4. Back-project those two pixel columns to metres using the
           depth intrinsics (average depth of all masked pixels in the
           band as the representative Z).

        Args:
            depth_meters: (H, W) float32 depth image in metres.
            cv_mask:      (H, W) uint8 segmentation mask (>0 = object).
            top_frac:     Fraction of BB height used for the top band (default 0.20).
            bot_frac:     Fraction of BB height used for the bottom band (default 0.10).
            outlier_std:  σ-multiplier for column-outlier rejection (default 2.0).
            min_pts:      Minimum masked pixels a band must have (default 5).

        Returns:
            dict with keys 'top_m' and 'bot_m' (metres), or None if the
            mask is empty.
        """
        # --- bounding box of the mask -----------------------------------
        ys, xs = np.where(cv_mask > 0)
        if len(xs) < min_pts:
            return None

        row_min, row_max = int(ys.min()), int(ys.max())  # image rows (v)
        bb_height = row_max - row_min
        if bb_height < 4:
            return None

        # Band row ranges (image rows increase downward)
        top_row_hi  = row_min + max(1, int(np.round(bb_height * top_frac)))
        bot_row_lo  = row_max - max(1, int(np.round(bb_height * bot_frac)))

        def _band_width_m(row_lo, row_hi):
            """Return physical width [m] for masked pixels in [row_lo, row_hi]."""
            # Pixel columns of masked pixels in the row slice
            in_band = (ys >= row_lo) & (ys <= row_hi)
            u_band  = xs[in_band]
            v_band  = ys[in_band]

            if len(u_band) < min_pts:
                return None

            # --- σ-outlier rejection on the column values ---------------
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

            u_left  = int(u_clean.min())
            u_right = int(u_clean.max())

            # Average depth of masked pixels in the band (valid only)
            z_band = depth_meters[v_clean, u_clean]
            z_valid = z_band[z_band > 0.1]
            if len(z_valid) == 0:
                return None
            z_avg = float(np.median(z_valid))

            # Back-project pixel span to metric width
            x_left  = (u_left  - self.intr['cx']) * z_avg / self.intr['fx']
            x_right = (u_right - self.intr['cx']) * z_avg / self.intr['fx']
            return float(abs(x_right - x_left))

        w_top = _band_width_m(row_min, top_row_hi)
        w_bot = _band_width_m(bot_row_lo, row_max)

        return {'top_m': w_top, 'bot_m': w_bot}


    def robust_mean(self, values):
        """Applies KDE-based outlier rejection and returns the mode of the distribution."""
        if not values or len(values) == 0:
            return None
        if len(values) < 3:
            return float(np.mean(values))
        return self._filter_kde_mode(values)

    def _filter_kde_mode(self, values):
        """Find the mode/peak of the distribution using Gaussian KDE."""
        data = np.array(values)
        try:
            kernel = stats.gaussian_kde(data)
            # Sample density over the range
            x_test = np.linspace(min(data), max(data), 200)
            y_test = kernel(x_test)
            return float(x_test[np.argmax(y_test)])
        except Exception:
            return float(np.median(data))

    def consolidate_all(self, estimates):
        """Consolidates all metrics using KDE-based outlier rejection."""
        results = {}
        for key in ['height', 'length', 'width', 'depth', 'top_w', 'bot_w']:
            results[key] = self.robust_mean(estimates[key])
        return results

    def plot_estimates(self, estimates, save_path):
        """Plots the raw distribution of estimates before rejection."""
        metrics = [k for k in ['height', 'length', 'width', 'top_w', 'bot_w'] if estimates[k]]
        if not metrics:
            return
            
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        if len(metrics) == 1: axes = [axes]
        
        for ax, key in zip(axes, metrics):
            data = np.array(estimates[key]) * 100 # Convert to cm
            mean = np.mean(data)
            std = np.std(data)
            
            ax.hist(data, bins=20, alpha=0.6, color='skyblue', edgecolor='black', label='Raw Samples')
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
            ax.axvspan(mean - std, mean + std, color='red', alpha=0.1, label=f'Std Dev: ±{std:.2f}')
            
            ax.set_title(f"Distribution: {key.upper()}")
            ax.set_xlabel("Dimension (cm)")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"  [Stats] Plot saved to → {save_path}")
        plt.close()


# ---------------------------------------------------------------------------
# Quick test – run directly: python object_dim_calc.py
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAMPLE_DIR = os.path.join(BASE_DIR, '..', 'data', 'sample_data', 'captured_images')
    DEPTH_SCALE = 1000.0   # PNG values are millimetres → divide by 1000 to get metres

    calc = ObjectDimensionCalculator()
    
    # Define range of frames to process
    start_idx = 0
    end_idx = 10
    indices = range(start_idx, end_idx + 1)
    
    estimates = {
        'height': [],
        'length': [],
        'width': [],
        'depth': [],
        'top_w': [],
        'bot_w': [],
    }
    
    depth_stack = []
    mask_stack  = []
    
    print(f"Processing frames {start_idx} to {end_idx} using KDE rejection strategy")

    for idx in indices:
        depth_path = os.path.join(SAMPLE_DIR,  f'{idx:04d}_depth.png')
        mask_path  = os.path.join(SAMPLE_DIR,  f'{idx:04d}_mask.png')

        if not os.path.exists(depth_path) or not os.path.exists(mask_path):
            continue

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        mask_raw  = cv2.imread(mask_path,  cv2.IMREAD_GRAYSCALE)

        if depth_raw is None or mask_raw is None:
            continue

        depth_m = depth_raw.astype(np.float32) / DEPTH_SCALE
        
        obj_pts = calc.get_object_points(depth_m, mask_raw)
        if obj_pts is None:
            continue

        depth_stack.append(depth_m)
        mask_stack.append(mask_raw)

        # --- Calculate individual estimates for this frame ---
        h_val = calc.get_height(depth_m, mask_raw, T_cam_to_base=CAMERA_TO_BASE)
        dims = calc.get_dimensions(depth_m, mask_raw)
        prof = calc.get_width_bb(depth_m, mask_raw)

        if h_val is not None:
            estimates['height'].append(h_val)
        if dims is not None:
            estimates['length'].append(dims[0])
            estimates['width'].append(dims[1])
            estimates['depth'].append(dims[2])
        if prof is not None:
            if prof.get('top_m') is not None: estimates['top_w'].append(prof['top_m'])
            if prof.get('bot_m')  is not None: estimates['bot_w'].append(prof['bot_m'])

    if not depth_stack:
        print("No valid frames found.")
    else:
        # 1. Plot raw stats before rejection
        stat_plot_path = os.path.join(BASE_DIR, 'measurement_stats.png')
        calc.plot_estimates(estimates, stat_plot_path)

        # 2. Filter (estimation) Outliers and Average using KDE
        final_results = calc.consolidate_all(estimates)

        final_h   = final_results.get('height')
        final_l   = final_results.get('length')
        final_w   = final_results.get('width')
        final_d   = final_results.get('depth')
        final_top = final_results.get('top_w')
        final_bot = final_results.get('bot_w')

        print(f"\n=== Averaged Results ({len(depth_stack)} frames) ===")
        if final_h is not None:
            label = "Robot Z (Vertical)" if CAMERA_TO_BASE is not None else "Camera Y"
            print(f"  Height Estimation ({label}): {final_h*100:.2f} cm")
        else:
            print("  Height Estimation – N/A")
            
        if final_l is not None:            
            top_cm = f"{final_top*100:.2f} cm" if final_top is not None else "N/A"
            bot_cm = f"{final_bot*100:.2f} cm" if final_bot is not None else "N/A"
            print(f"  Width at Top:  {top_cm}")
            print(f"  Width at Bottom:    {bot_cm}")
        else:
            print("  Dimensions – N/A")

        # ------------------------------------------------------------------
        # 2. Visualization (consensus cloud)
        # ------------------------------------------------------------------
        print("\nGenerating consensus point cloud for MeshLab...")
        depth_array = np.stack(depth_stack, axis=0)
        depth_array[depth_array == 0] = np.nan
        avg_depth = np.nanmean(depth_array, axis=0)
        avg_depth = np.nan_to_num(avg_depth)
        
        mask_array = np.stack(mask_stack, axis=0)
        avg_mask = (np.sum(mask_array > 0, axis=0) > (len(depth_stack) / 2)).astype(np.uint8) * 255
        
        h_img, w_img = avg_depth.shape
        u_all, v_all = np.meshgrid(np.arange(w_img), np.arange(h_img))
        z_all = avg_depth.flatten()
        valid = z_all > 0.1
        z_v, u_v, v_v = z_all[valid], u_all.flatten()[valid], v_all.flatten()[valid]

        pts_cam = np.stack([(u_v - DEPTH_INTRINSICS['cx']) * z_v / DEPTH_INTRINSICS['fx'],
                            (v_v - DEPTH_INTRINSICS['cy']) * z_v / DEPTH_INTRINSICS['fy'],
                            z_v], axis=-1)
        pts_robot = (CAMERA_TO_BASE @ np.hstack([pts_cam, np.ones((len(pts_cam), 1))]).T).T[:, :3]

        mask_flat = (avg_mask.flatten())[valid]
        rgb = np.full((len(pts_robot), 3), 180, dtype=np.uint8)  # Gray for scene
        rgb[mask_flat > 0] = [239, 68, 68]                        # Red for object

        out_path = os.path.join(BASE_DIR, 'debug_scene_robot_frame.ply')
        with open(out_path, 'w') as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts_robot)}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for i in range(len(pts_robot)):
                f.write(f"{pts_robot[i,0]:.6f} {pts_robot[i,1]:.6f} {pts_robot[i,2]:.6f} {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n")

        print(f"  Saved → {out_path}")
        print("  Colors: RED (object), GRAY (background)")

