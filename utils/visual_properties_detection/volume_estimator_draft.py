import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy import stats
import matplotlib.pyplot as plt

# --- CAMERA CONFIGURATION ---
# Values from intrinsics.txt
CAMERA_CONFIG = {
    'left': {
        'intrinsics': {
            'fx': 605.6497192382812, 'fy': 605.0811767578125,
            'cx': 325.6852722167969, 'cy': 253.1599884033203,
            'width': 640, 'height': 480
        },
        'translation': np.array([0.9572414526518586, -0.5733583795072371, 0.6132377553886849]),
        'quaternion': np.array([-0.5695525722846513, -0.6162914175300531, 0.39367624593843054, 0.37525160821315495]) # xyzw
    },
    'right': {
        'intrinsics': {
            'fx': 609.5201416015625, 'fy': 610.1154174804688,
            'cx': 326.4013366699219, 'cy': 242.26622009277344,
            'width': 640, 'height': 480
        },
        'translation': np.array([-0.9108059609829655, -0.37514940999326296, 0.5633414779217282]),
        'quaternion': np.array([-0.4378362214705174, 0.7096928424136695, -0.4618027699548845, 0.30228085327592835]) # xyzw
    }
}

def get_camera_to_base(side):
    """Builds 4x4 homogeneous matrix (camera → base) for given side."""
    config = CAMERA_CONFIG[side]
    r = Rotation.from_quat(config['quaternion']).as_matrix()
    t = config['translation']
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m



class ObjectDimensionCalculator:
    def __init__(self, intrinsics):
        self.intr = intrinsics

    # ------------------------------------------------------------------
    # Internal helper: deproject + mask-filter + outlier removal
    # ------------------------------------------------------------------
    def get_object_points(self, depth_meters, cv_mask):
        """Deproject the depth image into 3-D, keep only pixels where the
        mask is non-zero, and remove statistical outliers.
        """
        h, w = depth_meters.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        obj_px = cv_mask > 0                        
        z = depth_meters[obj_px]                    
        u_obj = u[obj_px]
        v_obj = v[obj_px]

        valid = z > 0.1                             # > 10 cm
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
        """Returns overall dimensions (X, Y, Z) in metres."""
        pts = self.get_object_points(depth_meters, cv_mask)
        if pts is None: return None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        aabb = pcd.get_axis_aligned_bounding_box()
        return aabb.get_max_bound() - aabb.get_min_bound()

    def get_height(self, depth_meters, cv_mask, T_cam_to_base=None, return_details=False):
        """Returns the physical height of the object in metres."""
        pts = self.get_object_points(depth_meters, cv_mask)
        if pts is None: return None

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

    def get_width_bb(self, depth_meters, cv_mask, top_frac=0.20, bot_frac=0.10, outlier_std=2.0, min_pts=5):
        """Estimate object width at the top and bottom in camera frame."""
        ys, xs = np.where(cv_mask > 0)
        if len(xs) < min_pts: return None

        row_min, row_max = int(ys.min()), int(ys.max())
        bb_height = row_max - row_min
        if bb_height < 4: return None

        def _band_width_m(row_lo, row_hi):
            in_band = (ys >= row_lo) & (ys <= row_hi)
            u_band, v_band = xs[in_band], ys[in_band]
            if len(u_band) < min_pts: return None

            u_mean, u_std = np.mean(u_band), np.std(u_band)
            if u_std > 0:
                keep = np.abs(u_band - u_mean) <= outlier_std * u_std
                u_clean, v_clean = u_band[keep], v_band[keep]
            else:
                u_clean, v_clean = u_band, v_band

            if len(u_clean) < min_pts: return None
            z_avg = float(np.median(depth_meters[v_clean, u_clean]))
            x_left = (int(u_clean.min()) - self.intr['cx']) * z_avg / self.intr['fx']
            x_right = (int(u_clean.max()) - self.intr['cx']) * z_avg / self.intr['fx']
            return float(abs(x_right - x_left))

        w_top = _band_width_m(row_min, row_min + max(1, int(bb_height * top_frac)))
        w_bot = _band_width_m(row_max - max(1, int(bb_height * bot_frac)), row_max)
        return {'top_m': w_top, 'bot_m': w_bot}

    def get_pca_volume(self, depth_meters, cv_mask, slice_height_m=0.004, height_limit=None, outlier_std=2.0):
        """Volume estimation using PCA alignment and slicing."""
        pts = self.get_object_points(depth_meters, cv_mask)
        if pts is None or len(pts) < 100: return None

        pts_mean = np.mean(pts, axis=0)
        _, _, Vh = np.linalg.svd(pts - pts_mean, full_matrices=False)
        pc1 = Vh[0] 
        if pc1[1] > 0: Vh[0] = -Vh[0]
            
        pts_pca = (pts - pts_mean) @ Vh.T
        h_vals = pts_pca[:, 0]
        h_min, h_max = np.percentile(h_vals, [1, 99])
        if h_max <= h_min: return None
            
        h_bins = np.arange(h_min, h_max, slice_height_m)
        total_vol, slice_data, acc_h = 0.0, [], 0.0

        for i in range(len(h_bins) - 1):
            h_lo, h_hi = h_bins[i], h_bins[i+1]
            thick = abs(h_hi - h_lo)
            if height_limit and (acc_h + thick) > height_limit: break
            
            mask = (h_vals >= h_lo) & (h_vals < h_hi)
            pts_slice = pts_pca[mask]
            if len(pts_slice) < 5: continue

            def _span(data):
                m, s = np.mean(data), np.std(data)
                clean = data[np.abs(data - m) <= outlier_std * s] if s > 0 else data
                return np.max(clean) - np.min(clean) if len(clean) >= 5 else None

            s2, s3 = _span(pts_slice[:,1]), _span(pts_slice[:,2])
            if s2 is None or s3 is None: continue
                
            diam = max(s2, s3)
            total_vol += np.pi * (diam/2.0)**2 * thick
            acc_h += thick
            slice_data.append({'rel_h': acc_h, 'diameter': diam})

        return total_vol, slice_data

    def get_cylindrical_volume(self, depth_meters, cv_mask, slice_height_m=0.005, height_limit=None):
        return self.get_pca_volume(depth_meters, cv_mask, slice_height_m, height_limit)

    def robust_mean(self, values):
        if not values: return None
        if len(values) < 3: return float(np.mean(values))
        try:
            data = np.array(values)
            kernel = stats.gaussian_kde(data)
            x = np.linspace(min(data), max(data), 200)
            return float(x[np.argmax(kernel(x))])
        except: return float(np.median(values))

    def calculate_consensus_volume(self, slices_l, slices_r, slice_height_m=0.005):
        if not slices_l and not slices_r: return 0.0, []
        bins = {}
        for s in slices_l + slices_r:
            idx = int(round(s['rel_h'] / slice_height_m))
            bins.setdefault(idx, []).append(s['diameter'])
            
        merged, total_vol = [], 0.0
        for idx in sorted(bins.keys()):
            diam = np.mean(bins[idx])
            total_vol += np.pi * (diam/2.0)**2 * slice_height_m
            merged.append({'rel_h': idx * slice_height_m, 'diameter': diam})
        return total_vol, merged

    def estimate_mass(self, volume_ml, label, filled=False):
        """Estimates mass in grams based on volume, label, and filling."""
        l_low = label.lower()
        m_cont = 0.0
        if 'cup' in l_low:
            # Quadratic fit for (200, 3), (500, 11), (600, 8)
            # Fits exactly: y = ax^2 + bx + c
            # Coefficients: a = -0.00014167, b = 0.125833, c = -16.5
            m_cont = -0.00014167 * (volume_ml**2) + 0.125833 * volume_ml - 16.5
            m_cont = max(0.1, m_cont)
        elif 'wineglass' in l_low:
            m_cont = 75.0 + (volume_ml * (61.0 / 354.0))
            m_cont = max(1.0, m_cont)
            
        rice_mass = (volume_ml * 0.9) * 0.9 if filled else 0.0
        return m_cont, m_cont + rice_mass

    def consolidate_all(self, estimates):
        results = {}
        for key, vals in estimates.items():
            results[key] = self.robust_mean(vals)
        return results

    def plot_estimates(self, estimates, save_path=None):
        metrics = [k for k in ['height', 'top_w', 'bot_w', 'volume'] if estimates.get(k)]
        if not metrics or not save_path: return
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        if len(metrics) == 1: axes = [axes]
        for ax, key in zip(axes, metrics):
            raw = np.array(estimates[key])
            data = raw * 1e6 if key == 'volume' else raw * 100
            unit = "mL" if key == 'volume' else "cm"
            mean, std = np.mean(data), np.std(data)
            ax.hist(data, bins=20, alpha=0.6, color='skyblue', edgecolor='black')
            ax.axvline(mean, color='red', linestyle='--')
            ax.set_title(f"Distribution: {key.upper()} (Mean: {mean:.2f} {unit})")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def estimate_object_mass_and_volume(frames_data, label, filled=False, camera_config=None):
    """ROS-friendly entry point for mass and volume estimation."""
    cfg = camera_config if camera_config else CAMERA_CONFIG
    camera_slices, camera_metrics = {}, {}

    for side in ['left', 'right']:
        if side not in frames_data: continue
        items = frames_data[side]
        if isinstance(items, tuple): items = [items]
        
        calc = ObjectDimensionCalculator(cfg[side]['intrinsics'])
        T_cam_to_base = get_camera_to_base(side) if side in cfg else np.eye(4)
        estimates = {'height': [], 'top_w': [], 'bot_w': [], 'volume': []}
        depth_stack, mask_stack = [], []
        
        for d_m, m_raw in items:
            h = calc.get_height(d_m, m_raw, T_cam_to_base=T_cam_to_base)
            p = calc.get_width_bb(d_m, m_raw)
            v = calc.get_cylindrical_volume(d_m, m_raw, height_limit=h)
            if h: estimates['height'].append(h)
            if p:
                if p.get('top_m'): estimates['top_w'].append(p['top_m'])
                if p.get('bot_m'): estimates['bot_w'].append(p['bot_m'])
            if v: estimates['volume'].append(v[0])
            depth_stack.append(d_m); mask_stack.append(m_raw)

        if not depth_stack: continue
        consolidated = calc.consolidate_all(estimates)
        camera_metrics[side] = consolidated

        depth_arr = np.stack(depth_stack, axis=0)
        depth_arr[depth_arr == 0] = np.nan
        with np.errstate(all='ignore'):
            avg_d = np.nan_to_num(np.nanmean(depth_arr, axis=0))
        mask_arr = np.stack(mask_stack, axis=0)
        avg_m = (np.sum(mask_arr > 0, axis=0) > (len(depth_stack)/2)).astype(np.uint8)*255
        
        v_res = calc.get_cylindrical_volume(avg_d, avg_m, height_limit=consolidated.get('height'))
        if v_res: camera_slices[side] = v_res[1]

    if 'left' in camera_slices and 'right' in camera_slices:
        fusion_calc = ObjectDimensionCalculator(cfg['left']['intrinsics'])
        f_vol, _ = fusion_calc.calculate_consensus_volume(camera_slices['left'], camera_slices['right'])
        avg_h = (camera_metrics['left']['height'] + camera_metrics['right']['height']) / 2.0
        avg_t = (camera_metrics['left']['top_w'] + camera_metrics['right']['top_w']) / 2.0
        avg_b = (camera_metrics['left']['bot_w'] + camera_metrics['right']['bot_w']) / 2.0
        r1, r2 = avg_t/2.0, avg_b/2.0
        v_frust = (1.0/3.0) * np.pi * avg_h * (r1**2 + r2**2 + r1*r2)
        
        vol_ml = ((v_frust + f_vol) / 2) * 1e6
        m_e, m_f = fusion_calc.estimate_mass(vol_ml, label, filled=filled)
        return {'volume_ml': vol_ml, 'mass_empty': m_e, 'mass_filled': m_f, 
                'height_cm': avg_h*100, 'top_width_cm': avg_t*100, 'label': label, 'is_filled': filled}
    return None

if __name__ == '__main__':
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAMPLE_DIR = os.path.join(BASE_DIR, 'depth_fusion', 'depth_fusion_red')
    DEPTH_SCALE, CURRENT_LABEL, IS_FILLED = 1000.0, 'cup', True
    frames_data = {'left': [], 'right': []}
    for side in ['left', 'right']:
        for i in range(4):
            dp, mp = os.path.join(SAMPLE_DIR, f'depth_{side}_{i}.png'), os.path.join(SAMPLE_DIR, f'mask_{side}_{CURRENT_LABEL}_0_0.png')
            if os.path.exists(dp) and os.path.exists(mp):
                frames_data[side].append((cv2.imread(dp, -1).astype(np.float32)/DEPTH_SCALE, cv2.imread(mp, 0)))

    results = estimate_object_mass_and_volume(frames_data, CURRENT_LABEL, filled=IS_FILLED)
    if results:
        print(f"\nRESULTS: {results['label'].upper()}\nVolume: {results['volume_ml']:.1f} mL\nEmpty Mass: {results['mass_empty']:.1f}g\nFilled Mass: {results['mass_filled']:.1f}g")
    else: print("Estimation failed.")
