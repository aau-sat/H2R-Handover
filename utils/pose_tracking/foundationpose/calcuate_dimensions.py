import numpy as np
import cv2
import open3d as o3d

# --- 1. HARDCODED INTRINSICS (From previous data) ---
COLOR_INTRINSICS = {
    'fx': 606.3970947265625, 'fy': 606.363037109375,
    'cx': 315.5717468261719, 'cy': 251.08265686035156,
    'width': 640, 'height': 480
}

DEPTH_INTRINSICS = {
    'fx': 386.0667419433594, 'fy': 386.0667419433594,
    'cx': 323.46771240234375, 'cy': 239.58531188964844,
    'width': 640, 'height': 480
}

# --- 2. TF DATA (Hardcoded from ROS topic) ---
# Format: [x, y, z] for translation, [x, y, z, w] for rotation

# Transform 1: camera_depth_optical_frame (Child) -> camera_depth_frame (Parent)
TF_DEPTH_OPT_TO_DEPTH = {
    't': [0.0, -0.0, -0.0],
    'q': [-0.5, 0.4999999999999999, -0.5, 0.5000000000000001]
}

# Transform 2: camera_depth_frame (Child) -> camera_link (Parent)
TF_DEPTH_TO_LINK = {
    't': [0.0, -0.0, -0.0],
    'q': [0.0, 0.0, 0.0, 1.0]
}

# Transform 3: camera_color_frame (Child) -> camera_link (Parent)
TF_COLOR_TO_LINK = {
    't': [-0.0003830830391962081, 0.014807513915002346, 0.00027857901295647025],
    'q': [0.007839435711503029, -0.0002547972835600376, 0.00028202691464684904, 0.9999691843986511]
}

# Transform 4: camera_color_optical_frame (Child) -> camera_color_frame (Parent)
TF_COLOR_OPT_TO_COLOR = {
    't': [0.0, -0.0, -0.0],
    'q': [-0.5, 0.4999999999999999, -0.5, 0.5000000000000001]
}


class ObjectDimensionCalculator:
    def __init__(self):
        # Configuration matches your provided RealSense data
        self.color_intr = COLOR_INTRINSICS
        self.depth_intr = DEPTH_INTRINSICS
        self.T_depth_to_color = self._get_full_transform_matrix()

    def _quat_to_mat(self, q, t):
        x, y, z, w = q
        tx, ty, tz = t
        R = np.array([
            [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
            [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
            [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
        ])
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = [tx, ty, tz]
        return mat

    def _get_full_transform_matrix(self):
        M_dopt_d = self._quat_to_mat(TF_DEPTH_OPT_TO_DEPTH['q'], TF_DEPTH_OPT_TO_DEPTH['t'])
        M_d_link = self._quat_to_mat(TF_DEPTH_TO_LINK['q'], TF_DEPTH_TO_LINK['t'])
        M_c_link = self._quat_to_mat(TF_COLOR_TO_LINK['q'], TF_COLOR_TO_LINK['t'])
        M_copt_c = self._quat_to_mat(TF_COLOR_OPT_TO_COLOR['q'], TF_COLOR_OPT_TO_COLOR['t'])
        
        T_dopt_to_link = M_d_link @ M_dopt_d
        T_link_to_copt = np.linalg.inv(M_copt_c) @ np.linalg.inv(M_c_link)
        return T_link_to_copt @ T_dopt_to_link

    def get_dimensions(self, depth_meters, cv_mask):
        """Returns (width, height, depth) in meters."""
        # 1. Deproject Depth to 3D
        
        h, w = depth_meters.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_meters
        x = (u - self.depth_intr['cx']) * z / self.depth_intr['fx']
        y = (v - self.depth_intr['cy']) * z / self.depth_intr['fy']
        
        points_depth_opt = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        valid_mask = points_depth_opt[:, 2] > 0.1 # Ignore points closer than 10cm or noise
        points_depth_opt = points_depth_opt[valid_mask]

        if len(points_depth_opt) == 0:
            return None

        # 2. Transform to Color Optical Frame
        points_hom = np.hstack((points_depth_opt, np.ones((len(points_depth_opt), 1))))
        points_color_opt = (self.T_depth_to_color @ points_hom.T)[:3, :].T
        
        # 3. Project back to 2D to sample the Mask
        z_c = points_color_opt[:, 2]
        valid_z = z_c > 0.01
        
        u_proj = (points_color_opt[valid_z, 0] / z_c[valid_z]) * self.color_intr['fx'] + self.color_intr['cx']
        v_proj = (points_color_opt[valid_z, 1] / z_c[valid_z]) * self.color_intr['fy'] + self.color_intr['cy']
        
        u_proj = np.rint(u_proj).astype(int)
        v_proj = np.rint(v_proj).astype(int)
        
        # 4. Filter by Mask
        cv_mask.astype(np.bool_)
        h_c, w_c = cv_mask.shape
        in_bounds = (u_proj >= 0) & (u_proj < w_c) & (v_proj >= 0) & (v_proj < h_c)
        
        final_valid_indices = np.where(valid_z)[0][in_bounds]
        mask_vals = cv_mask[v_proj[in_bounds], u_proj[in_bounds]]
        object_mask = mask_vals > 0 
        
        final_points = points_color_opt[final_valid_indices[object_mask]]
        
        if len(final_points) < 50: # Ensure we have a meaningful cloud
            return None

        # 5. Outlier Removal and Bounding Box
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_points)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_clean = pcd.select_by_index(ind)
        
        aabb = pcd_clean.get_axis_aligned_bounding_box()
        dims = aabb.get_max_bound() - aabb.get_min_bound()
        
        return dims # [Width, Height, Depth] in meters