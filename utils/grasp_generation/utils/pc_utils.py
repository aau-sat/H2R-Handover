
import numpy as np
import torch
from typing import Tuple, Dict

# Methods taken from GraspGen

def depth2points(
    depth: np.array,
    fx: int,
    fy: int,
    cx: int,
    cy: int,
    xmap: np.array = None,
    ymap: np.array = None,
    rgb: np.array = None,
    seg: np.array = None,
    mask: np.arange = None,
) -> Dict:
    """Compute point cloud from a depth image.

    Args:
        depth (np.array): 2D depth image.
        fx (int): Focal length x.
        fy (int): Focal length y.
        cx (int): Camera principal point x.
        cy (int): Camera principal point y.
        xmap (np.array, optional): precomputed meshgrid map X. Defaults to None.
        ymap (np.array, optional): precomputed meshgrid map Y. Defaults to None.
        rgb (np.array, optional): RGB imgae for colored point cloud. Defaults to None.
        seg (np.array, optional): Segmentation image corresponding to the rgb image Defaults to None.
        mask (np.arange, optional): Binary mask. 1 to keep. Defaults to None.

    Returns:
        Dict: Output is a dictionary with fields `xyz`, `rgb`, and `index`,
        where `xyz` and `rgb` are Nx3 numpy arrays and `index` is a Nx1 array
        to indicate which point is valid.
    """
    """convert depth image to point cloud"""
    if rgb is not None:
        assert rgb.shape[0] == depth.shape[0] and rgb.shape[1] == depth.shape[1]
    if xmap is not None:
        assert xmap.shape[0] == depth.shape[0] and xmap.shape[1] == depth.shape[1]
    if ymap is not None:
        assert ymap.shape[0] == depth.shape[0] and ymap.shape[1] == depth.shape[1]

    im_height, im_width = depth.shape[0], depth.shape[1]

    if xmap is None or ymap is None:
        ww = np.linspace(0, im_width - 1, im_width)
        hh = np.linspace(0, im_height - 1, im_height)
        xmap, ymap = np.meshgrid(ww, hh)

    pt2 = depth
    pt0 = (xmap - cx) * pt2 / fx
    pt1 = (ymap - cy) * pt2 / fy

    mask_depth = np.ma.getmaskarray(np.ma.masked_greater(pt2, 0))
    if mask is None:
        mask = mask_depth
    else:
        mask_semantic = np.ma.getmaskarray(np.ma.masked_equal(mask, 1))
        mask = mask_depth * mask_semantic

    index = mask.flatten().nonzero()[0]

    pt2_valid = pt2.flatten()[:, np.newaxis].astype(np.float32)
    pt0_valid = pt0.flatten()[:, np.newaxis].astype(np.float32)
    pt1_valid = pt1.flatten()[:, np.newaxis].astype(np.float32)
    pc_xyz = np.concatenate((pt0_valid, pt1_valid, pt2_valid), axis=1)
    if rgb is not None:
        r = rgb[:, :, 0].flatten()[:, np.newaxis]
        g = rgb[:, :, 1].flatten()[:, np.newaxis]
        b = rgb[:, :, 2].flatten()[:, np.newaxis]
        pc_rgb = np.concatenate((r, g, b), axis=1)
    else:
        pc_rgb = None

    if seg is not None:
        pc_seg = seg.flatten()[:, np.newaxis]
    else:
        pc_seg = None

    return {"xyz": pc_xyz, "rgb": pc_rgb, "seg": pc_seg, "index": index}

# Taken from GraspGen
def depth_and_segmentation_to_point_clouds_old(
    depth_image: np.ndarray,
    segmentation_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_image: np.ndarray = None,
    target_object_id: int = 1,
    remove_object_from_scene: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image and instance segmentation mask to scene and object point clouds.

    Args:
        depth_image: HxW depth image in meters
        segmentation_mask: HxW instance segmentation mask with integer labels
        fx, fy, cx, cy: Camera intrinsic parameters
        rgb_image: HxWx3 RGB image (optional, for colored point clouds)
        target_object_id: ID of the target object in the segmentation mask
        remove_object_from_scene: If True, removes object points from scene point cloud

    Returns:
        scene_pc: Nx3 point cloud of the entire scene (excluding object if remove_object_from_scene=True)
        object_pc: Mx3 point cloud of the target object only

    Raises:
        ValueError: If no target object found or multiple objects detected
    """
    # Check that segmentation mask contains the target object
    unique_ids = np.unique(segmentation_mask)
    if target_object_id not in unique_ids:
        raise ValueError(
            f"Target object ID {target_object_id} not found in segmentation mask. Available IDs: {unique_ids}"
        )

    # Check that only background (0) and one object (target_object_id) are present
    non_background_ids = unique_ids[unique_ids != 0]
    if len(non_background_ids) > 1:
        raise ValueError(
            f"Multiple objects detected in segmentation mask: {non_background_ids}. Please ensure only one object is present."
        )

    # Convert depth image to point cloud
    pts_data = depth2points(
        depth=depth_image,
        fx=int(fx),
        fy=int(fy),
        cx=int(cx),
        cy=int(cy),
        rgb=rgb_image,
        seg=segmentation_mask,
    )

    xyz = pts_data["xyz"]
    rgb = pts_data["rgb"]
    seg = pts_data["seg"]
    index = pts_data["index"]

    # Filter valid points (non-zero depth)
    xyz_valid = xyz[index]
    seg_valid = seg[index] if seg is not None else None
    rgb_valid = rgb[index] if rgb is not None else None

    # Scene point cloud (all valid points)
    scene_pc = xyz_valid

    # Object point cloud (only target object points)
    if seg_valid is not None:
        object_mask = seg_valid.flatten() == target_object_id
        object_pc = xyz_valid[object_mask]

        # Scene point cloud (optionally excluding object points)
        if remove_object_from_scene:
            scene_mask = ~object_mask  # Invert object mask to get scene-only points
            scene_pc = xyz_valid[scene_mask]
            print(
                f"Removed {np.sum(object_mask)} object points from scene point cloud"
            )
    else:
        raise ValueError("Segmentation data not available from depth2points")

    if len(object_pc) == 0:
        raise ValueError(f"No points found for target object ID {target_object_id}")

    print(f"Scene point cloud: {len(scene_pc)} points")
    print(f"Object point cloud: {len(object_pc)} points")

    return scene_pc, object_pc

def depth_and_segmentation_to_point_clouds(
    depth_image: np.ndarray,
    segmentation_mask_obj: np.ndarray,
    segmentation_mask_hand: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_image: np.ndarray = None,
    target_object_id: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image and instance segmentation mask to scene and object point clouds.

    Args:
        depth_image: HxW depth image in meters
        segmentation_mask_obj: HxW instance segmentation mask for the target object with integer labels
        segmentation_mask_hand: HxW instance segmentation mask for the hand with integer labels
        fx, fy, cx, cy: Camera intrinsic parameters
        rgb_image: HxWx3 RGB image (optional, for colored point clouds)
        target_object_id: ID of the target object in the segmentation mask
        remove_object_from_scene: If True, removes object points from scene point cloud

    Returns:
        scene_pc: Nx3 point cloud of the hand in the scene (excluding object if remove_object_from_scene=True)
        object_pc: Mx3 point cloud of the target object only

    Raises:
        ValueError: If no target object found or multiple objects detected
    """
    # Check that segmentation mask contains the target object
    unique_ids_obj = np.unique(segmentation_mask_obj)
    unique_ids_hand = np.unique(segmentation_mask_hand)
    if target_object_id not in unique_ids_obj:
        raise ValueError(
            f"Target object ID {target_object_id} not found in segmentation mask. Available IDs: {unique_ids_obj}"
        )
    if target_object_id not in unique_ids_hand:
        raise ValueError(
            f"Target object ID {target_object_id} not found in hand segmentation mask. Available IDs: {unique_ids_hand}"
        )
    
    # Check that only background (0) and one object (target_object_id) are present
    non_background_ids_obj = unique_ids_obj[unique_ids_obj != 0]
    non_background_ids_hand = unique_ids_hand[unique_ids_hand != 0]
    if len(non_background_ids_obj) > 1:
        raise ValueError(
            f"Multiple objects detected in segmentation mask: {non_background_ids_obj}. Please ensure only one object is present."
        )
    if len(non_background_ids_hand) > 1:
        raise ValueError(
            f"Multiple objects detected in hand segmentation mask: {non_background_ids_hand}. Please ensure only one object is present."
        )

    # Convert depth image to point cloud
    pts_data_obj = depth2points(
        depth=depth_image,
        fx=int(fx),
        fy=int(fy),
        cx=int(cx),
        cy=int(cy),
        rgb=rgb_image,
        seg=segmentation_mask_obj,
    )
    pts_data_hand = depth2points(
        depth=depth_image,
        fx=int(fx),
        fy=int(fy),
        cx=int(cx),
        cy=int(cy),
        rgb=rgb_image,
        seg=segmentation_mask_hand,
    )

    # Object point cloud (all valid points)
    xyz = pts_data_obj["xyz"]
    seg = pts_data_obj["seg"]
    index = pts_data_obj["index"]

    # Filter valid points (non-zero depth)
    xyz_valid_obj = xyz[index]
    seg_valid_obj = seg[index] if seg is not None else None
    
    # Hand point cloud (all valid points)
    xyz = pts_data_hand["xyz"]
    seg = pts_data_hand["seg"]
    index = pts_data_hand["index"]
    xyz_valid_hand = xyz[index]
    seg_valid_hand = seg[index] if seg is not None else None

    # Object point cloud (only target object points)
    if seg_valid_obj is not None:
        object_mask = seg_valid_obj.flatten() == target_object_id
        object_pc = xyz_valid_obj[object_mask]
    else:
        raise ValueError("Segmentation data not available from depth2points")
    
    if seg_valid_hand is not None:
        hand_mask = seg_valid_hand.flatten() == target_object_id
        hand_pc = xyz_valid_hand[hand_mask]
    else:
        raise ValueError("Hand segmentation data not available from depth2points")

    if len(object_pc) == 0:
        raise ValueError(f"No points found for target object ID {target_object_id}")
    if len(hand_pc) == 0:
        raise ValueError(f"No points found for target object ID {target_object_id} in hand segmentation")

    print(f"Object point cloud: {len(object_pc)} points")
    print(f"Hand point cloud: {len(hand_pc)} points")
    
    # # export pointclouds for debugging
    # o3d_obj_pc = o3d.geometry.PointCloud()
    # o3d_obj_pc.points = o3d.utility.Vector3dVector(object_pc)
    # o3d.io.write_point_cloud("debug_object_pc.ply", o3d_obj_pc)
    # o3d_hand_pc = o3d.geometry.PointCloud()
    # o3d_hand_pc.points = o3d.utility.Vector3dVector(hand_pc)
    # o3d.io.write_point_cloud("debug_hand_pc.ply", o3d_hand_pc)
    # # merged point cloud
    # merged_pc = o3d.geometry.PointCloud()
    # merged_pc.points = o3d.utility.Vector3dVector(np.concatenate([object_pc, hand_pc], axis=0))
    # o3d.io.write_point_cloud("debug_merged_pc.ply", merged_pc)

    return hand_pc, object_pc

# Taken from GraspGen
def knn_points(X: torch.Tensor, K: int, norm: int):
    """
    Computes the K-nearest neighbors for each point in the point cloud X.

    Args:
        X: (N, 3) tensor representing the point cloud.
        K: Number of nearest neighbors.

    Returns:
        dists: (N, K) tensor containing squared Euclidean distances to the K nearest neighbors.
        idxs: (N, K) tensor containing indices of the K nearest neighbors.
    """
    N, _ = X.shape

    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(X, X, p=norm)  # (N, N)

    # Ignore self-distance (optional, but avoids trivial zero distance)
    self_mask = torch.eye(N, device=X.device, dtype=torch.bool)
    dist_matrix.masked_fill_(self_mask, float("inf"))  # Set self-distances to inf

    # Get the indices of the K-nearest neighbors
    dists, idxs = torch.topk(dist_matrix, K, dim=1, largest=False)

    return dists, idxs

# Taken from GraspGen
def point_cloud_outlier_removal(
    obj_pc: torch.Tensor, threshold: float = 0.009, K: int = 30
) -> torch.Tensor:
    """
    Remove outliers from a point cloud. K-nearest neighbors is used to compute the distance to the nearest neighbor for each point.
    If the distance is greater than a threshold, the point is considered an outlier and removed.

    RANSAC can also be used.

    Args:
        obj_pc (torch.Tensor or np.ndarray): (N, 3) tensor or array representing the point cloud.
        threshold (float): Distance threshold for outlier detection. Points with mean distance to K nearest neighbors greater than this threshold are removed.
        K (int): Number of nearest neighbors to consider for outlier detection.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing filtered and removed point clouds.
    """
    # Convert numpy array to torch tensor if needed
    if isinstance(obj_pc, np.ndarray):
        obj_pc = torch.from_numpy(obj_pc)

    obj_pc = obj_pc.float()
    obj_pc = obj_pc.unsqueeze(0)

    nn_dists, _ = knn_points(obj_pc[0], K=K, norm=1)

    mask = nn_dists.mean(1) < threshold
    filtered_pc = obj_pc[0, mask]
    filtered_pc = filtered_pc.view(-1, 3)

    return filtered_pc
