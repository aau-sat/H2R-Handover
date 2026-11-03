import socket
import os
import torch
import numpy as np
import trimesh
import argparse
import time
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

def voxel_downsample(points, voxel_size=0.005):
    """ 
    Reduces point cloud density for significantly faster collision checking.
    """
    if len(points) == 0: return points
    # Round coordinates to the nearest voxel grid cell
    coords = (points / voxel_size).astype(int)
    # Keep only one point per voxel cell
    _, indices = np.unique(coords, axis=0, return_index=True)
    return points[indices]

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
    
    phi_deg = np.degrees(np.arctan2(vn[:, 0], vn[:, 2])) % 360
    theta_deg = np.degrees(np.arcsin(np.clip(-vn[:, 1], -1.0, 1.0)))
    
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
        cand_scores = scores[mask]

        if len(candidates) > 0:
            if expansion == 0.0:
                # Strict constraints already gave us grasps — return immediately,
                # no need for an extra iteration.
                return candidates, cand_scores, False
            if found_on_prev:
                # This is the bonus "one last step" after softening — return.
                total_expansion = expansion - phi_step  # expansion at first success
                print(
                    f"[AdaptivePhi] Constraints softened by ±{total_expansion:.1f}° on phi. "
                    f"Returning {len(candidates)} candidate(s) from final window "
                    f"[{a_min_mod:.1f}°, {a_max_mod:.1f}°]."
                )
                return candidates, cand_scores, True
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
                return grasp_poses[prev_mask], scores[prev_mask], softened

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
            return grasp_poses[mask_theta_only], scores[mask_theta_only], True

def filter_by_finger_orientation(grasp_poses, max_tilt_deg=20):
    """ Ensures fingers are horizontal (left-right) relative to camera Y. """
    finger_axis = grasp_poses[:, :3, 0] # X-axis of gripper
    mags = np.linalg.norm(finger_axis, axis=1, keepdims=True)
    finger_axis_n = finger_axis / (mags + 1e-8)
    
    y_component = np.abs(finger_axis_n[:, 1])
    threshold = np.sin(np.radians(max_tilt_deg))
    return y_component < threshold

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
                    gripper_name="default_gripper"):
    # Visualize scene point cloud - use RGB colors if available, otherwise use gray
    if scene_pc_colors is not None:
        visualize_pointcloud(
            viz, "scene_pc", scene_pc, scene_pc_colors, size=0.002
        )
    else:
        visualize_pointcloud(
            viz, "scene_pc", scene_pc, [128, 128, 128], size=0.002
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
        if top_grasp is not None:
            # Highlight top grasp
            visualize_grasp(
                viz,
                "top_grasp",
                top_grasp,
                color=[255, 0, 0],
                gripper_name=gripper_name,
                linewidth=2.0,
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

    # Initialize visualization if requested
    if args.visualize:
        vis = create_visualizer()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.settimeout(1.0) 
    if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)
    server.bind(SOCKET_PATH)
    server.listen(1)

    print(f"GraspGen Server (Optimized) | Voxel: {args.voxel_size}m | Margin: {args.collision_margin}m")

    try:
        while True:
            try:
                conn, _ = server.accept()
            except socket.timeout: continue

            with conn:
                start_time = time.time()
                
                # Receive scene and object point clouds
                header = recv_all(conn, 8)  # 2 int32 values: scene_size, object_size
                if not header: continue
                scene_size, object_size = np.frombuffer(header, dtype=np.int32)
                
                # Receive scene point cloud (scene_size x 3 float32)
                scene_data = recv_all(conn, scene_size * 3 * 4)
                if not scene_data: continue
                scene_pc = np.frombuffer(scene_data, dtype=np.float32).reshape((scene_size, 3))
                
                # Receive object point cloud (object_size x 3 float32)
                object_data = recv_all(conn, object_size * 3 * 4)
                if not object_data: continue
                object_pc = np.frombuffer(object_data, dtype=np.float32).reshape((object_size, 3))

                # Filter object points
                pc_filtered, _ = point_cloud_outlier_removal(torch.from_numpy(object_pc))
                pc_filtered_np = pc_filtered.numpy()

                # Speed up collision check: Combine and downsample environment
                env_pc = np.vstack([scene_pc, pc_filtered_np])
                env_pc_downsampled = voxel_downsample(env_pc, voxel_size=args.voxel_size)

                start_inference_time = time.time()
                # Run Inference
                grasps, conf = GraspGenSampler.run_inference(
                    pc_filtered_np, grasp_sampler, grasp_threshold=0.7, num_grasps=args.num_grasps
                )
                print('Inference completed in {:.2f}s.'.format(time.time() - start_inference_time))
                grasps_np, scores_np = grasps.cpu().numpy(), conf.cpu().numpy()
                # Ensure grasps have correct homogeneous coordinate
                grasps_np[:, 3, 3] = 1 # Nad added
                
                # centering
                # Process point clouds and grasps for consistent coordinate frame
                pc_filtered_np, grasps_np, T_center = (
                    process_point_cloud(
                        pc_filtered_np, grasps_np
                    )
                )
                scene_pc = tra.transform_points(scene_pc, T_center) # for visualization
                env_pc_downsampled = tra.transform_points(env_pc_downsampled, T_center) # for collision checking
                
                # crop environment pointcloud to a reasonable radius around the object to speed up collision checking
                env_pc_downsampled = env_pc_downsampled[np.linalg.norm(env_pc_downsampled, axis=1) < 0.5] # 0.5m radius cutoff

                # Parse ranges (args already give us floats via nargs=2)
                phi_range = tuple(args.phi_range)
                theta_range = tuple(args.theta_range)
                
                start_filter_time = time.time()
                # Filter 1: Spherical Constraints (adaptive phi softening)
                grasps_np, scores_np, phi_was_softened = apply_spherical_filter_adaptive_phi(
                    grasps_np, scores_np,
                    azimuth_range=phi_range,
                    elevation_range=theta_range,
                    phi_step=2.0,
                )
                if not phi_was_softened:
                    print(f"[Filter] Spherical: {len(grasps_np)} grasps kept with strict phi range.")
                else:
                    print(f"[Filter] Spherical (phi softened): {len(grasps_np)} grasps kept.")

                # Filter 2: Finger orientation
                grasps_oriented = []
                if len(grasps_np) > 0:
                    f_mask = filter_by_finger_orientation(grasps_np, max_tilt_deg=15)
                    grasps_oriented, scores_oriented = grasps_np[f_mask], scores_np[f_mask]
                    print('Completed intermediate filtering in {:.2f}s.'.format(time.time() - start_filter_time))

                
                # Filter 3: Collision Check (Optimized with downsampled PC)
                if len(grasps_oriented) > 0:
                    start_collision_filter_time = time.time()
                    c_mask = filter_colliding_grasps(
                        scene_pc=env_pc_downsampled,
                        grasp_poses=grasps_oriented,
                        gripper_collision_mesh=gripper_info.collision_mesh
                    )
                    print('Filtered collisions in {:.2f}s'.format(time.time() - start_collision_filter_time))
                    valid_grasps, valid_scores = grasps_oriented[c_mask], scores_oriented[c_mask]

                    if len(valid_grasps) > 0:
                        top_idx = np.argsort(valid_scores)[::-1]
                        top_poses = valid_grasps[top_idx]
                        
                        elapsed = time.time() - start_time
                        print(f"Success: {len(valid_grasps)} grasps found in {elapsed:.2f}s. Top Score: {valid_scores[top_idx[0]]:.3f}")
                        if args.visualize:
                            save_for_3d_viewer(
                                scene_pc=scene_pc, 
                                object_pc=pc_filtered_np, 
                                best_poses=top_poses, 
                                gripper_mesh=gripper_info.collision_mesh
                            )
                            grasp_colors = get_color_from_score(valid_scores, use_255_scale=True)
                            visualize_grasps(
                                viz=vis,
                                scene_pc=scene_pc,
                                scene_pc_colors=None,
                                object_pc=pc_filtered_np,
                                object_pc_colors=None,
                                grasps=valid_grasps,
                                top_grasp=top_poses[0],
                                grasps_colors=grasp_colors,
                                gripper_mesh=gripper_info.visual_mesh,
                                gripper_name=grasp_cfg.data.gripper_name
                            )
                        top_poses[0] = tra.inverse_matrix(T_center) @ top_poses[0] # Transform back to original frame
                        conn.sendall(top_poses[0].astype(np.float32).tobytes())
                        continue
              
                # Return empty if nothing found
                print("No valid grasps found.")
                conn.sendall(np.zeros(16, dtype=np.float32).tobytes())
                    
                if args.visualize:
                    grasp_colors = get_color_from_score(scores_np, use_255_scale=True)
                    visualize_grasps(
                        viz=vis,
                        scene_pc=scene_pc,
                        scene_pc_colors=None,
                        object_pc=pc_filtered_np,
                        object_pc_colors=None,
                        grasps=grasps_np,
                        top_grasp=None,
                        grasps_colors=grasp_colors,
                        gripper_mesh=gripper_info.visual_mesh,
                        gripper_name=grasp_cfg.data.gripper_name
                    )

    except KeyboardInterrupt:
        print("\nServer shutting down.")
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gripper_config", type=str, required=True)
    parser.add_argument("--phi_range", type=float, nargs=2, default=[0, 180],
                        metavar=("PHI_MIN", "PHI_MAX"),
                        help="Azimuth range in degrees, e.g. --phi_range 0 180")
    parser.add_argument("--theta_range", type=float, nargs=2, default=[-30, 0],
                        metavar=("THETA_MIN", "THETA_MAX"),
                        help="Elevation range in degrees, e.g. --theta_range -30 0")
    parser.add_argument("--num_grasps", type=int, default=400)
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size (m) for collision speed")
    parser.add_argument("--collision_margin", type=float, default=0.002, help="Padding around fingers")
    parser.add_argument("--visualize", action="store_true")
    
    args = parser.parse_args()
    run_server(args)