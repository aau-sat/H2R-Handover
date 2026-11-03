import os
import numpy as np
import csv

from config_motion import config_hand_motion

def scene_id_to_hierarchical_dir(scene_id: int) -> str:
    scene_id_str = f"{scene_id:08d}" # "12345678"
    hierarchical_dir = os.path.join(*(scene_id_str[i:i+2] for i in range(0, len(scene_id_str)-2, 2))) # "12/34/56/"
    return hierarchical_dir

def scene_id_to_scene_dir(scene_root_dir: str, scene_id: int) -> str:
    scene_path = os.path.join(scene_root_dir, scene_id_to_hierarchical_dir(scene_id))
    return scene_path

def remap_objects(object_names: list):
    # remove first 4 characters (no ids)
    remapped_file_names = [os.path.join(name[4:], name[4:] + ".usd") for name in object_names]
    return remapped_file_names

def get_moving_object_index(object_poses: np.ndarray, threshold: float=1e-4) -> int:
    num_objects = object_poses.shape[0]
    for obj_index in range(num_objects):
        poses = object_poses[obj_index]  # (T, 6)
        diffs = np.linalg.norm(poses[1:] - poses[:-1], axis=1)  # (T-1,)
        total_movement = np.sum(diffs)
        if total_movement > threshold:
            return obj_index
    raise ValueError("No moving object found in the provided object poses.")

def load_scene_data(scene_root_dir: str, scene_id: int, frame_interval: int=1):
    scene_dir = scene_id_to_scene_dir(scene_root_dir, scene_id)
    scene_data_path = os.path.join(scene_dir, f"{scene_id:08d}.npz")
    scene_data = dict(np.load(scene_data_path)) # "hand_name", "hand_side", "hand_path", "hand_pose", "object_names", "object_paths", "object_grasp_id", "object_poses", "endpoints", "source"
    source = scene_data["source"]
    scene_data["hand_side"] = scene_data["hand_side"].item()

    if source == "dexycb":
        scene_data["hand_pose"] = scene_data["hand_pose"][::frame_interval] # (T, 51)
    elif source == "genh2r":
        hand_pose = scene_data["hand_pose"][::frame_interval]
        scene_data["hand_pose"] = np.concatenate([hand_pose, np.tile(scene_data["hand_theta"], (hand_pose.shape[0], 1))], axis=1)
    else:
        raise NotImplementedError
    scene_data["object_names"] = scene_data["object_names"].tolist()

    return scene_data

def export_scene_data(scene_root_dir: str, objects_dir: str, scene_id_start: int, number_of_scenes: int=1000):
    csv_file_path = os.path.join(scene_root_dir, "scenes_data.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ['scene_id', 'scene_npz_path', 'hand_side', 'object_names', 'moving_object_name']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for scene_id in range(scene_id_start, number_of_scenes):
            scene_data = load_scene_data(scene_root_dir, scene_id)
            moving_object_index = get_moving_object_index(scene_data["object_poses"])
            moving_object_name = scene_data["object_names"][moving_object_index]
            scene_npz_path = os.path.join(scene_id_to_scene_dir(scene_root_dir, scene_id), f"{scene_id:08d}.npz")
            writer.writerow({
                'scene_id': scene_id,
                'scene_npz_path': scene_npz_path,
                'object_names': ";".join(scene_data["object_names"]),
                'moving_object_name': moving_object_name,
                'hand_side': scene_data["hand_side"],
            })

if __name__ == "__main__":
    export_scene_data(config_hand_motion.SCENE_DIR, config_hand_motion.OBJ_USD_DIR, 0)
