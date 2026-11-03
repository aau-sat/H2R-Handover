import argparse
import sys

import os
import torch
import numpy as np
import re
# from scipy.spatial.transform import Rotation as R
from typing import Optional

from config_motion import config_hand_motion

import threading

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.experience = config_hand_motion.ISAAC_EXPERIENCE

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg

import carb
import omni
from isaacsim.core.utils import extensions
from isaacsim.storage.native import is_file
from pxr import UsdPhysics
from scipy.spatial.transform import Rotation as R


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""

    # table_usd_path = os.path.join(config_hand_motion.STAGE_DIR, 'table/table.usd')
    # cfg_table = sim_utils.UsdFileCfg(usd_path=table_usd_path)
    # quat_from_euler = R.from_euler('xyz', (90, 0, 0), degrees=True).as_quat()
    # cfg_table.func("/World/Table", cfg_table, translation=(0.8, 0.0, 0.0), orientation=quat_from_euler)

def print_actor_properties(actor: Articulation):
    """Prints various properties of the given actor."""
    print("Number of Joints:", actor.num_joints)
    print("Joint Names:", actor.joint_names)
    print("Default Joint Positions:", actor.data.default_joint_pos)
    print("Default Joint Velocities:", actor.data.default_joint_vel)
    print("Default Root State:", actor.data.default_root_state)
    print("Current Joint Positions:", actor.data.joint_pos)
    print("Current Joint Velocities:", actor.data.joint_vel)
    print("Current Joint Position Targets:", actor.data.joint_pos_target)
    print("Current Joint Velocity Targets:", actor.data.joint_vel_target)
    print("Default Joint Pose Limits:", actor.data.default_joint_pos_limits)
    print("Current Joint Pose Limits:", actor.data.joint_pos_limits)
    print("Current Joint Velocity Limits:", actor.data.joint_vel_limits)
    print("Friction Coefficients:", actor.data.joint_friction_coeff)
    print("Default Friction Coefficients:", actor.data.default_joint_friction_coeff)

def print_joint_limits(actor: Articulation):
    for i, joint_name in enumerate(actor.joint_names):
        lower_limit, upper_limit = actor.data.joint_pos_limits[0, i]
        print(f"Joint: {joint_name}, Lower Limit: {lower_limit:.4f}, Upper Limit: {upper_limit:.4f}")

def print_joint_mapping(hand_pose_reordered: np.ndarray, joint_names: list, joint_names_sorted: list):
    print("Joint Mapping (Recorded Index -> Joint Name -> Isaac Index):")
    joint_names_dict = {name: idx for idx, name in enumerate(joint_names)}
    for recorded_index, joint_name in enumerate(joint_names_sorted):
        if joint_name in joint_names_dict:
            isaac_index = joint_names_dict[joint_name]
            print(f"Recorded Index: {recorded_index}, Joint Name: {joint_name}, Isaac Index: {isaac_index}")
        else:
            print(f"Recorded Index: {recorded_index}, Joint Name: {joint_name}, Isaac Index: Not Present")

def get_moving_object_index(object_poses: np.ndarray, threshold: float=1e-4) -> int:
    num_objects = object_poses.shape[0]
    for obj_index in range(num_objects):
        poses = object_poses[obj_index]  # (T, 6)
        diffs = np.linalg.norm(poses[1:] - poses[:-1], axis=1)  # (T-1,)
        total_movement = np.sum(diffs)
        if total_movement > threshold:
            return obj_index
    raise ValueError("No moving object found in the provided object poses.")

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

def load_scene_data(scene_root_dir: str, objects_dir: str, scene_id: int, table_height: float=0., stop_moving_frame: Optional[int]=None, frame_interval: int=1):
    scene_dir = scene_id_to_scene_dir(scene_root_dir, scene_id)
    scene_data_path = os.path.join(scene_dir, f"{scene_id:08d}.npz")
    scene_data = dict(np.load(scene_data_path)) # "hand_name", "hand_side", "hand_path", "hand_pose", "object_names", "object_paths", "object_grasp_id", "object_poses", "endpoints", "source"
    source = scene_data["source"]
    scene_data["hand_side"] = scene_data["hand_side"].item()
    # if scene_data["hand_side"] == "left":
    #     raise NotImplementedError("Left hand not supported yet.")
    
    if source == "dexycb":
        scene_data["hand_pose"] = scene_data["hand_pose"][::frame_interval] # (T, 51)
    elif source == "genh2r":
        hand_pose = scene_data["hand_pose"][::frame_interval]
        scene_data["hand_pose"] = np.concatenate([hand_pose, np.tile(scene_data["hand_theta"], (hand_pose.shape[0], 1))], axis=1)
    else:
        raise NotImplementedError
    scene_data["object_names"] = scene_data["object_names"].tolist()
    object_filenames = remap_objects(scene_data["object_names"])
    scene_data["object_paths"] = [os.path.join(objects_dir, object_path) for object_path in object_filenames]
    scene_data["object_poses"] = scene_data["object_poses"][:, ::frame_interval] # (#objects, T, 6)

    if stop_moving_frame is not None:
        scene_data["hand_pose"] = scene_data["hand_pose"][:stop_moving_frame]
        scene_data["object_poses"] = scene_data["object_poses"][:, :stop_moving_frame]
    if table_height != 0.:
        hand_nonzero_mask = np.any(scene_data["hand_pose"]!=0, axis=1)
        hand_nonzeros = np.where(hand_nonzero_mask)[0]
        hand_start_frame = hand_nonzeros[0]
        hand_end_frame = hand_nonzeros[-1]+1
        scene_data["hand_pose"][hand_start_frame:hand_end_frame, 2] += table_height
        scene_data["object_poses"][:, :, 2] += table_height
    
    if "endpoints" in scene_data:
        scene_data["endpoints"] //= frame_interval
    else:
        scene_data["endpoints"] = None

    return scene_data

def preprocess_trajectory(hand_poses, object_poses):
    # rotate wrist and objects by 90 degrees around x axis
    R_x_90 = R.from_euler('x', 90, degrees=True).as_matrix()  # (3, 3)
    num_frames = hand_poses.shape[0]
    for i in range(num_frames):
        # hand
        hand_rot = R.from_euler('xyz', hand_poses[i, 3:6], degrees=False).as_matrix()  # (3, 3)
        hand_rot = R_x_90 @ hand_rot
        hand_euler = R.from_matrix(hand_rot).as_euler('xyz', degrees=False)
        hand_poses[i, 3:6] = hand_euler
        # objects
        for j in range(object_poses.shape[0]):
            obj_rot = R.from_euler('xyz', object_poses[j, i, 3:6], degrees=False).as_matrix()  # (3, 3)
            obj_rot = R_x_90 @ obj_rot
            obj_euler = R.from_matrix(obj_rot).as_euler('xyz', degrees=False)
            object_poses[j, i, 3:6] = obj_euler
    return hand_poses, object_poses
 

robot_joint_pos_msg = list(config_hand_motion.robot_qpos.values())
robot_joint_vel_msg = [0.0] * len(config_hand_motion.robot_joint_names)
def main():
    simulation_app.update()

    # Initialize the simulation context (with IsaacLab, it's only 'torch' backend)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([-0.3, -3.6, 0.5], [-0.5, -1.0, 0.5])
    
    # Load stage
    stage_usd_path = config_hand_motion.ROBOT_STAGE_USD_FILE
    # get absolute path to the file, otherwise it will not work
    stage_usd_path = os.path.abspath(stage_usd_path)
    # make sure the file exists before we try to open it
    try:
        result = is_file(stage_usd_path)
    except:
        result = False

    if result:
        omni.usd.get_context().open_stage(stage_usd_path)
    else:
        carb.log_error(
            f"the usd path {stage_usd_path} could not be opened."
        )
        simulation_app.close()
        sys.exit()

    # Wait two frames so that stage starts loading
    simulation_app.update()
    simulation_app.update()
    print("Loading stage...")
    from isaacsim.core.utils.stage import is_stage_loading

    while is_stage_loading():
        simulation_app.update()
    
    # Design scene
    design_scene()
    
    # Load recorded motion data
    scene_data = load_scene_data(config_hand_motion.SCENE_DIR, config_hand_motion.OBJ_USD_DIR, config_hand_motion.SCENE_ID, table_height=1.0, stop_moving_frame=None, frame_interval=10)
    print('Objects:', scene_data['object_names'])

    # robot actor
    robot_translation = np.array([0.0, 0.0, 1.0])
    robot_actor = Articulation(ArticulationCfg(
        prim_path=config_hand_motion.ROBOT_PRIM_PATH,
        
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=config_hand_motion.robot_qpos,
            pos=robot_translation,
        ),
        actuators={
            
        }
    ))

    # joint names. joint7 is fixed
    hand_joint_names = config_hand_motion.hand_joint_names
    hand_qpos = {joint_name: 0.0 for joint_name in hand_joint_names}
    hand_usd_file = (scene_data["hand_side"] == "right") and config_hand_motion.HAND_RIGHT_USD_FILE or config_hand_motion.HAND_LEFT_USD_FILE
    hand_usd_path = os.path.join(config_hand_motion.HAND_USD_DIR, hand_usd_file + ('.usd' if not hand_usd_file.endswith('.usd') else ''))
    hand_mass_props = sim_utils.MassPropertiesCfg(mass=config_hand_motion.LEFT_MASS) if scene_data["hand_side"] == "left" else None
    hand_translation = np.array([0.7, 0.6, -0.077])
    transformhand_rotation = R.from_euler('xyz', (-90, 0, 0), degrees=True).as_quat()
    hand_actor = Articulation(ArticulationCfg(
        prim_path=f"/World/Hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=hand_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.0005, #0.005
                stabilization_threshold=0.05, # 0.0005
                fix_root_link=True
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
            mass_props=hand_mass_props,
            # fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={joint_name: hand_qpos[joint_name] for joint_name in hand_joint_names},
            pos=hand_translation,
            rot=transformhand_rotation,
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=hand_joint_names, # Match all joints
                stiffness=config_hand_motion.hand_joint_stiffness,
                damping=config_hand_motion.hand_joint_damping,
            ),
        }
    ))
    
    obj_joint_names = config_hand_motion.obj_joint_names
    obj_qpos = {joint_name: 0.0 for joint_name in obj_joint_names}
    obj_index = get_moving_object_index(scene_data["object_poses"])
    obj_usd_path = scene_data['object_paths'][obj_index]  # assuming first object
    object_translation = hand_translation
    object_rotation = transformhand_rotation
    obj_actor = Articulation(ArticulationCfg(
        prim_path=f"/World/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=obj_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
                fix_root_link=True
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
            # mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={joint_name: obj_qpos[joint_name] for joint_name in obj_joint_names}, # same reset pose as hand
            pos=object_translation,
            rot=object_rotation,
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=obj_joint_names, # Match all joints
                stiffness=config_hand_motion.obj_joint_stiffness, # Apply 0.1 stiffness to all joints
                damping=config_hand_motion.obj_joint_damping, # Apply 0 damping to all joints
            ),
        }
    ))

    # Play the simulator
    # sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # set robot initial pose
    robot_origins = torch.tensor([[0.0, 0.0, 0.0]]).to(sim.device)
    robot_root_state = robot_actor.data.default_root_state.clone()
    print("Default robot root state:", robot_root_state)
    robot_root_state[:, :3] += robot_origins
    robot_actor.write_root_pose_to_sim(robot_root_state[:, :7])
    robot_actor.write_root_velocity_to_sim(robot_root_state[:, 7:])
    # set joint positions with some noise
    robot_joint_pos, robot_joint_vel = robot_actor.data.default_joint_pos.clone(), robot_actor.data.default_joint_vel.clone()
    print("Default robot joint pos:", robot_joint_pos)
    robot_actor.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel)

    # set hand root pose
    origins = torch.tensor([[0.0, 0.0, 0.0]]).to(sim.device)
    root_state = hand_actor.data.default_root_state.clone()
    print("Default root state:", root_state)
    root_state[:, :3] += origins
    hand_actor.write_root_pose_to_sim(root_state[:, :7])
    hand_actor.write_root_velocity_to_sim(root_state[:, 7:])
    # set joint positions with some noise
    joint_pos, joint_vel = hand_actor.data.default_joint_pos.clone(), hand_actor.data.default_joint_vel.clone()
    print("Default joint pos:", joint_pos)
    hand_actor.write_joint_state_to_sim(joint_pos, joint_vel)
    
    print_actor_properties(hand_actor)
    print_joint_limits(hand_actor)
    
    hand = hand_actor  # alias
    
    # set object root pose
    obj_origins = torch.tensor([[0.0, 0.0, 0.0]]).to(sim.device)
    obj_root_state = obj_actor.data.default_root_state.clone()
    print("Object Default root state:", obj_root_state)
    obj_root_state[:, :3] += obj_origins
    obj_actor.write_root_pose_to_sim(obj_root_state[:, :7])
    obj_actor.write_root_velocity_to_sim(obj_root_state[:, 7:])
    # set joint positions with some noise
    obj_joint_pos, obj_joint_vel = obj_actor.data.default_joint_pos.clone(), obj_actor.data.default_joint_vel.clone()
    print("Object Default joint pos:", obj_joint_pos)
    obj_actor.write_joint_state_to_sim(obj_joint_pos, obj_joint_vel)

    print("[INFO] Starting grasp demo using existing control pipeline...")

    # clone to CUDA (if not already)
    joint_pos = hand.data.joint_pos.clone().to("cuda:0")
    joint_vel = torch.zeros_like(joint_pos).to("cuda:0")
    joint_limits = hand.data.joint_pos_limits.clone().to("cuda:0")
    obj_joint_pos = obj_actor.data.joint_pos.clone().to("cuda:0")
    obj_joint_vel = torch.zeros_like(obj_joint_pos).to("cuda:0")
    obj_joint_limits = obj_actor.data.joint_pos_limits.clone().to("cuda:0")

    robot_actor.reset()
    robot_actor.write_data_to_sim()
    robot_actor.update(0.1)
    
    # clear internal buffers
    hand_actor.reset()
    # -- write data to sim
    hand_actor.write_data_to_sim()
     # Update buffers
    hand_actor.update(0.1)
    
    obj_actor.reset()
    obj_actor.write_data_to_sim()
    obj_actor.update(0.1)
    
    hand_pose = scene_data['hand_pose']  # (T, 51)
    obj_pose = scene_data['object_poses'][obj_index]  # (T, 6)
    num_frames = hand_pose.shape[0]
    
    assert obj_pose.shape[0] == num_frames, "Object and hand motion length mismatch!"
    
    # debug info
    trans_seq = hand_pose[:, 0:3]  # (T, 3)
    euler_seq = hand_pose[:, 3:6]  # (T, 3) euler
    obj_trans_seq = scene_data['object_poses'][0, :, 0:3]  # (T, 3)
    obj_rot_seq = scene_data['object_poses'][0, :, 3:6]  # (T, 3) euler

    print('Isaac Sim data:')
    print('Joint limits for rotation joints:', joint_limits[0, 3:6])
    # print('Original rotation sequence (axis-angle) max:', np.max(rot_seq, axis=0))
    # print('Original rotation sequence (axis-angle) min:', np.min(rot_seq, axis=0))
    print('Recorded data:')
    print('Rotation sequence max:', np.max(euler_seq))
    print('Rotation sequence min:', np.min(euler_seq))
    # print('Joint sequence max:', np.max(pose_seq, axis=0))
    # print('Joint sequence min:', np.min(pose_seq, axis=0))
    
    # --- Step 1: sort the hand_joint_names array numerically ---
    joint_names_sorted = sorted(hand_joint_names, key=lambda x: int(re.findall(r'\d+', x)[0]))
    print('Mapping joints to pybullet recorded data format:')
    print('Sorted joint names:', joint_names_sorted)

    # --- Step 2: map recorded sequence to sorted joint_names array ---
    # recorded column i corresponds to joint_names_sorted[i]
    # so we can build a dictionary for lookup: recorded_index -> joint_name
    recorded_to_jointname = {i: joint_names_sorted[i] for i in range(len(joint_names_sorted))}
    print('Recorded to joint name mapping: ', {k: v for k, v in recorded_to_jointname.items()})

    # --- Step 3: create a dictionary of real indices ---
    # this dictionary maps joint name -> its real index in your Isaac model
    joint_names_dict = {name: idx for idx, name in enumerate(hand_joint_names)}
    print('Joint names to real Isaac indices mapping:', joint_names_dict)

    # --- Step 4–6: build reordered joint angle array ---
    # we will start with zeros and then fill each position
    hand_pose_reordered = np.zeros_like(hand_pose)

    for recorded_index, joint_name in recorded_to_jointname.items():
        if joint_name not in joint_names_dict:
            # joint not present in real model — skip
            continue
        real_index = joint_names_dict[joint_name]
        hand_pose_reordered[:, real_index] = hand_pose[:, recorded_index]

    # print_joint_mapping(hand_pose_reordered, hand_joint_names, joint_names_sorted)

    # Let robot settle for a bit
    for _ in range(100):
        sim.step(render=True)

    # object joints should be in order already
    # start motion
    new_pos = joint_pos.clone()
    new_obj_pos = obj_joint_pos.clone()
    
    # rate = node.create_rate(10)
    for frame in range(num_frames):
        # advance sim
        sim.step(render=True)
        
        
        # try new seq
        new_pos[0, 0:3] = torch.tensor(hand_pose_reordered[frame, 0:3], dtype=torch.float32, device="cuda:0")
        euler = hand_pose_reordered[frame, 3:6] #R.from_rotvec(hand_pose_reordered[frame, 3:6], degrees=False).as_euler('xyz', degrees=False)
        new_pos[0, 3:6] = torch.tensor(euler, dtype=torch.float32, device="cuda:0")
        new_pos[0, 6:] = torch.tensor(hand_pose_reordered[frame, 6:], dtype=torch.float32, device="cuda:0")

        # Write to sim
        hand.write_joint_state_to_sim(new_pos, joint_vel)
        
        new_obj_pos[0, 0:3] = torch.tensor(obj_pose[frame, 0:3], dtype=torch.float32, device="cuda:0")
        obj_euler = obj_pose[frame, 3:6] #R.from_rotvec(obj_pose[frame, 3:6], degrees=False).as_euler('xyz', degrees=False)
        new_obj_pos[0, 3:6] = torch.tensor(obj_euler, dtype=torch.float32, device="cuda:0")
        
        obj_actor.write_joint_state_to_sim(new_obj_pos, obj_joint_vel)
        
        # test robot motion
        if robot_joint_pos_msg is not None:
            # to tensors
            robot_joint_pos_tensor = torch.tensor(robot_joint_pos_msg, dtype=torch.float32, device="cuda:0")
            # robot_joint_vel_tensor = torch.tensor(robot_joint_vel_msg, dtype=torch.float32, device="cuda:0")
            # assign
            robot_joint_pos[0, :] = robot_joint_pos_tensor
            robot_actor.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel)
        
        
    for _ in range(num_frames):
        # advance sim
        sim.step(render=True)
        
    print("[INFO] Grasp demo finished.")
    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
