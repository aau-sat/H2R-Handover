# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import argparse
import sys
import time

from isaacsim import SimulationApp

# This sample loads a usd stage and starts simulation
CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RaytracedLighting"}

# Set up command line arguments
parser = argparse.ArgumentParser("Usd Load sample")
parser.add_argument(
    "--usd_path", type=str, help="Path to usd file, should be relative to your default assets folder", required=True
)
parser.add_argument("--headless", default=False, action="store_true", help="Run stage headless")
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")

args, unknown = parser.parse_known_args()
# Start the omniverse application
CONFIG["headless"] = args.headless
simulation_app = SimulationApp(launch_config=CONFIG)

import carb
import omni

from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import stage, extensions, nucleus
import omni.graph.core as og

import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd

from isaacsim.core.utils.prims import set_targets
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.nodes.scripts.utils import set_target_prims

# Locate Isaac Sim assets folder to load sample
from isaacsim.storage.native import get_assets_root_path, is_file

def publish_pointcloud_from_depth(camera: Camera, freq):
    print(type(camera))
    print('render product path:', camera._render_product_path)
    print('camera name:', camera.name)
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_pointcloud" # Set topic name to the camera's name
    queue_size = 1
    node_namespace = "wrist_camera/depth"
    frame_id = 'wrist_camera:depth'

    # Note, this pointcloud publisher will convert the Depth image to a pointcloud using the Camera intrinsics.
    # This pointcloud generation method does not support semantic labeled objects.
    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.DistanceToImagePlane.name
    )

    writer = rep.writers.get(rv + "ROS2PublishPointCloud")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return

# Enable ROS2 bridge extension
extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
simulation_context = SimulationContext(stage_units_in_meters=1.0)

assets_root_path = '' #get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()
usd_path = assets_root_path + args.usd_path

# make sure the file exists before we try to open it
try:
    result = is_file(usd_path)
except:
    result = False

if result:
    omni.usd.get_context().open_stage(usd_path)
else:
    carb.log_error(
        f"the usd path {usd_path} could not be opened, please make sure that {args.usd_path} is a valid usd file in {assets_root_path}"
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
print("Loading Complete")
# omni.timeline.get_timeline_interface().play()

# Initialize physics
approx_freq = 30
simulation_context.play()

camera_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/ur10_camera_holder_robotiq85/Robotiq_2F_85_edit/Robotiq_2F_85/base_link/Realsense/RSD455/Camera_Pseudo_Depth")
camera = Camera(prim_path=camera_prim.GetPath().pathString)
camera.initialize()
# publish_pointcloud_from_depth(camera, freq=10)

while simulation_app.is_running():
    simulation_context.step(render=True)
    # Run in realtime mode, we don't specify the step size
    # simulation_app.update()

# omni.timeline.get_timeline_interface().stop()
simulation_context.stop()
simulation_app.close()
