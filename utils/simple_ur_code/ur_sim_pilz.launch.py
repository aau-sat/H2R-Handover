#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
import os


def generate_launch_description():

    # --------------------------------------------------------
    # 1) Launch UR driver (fake hardware)
    # --------------------------------------------------------
    ur_driver_pkg = get_package_share_directory('ur_robot_driver')

    ur_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_driver_pkg, 'launch', 'ur_control.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur3e',
            'robot_ip': '127.0.0.1',
            'use_fake_hardware': 'true',
            'launch_rviz': 'false'
        }.items()
    )

    # --------------------------------------------------------
    # 2) Proper MoveIt configuration loading
    # --------------------------------------------------------
    moveit_config = (
        MoveItConfigsBuilder("ur3e", package_name="ur3e_moveit_config")
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"]
        )
        .to_moveit_configs()
    )

    move_group_launch = generate_move_group_launch(moveit_config)

    # --------------------------------------------------------
    # 3) Return full system
    # --------------------------------------------------------
    return LaunchDescription([
        ur_driver_launch,
        move_group_launch
    ])
