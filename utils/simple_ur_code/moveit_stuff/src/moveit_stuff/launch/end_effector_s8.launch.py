import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from launch_param_builder import ParameterBuilder
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None

def launch_setup(context, *args, **kwargs):
    moveit_config_pkg = "ur_custom_gripper_moveit_config"
    
    use_sim_time = LaunchConfiguration("use_sim_time")
    ur_type = LaunchConfiguration("ur_type")
    ur_type_value = ur_type.perform(context)
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    enable_pose_interpolation = LaunchConfiguration("enable_pose_interpolation")
    pose_interpolation_steps = LaunchConfiguration("pose_interpolation_steps")
    pose_interpolation_profile = LaunchConfiguration("pose_interpolation_profile")
    default_planning_pipeline = 'pilz_industrial_motion_planner'

    kinematics_params_file = os.path.join(
        get_package_share_directory(moveit_config_pkg),
        'calibration',
        f'{ur_type_value}.yaml'
    )

    if not os.path.exists(kinematics_params_file):
        raise RuntimeError(
            f"Kinematics file not found for ur_type='{ur_type_value}': {kinematics_params_file}"
        )

    xacro_mappings = {
        "ur_type": ur_type,
        "sim_gazebo": "false",
        "sim_ignition": "false",
        "use_fake_hardware": use_fake_hardware,
        "fake_sensor_commands": fake_sensor_commands, 
        "use_sim_time": use_sim_time,
        "kinematics_params": kinematics_params_file
    }

    moveit_config = (
        MoveItConfigsBuilder("ur", package_name=moveit_config_pkg)
        .robot_description(
            file_path="config/ur.urdf.xacro",
            mappings=xacro_mappings 
        )
        .robot_description_semantic(file_path="config/ur.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_scene_monitor(
            publish_robot_description=True, 
            publish_robot_description_semantic=True, 
            publish_planning_scene=True
        )
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"],
            default_planning_pipeline=default_planning_pipeline,
        )
        .to_moveit_configs()
    )
    moveit_config.planning_pipelines["ompl"]["ur_custom_manipulator"]["enforce_constrained_state_space"] = True
    moveit_config.planning_pipelines["ompl"]["ur_custom_manipulator"]["projection_evaluator"] = "joints(shoulder_pan_joint,shoulder_lift_joint)"
    pilz_planning_yaml = load_yaml(moveit_config_pkg, "config/pilz_industrial_motion_planner_planning.yaml")
    moveit_config.planning_pipelines["pilz_industrial_motion_planner"].update(pilz_planning_yaml)
    
    # moveit_config.robot_description_kinematics = {} # reset kinematics to enable inverse Jakobian instead of kinematics solver

    servo_params = {
        "moveit_servo": ParameterBuilder("ur_custom")
        .yaml("config/servo_config_s8.yaml")
        .yaml("config/pose_tracking_settings_s8.yaml")
        .to_dict()
    }

    pose_tracking_node = Node(
        package="moveit_stuff",
        executable="end_effector_s8",
        #executable="servo_node_main",
        # prefix=['xterm -e gdb -ex run --args'],
        output="screen",
        parameters=[
            servo_params,
            moveit_config.to_dict(),
            {"ur_type": ur_type},
            {"enable_pose_interpolation": enable_pose_interpolation},
            {"pose_interpolation_steps": pose_interpolation_steps},
            {"pose_interpolation_profile": pose_interpolation_profile},
        ],
    )
    
    return [pose_tracking_node]


def generate_launch_description():
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock"
    ))
    ld.add_action(DeclareLaunchArgument(
        "ur_type",
        default_value="ur7e",
        description="Type of UR robot (ur3e, ur7e, ur10e, ur12e)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="true",
        description="Whether to use fake hardware interface"
    ))
    ld.add_action(DeclareLaunchArgument(
        "fake_sensor_commands",
        default_value="true",
        description="Whether to use fake sensor commands (only relevant if use_fake_hardware is true)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "enable_pose_interpolation",
        default_value="true",
        description="Enable interpolation between the current pose and each target pose when using servoing"
    ))
    ld.add_action(DeclareLaunchArgument(
        "pose_interpolation_steps",
        default_value="1",
        description="Number of interpolation poses before each final target pose (0 disables interpolation steps)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "pose_interpolation_profile",
        default_value="linear",
        description="Interpolation profile for servo targets: linear | ease_in_quad | ease_in_cubic"
    ))

    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld