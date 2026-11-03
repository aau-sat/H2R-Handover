import yaml
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch_ros.actions import Node
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory
import os
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None

def launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration("use_sim_time")
    ur_type = LaunchConfiguration("ur_type")
    ur_type_value = ur_type.perform(context)
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    default_planning_pipeline = 'pilz_industrial_motion_planner'

    # Define package names for clarity
    moveit_config_pkg = "ur_custom_gripper_moveit_config"
    ur_custom_pkg = "ur_custom"

    joint_controllers_file = os.path.join(
        get_package_share_directory(ur_custom_pkg), 'config', 'ur_custom_controllers_gripper.yaml'
    )
    
    kinematics_params_file = os.path.join(
        get_package_share_directory(moveit_config_pkg), 'calibration', f'{ur_type_value}.yaml'
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
        "kinematics_params": kinematics_params_file
    }

    moveit_config = (
        MoveItConfigsBuilder("ur", package_name=moveit_config_pkg)
        .robot_description(
            file_path="config/ur.urdf.xacro",
            mappings=xacro_mappings 
        )
        .robot_description_semantic(file_path="config/ur_grasp_center.srdf")
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
    # moveit_config.planning_pipelines["ompl"]["gripper"]["enforce_constrained_state_space"] = True
    # moveit_config.planning_pipelines["ompl"]["gripper"]["projection_evaluator"] = "joints(gripper_finger_joint,gripper_finger2_joint)"
    pilz_planning_yaml = load_yaml(moveit_config_pkg, "config/pilz_industrial_motion_planner_planning.yaml")
    moveit_config.planning_pipelines["pilz_industrial_motion_planner"].update(pilz_planning_yaml)
                                     
    rviz_config_path = os.path.join(
        get_package_share_directory(moveit_config_pkg),
        "config",
        "moveit_upd.rviz",
    )

    # Nodes
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
        parameters=[
            moveit_config.to_dict(), # Pass full config to RViz
        ],
    )

    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[moveit_config.robot_description, joint_controllers_file],
        output='screen',
        remappings=[("~/robot_description", "/robot_description")],
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[moveit_config.robot_description],
        output='screen'
    )

    # Spawners
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    arm_scaled_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["scaled_joint_trajectory_controller", "--controller-manager", "/controller_manager"],
    )
    
    arm_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
    )
    
    forward_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "forward_position_controller",
            "-c",
            "/controller_manager",
            "--inactive",
        ],
    )
    
    forward_velocity_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "forward_velocity_controller",
            "-c",
            "/controller_manager",
            "--inactive",
        ],
    )

    gripper_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_position_controller", "--controller-manager", "/controller_manager"],
    )

    move_group_capabilities = {
        "capabilities": "pilz_industrial_motion_planner/MoveGroupSequenceAction pilz_industrial_motion_planner/MoveGroupSequenceService"
    }

    # Move Group Node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            move_group_capabilities,
            {"use_sim_time": use_sim_time}
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )
    
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
        # arguments=["--frame-id", "world", "--child-frame-id", "base_link"],
    )

    # Base
    base_json_path = os.path.join(get_package_share_directory('ur_custom_helper'), 'obstacles', 'base.json')

    # add_base_node = Node(
    #     package='ur_custom_helper',
    #     executable='add_obstacle_node', # Ensure this is marked executable in setup.py
    #     name='base_loader',
    #     parameters=[{'obstacle_json_path': base_json_path}]
    # )

    # Table
    table_json_path = os.path.join(get_package_share_directory('ur_custom_helper'), 'obstacles', 'table.json')

    # add_table_node = Node(
    #     package='ur_custom_helper',
    #     executable='add_obstacle_node', # Ensure this is marked executable in setup.py
    #     name='table_loader',
    #     parameters=[{'obstacle_json_path': table_json_path}]
    # )

    # Event Handlers
    delay_joint_state_broadcaster = RegisterEventHandler(
        OnProcessStart(target_action=controller_manager_node, on_start=[joint_state_broadcaster_spawner])
    )
    delay_scaled_arm_controller = RegisterEventHandler(
        OnProcessStart(target_action=joint_state_broadcaster_spawner, on_start=[arm_scaled_trajectory_controller_spawner])
    )
    delay_arm_controller = RegisterEventHandler(
        OnProcessStart(target_action=joint_state_broadcaster_spawner, on_start=[arm_trajectory_controller_spawner])
    )
    delay_gripper_position_controller = RegisterEventHandler(
        OnProcessStart(target_action=joint_state_broadcaster_spawner, on_start=[gripper_position_controller_spawner])
    )
    delay_forward_position_controller = RegisterEventHandler(
        OnProcessStart(target_action=joint_state_broadcaster_spawner, on_start=[forward_position_controller_spawner])
    )
    delay_forward_velocity_controller = RegisterEventHandler(
        OnProcessStart(target_action=joint_state_broadcaster_spawner, on_start=[forward_velocity_controller_spawner])
    )
    delay_rviz_node = RegisterEventHandler(
        OnProcessStart(target_action=robot_state_publisher, on_start=[rviz_node])
    )

    # delay_base_node = RegisterEventHandler(
    #     OnProcessStart(
    #         target_action=move_group_node,
    #         on_start=[add_base_node],
    #     )
    # )

    # delay_table_node = RegisterEventHandler(
    #     OnProcessStart(
    #         target_action=move_group_node,
    #         on_start=[add_table_node],
    #     )
    # )

    actions = [
        controller_manager_node,
        robot_state_publisher,
        static_tf,
        move_group_node,
        delay_joint_state_broadcaster,
        delay_scaled_arm_controller,
        delay_gripper_position_controller,
        delay_forward_position_controller,
        delay_forward_velocity_controller,
        delay_rviz_node,
    ]

    # Keep these available for quick enabling during experiments.
    # actions.append(delay_arm_controller)
    # actions.append(delay_base_node)
    # actions.append(delay_table_node)

    return actions


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
        description="Type of UR robot (ur3e, ur7e, ur10e)"
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

    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld