from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from launch_param_builder import ParameterBuilder
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    ld = LaunchDescription()

    moveit_config_pkg = "ur_custom_gripper_moveit_config"
    
    use_sim_time = LaunchConfiguration("use_sim_time")
    ur_type = LaunchConfiguration("ur_type")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    default_planning_pipeline = 'pilz_industrial_motion_planner'
    
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
        default_value="false",
        description="Whether to use fake sensor commands (only relevant if use_fake_hardware is true)"
    ))

    xacro_mappings = {
        "ur_type": ur_type,
        "sim_gazebo": "false",
        "sim_ignition": "false",
        "use_fake_hardware": use_fake_hardware,
        "fake_sensor_commands": fake_sensor_commands, 
        "use_sim_time": use_sim_time
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
    
    # moveit_config.robot_description_kinematics = {} # reset kinematics to enable inverse Jakobian instead of kinematics solver

    servo_params = {
        "moveit_servo": ParameterBuilder("ur_custom")
        .yaml("config/servo_config.yaml")
        .yaml("config/pose_tracking_settings.yaml")
        .to_dict()
    }

    # pose_tracking_node = Node(
    #     package="moveit_servo",
    #     executable="servo_pose_tracking_demo",
    #     #executable="servo_node_main",
    #     # prefix=['xterm -e gdb -ex run --args'],
    #     output="screen",
    #     parameters=[
    #         servo_params,
    #         moveit_config.robot_description,
    #         moveit_config.robot_description_semantic
    #     ],
    # )
    pose_tracking_node = Node(
        package="moveit_stuff",
        executable="estimated_goal_tracking",
        #executable="servo_node_main",
        # prefix=['xterm -e gdb -ex run --args'],
        output="screen",
        parameters=[
            servo_params,
            moveit_config.to_dict(),
        ],
    )
    
    ld.add_action(pose_tracking_node)
    return ld