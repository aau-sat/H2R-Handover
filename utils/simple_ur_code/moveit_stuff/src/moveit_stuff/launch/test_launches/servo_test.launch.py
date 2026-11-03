from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from launch_param_builder import ParameterBuilder

def generate_launch_description():
    moveit_config_pkg = "ur_custom_gripper_moveit_config"

    # moveit_config = MoveItConfigsBuilder("ur", package_name=moveit_config_pkg).to_moveit_configs()
    moveit_config = (MoveItConfigsBuilder("ur", package_name=moveit_config_pkg)
    .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"],
            default_planning_pipeline="pilz_industrial_motion_planner",
    )
    .to_moveit_configs())

    servo_params = {
        "moveit_servo": ParameterBuilder("ur_custom")
        .yaml("config/servo_config.yaml")
        .to_dict()
    }
    
    # The servo cpp interface demo
    # Creates the Servo node and publishes commands to it
    servo_node = Node(
        package="moveit_stuff",
        executable="servo_test",
        output="screen",
        parameters=[
            servo_params,
            moveit_config.to_dict(),
            # moveit_config.robot_description,
            # moveit_config.robot_description_semantic,
        ],
    )

    return LaunchDescription([servo_node])