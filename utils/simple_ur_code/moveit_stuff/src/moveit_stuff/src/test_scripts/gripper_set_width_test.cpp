
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <thread>
#include <mutex>
#include <deque>
#include <chrono>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <moveit_msgs/srv/get_motion_plan.hpp>
#include <moveit/robot_state/robot_state.h>


static const rclcpp::Logger LOGGER = rclcpp::get_logger("gripper_set_width_test");

using namespace std::chrono_literals;

float getGripperAngleFromWidth(float width, float grip_offset)
{
  float max_stroke = 0.140;
  float max_joint_limit = 0.7; // radians, finger_joint limits are 0 to 0.7

  width -= grip_offset;
  float target_width = std::min(std::max(width, 0.0f), max_stroke);
  float joint_angle = max_joint_limit * (1.0 - target_width / max_stroke);
  return joint_angle;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.use_intra_process_comms(false);
  rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("gripper_set_width_test", node_options);

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread executor_thread([&executor]() { executor.spin(); });

  // visualization
  static const std::string PLANNING_GROUP_ARM = "ur_custom_manipulator";
  static const std::string PLANNING_GROUP_GRIPPER = "gripper";
  moveit::planning_interface::MoveGroupInterface move_group(node, PLANNING_GROUP_ARM);
  moveit::planning_interface::MoveGroupInterface gripper_interface(node, PLANNING_GROUP_GRIPPER);
  gripper_interface.setPlanningPipelineId("ompl");
  gripper_interface.setPlannerId("RRTConnect");

  // Get default pipeline parameter
  std::string default_pipeline_id = move_group.getDefaultPlanningPipelineId(); // returns empty string
  if (default_pipeline_id.empty()) {
    if (node->get_parameter("move_group.default_planning_pipeline", default_pipeline_id)) {
      RCLCPP_INFO(LOGGER, "move_group.default_planning_pipeline param: %s", default_pipeline_id.c_str());
    } else if (node->get_parameter("default_planning_pipeline", default_pipeline_id)) {
      RCLCPP_INFO(LOGGER, "default_planning_pipeline param: %s", default_pipeline_id.c_str());
    } else {
      default_pipeline_id = "pilz_industrial_motion_planner";
      RCLCPP_WARN(LOGGER, "No default_planning_pipeline parameter found - defaulting to: %s", default_pipeline_id.c_str());
    }
  }
  // Get default planning config
  std::string default_planner_id = move_group.getDefaultPlannerId();
  if (default_planner_id.empty()) {
    if (node->get_parameter("pilz_industrial_motion_planner.default_planner_config", default_planner_id)) {
      RCLCPP_INFO(LOGGER, "pilz_industrial_motion_planner.default_planner_config param: %s", default_planner_id.c_str());
    } else if (node->get_parameter("default_planner_config", default_planner_id)) {
      RCLCPP_INFO(LOGGER, "default_planner_config param: %s", default_planner_id.c_str());
    } else {
      default_planner_id = "LIN";
      RCLCPP_WARN(LOGGER, "No default_planner_config parameter found - defaulting to: %s", default_planner_id.c_str());
    }
  }
  // Set planner ID for Pilz Industrial Motion Planner (use "PTP", "LIN", or "CIRC")
  // For point-to-point joint space motion, use "PTP"
  // For linear Cartesian motion, use "LIN"
  // "CIRC" doesn't work most of the time, and is not recommended
  if (default_pipeline_id == "pilz_industrial_motion_planner")
  {
    move_group.setPlannerId(default_planner_id);
  }
  const moveit::core::JointModelGroup* joint_model_group =
      move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP_ARM);

  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools(node, "base_link", "move_group_marker",
                                                      move_group.getRobotModel());
  visual_tools.deleteAllMarkers();

  // Load the planning scene monitor
  auto tf_buffer_monitor = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor;
  planning_scene_monitor = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(node, "robot_description", tf_buffer_monitor, "planning_scene_monitor");
  if (!planning_scene_monitor->getPlanningScene())
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Error in setting up the PlanningSceneMonitor.");
    exit(EXIT_FAILURE);
  }

  planning_scene_monitor->providePlanningSceneService();
  planning_scene_monitor->startSceneMonitor();
  planning_scene_monitor->startWorldGeometryMonitor(
      planning_scene_monitor::PlanningSceneMonitor::DEFAULT_COLLISION_OBJECT_TOPIC,
      planning_scene_monitor::PlanningSceneMonitor::DEFAULT_PLANNING_SCENE_WORLD_TOPIC,
      false /* skip octomap monitor */);
  planning_scene_monitor->startStateMonitor("joint_states");
  // planning_scene_monitor->setPlanningScenePublishingFrequency(25);
  planning_scene_monitor->startPublishingPlanningScene(planning_scene_monitor::PlanningSceneMonitor::UPDATE_SCENE);

  // Wait for Planning Scene Monitor to setup
  if (!planning_scene_monitor->waitForCurrentRobotState(node->now(), 5.0))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Error waiting for current robot state in PlanningSceneMonitor.");
    exit(EXIT_FAILURE);
  }

  auto scene_pub = node->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 10);

  // Open gripper in background
  std::thread([&gripper_interface]() {
        gripper_interface.setNamedTarget("open");
        gripper_interface.move();
      }).detach();

    std::this_thread::sleep_for(2s);

    float object_width = 0.04; // Example object width in meters
    float grip_offset = 0.005; // Example offset for gripper fingers in meters

    float target_joint_angle = getGripperAngleFromWidth(object_width, grip_offset);
    // RCLCPP_INFO_STREAM(LOGGER, "Setting gripper joint angle to: " << target_joint_angle << " radians for object width: " << object_width << " meters");

    // std::thread([&gripper_interface, target_joint_angle]() {
            gripper_interface.setJointValueTarget("finger_joint", target_joint_angle);
            gripper_interface.move();
    //       }).detach();
    
    // gripper_interface.setNamedTarget("closed");
    // gripper_interface.move();

    // std::this_thread::sleep_for(2s);

    executor.cancel();
    executor_thread.join();

    rclcpp::shutdown();
    return EXIT_SUCCESS;
}                  