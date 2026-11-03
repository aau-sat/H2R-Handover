/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, SRI International
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of SRI International nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Sachin Chitta, Dave Coleman, Mike Lautman */

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_state/robot_state.h>

#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>

#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>

#include <moveit_visual_tools/moveit_visual_tools.h>


// All source files that use ROS logging should define a file-specific
// static const rclcpp::Logger named LOGGER, located at the top of the file
// and inside the namespace with the narrowest scope (if there is one)
static const rclcpp::Logger LOGGER = rclcpp::get_logger("orient_gripper_test");

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);

  static const std::string marker_topic = "move_group_marker";

  auto move_group_node = rclcpp::Node::make_shared("orient_gripper_test", node_options);

  // We spin up a SingleThreadedExecutor for the current state monitor to get information
  // about the robot's state.
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(move_group_node);
  std::thread([&executor]() { executor.spin(); }).detach();

  // BEGIN_TUTORIAL
  //
  // Setup
  // ^^^^^
  //
  // MoveIt operates on sets of joints called "planning groups" and stores them in an object called
  // the ``JointModelGroup``. Throughout MoveIt, the terms "planning group" and "joint model group"
  // are used interchangeably.
  static const std::string PLANNING_GROUP_ARM = "ur_custom_manipulator";
  static const std::string PLANNING_GROUP_GRIPPER = "gripper";

 // Client class to conveniently use the ROS interfaces provided by the move_group node. 
  // It can be easily set up using just the name of the planning group you would like to control and plan for.
  moveit::planning_interface::MoveGroupInterface move_group(move_group_node, PLANNING_GROUP_ARM);
  moveit::planning_interface::MoveGroupInterface gripper_interface(move_group_node, PLANNING_GROUP_GRIPPER);

  // Debug
  // Check the robot model
  auto robot_model = move_group.getRobotModel();
  if (robot_model) {
    RCLCPP_INFO(LOGGER, "Robot model loaded successfully");
    RCLCPP_INFO(LOGGER, "Robot model name: %s", robot_model->getName().c_str());
    auto group_names = robot_model->getJointModelGroupNames();
    RCLCPP_INFO(LOGGER, "Joint model groups found: %zu", group_names.size());
    for (const auto& name : group_names) {
      RCLCPP_INFO(LOGGER, "  - %s", name.c_str());
    }
  } else {
    RCLCPP_ERROR(LOGGER, "Robot model is NULL!");
  }

  // Get default pipeline parameter
  std::string default_pipeline_id = move_group.getDefaultPlanningPipelineId(); // returns empty string
  if (default_pipeline_id.empty()) {
    if (move_group_node->get_parameter("move_group.default_planning_pipeline", default_pipeline_id)) {
      RCLCPP_INFO(LOGGER, "move_group.default_planning_pipeline param: %s", default_pipeline_id.c_str());
    } else if (move_group_node->get_parameter("default_planning_pipeline", default_pipeline_id)) {
      RCLCPP_INFO(LOGGER, "default_planning_pipeline param: %s", default_pipeline_id.c_str());
    } else {
      default_pipeline_id = "pilz_industrial_motion_planner";
      RCLCPP_WARN(LOGGER, "No default_planning_pipeline parameter found - defaulting to: %s", default_pipeline_id.c_str());
    }
  }

  // Get default planning config
  std::string default_planner_id = move_group.getDefaultPlannerId();
  if (default_planner_id.empty()) {
    if (move_group_node->get_parameter("pilz_industrial_motion_planner.default_planner_config", default_planner_id)) {
      RCLCPP_INFO(LOGGER, "pilz_industrial_motion_planner.default_planner_config param: %s", default_planner_id.c_str());
    } else if (move_group_node->get_parameter("default_planner_config", default_planner_id)) {
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

  // Set ompl planner for gripper control, since Pilz Industrial Motion Planner doesn't support it
  gripper_interface.setPlanningPipelineId("ompl");
  gripper_interface.setPlannerId("RRTConnect");

  // We will use the
  // :moveit_codedir:`PlanningSceneInterface<moveit_ros/planning_interface/planning_scene_interface/include/moveit/planning_scene_interface/planning_scene_interface.h>`
  // class to add and remove collision objects in our "virtual world" scene
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  // Raw pointers are frequently used to refer to the planning group for improved performance.
  const moveit::core::JointModelGroup* joint_model_group =
      move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP_ARM);

  // Visualization
  // ^^^^^^^^^^^^^
  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools(move_group_node, "world", marker_topic,
                                                      move_group.getRobotModel());

  visual_tools.deleteAllMarkers();

  /* Remote control is an introspection tool that allows users to step through a high level script */
  /* via buttons and keyboard shortcuts in RViz */
  visual_tools.loadRemoteControl();

  // RViz provides many types of markers, in this demo we will use text, cylinders, and spheres
  Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
  text_pose.translation().z() = 1.0;
  visual_tools.publishText(text_pose, "MoveGroupInterface_Demo", rvt::WHITE, rvt::XLARGE);

  // Batch publishing is used to reduce the number of messages being sent to RViz for large visualizations
  visual_tools.trigger();

  // Getting Basic Information
  // ^^^^^^^^^^^^^^^^^^^^^^^^^
  //
  // We can print the name of the reference frame for this robot.
  RCLCPP_INFO(LOGGER, "Planning frame: %s", move_group.getPlanningFrame().c_str());

  // We can also print the name of the end-effector link for this group.
  RCLCPP_INFO(LOGGER, "End effector link: %s", move_group.getEndEffectorLink().c_str());

  // We can get a list of all the groups in the robot:
  RCLCPP_INFO(LOGGER, "Available Planning Groups:");
  std::copy(move_group.getJointModelGroupNames().begin(), move_group.getJointModelGroupNames().end(), std::ostream_iterator<std::string>(std::cout, ", "));

  // Start the demo
  // ^^^^^^^^^^^^^^^^^^^^^^^^^
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");

  auto current_pose = move_group.getCurrentPose();

  std::vector<geometry_msgs::msg::Pose> gripper_poses;

  geometry_msgs::msg::Pose pose = current_pose.pose;
  pose.position.x = -0.15549154579639435;
  pose.position.y = 0.5039573907852173;
  pose.position.z = 0.2858544588088989;
  pose.orientation.x = -0.8199933767318726;
  pose.orientation.y = 0.004146444145590067;
  pose.orientation.z = -0.05828935652971268;
  pose.orientation.w = 0.5693821310997009;
  gripper_poses.push_back(pose);

  pose.position.x = -0.07027453184127808;
  pose.position.y = 0.855571985244751;
  pose.position.z = 0.16768424212932587;
  pose.orientation.x = 0.67822265625;
  pose.orientation.y = 0.6802109479904175;
  pose.orientation.z = 0.2019793540239334;
  pose.orientation.w = -0.19113188982009888;
  gripper_poses.push_back(pose);

  pose.position.x = 0.018008608371019363;
  pose.position.y = 0.616560161113739;
  pose.position.z = 0.21855834126472473;
  pose.orientation.x = -0.3880927562713623;
  pose.orientation.y = -0.6469264030456543;
  pose.orientation.z = -0.2983485460281372;
  pose.orientation.w = 0.5846865773200989;
  gripper_poses.push_back(pose);

  const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();
  std::vector<double> joint_values;
  auto robot_state = move_group.getCurrentState();

  for (const auto& gripper_pose: gripper_poses)
  {
      move_group.setPoseTarget(gripper_pose);

      moveit::planning_interface::MoveGroupInterface::Plan my_plan;
      bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

      RCLCPP_INFO(LOGGER, "Visualizing plan 1 (pose goal) %s", success ? "" : "FAILED");
      if (!success) {
        RCLCPP_INFO(LOGGER, "Trying from inverse kinematics solution...");
        if(robot_state->setFromIK(joint_model_group, gripper_pose))
        {
          RCLCPP_INFO(LOGGER, "Successfully retrieved IK Solution!");
          robot_state->copyJointGroupPositions(joint_model_group, joint_values);
          for (std::size_t i = 0; i < joint_names.size(); ++i)
          {
            RCLCPP_INFO(LOGGER, "Joint %s: %f", joint_names[i].c_str(), joint_values[i]);
            move_group.setJointValueTarget(joint_values);
          }
        }
        else {
          RCLCPP_ERROR(LOGGER, "Cannot solve IK");
        }
      }

      std::string pose_name = "pose" + std::to_string(&gripper_pose - &gripper_poses[0]);
      visual_tools.publishAxisLabeled(gripper_pose, pose_name);
      visual_tools.publishText(text_pose, "Pose_Goal", rvt::WHITE, rvt::XLARGE);
      visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
      visual_tools.trigger();

      visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

      visual_tools.deleteAllMarkers();
      visual_tools.trigger();
  }

  // Cartesian Paths
  const double jump_threshold = 0.0;
  const double eef_step = 0.01;
  moveit_msgs::msg::RobotTrajectory trajectory;
  std::vector<geometry_msgs::msg::Pose> waypoints;
  // waypoints.push_back(current_pose.pose);
  for (const auto& gripper_pose: gripper_poses)
  {
      waypoints.push_back(gripper_pose);
      double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
      RCLCPP_INFO(LOGGER, "Visualizing plan 4 (Cartesian path) (%.2f%% achieved)", fraction * 100.0);

      // Visualize the plan in RViz
      visual_tools.deleteAllMarkers();
      visual_tools.publishText(text_pose, "Cartesian_Path", rvt::WHITE, rvt::XLARGE);
      visual_tools.publishPath(waypoints, rvt::LIME_GREEN, rvt::SMALL);
      for (std::size_t i = 0; i < waypoints.size(); ++i)
        visual_tools.publishAxisLabeled(waypoints[i], "pt" + std::to_string(i), rvt::SMALL);
      visual_tools.trigger();
      visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

      waypoints.pop_back();
  }
  visual_tools.deleteAllMarkers();
  visual_tools.trigger();

  // execute a single plan with gripper control
  std::thread([&gripper_interface]() {
    gripper_interface.setJointValueTarget({0.8});
    // gripper_group.setNamedTarget("closed"); // or setJointValueTarget
    gripper_interface.move();
  }).detach();

  move_group.setPoseTarget(gripper_poses[0]);
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

  RCLCPP_INFO(LOGGER, "Visualizing plan (pose goal) %s", success ? "" : "FAILED");
  std::string pose_name = "pose_final";
  visual_tools.publishAxisLabeled(gripper_poses[0], pose_name);
  visual_tools.publishText(text_pose, "Pose_Goal", rvt::WHITE, rvt::XLARGE);
  visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
  visual_tools.trigger();

  if (success) {
    move_group.execute(my_plan);
  }
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  visual_tools.deleteAllMarkers();
  visual_tools.trigger();

  rclcpp::shutdown();
  return 0;
}
