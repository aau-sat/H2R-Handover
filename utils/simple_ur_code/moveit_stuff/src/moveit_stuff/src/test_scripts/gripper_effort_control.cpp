/*******************************************************************************
 *      Title     : pose_tracking_example.cpp
 *      Project   : moveit_servo
 *      Created   : 09/04/2020
 *      Author    : Adam Pettinger
 *
 * BSD 3-Clause License
 *
 * Copyright (c) 2019, Los Alamos National Security, LLC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/

#include <std_msgs/msg/int8.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>

#include <thread>
#include <chrono> // Added for time literals
#include <atomic>
#include <string>

#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/move_group_interface/move_group_interface.h>

// --- ADDED FOR CONTROLLER SWITCHING ---
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <control_msgs/action/gripper_command.hpp>
#include <control_msgs/msg/dynamic_joint_state.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
// --------------------------------------

static const rclcpp::Logger LOGGER = rclcpp::get_logger("moveit_servo_tracking_test");

using GripperCommand = control_msgs::action::GripperCommand;
using GoalHandleGripper = rclcpp_action::ClientGoalHandle<GripperCommand>;

namespace
{
std::string objectDetectionStatusToString(int status)
{
  switch (status)
  {
    case 0:
      return "MOVING";
    case 1:
      return "OBJECT_DETECTED_OPENING";
    case 2:
      return "OBJECT_DETECTED_CLOSING";
    case 3:
      return "AT_REQUESTED_POSITION";
    default:
      return "UNKNOWN(" + std::to_string(status) + ")";
  }
}
}  // namespace

// --------------------------------------

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("pose_tracking_test");

  static const std::string marker_topic = "move_group_marker";

  // --- MODIFIED: Upgraded to MultiThreadedExecutor ---
  // This is safer for moveit_servo and ensures our service response can be 
  // processed while the main thread waits.
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread executor_thread([&executor]() { executor.spin(); });
  // ---------------------------------------------------

  // move group interfaces
  static const std::string PLANNING_GROUP_ARM = "ur_custom_manipulator";
  static const std::string PLANNING_GROUP_GRIPPER = "gripper";
  moveit::planning_interface::MoveGroupInterface move_group(node, PLANNING_GROUP_ARM);
  moveit::planning_interface::MoveGroupInterface gripper_interface(node, PLANNING_GROUP_GRIPPER);

  // ompl planner can handle gripper control
  gripper_interface.setPlanningPipelineId("ompl");
  gripper_interface.setPlannerId("RRTConnect");

  // visualization
  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools(node, "base_link", marker_topic,
                                                      move_group.getRobotModel());
  visual_tools.deleteAllMarkers();

  // Load the planning scene monitor
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor;
  planning_scene_monitor = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(node, "robot_description");
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
  planning_scene_monitor->startStateMonitor("/joint_states");
  planning_scene_monitor->startPublishingPlanningScene(planning_scene_monitor::PlanningSceneMonitor::UPDATE_SCENE);

  // Print Robotiq custom state interfaces, including object detection status.
  auto last_object_detection_status = std::make_shared<std::atomic<int>>(-1);
  auto dynamic_joint_state_sub = node->create_subscription<control_msgs::msg::DynamicJointState>(
      "/dynamic_joint_states", rclcpp::QoS(10),
      [last_object_detection_status](const control_msgs::msg::DynamicJointState::SharedPtr msg) {
        for (size_t joint_idx = 0; joint_idx < msg->joint_names.size() && joint_idx < msg->interface_values.size();
             ++joint_idx)
        {
          const auto& joint_name = msg->joint_names[joint_idx];
          const auto& interface_value = msg->interface_values[joint_idx];

          for (size_t i = 0; i < interface_value.interface_names.size() && i < interface_value.values.size(); ++i)
          {
            if (interface_value.interface_names[i] != "object_detection_status")
            {
              continue;
            }

            const int status = static_cast<int>(interface_value.values[i]);
            const int previous_status = last_object_detection_status->exchange(status);
            if (status != previous_status)
            {
              RCLCPP_INFO(LOGGER, "[%s] object_detection_status = %s (%d)", joint_name.c_str(),
                          objectDetectionStatusToString(status).c_str(), status);
            }
          }
        }
      });

  // Wait for Planning Scene Monitor to setup
  if (!planning_scene_monitor->waitForCurrentRobotState(node->now(), 5.0 /* seconds */))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Error waiting for current robot state in PlanningSceneMonitor.");
    exit(EXIT_FAILURE);
  }

  move_group.setStartStateToCurrentState();

  auto goal_msg = GripperCommand::Goal();
  auto send_goal_options = rclcpp_action::Client<GripperCommand>::SendGoalOptions();

  float target_joint_angle = 0.25; //0.69 for closed

  auto action_client = rclcpp_action::create_client<GripperCommand>(node, "/gripper_position_controller/gripper_cmd");
  // wait for the action server to be available
  if (!action_client->wait_for_action_server(std::chrono::seconds(5))) {
    RCLCPP_ERROR(LOGGER, "Gripper action server not available after waiting");
    executor.cancel();
    executor_thread.join();
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
  
  goal_msg.command.position = target_joint_angle;
  goal_msg.command.max_effort = 0.1;             // Target force in Newtons
  action_client->async_send_goal(goal_msg, send_goal_options);
    
  std::this_thread::sleep_for(std::chrono::milliseconds(5000));

  (void)dynamic_joint_state_sub;

  // Kill executor thread before shutdown
  executor.cancel();
  executor_thread.join();

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}