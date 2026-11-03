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
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>

#include <thread>
#include <chrono> // Added for time literals
#include <atomic>
#include <future>
#include <string>
#include <cmath>

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

using namespace std::chrono_literals;

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

  // Track if object has been detected during closed-loop grasping
  auto object_detected_during_grasp = std::make_shared<std::atomic<bool>>(false);

  // Track tactile sensor contacts
  auto alfredo_contact = std::make_shared<std::atomic<bool>>(false);             // Right sensor
  auto giovanni_contact = std::make_shared<std::atomic<bool>>(false);           // Left sensor
  auto tactile_detected = std::make_shared<std::atomic<bool>>(false);    // Both sensors contacted
  auto tactile_stop_requested = std::make_shared<std::atomic<bool>>(false); // Flag to stop grasping loop if tactile contact detected

  // Print Robotiq custom state interfaces, including object detection status.
  auto last_object_detection_status = std::make_shared<std::atomic<int>>(-1);
  auto dynamic_joint_state_sub = node->create_subscription<control_msgs::msg::DynamicJointState>(
      "/dynamic_joint_states", rclcpp::SensorDataQoS(),
      [last_object_detection_status, object_detected_during_grasp](const control_msgs::msg::DynamicJointState::SharedPtr msg) {
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
            // const int previous_status = last_object_detection_status->exchange(status);
            // if (status != previous_status)
            // {
              RCLCPP_INFO(LOGGER, "[%s] object_detection_status = %s (%d)", joint_name.c_str(),
                          objectDetectionStatusToString(status).c_str(), status);
            // }
            
            // Set flag if object detected while closing (status 2 = OBJECT_DETECTED_CLOSING)
            if (status == 2) {
              object_detected_during_grasp->store(true);
            }
          }
        }
      });

  // Subscribe to tactical sensor: ALFREDO (right)
  auto alfredo_sub = node->create_subscription<std_msgs::msg::Bool>(
      "/ALFREDO/Contact", rclcpp::SensorDataQoS(),
      [tactile_stop_requested](const std_msgs::msg::Bool::SharedPtr msg) {
        if (msg->data) {
          tactile_stop_requested->store(true);
          RCLCPP_INFO(LOGGER, "Tactile contact detected on ALFREDO (right) sensor");
        }
      });

  // Subscribe to tactical sensor: GIOVANNI (left)
  auto giovanni_sub = node->create_subscription<std_msgs::msg::Bool>(
      "/GIOVANNI/Contact", rclcpp::SensorDataQoS(),
      [tactile_stop_requested](const std_msgs::msg::Bool::SharedPtr msg) {
        if (msg->data) {
          tactile_stop_requested->store(true);
          RCLCPP_INFO(LOGGER, "Tactile contact detected on GIOVANNI (left) sensor");
        }
      });

  // Wait for Planning Scene Monitor to setup
  if (!planning_scene_monitor->waitForCurrentRobotState(node->now(), 5.0 /* seconds */))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Error waiting for current robot state in PlanningSceneMonitor.");
    exit(EXIT_FAILURE);
  }

  move_group.setStartStateToCurrentState();

  // Parameters for closed-loop grasping
  float target_joint_angle = 0.3;  // Middle of the range (0.69 for fully closed)
  float delta = 0.25;                // Total range around target
  int num_steps = 50;                // Number of discrete steps in the range
  constexpr float min_joint_angle = 0.0; // Fully open
  constexpr float max_joint_angle = 0.69; // Fully closed

  const bool use_joint_angle_interval =
      node->declare_parameter<bool>("use_joint_angle_interval", true);
  
  // Create discrete range of joint angles (non-linear distribution)
  // We use a mapping v = 1 - (1 - u)^exponent with u in [0,1].
  // exponent > 1 produces larger steps near the start and finer steps near the end.
  const double non_linear_exponent = 5.0; // tune this to change curvature
  std::vector<float> joint_angle_range;
  if (use_joint_angle_interval)
  {
    const float range_start = target_joint_angle - delta / 2.0f;
    for (int i = 0; i < num_steps; ++i) {
      double u = (num_steps == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(num_steps - 1);
      double v = 1.0 - std::pow(1.0 - u, non_linear_exponent);
      float angle = range_start + static_cast<float>(v * delta);
      angle = std::max(min_joint_angle, std::min(max_joint_angle, angle)); // Clamp to valid range
      joint_angle_range.push_back(angle);
      RCLCPP_INFO(LOGGER, "Step %d: joint angle = %.4f", i, angle);
    }
  }
  else
  {
    const float final_joint_angle = std::max(min_joint_angle, std::min(max_joint_angle, target_joint_angle));
    joint_angle_range.push_back(final_joint_angle);
    RCLCPP_INFO(LOGGER, "Single-step mode enabled: joint angle = %.4f", final_joint_angle);
  }

  // Create gripper action client
  auto action_client = rclcpp_action::create_client<GripperCommand>(node, "/gripper_position_controller/gripper_cmd");
  
  // Wait for the action server to be available
  if (!action_client->wait_for_action_server(std::chrono::seconds(5))) {
    RCLCPP_ERROR(LOGGER, "Gripper action server not available after waiting");
    executor.cancel();
    executor_thread.join();
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
  
  RCLCPP_INFO(LOGGER, "Starting closed-loop grasping with %lu discrete steps", joint_angle_range.size());

  const auto wait_for_finger_joint_position = [&gripper_interface](double target_position,
                                                            double epsilon,
                                                            std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;

    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline)
    {
      auto current_state_ptr = gripper_interface.getCurrentState(0.1);
      if (current_state_ptr)
      {
        const double current_position = current_state_ptr->getVariablePosition("finger_joint");
        if (std::isfinite(current_position) && std::abs(current_position - target_position) <= epsilon)
        {
          return true;
        }
      }

      std::this_thread::sleep_for(10ms);
    }

    return false;
  };
  
  // Execute closed-loop grasping loop
  for (size_t step = 0; step < joint_angle_range.size(); ++step) {
    // sleep for a short duration to allow any previous action to start processing
    // std::this_thread::sleep_for(std::chrono::milliseconds(5));
    float current_angle = joint_angle_range[step];
    RCLCPP_INFO(LOGGER, "Grasping step %lu/%lu: sending joint angle = %.4f", 
                step + 1, joint_angle_range.size(), current_angle);
    
    // Reset detection flags for this step
    object_detected_during_grasp->store(false);
    tactile_detected->store(false);
    
    // Create and send goal
    auto goal_msg = GripperCommand::Goal();
    goal_msg.command.position = current_angle;
    goal_msg.command.max_effort = 0.1;  // Target force in Newtons
    
    // Send goal asynchronously and get the future
    auto goal_handle_future = action_client->async_send_goal(goal_msg);
    
    // Wait for goal handle to be ready
    auto goal_handle = goal_handle_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(LOGGER, "Failed to send goal at step %lu", step + 1);
      break;
    }
    
    bool cancelled = false;
    const auto step_timeout = std::chrono::seconds(3);
    const auto step_deadline = std::chrono::steady_clock::now() + step_timeout;
    bool reached_target = false;
    while (rclcpp::ok() && std::chrono::steady_clock::now() < step_deadline)
    {
        if (tactile_stop_requested->load())
        {
            action_client->async_cancel_goal(goal_handle);
            cancelled = true;
            break;
        }

        if (object_detected_during_grasp->load()) 
        {
            action_client->async_cancel_goal(goal_handle);
            cancelled = true;
            break;
        }
      
        if (wait_for_finger_joint_position(goal_msg.command.position, 0.01, 100ms))
        {
            reached_target = true;
            break;
        }
    }

    auto current_state_ptr = gripper_interface.getCurrentState(0.1);
    if (current_state_ptr)
    {
      const double current_position = current_state_ptr->getVariablePosition("finger_joint");
      RCLCPP_INFO(LOGGER, "Current finger_joint value: %.4f, desired: %.4f",
                  current_position, goal_msg.command.position);
    }

    if (cancelled)
    {
      RCLCPP_INFO(LOGGER, "Gripper goal cancelled at step %lu", step + 1);
    }
    else if (!reached_target)
    {
      RCLCPP_WARN(LOGGER, "Timed out waiting for finger_joint to reach %.4f at step %lu",
                  goal_msg.command.position, step + 1);
      break;
    }

    RCLCPP_INFO(LOGGER, "Gripper step %lu completed", step + 1);
    
    // Check for paired tactile contact (both sensors)
    if (tactile_stop_requested->load()) {
      tactile_detected->store(true);
      RCLCPP_INFO(LOGGER, "Tactile detection at step %lu! At least one sensor detected contact - exiting loop.", step + 1);
      break;
    }
    // Check if object was detected during this step
    if (object_detected_during_grasp->load()) {
      RCLCPP_INFO(LOGGER, "Object detected at step %lu! Grasp successful - exiting loop.", step + 1);
      break;
    } else if (step + 1 == joint_angle_range.size()) {
      RCLCPP_INFO(LOGGER, "Reached end of joint angle range (step %lu). Grasping complete.", step + 1);
    } else {
      RCLCPP_INFO(LOGGER, "No object detected at step %lu. Continuing to next step.", step + 1);
    }
  }
  
  RCLCPP_INFO(LOGGER, "Closed-loop grasping finished. Object secured: %s, Tactile contact: %s", 
              object_detected_during_grasp->load() ? "yes" : "no",
              tactile_detected->load() ? "yes" : "no");

  (void)dynamic_joint_state_sub;
  (void)alfredo_sub;
  (void)giovanni_sub;

  // Kill executor thread before shutdown
  executor.cancel();
  executor_thread.join();

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}