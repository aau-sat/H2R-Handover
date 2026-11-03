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

#include <moveit_servo/servo.h>
#include <moveit_servo/pose_tracking.h>
#include <moveit_servo/status_codes.h>
#include <moveit_servo/servo_parameters.h>
#include <moveit_servo/make_shared_from_pool.h>
#include <thread>
#include <chrono> // Added for time literals

#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/move_group_interface/move_group_interface.h>

// --- ADDED FOR CONTROLLER SWITCHING ---
#include <controller_manager_msgs/srv/switch_controller.hpp>
// --------------------------------------

static const rclcpp::Logger LOGGER = rclcpp::get_logger("moveit_servo_tracking_test");

// Class for monitoring status of moveit_servo
class StatusMonitor
{
public:
  StatusMonitor(const rclcpp::Node::SharedPtr& node, const std::string& topic)
  {
    sub_ = node->create_subscription<std_msgs::msg::Int8>(topic, rclcpp::SystemDefaultsQoS(),
                                                          [this](const std_msgs::msg::Int8::ConstSharedPtr& msg) {
                                                            return statusCB(msg);
                                                          });
  }

private:
  void statusCB(const std_msgs::msg::Int8::ConstSharedPtr& msg)
  {
    moveit_servo::StatusCode latest_status = static_cast<moveit_servo::StatusCode>(msg->data);
    if (latest_status != status_)
    {
      status_ = latest_status;
      const auto& status_str = moveit_servo::SERVO_STATUS_CODE_MAP.at(status_);
      RCLCPP_INFO_STREAM(LOGGER, "Servo status: " << status_str);
    }
  }

  moveit_servo::StatusCode status_ = moveit_servo::StatusCode::INVALID;
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr sub_;
};

bool switch_controllers(rclcpp::Node::SharedPtr node)
{
  auto switch_client = node->create_client<controller_manager_msgs::srv::SwitchController>(
      "/controller_manager/switch_controller");

  // Wait for the service
  while (!switch_client->wait_for_service(std::chrono::seconds(2))) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(LOGGER, "Interrupted while waiting for switch_controller service.");
      return false;
    }
    RCLCPP_INFO(LOGGER, "Waiting for /controller_manager/switch_controller service...");
  }

  // Populate the request
  auto request = std::make_shared<controller_manager_msgs::srv::SwitchController::Request>();
  request->deactivate_controllers = {"scaled_joint_trajectory_controller"};
  request->activate_controllers = {"forward_position_controller"};
  request->strictness = controller_manager_msgs::srv::SwitchController::Request::STRICT;
  request->timeout.sec = 5;

  // Send request asynchronously
  auto result_future = switch_client->async_send_request(request);

  // Wait for the background executor to process the response
  if (result_future.wait_for(std::chrono::seconds(6)) == std::future_status::ready) {
    auto response = result_future.get();
    if (response->ok) {
      RCLCPP_INFO(LOGGER, "Controllers switched successfully!");
      return true;
    } else {
      RCLCPP_ERROR(LOGGER, "Controller manager rejected the switch request.");
      return false;
    }
  } else {
    RCLCPP_ERROR(LOGGER, "Failed to call switch_controller service (timeout).");
    return false;
  }
}
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

  auto servo_parameters = moveit_servo::ServoParameters::makeServoParameters(node);
  if (servo_parameters == nullptr)
  {
    RCLCPP_FATAL(LOGGER, "Could not get servo parameters!");
    exit(EXIT_FAILURE);
  }

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
  planning_scene_monitor->startStateMonitor(servo_parameters->joint_topic);
  planning_scene_monitor->startPublishingPlanningScene(planning_scene_monitor::PlanningSceneMonitor::UPDATE_SCENE);

  // Wait for Planning Scene Monitor to setup
  if (!planning_scene_monitor->waitForCurrentRobotState(node->now(), 5.0 /* seconds */))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Error waiting for current robot state in PlanningSceneMonitor.");
    exit(EXIT_FAILURE);
  }

  move_group.setStartStateToCurrentState();

  RCLCPP_INFO(LOGGER, "Attempting to switch controllers before starting movement...");
  if (!switch_controllers(node))
  {
    RCLCPP_ERROR(LOGGER, "Aborting execution because controller switch failed.");
    executor.cancel();
    executor_thread.join();
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  // Create the pose tracker
  moveit_servo::PoseTracking tracker(node, servo_parameters, planning_scene_monitor);

  // Make a publisher for sending pose commands
  auto target_pose_pub =
      node->create_publisher<geometry_msgs::msg::PoseStamped>("target_pose", rclcpp::SystemDefaultsQoS());

  // Subscribe to servo status (and log it when it changes)
  StatusMonitor status_monitor(node, servo_parameters->status_topic);

  Eigen::Vector3d lin_tol{ 0.001, 0.001, 0.001 };
  double rot_tol = 0.01;

  // Get the current EE transform from world to grasp_center using TF2
  geometry_msgs::msg::TransformStamped current_ee_tf;
  
  // The transform from the MoveIt planning frame to robot_link_command_frame
  tracker.getCommandFrameTransform(current_ee_tf);

  // Convert it to a Pose
  geometry_msgs::msg::PoseStamped target_pose;
  target_pose.header.frame_id = current_ee_tf.header.frame_id;  
  target_pose.pose.position.x = current_ee_tf.transform.translation.x;
  target_pose.pose.position.y = current_ee_tf.transform.translation.y;
  target_pose.pose.position.z = current_ee_tf.transform.translation.z;
  target_pose.pose.orientation = current_ee_tf.transform.rotation;

  RCLCPP_INFO_STREAM(LOGGER, "Publishing target pose in frame: " << target_pose.header.frame_id);
  RCLCPP_INFO_STREAM(LOGGER, "Target Position: [" << target_pose.pose.position.x << ", " 
                                                    << target_pose.pose.position.y << ", " 
                                                    << target_pose.pose.position.z << "]");

  tracker.resetTargetPose();

  // Publish target pose
  target_pose.header.stamp = node->now();
  target_pose_pub->publish(target_pose);

  // visualize the target pose in RViz
  std::string pose_name = "pose1";
  visual_tools.publishAxisLabeled(target_pose.pose, pose_name);
  visual_tools.publishText(target_pose.pose, "Pose_Goal", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();

  // Run the pose tracking in a new thread
  std::thread move_to_pose_thread([&tracker, &lin_tol, &rot_tol] {
    moveit_servo::PoseTrackingStatusCode tracking_status =
        tracker.moveToPose(lin_tol, rot_tol, 0.1 /* target pose timeout */);
    RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
                                   << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
  });

  float radius = 0.2;
  float angular_speed = 0.1;  // radians per second
  float angle = 0.0;

  rclcpp::WallRate loop_rate(100);
  geometry_msgs::msg::PoseStamped circular_pose = target_pose;
  
  for (size_t i = 0; i < 5000; ++i)
  {
    if (i == 100) {
      // We launch this in a separate thread so it doesn't 
      // pause the high-frequency 'target_pose_pub' stream.
      std::thread([&gripper_interface]() {
        gripper_interface.setJointValueTarget({0.8});
        // gripper_group.setNamedTarget("closed"); // or setJointValueTarget
        gripper_interface.move();
      }).detach();
    }

    // visualize in RVIZ
    visual_tools.deleteAllMarkers();
    visual_tools.trigger();
    std::string current_pose_name = "pose" + std::to_string(i);
    visual_tools.publishAxisLabeled(circular_pose.pose, current_pose_name);
    visual_tools.publishText(circular_pose.pose, "Pose_Goal", rvt::WHITE, rvt::XLARGE);
    visual_tools.trigger();

    // make a circle in the air
    angle += angular_speed / 50.0;  // 50 Hz loop rate (though WallRate is set to 10Hz)
    circular_pose.pose.position.x = target_pose.pose.position.x + radius * cos(angle);
    circular_pose.pose.position.y = target_pose.pose.position.y + radius * sin(angle);
    circular_pose.pose.position.z = target_pose.pose.position.z;

    circular_pose.header.stamp = node->now();
    target_pose_pub->publish(circular_pose);

    loop_rate.sleep();
  }

  // Make sure the tracker is stopped and clean up
  move_to_pose_thread.join();

  // Kill executor thread before shutdown
  executor.cancel();
  executor_thread.join();

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}