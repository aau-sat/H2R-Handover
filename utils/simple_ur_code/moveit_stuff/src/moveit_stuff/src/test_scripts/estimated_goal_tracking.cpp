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
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <moveit_servo/servo.h>
#include <moveit_servo/pose_tracking.h>
#include <moveit_servo/status_codes.h>
#include <moveit_servo/servo_parameters.h>
#include <moveit_servo/servo_parameters.h>
#include <moveit_servo/make_shared_from_pool.h>
#include <thread>
#include <mutex>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <controller_manager_msgs/srv/switch_controller.hpp>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("moveit_servo_tracking_test");

// Helper function to convert geometry_msgs::msg::Pose to Eigen::Isometry3d (4x4 transformation matrix)
Eigen::Isometry3d poseToMatrix(const geometry_msgs::msg::Pose& pose)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  T.linear() = Eigen::Quaterniond(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z).toRotationMatrix();
  return T;
}

// Helper function to convert geometry_msgs::msg::Transform to Eigen::Isometry3d
Eigen::Isometry3d transformToMatrix(const geometry_msgs::msg::Transform& transform)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(transform.translation.x, transform.translation.y, transform.translation.z);
  T.linear() = Eigen::Quaterniond(transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z).toRotationMatrix();
  return T;
}

// Helper function to convert Eigen::Isometry3d to geometry_msgs::msg::Pose
geometry_msgs::msg::Pose matrixToPose(const Eigen::Isometry3d& T)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = T.translation().x();
  pose.position.y = T.translation().y();
  pose.position.z = T.translation().z();
  
  Eigen::Quaterniond q(T.linear());
  pose.orientation.w = q.w();
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  return pose;
}

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

// Class for subscribing to goal poses
class GoalPoseSubscriber
{
public:
  GoalPoseSubscriber(
      const rclcpp::Node::SharedPtr& node,
      geometry_msgs::msg::PoseStamped current_pose,
      const std::shared_ptr<tf2_ros::Buffer>& tf_buffer,
      const std::string& topic = "goal_pose",
      const std::string& base_frame = "base",
      const std::string& base_link_frame = "base_link",
      const std::string& command_frame = "tool0",
      const std::string& grasp_frame = "grasp_center"
    )
  {
    received_pose_ = false;
    tf_buffer_ = tf_buffer;
    base_frame_ = base_frame;
    base_link_frame_ = base_link_frame;
    command_frame_ = command_frame;
    grasp_frame_ = grasp_frame;

    sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        topic, rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg) { return goalPoseCB(msg); });
    current_pose_ = current_pose;
  }

  // returns approach position in the planning frame
  void getApproachPose(geometry_msgs::msg::PoseStamped& goal, geometry_msgs::msg::PoseStamped& current_pose, Eigen::Vector3d grasp_center_shift)
  {
    // planning frame is world/base_link
    // latest_pose_ is the goal pose in planning frame
    // current_pose_ is the current EE pose in planning frame
    // get unit vector difference between goal and ee pose, in planning frame
    // subtract 20 cm distance from goal vector to obtain safe approach point
    // by computing x, y, z components with the difference unit vector
    Eigen::Vector3d T_goal = poseToMatrix(goal.pose).translation();
    Eigen::Vector3d T_current = poseToMatrix(current_pose.pose).translation();
    Eigen::Vector3d translation_diff = T_goal - T_current;
    double distance = translation_diff.norm();
    if (distance > 0.0001) // avoid division by zero
    {
      Eigen::Vector3d unit_vector = translation_diff / distance;
      double approach_distance = 0.2; //std::max(0.0, distance - 0.2); // subtract 20 cm, but don't go negative
      approach_distance = approach_distance + grasp_center_shift.norm();
      // RCLCPP_INFO_STREAM(LOGGER, "Distance to goal: " << distance << ", approach distance: " << approach_distance << ", grasp center shift: " << grasp_center_shift.norm());
      Eigen::Vector3d approach_vector = unit_vector * approach_distance;
      Eigen::Vector3d T_approach = T_goal - approach_vector; // approach point is along the line from current to goal, but 20 cm away from the goal
      goal.pose.position.x = T_approach.x();
      goal.pose.position.y = T_approach.y();
      goal.pose.position.z = T_approach.z();
      // Eigen::Isometry3d T_approach = Eigen::Isometry3d::Identity();
      // T_approach.translation() = T_goal.translation() - approach_vector; // approach point is along the line from current to goal, but 20 cm away from the goal
      // return T_approach;
    }
  }

  bool getLatestPose(geometry_msgs::msg::PoseStamped& pose, geometry_msgs::msg::PoseStamped& current_pose)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!received_pose_)
      return false;

    try
    {
        // get transform from grasp_center to tool0 (static transform)
        geometry_msgs::msg::TransformStamped T_grasp_to_tool0_msg = 
            tf_buffer_->lookupTransform(grasp_frame_, command_frame_, rclcpp::Time(0));
        Eigen::Vector3d grasp_center_shift = transformToMatrix(T_grasp_to_tool0_msg.transform).translation();

        // Get transform from base to base_link (dynamic transform)
        // (target_frame, source_frame, time)
        geometry_msgs::msg::TransformStamped T_baselink_to_base_msg = 
            tf_buffer_->lookupTransform(base_link_frame_, base_frame_, rclcpp::Time(0));

        // Convert base->base_link to matrix
        Eigen::Isometry3d T_baselink_to_base_ = transformToMatrix(T_baselink_to_base_msg.transform);

        // Convert incoming pose to transformation matrix
        // T_base is the gripper pose in base frame
        Eigen::Isometry3d T_base = poseToMatrix(latest_pose_.pose);

        // Then: apply base->base_link transform: T_out = T_baselink_to_base @ T_base
        Eigen::Isometry3d T_out = T_baselink_to_base_ * T_base;

        // Convert back to pose, output in base_link frame
        pose.header.frame_id = "world";//base_link_frame_;
        pose.header.stamp = latest_pose_.header.stamp;
        pose.pose = matrixToPose(T_out);
        pose.pose.orientation = current_pose_.pose.orientation;  // Keep original orientation from incoming message
        getApproachPose(pose, current_pose, grasp_center_shift); // modify pose to be the approach pose
        return true;
    }
    catch (const tf2::TransformException& ex)
    {
      RCLCPP_WARN_STREAM(LOGGER, "Transform lookup failed: " << ex.what());
      return false;
    }
  }

  bool hasReceivedPose() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_pose_;
  }

private:
  void goalPoseCB(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // latest_pose_ = *msg; // full update
    latest_pose_.header = msg->header;
    latest_pose_.pose.position = msg->pose.position;
    latest_pose_.pose.orientation = msg->pose.orientation;  // Keep orientation from incoming message
    if (!received_pose_)
    {
      received_pose_ = true;
      RCLCPP_INFO_STREAM(LOGGER, "Received first goal pose in frame: " << msg->header.frame_id);
    }
  }

  mutable std::mutex mutex_;
  geometry_msgs::msg::PoseStamped current_pose_;
  geometry_msgs::msg::PoseStamped latest_pose_;
  bool received_pose_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_static_sub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::string base_frame_;
  std::string base_link_frame_;
  std::string command_frame_;
  std::string grasp_frame_;
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

void add_collision_objects(rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr scene_pub)
{
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = "base_link";
    collision_object.id = "base_box";

    // Make a box and put it in the way
    shape_msgs::msg::SolidPrimitive base;
    base.type = base.BOX;
    base.dimensions = { 1.40, 0.80, 0.9 };
    geometry_msgs::msg::Pose base_pose;
    base_pose.position.x = 0.0;
    base_pose.position.y = -0.17;
    base_pose.position.z = -0.451;
    base_pose.orientation.w = 1.0;
    base_pose.orientation.x = 0.0;
    base_pose.orientation.y = 0.0;
    base_pose.orientation.z = 0.0;

    // Add the box as a collision object
    collision_object.primitives.push_back(base);
    collision_object.primitive_poses.push_back(base_pose);
    collision_object.operation = collision_object.ADD;

    shape_msgs::msg::SolidPrimitive table;
    table.type = table.BOX;
    table.dimensions = { 1.75, 1.0, 1.0 };
    geometry_msgs::msg::Pose table_pose;
    table_pose.position.x = 0.0;
    table_pose.position.y = 0.6799999999999999;
    table_pose.position.z = -0.6957919745062902;
    table_pose.orientation.w = 1.0;
    table_pose.orientation.x = 0.0;
    table_pose.orientation.y = 0.0;
    table_pose.orientation.z = 0.0;

    // Add the table as a collision object
    moveit_msgs::msg::CollisionObject table_object;
    table_object.header.frame_id = "base_link";
    table_object.id = "table";
    table_object.primitives.push_back(table);
    table_object.primitive_poses.push_back(table_pose);
    table_object.operation = table_object.ADD;

    // Create the message to publish the collision object
    moveit_msgs::msg::PlanningSceneWorld psw;
    psw.collision_objects.push_back(collision_object);
    psw.collision_objects.push_back(table_object);
    moveit_msgs::msg::PlanningScene ps;
    ps.is_diff = true;
    ps.world = psw;

    // Publish the collision object to the planning scene
    scene_pub->publish(ps);
}

/**
 * Instantiate the pose tracking interface.
 * Send a pose slightly different from the starting pose
 * Then keep updating the target pose a little bit
 */
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("pose_tracking_test");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread executor_thread([&executor]() { executor.spin(); });

  // visualization
  static const std::string PLANNING_GROUP = "ur_custom_manipulator";
  static const std::string PLANNING_GROUP_GRIPPER = "gripper";
  moveit::planning_interface::MoveGroupInterface move_group(node, PLANNING_GROUP);
  moveit::planning_interface::MoveGroupInterface gripper_interface(node, PLANNING_GROUP_GRIPPER);
  // ompl planner can handle gripper control
  gripper_interface.setPlanningPipelineId("ompl");
  gripper_interface.setPlannerId("RRTConnect");

  // open gripper
  std::thread([&gripper_interface]() {
        gripper_interface.setNamedTarget("open");
        gripper_interface.move();
      }).detach();

  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools(node, "world", "move_group_marker",
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

  auto scene_pub = node->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 10);
  add_collision_objects(scene_pub);

  // Wait for Planning Scene Monitor to setup
  if (!planning_scene_monitor->waitForCurrentRobotState(node->now(), 5.0 /* seconds */))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Error waiting for current robot state in PlanningSceneMonitor.");
    exit(EXIT_FAILURE);
  }

  RCLCPP_INFO(LOGGER, "Attempting to switch controllers before starting movement...");
  if (!switch_controllers(node))
  {
    RCLCPP_ERROR(LOGGER, "Aborting execution because controller switch failed.");
    executor.cancel();
    executor_thread.join();
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }


  // Create the pose tracker
  moveit_servo::PoseTracking tracker(node, servo_parameters, planning_scene_monitor);

  // Get the current EE transform from world to grasp_center using TF2
  geometry_msgs::msg::TransformStamped current_ee_tf;
  // The transform from the MoveIt planning frame to robot_link_command_frame
  tracker.getCommandFrameTransform(current_ee_tf);
  // Convert it to a Pose
  geometry_msgs::msg::PoseStamped current_pose;
  current_pose.header.frame_id = current_ee_tf.header.frame_id;  
  current_pose.pose.position.x = current_ee_tf.transform.translation.x;
  current_pose.pose.position.y = current_ee_tf.transform.translation.y;
  current_pose.pose.position.z = current_ee_tf.transform.translation.z;
  current_pose.pose.orientation = current_ee_tf.transform.rotation;

  // Create TF2 buffer and listener for dynamic frame transformations
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  // Subscribe to goal_pose topic
  GoalPoseSubscriber goal_pose_subscriber(node, current_pose, tf_buffer, "goal_pose");

  // Make a publisher for sending pose commands to the tracker
  auto target_pose_pub =
      node->create_publisher<geometry_msgs::msg::PoseStamped>("target_pose", rclcpp::SystemDefaultsQoS());

  // Subscribe to servo status (and log it when it changes)
  StatusMonitor status_monitor(node, servo_parameters->status_topic);

  Eigen::Vector3d lin_tol{ 0.001, 0.001, 0.001 };
  double rot_tol = 0.01;

  // Wait for the first goal pose to arrive
  RCLCPP_INFO_STREAM(LOGGER, "Waiting for goal_pose topic...");
  rclcpp::WallRate wait_rate(10);
  while (rclcpp::ok() && !goal_pose_subscriber.hasReceivedPose())
  {
    wait_rate.sleep();
  }

  if (!rclcpp::ok())
  {
    RCLCPP_ERROR_STREAM(LOGGER, "ROS shutdown while waiting for goal pose.");
    exit(EXIT_FAILURE);
  }

  RCLCPP_INFO_STREAM(LOGGER, "Goal pose received, starting tracking...");

  // Get initial target pose
  geometry_msgs::msg::PoseStamped target_pose;
  goal_pose_subscriber.getLatestPose(target_pose, current_pose);

  RCLCPP_INFO_STREAM(LOGGER, "Initial target pose in frame: " << target_pose.header.frame_id);
  RCLCPP_INFO_STREAM(LOGGER, "Target Position: [" << target_pose.pose.position.x << ", " 
                                                    << target_pose.pose.position.y << ", " 
                                                    << target_pose.pose.position.z << "]");

  // resetTargetPose() can be used to clear the target pose and wait for a new one, e.g. when moving between multiple
  tracker.resetTargetPose();

  // Publish initial target pose
  target_pose.header.stamp = node->now();

  // visualize the target pose in RViz
  std::string pose_name = "pose0";
  visual_tools.publishAxisLabeled(target_pose.pose, pose_name);
  visual_tools.publishText(target_pose.pose, "Pose_Goal", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();

  // Run the pose tracking in a new thread
  moveit_servo::PoseTrackingStatusCode tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
  std::thread move_to_pose_thread([&tracker, &lin_tol, &rot_tol, &tracking_status] {
    tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1 /* target pose timeout */);
    RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
                                   << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
  });

  rclcpp::WallRate loop_rate(50);
  geometry_msgs::msg::PoseStamped current_goal_pose = target_pose;
  int pose_index = 0;

  // Main loop: track incoming goal poses in real time
  while (rclcpp::ok())
  {
    if (tracking_status != moveit_servo::PoseTrackingStatusCode::INVALID)
    {
      // TODO: check the actual reason why thread has exited
      tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
      
      // Join the previous thread before starting a new one
      if (move_to_pose_thread.joinable())
      {
        move_to_pose_thread.join();
      }
      tracker.resetTargetPose();
      
      // Restart the tracker with a new thread
      move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status] {
        tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1 /* target pose timeout */);
        RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
                                   << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
      });
    }

    tracker.getCommandFrameTransform(current_ee_tf);
    geometry_msgs::msg::PoseStamped current_pose;
    current_pose.header.frame_id = current_ee_tf.header.frame_id;  
    current_pose.pose.position.x = current_ee_tf.transform.translation.x;
    current_pose.pose.position.y = current_ee_tf.transform.translation.y;
    current_pose.pose.position.z = current_ee_tf.transform.translation.z;
    current_pose.pose.orientation = current_ee_tf.transform.rotation;

    // Get the latest goal pose from the subscriber
    geometry_msgs::msg::PoseStamped new_goal_pose;
    if (goal_pose_subscriber.getLatestPose(new_goal_pose, current_pose))
    {
      // Check if the pose has changed significantly
      double position_diff = sqrt(pow(new_goal_pose.pose.position.x - current_goal_pose.pose.position.x, 2) +
                                  pow(new_goal_pose.pose.position.y - current_goal_pose.pose.position.y, 2) +
                                  pow(new_goal_pose.pose.position.z - current_goal_pose.pose.position.z, 2));

      if (position_diff > 0.001)  // Only update if position changed by more than 1 mm (tune as needed)
      {
        tracker.resetTargetPose();
        current_goal_pose = new_goal_pose;
        current_goal_pose.header.stamp = node->now();

        // Visualize in RVIZ
        visual_tools.deleteAllMarkers();
        visual_tools.trigger();
        pose_name = "pose" + std::to_string(pose_index);
        visual_tools.publishAxisLabeled(current_goal_pose.pose, pose_name);
        visual_tools.publishText(current_goal_pose.pose, "Pose_Goal", rvt::WHITE, rvt::XLARGE);
        visual_tools.trigger();

        // RCLCPP_INFO_STREAM(LOGGER, "Updated goal pose [" << pose_index << "]: ["
        //                                                  << current_goal_pose.pose.position.x << ", "
        //                                                  << current_goal_pose.pose.position.y << ", "
        //                                                  << current_goal_pose.pose.position.z << "]");
        pose_index++;
      }
    }
    target_pose_pub->publish(current_goal_pose);

    loop_rate.sleep();
  }

  move_to_pose_thread.join();

  executor.cancel();
  executor_thread.join();

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}