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
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <moveit_servo/servo.h>
#include <moveit_servo/pose_tracking.h>
#include <moveit_servo/status_codes.h>
#include <moveit_servo/servo_parameters.h>
#include <moveit_servo/make_shared_from_pool.h>
#include <thread>
#include <mutex>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <controller_manager_msgs/srv/switch_controller.hpp>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("generated_goal_tracking_test");

bool evaluate_pose_manipulability(const moveit::planning_interface::MoveGroupInterface& move_group,
                                  const geometry_msgs::msg::PoseStamped& target_pose,
                                  double& manipulability,
                                  double& min_singular_value)
{
  manipulability = 0.0;
  min_singular_value = 0.0;

  auto current_state_ptr = move_group.getCurrentState(1.0);
  if (!current_state_ptr)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[PoseQuality] Could not get current robot state");
    return false;
  }

  moveit::core::RobotState ik_state(*current_state_ptr);
  const moveit::core::JointModelGroup* jmg = ik_state.getJointModelGroup(move_group.getName());
  if (!jmg)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[PoseQuality] Joint model group not found: " << move_group.getName());
    return false;
  }

  std::string eef_link = move_group.getEndEffectorLink();
  if (eef_link.empty())
  {
    const auto& link_names = jmg->getLinkModelNames();
    if (link_names.empty())
    {
      RCLCPP_WARN_STREAM(LOGGER, "[PoseQuality] No link names found in joint model group");
      return false;
    }
    eef_link = link_names.back();
  }

  const bool ik_found = ik_state.setFromIK(jmg, target_pose.pose, eef_link, 0.05);
  if (!ik_found)
  {
    return false;
  }

  const auto* eef_link_model = ik_state.getLinkModel(eef_link);
  if (!eef_link_model)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[PoseQuality] Could not get link model for end effector: " << eef_link);
    return false;
  }

  const Eigen::Vector3d reference_point_position = Eigen::Vector3d::Zero();
  Eigen::MatrixXd jacobian;
  ik_state.getJacobian(jmg, eef_link_model, reference_point_position, jacobian);

  if (jacobian.rows() == 0 || jacobian.cols() == 0)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[PoseQuality] Empty Jacobian for IK solution");
    return false;
  }

  const Eigen::MatrixXd jj_t = jacobian * jacobian.transpose();
  const double det = std::max(0.0, jj_t.determinant());
  manipulability = std::sqrt(det);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (svd.singularValues().size() > 0)
  {
    min_singular_value = svd.singularValues().minCoeff();
  }

  return true;
}

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

// Class for subscribing to goal poses and handling TF transformations
class GoalPoseSubscriber
{
public:
  GoalPoseSubscriber(const rclcpp::Node::SharedPtr& node,
                     const std::shared_ptr<tf2_ros::Buffer>& tf_buffer,
                     const std::string& gripper_pose_topic = "gripper_goal_pose",
                     const std::string& base_frame = "base",
                     const std::string& base_link_frame = "base_link",
                     const std::string& camera_frame = "right_color_optical_frame")
  {
    node_ = node;
    tf_buffer_ = tf_buffer;
    base_frame_ = base_frame;
    base_link_frame_ = base_link_frame;
    camera_frame_ = camera_frame;
    T_base_to_cam_received_ = false;
    received_pose_ = false;
    
    // Subscribe to goal poses (gripper poses in camera frame)
    gripper_pose_sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        gripper_pose_topic, rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg) { return onPose(msg); });
    
    // Subscribe to static TF messages to get base->camera transform
    tf_static_sub_ = node->create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf_static", rclcpp::SystemDefaultsQoS(),
        [this](const tf2_msgs::msg::TFMessage::ConstSharedPtr& msg) { return onTFStatic(msg); });
  }

  bool getLatestPose(geometry_msgs::msg::PoseStamped& pose, const geometry_msgs::msg::PoseStamped& current_pose)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!received_pose_)
    {
      RCLCPP_WARN_STREAM(LOGGER, "Gripper pose not received yet");
      return false;
    }

    if (!T_base_to_cam_received_)
    {
      RCLCPP_WARN_STREAM(LOGGER, "Static TF not received yet");
      return false;
    }

    try
    {
      // Get transform from base to base_link (dynamic transform)
      // (target_frame, source_frame, time)
      geometry_msgs::msg::TransformStamped T_baselink_to_base_msg = 
          tf_buffer_->lookupTransform(base_link_frame_, base_frame_, rclcpp::Time(0));
      
      // Convert base->base_link to matrix
      Eigen::Isometry3d T_baselink_to_base_ = transformToMatrix(T_baselink_to_base_msg.transform);
      
      // Convert incoming pose to transformation matrix
      // T_in is the gripper pose in camera frame
      Eigen::Isometry3d T_in = poseToMatrix(latest_pose_.pose);
      
      // Transform chain: camera_optical_frame -> base -> base_link
      // First: apply base->camera transform: T_base = T_base_to_cam @ T_in
      Eigen::Isometry3d T_base;
      if (latest_pose_.header.frame_id == camera_frame_)
      {
        T_base = T_base_to_cam_ * T_in;
      }
      else if (latest_pose_.header.frame_id == base_frame_)
      {
        T_base = T_in;  // Already in base frame
        RCLCPP_INFO_STREAM(LOGGER, "Received pose already in base frame, skipping camera->base transform");
      }
      else
      {
        RCLCPP_WARN_STREAM(LOGGER, "Received pose in unexpected frame: " << latest_pose_.header.frame_id);
        return false;
      }
      
      // Then: apply base->base_link transform: T_out = T_baselink_to_base @ T_base
      Eigen::Isometry3d T_out = T_baselink_to_base_ * T_base;
      
      // Apply gripper offset (0.25m in Z direction)
      // T_gripper_offset = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0.25; 0, 0, 0, 1]
      Eigen::Isometry3d T_gripper_offset = Eigen::Isometry3d::Identity();
      // T_gripper_offset.translation().z() = 0.25;  // 25 cm offset
      
      // // Apply gripper offset: T_final = T_out @ T_gripper_offset
      // T_out = T_out * T_gripper_offset;
      
      // Convert back to pose, output in base_link frame
      pose.header.frame_id = base_link_frame_;
      pose.header.stamp = latest_pose_.header.stamp;
      pose.pose = matrixToPose(T_out);

      // Compare current and goal end-effector Y-axis up/down direction in the same frame.
      // If one points up (+Z) and the other down (-Z), flip goal by 180 degrees around Z.
      Eigen::Quaterniond q_current(current_pose.pose.orientation.w,
                                   current_pose.pose.orientation.x,
                                   current_pose.pose.orientation.y,
                                   current_pose.pose.orientation.z);
      Eigen::Quaterniond q_goal(pose.pose.orientation.w,
                                pose.pose.orientation.x,
                                pose.pose.orientation.y,
                                pose.pose.orientation.z);
      q_current.normalize();
      q_goal.normalize();

      const Eigen::Vector3d current_y_axis = q_current * Eigen::Vector3d::UnitY();
      const Eigen::Vector3d goal_y_axis = q_goal * Eigen::Vector3d::UnitY();

      const double current_y_z = current_y_axis.dot(Eigen::Vector3d::UnitZ());
      const double goal_y_z = goal_y_axis.dot(Eigen::Vector3d::UnitZ());
      const double updown_eps = 1e-4;

      // Strictly opposite up/down direction: one positive and one negative along world Z.
      if ((std::abs(current_y_z) > updown_eps) &&
          (std::abs(goal_y_z) > updown_eps) &&
          ((current_y_z > 0.0 && goal_y_z < 0.0) || (current_y_z < 0.0 && goal_y_z > 0.0)))
      {
        RCLCPP_INFO_STREAM(LOGGER, "Flipping goal pose by 180 degrees around Z-axis to match Y-axis up/down direction");
        const Eigen::AngleAxisd flip_rotation(M_PI, Eigen::Vector3d::UnitZ());
        const Eigen::Quaterniond q_flip(flip_rotation);
        const Eigen::Quaterniond q_flipped = q_goal * q_flip;

        pose.pose.orientation.w = q_flipped.w();
        pose.pose.orientation.x = q_flipped.x();
        pose.pose.orientation.y = q_flipped.y();
        pose.pose.orientation.z = q_flipped.z();
      }

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

  bool hasReceivedTransform() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return T_base_to_cam_received_;
  }

private:
  void onTFStatic(const tf2_msgs::msg::TFMessage::ConstSharedPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (T_base_to_cam_received_)
      return;  // Already received the transform

    for (const auto& t : msg->transforms)
    {
      // RCLCPP_INFO_STREAM(LOGGER, "Received static TF: " << t.header.frame_id << " → " << t.child_frame_id);
      // Match Python code: find where header.frame_id == base_frame AND child_frame_id == camera_frame
      if (t.header.frame_id == base_frame_ && t.child_frame_id == camera_frame_)
      {
        T_base_to_cam_ = transformToMatrix(t.transform);
        T_base_to_cam_received_ = true;
        
        RCLCPP_INFO_STREAM(LOGGER, "Received static TF " << base_frame_ << " → " << camera_frame_);
        RCLCPP_INFO_STREAM(LOGGER, "Transform translation: [" 
                                   << t.transform.translation.x << ", "
                                   << t.transform.translation.y << ", "
                                   << t.transform.translation.z << "]");
        break;
      }
    }
  }

  void onPose(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!T_base_to_cam_received_)
    {
      RCLCPP_WARN_STREAM(LOGGER, "Static TF not received yet");
      return;
    }

    latest_pose_ = *msg;
    received_pose_ = true;
  }

  mutable std::mutex mutex_;
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::string base_frame_;
  std::string base_link_frame_;
  std::string camera_frame_;
  Eigen::Isometry3d T_base_to_cam_ = Eigen::Isometry3d::Identity();
  geometry_msgs::msg::PoseStamped latest_pose_;
  bool T_base_to_cam_received_;
  bool received_pose_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gripper_pose_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_static_sub_;
};

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

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread executor_thread([&executor]() { executor.spin(); });

  // visualization
  static const std::string PLANNING_GROUP_ARM = "ur_custom_manipulator";
  static const std::string PLANNING_GROUP_GRIPPER = "gripper";
  moveit::planning_interface::MoveGroupInterface move_group(node, PLANNING_GROUP_ARM);
  moveit::planning_interface::MoveGroupInterface gripper_interface(node, PLANNING_GROUP_GRIPPER);

  // ompl planner can handle gripper control
  gripper_interface.setPlanningPipelineId("ompl");
  gripper_interface.setPlannerId("RRTConnect");

  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools(node, "base_link", "move_group_marker",
                                                      move_group.getRobotModel());
  visual_tools.deleteAllMarkers();

  auto servo_parameters = moveit_servo::ServoParameters::makeServoParameters(node);

  if (servo_parameters == nullptr)
  {
    RCLCPP_FATAL(LOGGER, "Could not get servo parameters!");
    exit(EXIT_FAILURE);
  }

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
  planning_scene_monitor->startStateMonitor(servo_parameters->joint_topic);
  // planning_scene_monitor->setPlanningScenePublishingFrequency(25);
  planning_scene_monitor->startPublishingPlanningScene(planning_scene_monitor::PlanningSceneMonitor::UPDATE_SCENE);

  // Wait for Planning Scene Monitor to setup
  if (!planning_scene_monitor->waitForCurrentRobotState(node->now(), 5.0 /* seconds */))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Error waiting for current robot state in PlanningSceneMonitor.");
    exit(EXIT_FAILURE);
  }

  auto scene_pub = node->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 10);
  add_collision_objects(scene_pub);

  RCLCPP_INFO(LOGGER, "Attempting to switch controllers before starting movement...");
  if (!switch_controllers(node))
  {
    RCLCPP_WARN(LOGGER, "Couldn't switch controllers.");
    // RCLCPP_ERROR(LOGGER, "Aborting execution because controller switch failed.");
    // executor.cancel();
    // executor_thread.join();
    // rclcpp::shutdown();
    // return EXIT_FAILURE;
  }

  move_group.setStartStateToCurrentState();

  // Create the pose tracker
  moveit_servo::PoseTracking tracker(node, servo_parameters, planning_scene_monitor);

  // Create TF2 buffer and listener for dynamic frame transformations
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  // Subscribe to gripper goal poses and handle TF transformations
  // The class will listen to /tf_static for base->camera and query TF2 for base->base_link
  GoalPoseSubscriber goal_pose_subscriber(node, tf_buffer, "gripper_goal_pose", 
                                         "base", "base_link", "right_color_optical_frame");

  // Make a publisher for sending pose commands to the tracker
  auto target_pose_pub =
      node->create_publisher<geometry_msgs::msg::PoseStamped>("target_pose", rclcpp::SystemDefaultsQoS());

  // Subscribe to servo status (and log it when it changes)
  StatusMonitor status_monitor(node, servo_parameters->status_topic);

  Eigen::Vector3d lin_tol{ 0.001, 0.001, 0.001 };
  double rot_tol = 0.01;

  // Wait for the first goal pose and static TF to arrive
  RCLCPP_INFO_STREAM(LOGGER, "Waiting for gripper_goal_pose topic and static TF transform...");
  rclcpp::WallRate wait_rate(10);
  while (rclcpp::ok() && (!goal_pose_subscriber.hasReceivedPose() || !goal_pose_subscriber.hasReceivedTransform()))
  {
    wait_rate.sleep();
  }

  if (!rclcpp::ok())
  {
    RCLCPP_ERROR_STREAM(LOGGER, "ROS shutdown while waiting for goal pose or transform.");
    exit(EXIT_FAILURE);
  }

  RCLCPP_INFO_STREAM(LOGGER, "Gripper goal pose and static TF received, starting tracking...");

  auto log_pose_quality = [&move_group](const geometry_msgs::msg::PoseStamped& pose_msg) {
    double manipulability = 0.0;
    double min_singular_value = 0.0;
    const bool ik_ok = evaluate_pose_manipulability(move_group, pose_msg, manipulability, min_singular_value);

    if (!ik_ok)
    {
      RCLCPP_WARN_STREAM(LOGGER, "[PoseQuality] INVALID (IK failed) for received grasp pose");
      return;
    }

    // Heuristic thresholds (robot/config dependent):
    // - BAD: close to singularity or very low dexterity
    // - MARGINAL: usable but limited dexterity
    // - GOOD: comfortable dexterity margin
    static constexpr double kBadManipulability = 0.005;
    static constexpr double kBadMinSingularValue = 0.03;
    static constexpr double kGoodManipulability = 0.02;
    static constexpr double kGoodMinSingularValue = 0.08;

    std::string quality_label = "MARGINAL";
    if (manipulability <= kBadManipulability || min_singular_value <= kBadMinSingularValue)
    {
      quality_label = "BAD";
    }
    else if (manipulability >= kGoodManipulability && min_singular_value >= kGoodMinSingularValue)
    {
      quality_label = "GOOD";
    }

    RCLCPP_INFO_STREAM(LOGGER,
                       "[PoseQuality] " << quality_label
                       << " | IK=OK, manipulability=" << manipulability
                       << ", min_singular_value=" << min_singular_value);
  };

  // Get initial target pose
  geometry_msgs::msg::PoseStamped target_pose;
  geometry_msgs::msg::PoseStamped current_pose = move_group.getCurrentPose();
  goal_pose_subscriber.getLatestPose(target_pose, current_pose);
  rclcpp::Time last_goal_stamp = target_pose.header.stamp;

  log_pose_quality(target_pose);

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
  std::thread move_to_pose_thread([&tracker, &lin_tol, &rot_tol] {
    moveit_servo::PoseTrackingStatusCode tracking_status =
        tracker.moveToPose(lin_tol, rot_tol, 0.1 /* target pose timeout */);
    RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
                                   << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
  });

  std::thread([&gripper_interface]() {
        gripper_interface.setJointValueTarget({0.8});
        // gripper_group.setNamedTarget("closed"); // or setJointValueTarget
        gripper_interface.move();
      }).detach();

  target_pose_pub->publish(target_pose);

  rclcpp::WallRate loop_rate(100);

  // Main loop: track incoming goal poses in real time
  while (rclcpp::ok())
  {
    geometry_msgs::msg::PoseStamped latest_goal_pose;
    geometry_msgs::msg::PoseStamped current_pose_loop = move_group.getCurrentPose();
    if (goal_pose_subscriber.getLatestPose(latest_goal_pose, current_pose_loop))
    {
      if (latest_goal_pose.header.stamp != last_goal_stamp)
      {
        target_pose = latest_goal_pose;
        last_goal_stamp = latest_goal_pose.header.stamp;
        log_pose_quality(target_pose);
      }
    }

    target_pose.header.stamp = node->now();
    target_pose_pub->publish(target_pose);
    
    loop_rate.sleep();
  }

  move_to_pose_thread.join();

  executor.cancel();
  executor_thread.join();

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}