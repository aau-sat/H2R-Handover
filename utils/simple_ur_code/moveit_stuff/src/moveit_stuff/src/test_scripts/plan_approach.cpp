
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/bool.hpp>
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
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("plan_approach_node");

// Configuration parameters for handover reaction
struct HandoverConfig
{
  double approach_distance = 0.30;        // 20 cm from object for initial approach
  double tracking_distance = 0.20;        // 10 cm from object for servo tracking
  int velocity_estimation_frames = 5;     // Number of frames to estimate velocity
  int prediction_frames = 3;              // Predict object pose for N frames ahead
};


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

// Class for subscribing to goal poses and estimating object velocity
class GoalPoseSubscriber
{
public:
  GoalPoseSubscriber(
      const rclcpp::Node::SharedPtr& node,
      geometry_msgs::msg::PoseStamped current_pose,
      const std::shared_ptr<tf2_ros::Buffer>& tf_buffer,
      const HandoverConfig& config,
      const std::string& topic = "goal_pose",
      const std::string& base_frame = "base",
      const std::string& base_link_frame = "base_link",
      const std::string& command_frame = "tool0",
      const std::string& grasp_frame = "grasp_center"
    )
  {
    received_pose_ = false;
    handover_detected_ = false;
    tf_buffer_ = tf_buffer;
    base_frame_ = base_frame;
    base_link_frame_ = base_link_frame;
    command_frame_ = command_frame;
    grasp_frame_ = grasp_frame;
    config_ = config;
    start_approach_ = false;
    last_pose_time_ = std::chrono::steady_clock::now();

    handover_sub_ = node->create_subscription<std_msgs::msg::Bool>("/handover_detected", rclcpp::SystemDefaultsQoS(),
        [this](const std_msgs::msg::Bool::ConstSharedPtr& msg) {
          return handoverCB(msg);
        });
    pose_sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        topic, rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg) { return goalPoseCB(msg); });
    current_pose_ = current_pose;
  }

  void setCurrentPose(const geometry_msgs::msg::PoseStamped& pose)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_pose_ = pose;
  }

  // Internal method: Estimate velocity from stored pose history (assumes mutex is already locked)
  Eigen::Vector3d estimateVelocityInternal()
  {
    if (pose_history_.size() < config_.velocity_estimation_frames)
    {
      return Eigen::Vector3d::Zero();
    }

    // Get vectors at the end of the history
    const auto& oldest_pose = pose_history_.front();
    const auto& newest_pose = pose_history_.back();
    
    Eigen::Vector3d position_diff = 
        Eigen::Vector3d(newest_pose.position.x - oldest_pose.position.x,
                       newest_pose.position.y - oldest_pose.position.y,
                       newest_pose.position.z - oldest_pose.position.z);
    
    double time_diff = (pose_history_.size() - 1) * frame_time_;
    if (time_diff < 0.001) return Eigen::Vector3d::Zero();
    
    Eigen::Vector3d velocity = position_diff / time_diff;
    return velocity;
  }

  // Estimate velocity from stored pose history (thread-safe)
  Eigen::Vector3d estimateVelocity()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return estimateVelocityInternal();
  }

  // Internal method: Predict future object pose based on current velocity (assumes mutex is already locked)
  geometry_msgs::msg::PoseStamped predictFutureObjectPoseInternal(const geometry_msgs::msg::PoseStamped& current_object_pose)
  {
    Eigen::Vector3d velocity = estimateVelocityInternal();
    double prediction_time = config_.prediction_frames * frame_time_;
    
    geometry_msgs::msg::PoseStamped predicted_pose = current_object_pose;
    predicted_pose.pose.position.x += velocity.x() * prediction_time;
    predicted_pose.pose.position.y += velocity.y() * prediction_time;
    predicted_pose.pose.position.z += velocity.z() * prediction_time;
    
    return predicted_pose;
  }

  // Predict future object pose based on current velocity (thread-safe)
  geometry_msgs::msg::PoseStamped predictFutureObjectPose(const geometry_msgs::msg::PoseStamped& current_object_pose)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return predictFutureObjectPoseInternal(current_object_pose);
  }

  void getApproachPose(geometry_msgs::msg::PoseStamped& goal, 
                    const geometry_msgs::msg::PoseStamped& current_pose, 
                    Eigen::Vector3d grasp_center_shift,
                    double approach_distance)
  {
    Eigen::Vector3d T_goal = poseToMatrix(goal.pose).translation();
    Eigen::Vector3d T_current = poseToMatrix(current_pose.pose).translation();
    
    Eigen::Vector3d ray = T_goal - T_current;
    double distance = ray.norm();

    if (distance > 0.0001) 
    {
        Eigen::Vector3d unit_ray = ray / distance;
        
        // We want the physical fingers (grasp_center) to be 'approach_distance' away.
        // Therefore, the flange (tool0) must be 'approach_distance + shift' away.
        double offset_from_target = approach_distance + grasp_center_shift.norm();
        
        Eigen::Vector3d T_flange_target = T_goal - (unit_ray * offset_from_target);

        goal.pose.position.x = T_flange_target.x();
        goal.pose.position.y = T_flange_target.y();
        goal.pose.position.z = T_flange_target.z();
    }
  }
  // returns approach position in the planning frame
  // void getApproachPose(geometry_msgs::msg::PoseStamped& goal, 
  //                     geometry_msgs::msg::PoseStamped& current_pose, 
  //                     Eigen::Vector3d grasp_center_shift,
  //                     double approach_distance)
  // {
  //   // planning frame is world/base_link
  //   // latest_pose_ is the goal pose in planning frame
  //   // current_pose_ is the current EE pose in planning frame
  //   // get unit vector difference between goal and ee pose, in planning frame
  //   // subtract approach_distance from goal vector to obtain safe approach point
  //   Eigen::Vector3d T_goal = poseToMatrix(goal.pose).translation();
  //   Eigen::Vector3d T_current = poseToMatrix(current_pose.pose).translation();
  //   Eigen::Vector3d translation_diff = T_goal - T_current;
  //   double distance = translation_diff.norm();
  //   if (distance > 0.0001) // avoid division by zero
  //   {
  //     Eigen::Vector3d unit_vector = translation_diff / distance;
  //     double total_approach_distance = approach_distance + grasp_center_shift.norm();
  //     Eigen::Vector3d approach_vector = unit_vector * total_approach_distance;
  //     Eigen::Vector3d T_approach = T_goal - approach_vector; // approach point is along the line from current to goal
  //     goal.pose.position.x = T_approach.x();
  //     goal.pose.position.y = T_approach.y();
  //     goal.pose.position.z = T_approach.z();
  //   }
  // }

  bool getApproachGoalPose(geometry_msgs::msg::PoseStamped& pose, 
                          geometry_msgs::msg::PoseStamped& current_pose)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!start_approach_) {
      RCLCPP_INFO_STREAM(LOGGER, "Handover not detected or frame time not estimated yet. Cannot compute approach goal pose.");
      return false;
    }
    try
    {
      // unit vector
        // Eigen::Vector3d grasp_center_shift = Eigen::Vector3d::Zero();
        geometry_msgs::msg::TransformStamped T_grasp_to_tool0_msg = 
            tf_buffer_->lookupTransform(grasp_frame_, command_frame_, rclcpp::Time(0));
        Eigen::Vector3d grasp_center_shift = transformToMatrix(T_grasp_to_tool0_msg.transform).translation();

        // Get transform from base to base_link (dynamic transform)
        geometry_msgs::msg::TransformStamped T_baselink_to_base_msg = 
            tf_buffer_->lookupTransform(base_link_frame_, base_frame_, rclcpp::Time(0));

        // Convert base->base_link to matrix
        Eigen::Isometry3d T_baselink_to_base_ = transformToMatrix(T_baselink_to_base_msg.transform);

        // Predict future object pose based on velocity estimation (use internal method - mutex already locked)
        geometry_msgs::msg::PoseStamped predicted_object_pose = predictFutureObjectPoseInternal(latest_pose_);
        
        // Convert predicted pose to transformation matrix
        Eigen::Isometry3d T_base = poseToMatrix(predicted_object_pose.pose);

        // Apply base->base_link transform
        Eigen::Isometry3d T_out = T_baselink_to_base_ * T_base;

        // Convert back to pose, output in world frame
        pose.header.frame_id = base_link_frame_;
        pose.header.stamp = predicted_object_pose.header.stamp;
        pose.pose = matrixToPose(T_out);
        
        // Cup the height to starting end effector height (cap at maximum, allow going lower)
        // if (pose.pose.position.z > current_pose_.pose.position.z)
        // {
        //     pose.pose.position.z = current_pose_.pose.position.z;
        // }
        
        pose.pose.orientation = current_pose_.pose.orientation;  // Keep original orientation
        
        // Modify pose to be the approach pose (40 cm away, facing object)
        getApproachPose(pose, current_pose, grasp_center_shift, config_.approach_distance); 
        
        // Orient gripper to face the predicted object
        geometry_msgs::msg::PoseStamped transformed_predicted_pose;
        transformed_predicted_pose.pose = matrixToPose(T_out);
        orientGripperTowardObject(pose, transformed_predicted_pose);

        RCLCPP_INFO_STREAM(LOGGER, "Computed approach goal pose based on predicted object pose and velocity estimation.");
        
        return true;
    }
    catch (const tf2::TransformException& ex)
    {
      RCLCPP_WARN_STREAM(LOGGER, "Transform lookup failed: " << ex.what());
      return false;
    }
  }

  bool getTrackingGoalPose(geometry_msgs::msg::PoseStamped& pose, 
                          geometry_msgs::msg::PoseStamped& current_pose)
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
        geometry_msgs::msg::TransformStamped T_baselink_to_base_msg = 
            tf_buffer_->lookupTransform(base_link_frame_, base_frame_, rclcpp::Time(0));

        // Convert base->base_link to matrix
        Eigen::Isometry3d T_baselink_to_base_ = transformToMatrix(T_baselink_to_base_msg.transform);

        // Predict future object pose based on velocity estimation (use internal method - mutex already locked)
        // geometry_msgs::msg::PoseStamped predicted_object_pose = predictFutureObjectPoseInternal(latest_pose_);
        
        // Convert predicted pose to transformation matrix
        Eigen::Isometry3d T_base = poseToMatrix(latest_pose_.pose);

        // Apply base->base_link transform
        Eigen::Isometry3d T_out = T_baselink_to_base_ * T_base;

        // Convert back to pose, output in world frame
        pose.header.frame_id = base_link_frame_;
        pose.header.stamp = latest_pose_.header.stamp;
        pose.pose = matrixToPose(T_out);
        
        // Cup the height to starting end effector height (cap at maximum, allow going lower)
        // if (pose.pose.position.z > current_pose_.pose.position.z)
        // {
        //     pose.pose.position.z = current_pose_.pose.position.z;
        // }
        
        pose.pose.orientation = current_pose_.pose.orientation;  // Keep original orientation
        
        // Modify pose to be the tracking pose (20 cm away, facing object)
        getApproachPose(pose, current_pose, grasp_center_shift, config_.tracking_distance);

        geometry_msgs::msg::PoseStamped transformed_predicted_pose;
        transformed_predicted_pose.pose = matrixToPose(T_out);
        orientGripperTowardObject(pose, transformed_predicted_pose);
        
        return true;
    }
    catch (const tf2::TransformException& ex)
    {
      RCLCPP_WARN_STREAM(LOGGER, "Transform lookup failed: " << ex.what());
      return false;
    }
  }

  bool readyToApproach() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return start_approach_;
  }

  bool hasReceivedPose() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_pose_;
  }

  bool hasHandoverDetected() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return handover_detected_;
  }

private:
  void handoverCB(const std_msgs::msg::Bool::ConstSharedPtr& msg)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (msg->data)
      {
        RCLCPP_INFO_STREAM(LOGGER, "Handover detected! Starting approach to goal pose...");
        handover_detected_ = true;
        handover_sub_.reset();  // No longer need to listen for handover after it's detected
      }
      else
      {
        handover_detected_ = false;
      }
    }
  }

  void goalPoseCB(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    last_pose_time_ = std::chrono::steady_clock::now();
    
    if (frame_time_ == 0.0 && received_pose_)
    {
      auto now = std::chrono::steady_clock::now();
      frame_time_ = std::chrono::duration<double>(now - last_pose_time_).count();
      RCLCPP_INFO_STREAM(LOGGER, "Estimated frame time: " << frame_time_ << " seconds");
    }
    if (!received_pose_)
    {
      received_pose_ = true;
      RCLCPP_INFO_STREAM(LOGGER, "Received first goal pose in frame: " << msg->header.frame_id);
    }

    latest_pose_ = *msg;
    
    // Maintain a history of poses for velocity estimation
    pose_history_.push_back(msg->pose);
    if (pose_history_.size() > static_cast<size_t>(config_.velocity_estimation_frames))
    {
      pose_history_.pop_front();
    }

    if (handover_detected_ && frame_time_ > 0.0)
    {
      start_approach_ = true;
    }
  }

  void orientGripperTowardObject(geometry_msgs::msg::PoseStamped& gripper_pose, 
                               const geometry_msgs::msg::PoseStamped& object_pose)
  {
    Eigen::Vector3d gripper_pos(gripper_pose.pose.position.x, 
                                gripper_pose.pose.position.y, 
                                gripper_pose.pose.position.z);
    Eigen::Vector3d object_pos(object_pose.pose.position.x, 
                               object_pose.pose.position.y, 
                               object_pose.pose.position.z);
    
    Eigen::Vector3d z_axis = (object_pos - gripper_pos).normalized();
    
    // Choose a temporary world "Up" vector. 
    // If the gripper is pointing nearly vertically, use X as the temp reference.
    Eigen::Vector3d temp_up(0, 0, 1);
    if (std::abs(z_axis.dot(temp_up)) > 0.99) {
        temp_up = Eigen::Vector3d(0, 1, 0);
    }

    // Standard Gram-Schmidt/Look-At construction
    Eigen::Vector3d x_axis = temp_up.cross(z_axis).normalized();
    Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
    
    Eigen::Matrix3d rot_matrix;
    rot_matrix.col(0) = x_axis;
    rot_matrix.col(1) = y_axis;
    rot_matrix.col(2) = z_axis; // This ensures Gripper Z points at object
    
    Eigen::Quaterniond q(rot_matrix);
    gripper_pose.pose.orientation.w = q.w();
    gripper_pose.pose.orientation.x = q.x();
    gripper_pose.pose.orientation.y = q.y();
    gripper_pose.pose.orientation.z = q.z();
}

  mutable std::mutex mutex_;
  geometry_msgs::msg::PoseStamped current_pose_;
  geometry_msgs::msg::PoseStamped latest_pose_;
  bool received_pose_;
  bool handover_detected_;
  bool start_approach_;
  std::deque<geometry_msgs::msg::Pose> pose_history_;
  std::chrono::steady_clock::time_point last_pose_time_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr handover_sub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::string base_frame_;
  std::string base_link_frame_;
  std::string command_frame_;
  std::string grasp_frame_;
  HandoverConfig config_;
  double frame_time_ = 0.0;
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

bool planApproach(moveit::planning_interface::MoveGroupInterface& move_group,
                           const moveit::core::JointModelGroup* joint_model_group,
                           const geometry_msgs::msg::PoseStamped& target_pose,
                           moveit::planning_interface::MoveGroupInterface::Plan& plan,
                           moveit_visual_tools::MoveItVisualTools& visual_tools)
{
  RCLCPP_INFO_STREAM(LOGGER, "Planning approach motion with Piltz planner...");
  
  // Configure planner
  move_group.setPoseTarget(target_pose);
  // set high goal tolerance to allow for some error in the approach pose (since we will switch to servo at the end)
  move_group.setGoalPositionTolerance(0.02); // 2 cm position tolerance
  // move_group.setMaxVelocityScalingFactor(0.7);
  // move_group.setMaxAccelerationScalingFactor(0.3);
  
  // Plan motion
  bool success = (move_group.plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
  
  if (success)
  {
    RCLCPP_INFO_STREAM(LOGGER, "Piltz plan found! Executing approach motion...");
    
    // Visualize plan
    namespace rvt = rviz_visual_tools;
    visual_tools.publishAxisLabeled(target_pose.pose, "approach_target");
    visual_tools.publishTrajectoryLine(plan.trajectory_, joint_model_group);
    visual_tools.trigger();
    
    // Execute plan
    bool execute_success = (move_group.execute(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    return execute_success;
  }
  else
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Could not find Piltz plan for approach!");
    return false;
  }
}

// https://groups.google.com/g/moveit-users/c/MOoFxy2exT4
void planCartesianPath(moveit::planning_interface::MoveGroupInterface& move_group,
  float velocity_scaling_factor,
  float acceleration_scaling_factor,
  const geometry_msgs::msg::PoseStamped& goal_pose,
  moveit_visual_tools::MoveItVisualTools& visual_tools,
  const moveit::core::JointModelGroup* joint_model_group)
{
    const double jump_threshold = 0.0;
    const double eef_step = 0.01;

    moveit_msgs::msg::RobotTrajectory trajectory;
    std::vector<geometry_msgs::msg::Pose> waypoints;

    move_group.setPlanningPipelineId("ompl");
    move_group.setPlannerId("RRTConnect");

    waypoints.push_back(goal_pose.pose);

    double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
    RCLCPP_INFO_STREAM(LOGGER, "Cartesian path computed with success rate: " << fraction * 100.0 << "%");

    if (fraction > 0.5)
    {
      // The trajectory needs to be modified so it will include velocities as well.
      // First create a RobotTrajectory object
      robot_trajectory::RobotTrajectory rt(move_group.getCurrentState()->getRobotModel(), "ur_custom_manipulator");
      // Get a RobotTrajectory from trajectory
      rt.setRobotTrajectoryMsg(*move_group.getCurrentState(), trajectory);
      // Create a IterativeParabolicTimeParameterization object
      trajectory_processing::IterativeParabolicTimeParameterization iptp;
      // Compute computeTimeStamps
      bool success = iptp.computeTimeStamps(rt, velocity_scaling_factor, acceleration_scaling_factor);
      RCLCPP_INFO(LOGGER, "Computed time stamp %s", success ? "SUCCEDED" : "FAILED");
      // Get RobotTrajectory_msg from RobotTrajectory
      rt.getRobotTrajectoryMsg(trajectory);
      // Check trajectory_msg for velocities not empty
      if (trajectory.joint_trajectory.points.empty() || trajectory.joint_trajectory.points[0].velocities.empty())
      {
        RCLCPP_ERROR_STREAM(LOGGER, "Cartesian planner: trajectory points do not contain velocity information!");
        return;
      }

      moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
      cartesian_plan.trajectory_ = trajectory;
      RCLCPP_INFO_STREAM(LOGGER, "Cartesian plan found! Executing approach motion...");
      visual_tools.publishTrajectoryLine(cartesian_plan.trajectory_, joint_model_group);
      visual_tools.trigger();
      bool execute_success = (move_group.execute(cartesian_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
      if (!execute_success)
        RCLCPP_ERROR_STREAM(LOGGER, "Failed to execute Cartesian approach motion!");
    }
    move_group.setPlanningPipelineId("pilz_industrial_motion_planner");
    move_group.setPlannerId("LIN");
}

/**
 * Main function: Handover reaction node with two-phase approach
 * Phase 1: Use Piltz planner for initial 40 cm approach motion
 * Phase 2: Switch to MoveIT servo for 20 cm tracking with velocity-based prediction
 */
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.use_intra_process_comms(false);
  rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("handover_reaction_node", node_options);

  // Configuration
  HandoverConfig config;
  config.approach_distance = 0.50;        // 40 cm for Piltz approach
  config.tracking_distance = 0.40;        // 20 cm for servo tracking
  config.velocity_estimation_frames = 5;  // Estimate velocity from 5 frames
  config.prediction_frames = 3;           // Predict 3 frames ahead

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
  if (!planning_scene_monitor->waitForCurrentRobotState(node->now(), 5.0))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Error waiting for current robot state in PlanningSceneMonitor.");
    exit(EXIT_FAILURE);
  }

  auto scene_pub = node->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 10);
  add_collision_objects(scene_pub);

  // Open gripper in background
  std::thread([&gripper_interface]() {
        gripper_interface.setNamedTarget("open");
        gripper_interface.move();
      }).detach();

  // Create TF2 buffer and listener for dynamic frame transformations
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  // Get the current EE transform
  geometry_msgs::msg::TransformStamped current_ee_tf;
  auto current_pose = move_group.getCurrentPose();

  // Subscribe to goal_pose topic and object poses
  GoalPoseSubscriber goal_pose_subscriber(node, current_pose, tf_buffer, config, "goal_pose");

  // Wait for handover detection and first goal pose
  RCLCPP_INFO_STREAM(LOGGER, "Waiting for handover detection...");
  rclcpp::WallRate wait_rate(10);
  while (rclcpp::ok() && !goal_pose_subscriber.hasHandoverDetected())
  {
    wait_rate.sleep();
  }

  if (!rclcpp::ok())
  {
    RCLCPP_ERROR_STREAM(LOGGER, "ROS shutdown while waiting for handover detection.");
    exit(EXIT_FAILURE);
  }

  RCLCPP_INFO_STREAM(LOGGER, "Waiting for the signal to start approach...");
  while (rclcpp::ok() && !goal_pose_subscriber.readyToApproach())
  {
    wait_rate.sleep();
  }

  if (!rclcpp::ok())
  {
    RCLCPP_ERROR_STREAM(LOGGER, "ROS shutdown while waiting for goal pose.");
    exit(EXIT_FAILURE);
  }

  RCLCPP_INFO_STREAM(LOGGER, "Goal pose received! Starting approach phase with planner...");

  // ===== PHASE 1: PILTZ PLANNER APPROACH (40 cm distance) =====

  // Get approach goal pose (40 cm from object)
  geometry_msgs::msg::PoseStamped approach_goal_pose;
  if (!goal_pose_subscriber.getApproachGoalPose(approach_goal_pose, current_pose))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Failed to compute approach goal pose!");
    executor.cancel();
    executor_thread.join();
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }

  RCLCPP_INFO_STREAM(LOGGER, "Approach goal pose: [" 
                        << approach_goal_pose.pose.position.x << ", "
                        << approach_goal_pose.pose.position.y << ", "
                        << approach_goal_pose.pose.position.z << "]");
  RCLCPP_INFO(LOGGER, "End effector link for global planning: %s", move_group.getEndEffectorLink().c_str());

  // Plan and execute approach motion with Piltz
  moveit::planning_interface::MoveGroupInterface::Plan approach_plan;
  move_group.setStartStateToCurrentState();
  float velocity_scaling_factor = 0.5;
  float acceleration_scaling_factor = 0.3;
  
  if (!planApproach(move_group, joint_model_group, approach_goal_pose, approach_plan, visual_tools))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Failed to execute Piltz approach motion!");
    // executor.cancel();
    // executor_thread.join();
    // rclcpp::shutdown();
    // return EXIT_FAILURE;
    RCLCPP_WARN_STREAM(LOGGER, "Trying Cartesian planning...");
    // If Piltz approach fails, try a Cartesian path as a fallback
    planCartesianPath(move_group, velocity_scaling_factor, acceleration_scaling_factor, approach_goal_pose, visual_tools, joint_model_group);
  }
  else {
    RCLCPP_INFO_STREAM(LOGGER, "Piltz approach completed! Switching to servo tracking phase...");
  }

  // Add small delay after Piltz motion
  // std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // ===== PHASE 2: SERVO TRACKING (20 cm distance) =====
  
  // Switch controllers for servo
  RCLCPP_INFO(LOGGER, "Switching controllers to enable MoveIT Servo...");
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

  // Make a publisher for sending pose commands to the tracker
  auto target_pose_pub =
      node->create_publisher<geometry_msgs::msg::PoseStamped>("target_pose", rclcpp::SystemDefaultsQoS());

  // Subscribe to servo status
  StatusMonitor status_monitor(node, servo_parameters->status_topic);

  // Get current EE transform for servo
  tracker.getCommandFrameTransform(current_ee_tf);
  current_pose.header.frame_id = current_ee_tf.header.frame_id;  
  current_pose.pose.position.x = current_ee_tf.transform.translation.x;
  current_pose.pose.position.y = current_ee_tf.transform.translation.y;
  current_pose.pose.position.z = current_ee_tf.transform.translation.z;
  current_pose.pose.orientation = current_ee_tf.transform.rotation;

  goal_pose_subscriber.setCurrentPose(current_pose);

  // Tolerance for servo tracking
  Eigen::Vector3d lin_tol{ 0.002, 0.002, 0.002 };  // 2 mm tolerance
  double rot_tol = 0.01;  // 0.57 degree tolerance

  RCLCPP_INFO_STREAM(LOGGER, "Starting servo pose tracking at 20 cm distance with velocity prediction...");

  // Get initial tracking goal pose (20 cm from object)
  geometry_msgs::msg::PoseStamped target_pose;
  if (!goal_pose_subscriber.getTrackingGoalPose(target_pose, current_pose))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Failed to compute tracking goal pose!");
    executor.cancel();
    executor_thread.join();
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }

  // Initialize tracking
  tracker.resetTargetPose();

  // Publish initial target pose
  target_pose.header.stamp = node->now();

  // Visualize the target pose in RViz
  std::string pose_name = "tracking_pose_0";
  visual_tools.publishAxisLabeled(target_pose.pose, pose_name);
  visual_tools.publishText(target_pose.pose, "Tracking_Goal", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();

  // Run the pose tracking in a new thread
  moveit_servo::PoseTrackingStatusCode tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
  std::thread move_to_pose_thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
    tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
    RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
                                   << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
  });

  rclcpp::WallRate loop_rate(100);  // 100 Hz tracking loop
  geometry_msgs::msg::PoseStamped current_goal_pose = target_pose;
  int pose_index = 0;

  // Main servo tracking loop
  while (rclcpp::ok())
  {
    if (tracking_status != moveit_servo::PoseTrackingStatusCode::INVALID)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Tracking thread exited, restarting...");
      tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
      
      if (move_to_pose_thread.joinable())
      {
        move_to_pose_thread.join();
      }
      tracker.resetTargetPose();
      
      // Restart tracker
      move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
        tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
        RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
                                       << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
      });
    }

    // Get current EE pose
    tracker.getCommandFrameTransform(current_ee_tf);
    current_pose.header.frame_id = current_ee_tf.header.frame_id;  
    current_pose.pose.position.x = current_ee_tf.transform.translation.x;
    current_pose.pose.position.y = current_ee_tf.transform.translation.y;
    current_pose.pose.position.z = current_ee_tf.transform.translation.z;
    current_pose.pose.orientation = current_ee_tf.transform.rotation;

    // Get updated tracking goal pose with predicted object position
    geometry_msgs::msg::PoseStamped new_goal_pose;
    if (goal_pose_subscriber.getTrackingGoalPose(new_goal_pose, current_pose))
    {
      // Check if pose changed significantly
      double position_diff = sqrt(pow(new_goal_pose.pose.position.x - current_goal_pose.pose.position.x, 2) +
                                  pow(new_goal_pose.pose.position.y - current_goal_pose.pose.position.y, 2) +
                                  pow(new_goal_pose.pose.position.z - current_goal_pose.pose.position.z, 2));

      // if (position_diff > 0.01)  // Update if position changed by >5 cm
      // {
      //   tracker.resetTargetPose();
        current_goal_pose = new_goal_pose;
        current_goal_pose.header.stamp = node->now();

        // Visualize updated goal
        visual_tools.deleteAllMarkers();
        visual_tools.trigger();
        pose_name = "tracking_pose_" + std::to_string(pose_index);
        visual_tools.publishAxisLabeled(current_goal_pose.pose, pose_name);
        visual_tools.publishText(current_goal_pose.pose, "Tracking_Goal", rvt::WHITE, rvt::XLARGE);
        visual_tools.trigger();

        pose_index++;
      // }
    }

    // Publish current tracking goal
    target_pose_pub->publish(current_goal_pose);

    loop_rate.sleep();
  }

  move_to_pose_thread.join();

  executor.cancel();
  executor_thread.join();

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
