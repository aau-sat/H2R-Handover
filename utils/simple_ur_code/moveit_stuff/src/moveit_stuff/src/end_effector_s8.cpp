#include <rclcpp/rclcpp.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <future>
#include <algorithm>
#include <cctype>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <deque>
#include <map>
#include <unordered_set>

#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <moveit_servo/servo.h>
#include <moveit_servo/pose_tracking.h>
#include <moveit_servo/status_codes.h>
#include <moveit_servo/servo_parameters.h>
#include <moveit_servo/servo_parameters.h>
#include <moveit_servo/make_shared_from_pool.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <moveit_msgs/srv/get_motion_plan.hpp>
#include <moveit/collision_detection/collision_matrix.h>
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <moveit/robot_state/conversions.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_state/robot_state.h>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("end_effector_s8");

struct ServoExecutionContext
{
  bool enabled = false;
  std::shared_ptr<moveit_servo::PoseTracking> tracker;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_pub;
  rclcpp::Node::SharedPtr node;
  double linear_tolerance_m = 0.001;
  double angular_tolerance_rad = 0.01;
};

struct PoseInterpolationConfig
{
  bool enabled = false;
  int steps = 0;
  std::string profile = "linear";
};

struct MotionRecord
{
  int run_id;
  int repetition;
  int target_id;
  double x_mm;
  double y_mm;
  double z_mm;
  double qx;
  double qy;
  double qz;
  double qw;
  std::string start_time;
  std::string end_time;
  long long motion_time_ms;
};

struct AccuracyRecord
{
  int run_id;
  int repetition;
  int target_id;
  double world_dx_mm;
  double world_dy_mm;
  double world_dz_mm;
  double world_dnorm_mm;
  double world_dang_deg;
  double custom_dx_mm;
  double custom_dy_mm;
  double custom_dz_mm;
  double custom_dnorm_mm;
  double custom_dang_deg;
  std::string start_time;
  std::string end_time;
  long long motion_time_ms;
};

geometry_msgs::msg::Quaternion eigenQuatToMsg(const Eigen::Quaterniond& q)
{
  geometry_msgs::msg::Quaternion q_msg;
  q_msg.w = q.w();
  q_msg.x = q.x();
  q_msg.y = q.y();
  q_msg.z = q.z();
  return q_msg;
}

Eigen::Isometry3d poseMsgToIsometry(const geometry_msgs::msg::Pose& pose)
{
  Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();

  Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  if (q.norm() < 1e-9)
  {
    q = Eigen::Quaterniond::Identity();
  }
  q.normalize();

  iso.linear() = q.toRotationMatrix();
  iso.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  return iso;
}

geometry_msgs::msg::Pose isometryToPoseMsg(const Eigen::Isometry3d& iso)
{
  geometry_msgs::msg::Pose pose;
  Eigen::Quaterniond q(iso.linear());
  q.normalize();

  pose.position.x = iso.translation().x();
  pose.position.y = iso.translation().y();
  pose.position.z = iso.translation().z();
  pose.orientation = eigenQuatToMsg(q);
  return pose;
}

geometry_msgs::msg::Pose composeWorldPoseFromHome(const geometry_msgs::msg::Pose& home_pose,
                                                  const geometry_msgs::msg::Pose& home_relative_pose)
{
  geometry_msgs::msg::Pose world_pose;

  // Custom frame origin is startup pose, but translation axes stay aligned with base_link.
  world_pose.position.x = home_pose.position.x + home_relative_pose.position.x;
  world_pose.position.y = home_pose.position.y + home_relative_pose.position.y;
  world_pose.position.z = home_pose.position.z + home_relative_pose.position.z;

  // Keep orientation relative to startup pose so custom identity maps to startup orientation.
  Eigen::Quaterniond q_home(home_pose.orientation.w, home_pose.orientation.x, home_pose.orientation.y, home_pose.orientation.z);
  if (q_home.norm() < 1e-9)
  {
    q_home = Eigen::Quaterniond::Identity();
  }
  q_home.normalize();

  Eigen::Quaterniond q_rel(home_relative_pose.orientation.w,
                           home_relative_pose.orientation.x,
                           home_relative_pose.orientation.y,
                           home_relative_pose.orientation.z);
  if (q_rel.norm() < 1e-9)
  {
    q_rel = Eigen::Quaterniond::Identity();
  }
  q_rel.normalize();

  Eigen::Quaterniond q_world = q_home * q_rel;
  q_world.normalize();
  world_pose.orientation = eigenQuatToMsg(q_world);
  return world_pose;
}

static std::string toLowerAscii(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

static double applyInterpolationProfile(double alpha_linear, const std::string& profile)
{
  const double a = std::clamp(alpha_linear, 0.0, 1.0);
  const std::string p = toLowerAscii(profile);

  if (p == "linear")
  {
    return a;
  }
  if (p == "ease_in_quad")
  {
    return a * a;
  }
  if (p == "ease_in_cubic")
  {
    return a * a * a;
  }

  RCLCPP_WARN_STREAM(LOGGER,
                     "Unknown interpolation profile '" << profile
                                                        << "'. Falling back to linear.");
  return a;
}

static geometry_msgs::msg::Pose interpolatePoseLinearWithSlerp(const geometry_msgs::msg::Pose& start_pose,
                                                               const geometry_msgs::msg::Pose& goal_pose,
                                                               double alpha)
{
  alpha = std::clamp(alpha, 0.0, 1.0);

  geometry_msgs::msg::Pose out;
  out.position.x = start_pose.position.x + alpha * (goal_pose.position.x - start_pose.position.x);
  out.position.y = start_pose.position.y + alpha * (goal_pose.position.y - start_pose.position.y);
  out.position.z = start_pose.position.z + alpha * (goal_pose.position.z - start_pose.position.z);

  Eigen::Quaterniond q_start(start_pose.orientation.w,
                             start_pose.orientation.x,
                             start_pose.orientation.y,
                             start_pose.orientation.z);
  Eigen::Quaterniond q_goal(goal_pose.orientation.w,
                            goal_pose.orientation.x,
                            goal_pose.orientation.y,
                            goal_pose.orientation.z);
  if (q_start.norm() < 1e-9)
  {
    q_start = Eigen::Quaterniond::Identity();
  }
  if (q_goal.norm() < 1e-9)
  {
    q_goal = Eigen::Quaterniond::Identity();
  }
  q_start.normalize();
  q_goal.normalize();

  const Eigen::Quaterniond q_interp = q_start.slerp(alpha, q_goal);
  out.orientation = eigenQuatToMsg(q_interp);
  return out;
}

geometry_msgs::msg::Pose expressWorldPoseInHomeFrame(const geometry_msgs::msg::Pose& home_pose,
                                                     const geometry_msgs::msg::Pose& world_pose)
{
  geometry_msgs::msg::Pose pose_in_custom;
  // Custom frame origin is home pose, but axes stay aligned with base_link axes.
  pose_in_custom.position.x = world_pose.position.x - home_pose.position.x;
  pose_in_custom.position.y = world_pose.position.y - home_pose.position.y;
  pose_in_custom.position.z = world_pose.position.z - home_pose.position.z;

  // Report orientation relative to startup pose so startup is identity in custom frame.
  Eigen::Quaterniond q_home(home_pose.orientation.w, home_pose.orientation.x, home_pose.orientation.y, home_pose.orientation.z);
  if (q_home.norm() < 1e-9)
  {
    q_home = Eigen::Quaterniond::Identity();
  }
  q_home.normalize();

  Eigen::Quaterniond q_world(world_pose.orientation.w, world_pose.orientation.x, world_pose.orientation.y, world_pose.orientation.z);
  if (q_world.norm() < 1e-9)
  {
    q_world = Eigen::Quaterniond::Identity();
  }
  q_world.normalize();

  Eigen::Quaterniond q_custom = q_home.conjugate() * q_world;
  q_custom.normalize();
  pose_in_custom.orientation = eigenQuatToMsg(q_custom);
  return pose_in_custom;
}

std::string toIso8601Utc(const std::chrono::system_clock::time_point& time_point)
{
  const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_point.time_since_epoch()).count();
  const auto sec_part = static_cast<std::time_t>(total_ms / 1000);
  const auto ms_part = static_cast<int>(total_ms % 1000);

  std::tm tm_utc{};
  gmtime_r(&sec_part, &tm_utc);

  std::ostringstream oss;
  oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%S")
      << "." << std::setw(3) << std::setfill('0') << ms_part << "Z";
  return oss.str();
}

Eigen::Quaterniond orientationPointingToCenter(const Eigen::Vector3d& target_world,
                                               const Eigen::Vector3d& center_world)
{
  // Constrain only local Z to point at center; choose remaining axes via stable projection.
  Eigen::Vector3d z_axis = center_world - target_world;
  if (z_axis.norm() < 1e-9)
  {
    return Eigen::Quaterniond::Identity();
  }
  z_axis.normalize();

  Eigen::Vector3d x_ref(1.0, 0.0, 0.0);
  if (std::abs(z_axis.dot(x_ref)) > 0.95)
  {
    x_ref = Eigen::Vector3d(0.0, 1.0, 0.0);
  }

  Eigen::Vector3d x_axis = x_ref - z_axis * x_ref.dot(z_axis);
  if (x_axis.norm() < 1e-9)
  {
    return Eigen::Quaterniond::Identity();
  }
  x_axis.normalize();

  Eigen::Vector3d y_axis = z_axis.cross(x_axis);
  y_axis.normalize();

  Eigen::Matrix3d rotation;
  rotation.col(0) = x_axis;
  rotation.col(1) = y_axis;
  rotation.col(2) = z_axis;

  Eigen::Quaterniond q(rotation);
  q.normalize();
  return q;
}

bool executePoseWithServoTracking(moveit::planning_interface::MoveGroupInterface& move_group,
                                  const ServoExecutionContext& servo_ctx,
                                  const geometry_msgs::msg::Pose& target_pose,
                                  std::chrono::system_clock::time_point& start_tp,
                                  std::chrono::system_clock::time_point& end_tp,
                                  geometry_msgs::msg::PoseStamped& reached_pose)
{
  if (!servo_ctx.enabled || !servo_ctx.tracker || !servo_ctx.target_pose_pub || !servo_ctx.node)
  {
    RCLCPP_ERROR(LOGGER, "Servo execution requested but servo context is not fully initialized.");
    return false;
  }

  geometry_msgs::msg::PoseStamped target_pose_msg;
  target_pose_msg.header.frame_id = move_group.getPlanningFrame();
  target_pose_msg.pose = target_pose;

  servo_ctx.tracker->resetTargetPose();
  // Publish the static target a few times to reliably latch it in the tracker callback chain.
  for (int i = 0; i < 5; ++i)
  {
    target_pose_msg.header.stamp = servo_ctx.node->now();
    servo_ctx.target_pose_pub->publish(target_pose_msg);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  Eigen::Vector3d lin_tol{servo_ctx.linear_tolerance_m, servo_ctx.linear_tolerance_m, servo_ctx.linear_tolerance_m};

  start_tp = std::chrono::system_clock::now();
  const auto tracking_status = servo_ctx.tracker->moveToPose(lin_tol, servo_ctx.angular_tolerance_rad, 0.1);
  end_tp = std::chrono::system_clock::now();

  if (tracking_status != moveit_servo::PoseTrackingStatusCode::SUCCESS)
  {
    const auto it = moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.find(tracking_status);
    if (it != moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.end())
    {
      RCLCPP_ERROR(LOGGER, "Servo tracking failed with status: %s", it->second.c_str());
    }
    else
    {
      RCLCPP_ERROR(LOGGER, "Servo tracking failed with unknown status code: %d", static_cast<int>(tracking_status));
    }
    return false;
  }

  reached_pose = move_group.getCurrentPose();
  return true;
}

bool executeInterpolatedServoTracking(moveit::planning_interface::MoveGroupInterface& move_group,
                                       const ServoExecutionContext& servo_ctx,
                                       const PoseInterpolationConfig& interpolation_cfg,
                                       const geometry_msgs::msg::Pose& target_pose,
                                       std::chrono::system_clock::time_point& start_tp,
                                       std::chrono::system_clock::time_point& end_tp,
                                       geometry_msgs::msg::PoseStamped& reached_pose)
{
  if (!interpolation_cfg.enabled || interpolation_cfg.steps <= 0)
  {
    return executePoseWithServoTracking(move_group, servo_ctx, target_pose, start_tp, end_tp, reached_pose);
  }

  const geometry_msgs::msg::Pose start_pose = move_group.getCurrentPose().pose;
  bool started = false;
  std::chrono::system_clock::time_point sequence_start_tp;
  geometry_msgs::msg::PoseStamped step_reached_pose;

  for (int step = 1; step <= interpolation_cfg.steps; ++step)
  {
    const double alpha_linear = static_cast<double>(step) / static_cast<double>(interpolation_cfg.steps + 1);
    const double alpha = applyInterpolationProfile(alpha_linear, interpolation_cfg.profile);
    const geometry_msgs::msg::Pose interpolated_pose = interpolatePoseLinearWithSlerp(start_pose, target_pose, alpha);

    std::chrono::system_clock::time_point step_start_tp;
    std::chrono::system_clock::time_point step_end_tp;
    if (!executePoseWithServoTracking(move_group, servo_ctx, interpolated_pose, step_start_tp, step_end_tp, step_reached_pose))
    {
      return false;
    }

    if (!started)
    {
      sequence_start_tp = step_start_tp;
      started = true;
    }
    end_tp = step_end_tp;
  }

  std::chrono::system_clock::time_point final_start_tp;
  std::chrono::system_clock::time_point final_end_tp;
  if (!executePoseWithServoTracking(move_group, servo_ctx, target_pose, final_start_tp, final_end_tp, step_reached_pose))
  {
    return false;
  }

  if (!started)
  {
    sequence_start_tp = final_start_tp;
  }

  start_tp = sequence_start_tp;
  end_tp = final_end_tp;
  reached_pose = step_reached_pose;
  return true;
}

bool planAndExecutePose(moveit::planning_interface::MoveGroupInterface& move_group,
                        const rclcpp::Node::SharedPtr& node,
                        const rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr& ik_client,
                        const ServoExecutionContext& servo_ctx,
                        const PoseInterpolationConfig& interpolation_cfg,
                        const geometry_msgs::msg::Pose& target_pose,
                        std::chrono::system_clock::time_point& start_tp,
                        std::chrono::system_clock::time_point& end_tp,
                        geometry_msgs::msg::PoseStamped& reached_pose)
{
  if (servo_ctx.enabled)
  {
    return executeInterpolatedServoTracking(
        move_group, servo_ctx, interpolation_cfg, target_pose, start_tp, end_tp, reached_pose);
  }

  if (!ik_client)
  {
    RCLCPP_ERROR(LOGGER, "IK client is not available.");
    return false;
  }

  auto request = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
  request->ik_request.group_name = move_group.getName();
  request->ik_request.ik_link_name = move_group.getEndEffectorLink();
  request->ik_request.pose_stamped.header.frame_id = move_group.getPlanningFrame();
  request->ik_request.pose_stamped.header.stamp = node->now();
  request->ik_request.pose_stamped.pose = target_pose;
  request->ik_request.avoid_collisions = true;
  request->ik_request.timeout.sec = 1;
  request->ik_request.timeout.nanosec = 0;

  const auto current_state = move_group.getCurrentState();
  if (!current_state)
  {
    RCLCPP_ERROR(LOGGER, "Could not get current robot state for IK seed.");
    return false;
  }
  moveit::core::robotStateToRobotStateMsg(*current_state, request->ik_request.robot_state);

  auto future = ik_client->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
  {
    RCLCPP_ERROR(LOGGER, "Timed out waiting for IK service response.");
    return false;
  }

  auto response = future.get();
  if (response->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
  {
    RCLCPP_ERROR(LOGGER, "IK failed with MoveIt error code %d.", response->error_code.val);
    return false;
  }

  const auto& solution = response->solution.joint_state;
  std::map<std::string, double> joint_targets;
  const auto group_joint_names = move_group.getJointNames();
  const std::unordered_set<std::string> group_joint_set(group_joint_names.begin(), group_joint_names.end());
  for (std::size_t idx = 0; idx < solution.name.size() && idx < solution.position.size(); ++idx)
  {
    const auto& joint_name = solution.name[idx];
    if (group_joint_set.count(joint_name) > 0)
    {
      joint_targets[joint_name] = solution.position[idx];
    }
  }

  if (joint_targets.empty())
  {
    RCLCPP_ERROR(LOGGER, "IK returned no joint targets for planning group %s.", move_group.getName().c_str());
    return false;
  }

  if (!move_group.setJointValueTarget(joint_targets))
  {
    RCLCPP_ERROR(LOGGER, "Failed to set joint targets from IK solution.");
    return false;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const auto plan_result = move_group.plan(plan);
  if (plan_result != moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_ERROR(LOGGER, "Planning to IK solution failed.");
    return false;
  }

  start_tp = std::chrono::system_clock::now();
  const auto execute_result = move_group.execute(plan);
  end_tp = std::chrono::system_clock::now();

  move_group.stop();

  if (execute_result != moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_ERROR(LOGGER, "Execution failed.");
    return false;
  }

  reached_pose = move_group.getCurrentPose();
  return true;
}

bool planAndExecuteJointTargets(moveit::planning_interface::MoveGroupInterface& move_group,
                                const std::map<std::string, double>& joint_targets,
                                const std::string& group_label)
{
  if (joint_targets.empty())
  {
    return true;
  }

  move_group.setStartStateToCurrentState();
  if (!move_group.setJointValueTarget(joint_targets))
  {
    RCLCPP_ERROR(LOGGER, "Failed to set %s joint targets.", group_label.c_str());
    return false;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const auto plan_result = move_group.plan(plan);
  if (plan_result != moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_ERROR(LOGGER, "Planning failed for %s homing motion.", group_label.c_str());
    return false;
  }

  const auto exec_result = move_group.execute(plan);
  move_group.stop();
  if (exec_result != moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_ERROR(LOGGER, "Execution failed for %s homing motion.", group_label.c_str());
    return false;
  }

  return true;
}


void add_collision_objects(
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr scene_pub,
  const planning_scene_monitor::PlanningSceneMonitorPtr& planning_scene_monitor)
{
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = "base_link";
    collision_object.id = "base_box";

    // Make a box and put it in the way
    shape_msgs::msg::SolidPrimitive base;
    base.type = base.BOX;
    // base.dimensions = { 1.40, 0.80, 0.9 };
    // geometry_msgs::msg::Pose base_pose;
    // base_pose.position.x = 0.0;
    // base_pose.position.y = 0.0; //-0.17;
    // base_pose.position.z = -0.451;
    // base_pose.orientation.w = 1.0;
    // base_pose.orientation.x = 0.0;
    // base_pose.orientation.y = 0.0;
    // base_pose.orientation.z = 0.0;
    base.dimensions = { 0.21, 0.21, 0.025 };
    geometry_msgs::msg::Pose base_pose;
    base_pose.position.x = 0.0;
    base_pose.position.y = 0.0; //-0.17;
    base_pose.position.z = -0.0125;
    base_pose.orientation.w = 1.0;
    base_pose.orientation.x = 0.0;
    base_pose.orientation.y = 0.0;
    base_pose.orientation.z = 0.0;

    // Add the box as a collision object
    collision_object.primitives.push_back(base);
    collision_object.primitive_poses.push_back(base_pose);
    collision_object.operation = collision_object.ADD;

    // secondary base
    base.dimensions = { 1.08, 0.59, 0.025 };
    base_pose.position.x = 0.0;
    base_pose.position.y = 0.0; //-0.17;
    base_pose.position.z = -0.0375;
    collision_object.primitives.push_back(base);
    collision_object.primitive_poses.push_back(base_pose);

    // third base
    base.dimensions = { 1.08, 0.765, 0.04 };
    base_pose.position.x = 0.0;
    base_pose.position.y = 0.0; //-0.17;
    base_pose.position.z = -0.07;
    collision_object.primitives.push_back(base);
    collision_object.primitive_poses.push_back(base_pose);

    shape_msgs::msg::SolidPrimitive table;
    table.type = table.BOX;
    table.dimensions = { 1.75, 1.0, 0.86 }; // { 1.75, 1.0, 0.86 };
    geometry_msgs::msg::Pose table_pose;
    table_pose.position.x = 0.0;
    table_pose.position.y = 0.6799999999999999;
    table_pose.position.z = -0.52; //-0.5; 
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

    // Merge into existing ACM instead of replacing it.
    if (planning_scene_monitor && planning_scene_monitor->getPlanningScene())
    {
      planning_scene_monitor::LockedPlanningSceneRO scene(planning_scene_monitor);
      collision_detection::AllowedCollisionMatrix acm = scene->getAllowedCollisionMatrix();
      acm.setEntry("base_link_inertia", "base_box", true);
      acm.getMessage(ps.allowed_collision_matrix);
    }
    else
    {
      RCLCPP_WARN(LOGGER, "PlanningSceneMonitor unavailable while updating ACM; publishing collision objects without ACM diff.");
    }

    // Publish the collision object to the planning scene
    scene_pub->publish(ps);
}

void publishReachabilityEnvelope(moveit_visual_tools::MoveItVisualTools& visual_tools,
                                 double reach_radius_m,
                                 int azimuth_samples = 24,
                                 int elevation_samples = 24)
{
  namespace rvt = rviz_visual_tools;

  geometry_msgs::msg::Point origin;
  origin.x = 0.0;
  origin.y = 0.0;
  origin.z = 0.0;

  const double min_elev = -M_PI / 2.0;
  const double max_elev = M_PI / 2.0;

  for (int i = 0; i < elevation_samples; ++i)
  {
    const double t = (elevation_samples <= 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(elevation_samples - 1);
    const double elev = min_elev + t * (max_elev - min_elev);

    for (int j = 0; j < azimuth_samples; ++j)
    {
      const double az = (2.0 * M_PI * static_cast<double>(j)) / static_cast<double>(azimuth_samples);

      geometry_msgs::msg::Point tip;
      tip.x = reach_radius_m * std::cos(elev) * std::cos(az);
      tip.y = reach_radius_m * std::cos(elev) * std::sin(az);
      tip.z = reach_radius_m * std::sin(elev);

      visual_tools.publishArrow(origin, tip, rvt::TRANSLUCENT_LIGHT, rvt::SMALL);
    }
  }

  geometry_msgs::msg::Pose label_pose;
  label_pose.orientation.w = 1.0;
  label_pose.position.x = 0.0;
  label_pose.position.y = 0.0;
  label_pose.position.z = reach_radius_m + 0.08;
  visual_tools.publishText(label_pose, "UR7e Reach Envelope (D=1.146 m)", rvt::WHITE, rvt::LARGE);
  visual_tools.trigger();
}


bool switch_controllers(rclcpp::Node::SharedPtr node, bool reverse = false)
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
  if (!reverse) {
    request->deactivate_controllers = {"scaled_joint_trajectory_controller"};
    request->activate_controllers = {"forward_position_controller"}; //{"forward_position_controller"};
  } else {
    request->deactivate_controllers = {"forward_position_controller"}; // {"forward_position_controller"};
    request->activate_controllers = {"scaled_joint_trajectory_controller"};
  }
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

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.use_intra_process_comms(false);
  rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("end_effector_s8", node_options);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();

  const std::string output_csv =
      node->declare_parameter<std::string>("output_csv", "s8_submission_single_run.csv");
    const std::string accuracy_csv =
      node->declare_parameter<std::string>("accuracy_csv", "s8_accuracy_single_run.csv");
  const int run_id = node->declare_parameter<int>("run_id", 1);
  const int repetitions = node->declare_parameter<int>("repetitions", 3);
  const double tool_length_m = 0.077; //0.28;
  const double reach_radius_m = 0.800; //0.946 + tool_length_m;
  const double h_reach_radius_m = 0.750;
  const double default_reach_mm = reach_radius_m * 1000.0;
  const double default_reach_mm_h = h_reach_radius_m * 1000.0;

  const double r_mm = node->declare_parameter<double>("r_mm", default_reach_mm);
  const double h_mm = node->declare_parameter<double>("h_mm", default_reach_mm_h);
  const double planning_time_s = node->declare_parameter<double>("planning_time_s", 2.0);
  const double velocity_scaling = node->declare_parameter<double>("velocity_scaling", 0.2); // 0.20
  const double acceleration_scaling = node->declare_parameter<double>("acceleration_scaling", 0.2); // 0.20
  const double motion_time_limit_ms = node->declare_parameter<double>("motion_time_limit_ms", 5000.0);
  const double target_hold_time_s = node->declare_parameter<double>("target_hold_time_s", 1.0);
  const bool point_to_center_orientation = node->declare_parameter<bool>("point_to_center_orientation", false);
  const double target_offset_y_m = node->declare_parameter<double>("target_offset_y_m", 0.0); //0.1
  const double target_offset_z_m = node->declare_parameter<double>("target_offset_z_m", 0.0); // 0.1624985635280609
  const bool use_pid_servoing = node->declare_parameter<bool>("use_pid_servoing", true);
  const double servo_linear_tolerance_m = node->declare_parameter<double>("servo_linear_tolerance_m", 0.005); // 0.007
  const double servo_angular_tolerance_rad = node->declare_parameter<double>("servo_angular_tolerance_rad", 0.05); // 0.07
  const bool enable_pose_interpolation = node->declare_parameter<bool>("enable_pose_interpolation", true);
  const int pose_interpolation_steps = node->declare_parameter<int>("pose_interpolation_steps", 1);
  const std::string pose_interpolation_profile =
      node->declare_parameter<std::string>("pose_interpolation_profile", "linear");

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
      default_planner_id = "PTP";
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
  moveit_visual_tools::MoveItVisualTools reach_visual_tools(node, "shoulder_link", "reachability_marker",
                                                            move_group.getRobotModel());
  visual_tools.deleteAllMarkers();
  reach_visual_tools.deleteAllMarkers();
  publishReachabilityEnvelope(reach_visual_tools, reach_radius_m); // UR3e max reach + tool length

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
  planning_scene_monitor->startPublishingPlanningScene(planning_scene_monitor::PlanningSceneMonitor::UPDATE_SCENE);

  // Wait for Planning Scene Monitor to setup
  if (!planning_scene_monitor->waitForCurrentRobotState(node->now(), 5.0))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Error waiting for current robot state in PlanningSceneMonitor.");
    exit(EXIT_FAILURE);
  }

  auto ik_client = node->create_client<moveit_msgs::srv::GetPositionIK>("compute_ik");
  if (!ik_client->wait_for_service(std::chrono::seconds(5)))
  {
    RCLCPP_FATAL(LOGGER, "IK service /compute_ik is not available.");
    exit(EXIT_FAILURE);
  }

  ServoExecutionContext servo_ctx;
  if (use_pid_servoing)
  {
    servo_ctx.enabled = true;
    servo_ctx.node = node;
    servo_ctx.linear_tolerance_m = servo_linear_tolerance_m;
    servo_ctx.angular_tolerance_rad = servo_angular_tolerance_rad;
    servo_ctx.tracker = std::make_shared<moveit_servo::PoseTracking>(node, servo_parameters, planning_scene_monitor);
    servo_ctx.target_pose_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>("target_pose", rclcpp::SystemDefaultsQoS());
    RCLCPP_INFO(LOGGER,
                "Using MoveIt Servo pose tracking backend (lin_tol=%.4f m, ang_tol=%.4f rad).",
                servo_linear_tolerance_m,
                servo_angular_tolerance_rad);
  }
  else
  {
    RCLCPP_INFO(LOGGER, "Using global planning backend (IK + joint-value planning).\n");
  }

  const PoseInterpolationConfig pose_interpolation_cfg{
      enable_pose_interpolation && use_pid_servoing,
      pose_interpolation_steps,
      pose_interpolation_profile};

  auto scene_pub = node->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 10);
  add_collision_objects(scene_pub, planning_scene_monitor);

//   move_group.setPoseReferenceFrame("base_link");
  move_group.setPlanningTime(planning_time_s);
  move_group.setMaxVelocityScalingFactor(velocity_scaling);
  move_group.setMaxAccelerationScalingFactor(acceleration_scaling);
  move_group.setStartStateToCurrentState();

  const std::vector<std::pair<std::string, double>> requested_home_joints = {
      // From provided joint state; all gripper-related joints are intentionally ignored.
      {"shoulder_pan_joint", -0.5132342874483838},
      {"wrist_3_joint", 0.041397262089919354},
      {"wrist_2_joint", -0.5137651185489339},
      {"wrist_1_joint", -3.1935258478824666},
      {"elbow_joint", -2.4380439227340647},
      {"shoulder_lift_joint", -0.6611059734955747}
    };

  const auto arm_joint_names = move_group.getJointNames();
  const std::unordered_set<std::string> arm_joint_set(arm_joint_names.begin(), arm_joint_names.end());

  std::map<std::string, double> arm_home_joint_targets;
  for (const auto& [joint_name, joint_value] : requested_home_joints)
  {
    if (arm_joint_set.count(joint_name) > 0)
    {
      arm_home_joint_targets[joint_name] = joint_value;
    }
    else
    {
      RCLCPP_WARN(LOGGER, "Requested home joint '%s' was not found in arm/gripper planning groups.", joint_name.c_str());
    }
  }

  RCLCPP_INFO(LOGGER, "Moving robot to requested joint-space home position.");
  if (!planAndExecuteJointTargets(move_group, arm_home_joint_targets, "arm"))
  {
    RCLCPP_FATAL(LOGGER, "Failed to move arm to requested home joint position.");
    rclcpp::shutdown();
    return 1;
  }

  if (default_pipeline_id == "pilz_industrial_motion_planner")
  {
    default_planner_id = "LIN";
    move_group.setPlannerId(default_planner_id);
  }

  move_group.setStartStateToCurrentState();

  geometry_msgs::msg::Pose reporting_home_pose = move_group.getCurrentPose().pose;
  geometry_msgs::msg::Pose home_pose = reporting_home_pose;

  const Eigen::Vector3d base_center_m(0.0,
                                      r_mm / 2.0 / 1000.0 + target_offset_y_m,
                                      target_offset_z_m);
  if (point_to_center_orientation)
  {
    const Eigen::Vector3d home_position(home_pose.position.x, home_pose.position.y, home_pose.position.z);
    const Eigen::Quaterniond q_home_to_center = orientationPointingToCenter(home_position, base_center_m);

    geometry_msgs::msg::Pose home_oriented_pose = home_pose;
    home_oriented_pose.orientation = eigenQuatToMsg(q_home_to_center);

    std::chrono::system_clock::time_point orient_start_tp;
    std::chrono::system_clock::time_point orient_end_tp;
    geometry_msgs::msg::PoseStamped reached_oriented_home;

    RCLCPP_INFO(LOGGER, "Orienting startup tool0 pose toward center (0, r/2, 0) in base_link frame.");
    if (!planAndExecutePose(move_group,
                            node,
                            ik_client,
                            servo_ctx,
                            pose_interpolation_cfg,
                            home_oriented_pose,
                            orient_start_tp,
                            orient_end_tp,
                            reached_oriented_home))
    {
      RCLCPP_FATAL(LOGGER, "Failed to orient startup pose toward center.");
      rclcpp::shutdown();
      return 1;
    }

    home_pose = reached_oriented_home.pose;
  }

  const std::array<Eigen::Vector3d, 6> custom_targets_m = {
      Eigen::Vector3d(0.0, +r_mm / 2.0, 0.0) / 1000.0,
      Eigen::Vector3d(-r_mm / 2.0, +r_mm / 2.0, 0.0) / 1000.0,
      Eigen::Vector3d(+r_mm / 2.0, +r_mm / 2.0, 0.0) / 1000.0,
      Eigen::Vector3d(0.0, +r_mm / 2.0, +h_mm / 4.0) / 1000.0,
      Eigen::Vector3d(-r_mm / 2.0, +r_mm / 2.0, +h_mm / 4.0) / 1000.0,
      Eigen::Vector3d(+r_mm / 2.0, +r_mm / 2.0, +h_mm / 4.0) / 1000.0
    };

  for (std::size_t i = 0; i < custom_targets_m.size(); ++i)
  {
    geometry_msgs::msg::Pose target_pose_custom;
    target_pose_custom.orientation.w = 1.0;

    const Eigen::Vector3d custom_target = custom_targets_m[i];
    target_pose_custom.position.x = custom_target.x();
    target_pose_custom.position.y = custom_target.y() + target_offset_y_m;
    target_pose_custom.position.z = custom_target.z() + target_offset_z_m;

    const geometry_msgs::msg::Pose target_pose_world = composeWorldPoseFromHome(home_pose, target_pose_custom);

    const std::string axis_label = "s8_t" + std::to_string(i + 1);
    visual_tools.publishAxisLabeled(target_pose_world, axis_label, rviz_visual_tools::SMALL);

    geometry_msgs::msg::Pose text_pose = target_pose_world;
    text_pose.position.z += 0.05;
    visual_tools.publishText(text_pose, "T" + std::to_string(i + 1), rviz_visual_tools::WHITE, rviz_visual_tools::SMALL);
  }

  visual_tools.trigger();
  RCLCPP_INFO(LOGGER, "Published 6 S8 target markers from custom frame targets (rendered in base_link frame).");

  std::vector<MotionRecord> records;
  records.reserve(static_cast<std::size_t>(repetitions) * custom_targets_m.size());
  std::vector<AccuracyRecord> accuracy_records;
  accuracy_records.reserve(static_cast<std::size_t>(repetitions) * custom_targets_m.size());

  RCLCPP_INFO(LOGGER, "s8 run started with targets defined in custom frame and planned in base_link frame. Repetitions=%d", repetitions);

  for (int rep = 1; rep <= repetitions; ++rep)
  {
    RCLCPP_INFO(LOGGER, "Starting repetition %d/%d", rep, repetitions);

    if (use_pid_servoing && rep > 1)
    {
      if (!switch_controllers(node, true))
      {
        RCLCPP_ERROR(LOGGER, "Aborting execution because joint-space controller switch failed before repetition %d.", rep);
        rclcpp::shutdown();
        return EXIT_FAILURE;
      }
    }

    // Ensure the trial starts from the same joint-space pose each repetition.
    if (default_pipeline_id == "pilz_industrial_motion_planner")
    {
      move_group.setPlannerId("PTP");
    }
    if (!planAndExecuteJointTargets(move_group, arm_home_joint_targets, "arm"))
    {
      RCLCPP_FATAL(LOGGER, "Failed to move to repetition start pose for repetition %d.", rep);
      rclcpp::shutdown();
      return 1;
    }
    if (default_pipeline_id == "pilz_industrial_motion_planner")
    {
      move_group.setPlannerId(default_planner_id);
    }

    if (use_pid_servoing)
    {
      if (!switch_controllers(node))
      {
        RCLCPP_ERROR(LOGGER, "Aborting execution because servo controller switch failed before repetition %d targets.", rep);
        rclcpp::shutdown();
        return EXIT_FAILURE;
      }
    }

    for (std::size_t i = 0; i < custom_targets_m.size(); ++i)
    {
      geometry_msgs::msg::Pose target_pose_custom;
      target_pose_custom.orientation.w = 1.0;

      const Eigen::Vector3d custom_target = custom_targets_m[i];
      target_pose_custom.position.x = custom_target.x();
      target_pose_custom.position.y = custom_target.y() + target_offset_y_m;
      target_pose_custom.position.z = custom_target.z() + target_offset_z_m;

      const geometry_msgs::msg::Pose target_pose_world = composeWorldPoseFromHome(home_pose, target_pose_custom);

      std::chrono::system_clock::time_point start_tp;
      std::chrono::system_clock::time_point end_tp;
      geometry_msgs::msg::PoseStamped reached_target_world;

      RCLCPP_INFO(LOGGER, "Executing repetition %d/%d target %zu/6", rep, repetitions, i + 1);
      if (!planAndExecutePose(move_group,
                              node,
                              ik_client,
                              servo_ctx,
                              pose_interpolation_cfg,
                              target_pose_world,
                              start_tp,
                              end_tp,
                              reached_target_world))
      {
        RCLCPP_FATAL(LOGGER, "Failed to complete outbound motion for repetition %d target %zu.", rep, i + 1);
        rclcpp::shutdown();
        return 1;
      }

    const auto outbound_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_tp - start_tp).count();
    if (outbound_duration > motion_time_limit_ms)
    {
      RCLCPP_WARN(
          LOGGER,
          "Target %zu outbound motion took %lld ms (> %.0f ms threshold).",
          i + 1,
          static_cast<long long>(outbound_duration),
          motion_time_limit_ms);
    }

    const geometry_msgs::msg::Pose reached_target_home = expressWorldPoseInHomeFrame(reporting_home_pose, reached_target_world.pose);
    const geometry_msgs::msg::Pose ideal_target_home = target_pose_custom;

    RCLCPP_INFO(
      LOGGER,
      "Target %zu frame report | home_world [m]: (%.4f, %.4f, %.4f) | home_custom [m]: (%.4f, %.4f, %.4f)",
      i + 1,
      reporting_home_pose.position.x,
      reporting_home_pose.position.y,
      reporting_home_pose.position.z,
      0.0,
      0.0,
      0.0);
    RCLCPP_INFO(
      LOGGER,
      "Target %zu frame report | reached_world [m]: (%.4f, %.4f, %.4f) | reached_custom [m]: (%.4f, %.4f, %.4f)",
      i + 1,
      reached_target_world.pose.position.x,
      reached_target_world.pose.position.y,
      reached_target_world.pose.position.z,
      reached_target_home.position.x,
      reached_target_home.position.y,
      reached_target_home.position.z);

    const Eigen::Vector3d world_pos_error(
      reached_target_world.pose.position.x - target_pose_world.position.x,
      reached_target_world.pose.position.y - target_pose_world.position.y,
      reached_target_world.pose.position.z - target_pose_world.position.z);

    const Eigen::Vector3d custom_pos_error(
      reached_target_home.position.x - ideal_target_home.position.x,
      reached_target_home.position.y - ideal_target_home.position.y,
      reached_target_home.position.z - ideal_target_home.position.z);

    Eigen::Quaterniond q_reached_world(
      reached_target_world.pose.orientation.w,
      reached_target_world.pose.orientation.x,
      reached_target_world.pose.orientation.y,
      reached_target_world.pose.orientation.z);
    Eigen::Quaterniond q_target_world(
      target_pose_world.orientation.w,
      target_pose_world.orientation.x,
      target_pose_world.orientation.y,
      target_pose_world.orientation.z);
    if (q_reached_world.norm() < 1e-9)
    {
      q_reached_world = Eigen::Quaterniond::Identity();
    }
    if (q_target_world.norm() < 1e-9)
    {
      q_target_world = Eigen::Quaterniond::Identity();
    }
    q_reached_world.normalize();
    q_target_world.normalize();
    const double world_ang_error_deg = q_reached_world.angularDistance(q_target_world) * 180.0 / M_PI;

    Eigen::Quaterniond q_reached_custom(
      reached_target_home.orientation.w,
      reached_target_home.orientation.x,
      reached_target_home.orientation.y,
      reached_target_home.orientation.z);
    Eigen::Quaterniond q_ideal_custom(
      ideal_target_home.orientation.w,
      ideal_target_home.orientation.x,
      ideal_target_home.orientation.y,
      ideal_target_home.orientation.z);
    if (q_reached_custom.norm() < 1e-9)
    {
      q_reached_custom = Eigen::Quaterniond::Identity();
    }
    if (q_ideal_custom.norm() < 1e-9)
    {
      q_ideal_custom = Eigen::Quaterniond::Identity();
    }
    q_reached_custom.normalize();
    q_ideal_custom.normalize();
    const double custom_ang_error_deg = q_reached_custom.angularDistance(q_ideal_custom) * 180.0 / M_PI;

    RCLCPP_INFO(
      LOGGER,
      "Target %zu ideal-vs-taken | world dpos [mm]: (%.2f, %.2f, %.2f), norm=%.2f, dang=%.2f deg",
      i + 1,
      world_pos_error.x() * 1000.0,
      world_pos_error.y() * 1000.0,
      world_pos_error.z() * 1000.0,
      world_pos_error.norm() * 1000.0,
      world_ang_error_deg);
    RCLCPP_INFO(
      LOGGER,
      "Target %zu ideal-vs-reported | custom dpos [mm]: (%.2f, %.2f, %.2f), norm=%.2f, dang=%.2f deg",
      i + 1,
      custom_pos_error.x() * 1000.0,
      custom_pos_error.y() * 1000.0,
      custom_pos_error.z() * 1000.0,
      custom_pos_error.norm() * 1000.0,
      custom_ang_error_deg);

    const Eigen::Vector3d reached_home(
      reached_target_home.position.x,
      reached_target_home.position.y,
      reached_target_home.position.z);
    Eigen::Quaterniond q_reached(
      reached_target_home.orientation.w,
      reached_target_home.orientation.x,
      reached_target_home.orientation.y,
      reached_target_home.orientation.z);
    if (q_reached.norm() < 1e-9)
    {
      q_reached = Eigen::Quaterniond::Identity();
    }
    q_reached.normalize();

      records.push_back(MotionRecord{
          run_id,
          rep,
          static_cast<int>(i + 1),
          reached_home.x() * 1000.0,
          reached_home.y() * 1000.0,
          reached_home.z() * 1000.0,
          q_reached.x(),
          q_reached.y(),
          q_reached.z(),
          q_reached.w(),
          toIso8601Utc(start_tp),
          toIso8601Utc(end_tp),
          outbound_duration});

      accuracy_records.push_back(AccuracyRecord{
          run_id,
          rep,
          static_cast<int>(i + 1),
          world_pos_error.x() * 1000.0,
          world_pos_error.y() * 1000.0,
          world_pos_error.z() * 1000.0,
          world_pos_error.norm() * 1000.0,
          world_ang_error_deg,
          custom_pos_error.x() * 1000.0,
          custom_pos_error.y() * 1000.0,
          custom_pos_error.z() * 1000.0,
          custom_pos_error.norm() * 1000.0,
          custom_ang_error_deg,
          toIso8601Utc(start_tp),
          toIso8601Utc(end_tp),
          outbound_duration});

    if (target_hold_time_s > 0.0)
    {
      RCLCPP_INFO(LOGGER, "Holding target %zu for %.2f s before return motion.", i + 1, target_hold_time_s);
      std::this_thread::sleep_for(std::chrono::duration<double>(target_hold_time_s));
    }

    std::chrono::system_clock::time_point return_start_tp;
    std::chrono::system_clock::time_point return_end_tp;
    geometry_msgs::msg::PoseStamped reached_home_world;

      if (!planAndExecutePose(move_group,
                              node,
                              ik_client,
                              servo_ctx,
                              pose_interpolation_cfg,
                              home_pose,
                              return_start_tp,
                              return_end_tp,
                              reached_home_world))
      {
        RCLCPP_FATAL(LOGGER, "Failed to return to home pose after repetition %d target %zu.", rep, i + 1);
        rclcpp::shutdown();
        return 1;
      }

    const auto return_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(return_end_tp - return_start_tp).count();
      if (return_duration > motion_time_limit_ms)
      {
        RCLCPP_ERROR(
            LOGGER,
            "Error! Repetition %d target %zu return motion took %lld ms (> %.0f ms threshold).",
            rep,
            i + 1,
            static_cast<long long>(return_duration),
            motion_time_limit_ms);
        exit(EXIT_FAILURE);
      }
    }
  }

  if (use_pid_servoing)
  {
    if (!switch_controllers(node, true))
    {
      RCLCPP_ERROR(LOGGER, "Aborting execution because controller switch failed.");
      rclcpp::shutdown();
      return EXIT_FAILURE;
    }
  }

  std::ofstream csv(output_csv, std::ios::out | std::ios::trunc);
  if (!csv.is_open())
  {
    RCLCPP_FATAL(LOGGER, "Could not open output CSV: %s", output_csv.c_str());
    rclcpp::shutdown();
    return 1;
  }

  csv << "run_id,repetition,target_id,x,y,z,qx,qy,qz,qw,start_time,end_time,motion_time_ms\n";
  csv << std::fixed << std::setprecision(6);
  for (const auto& row : records)
  {
    csv << row.run_id << ","
        << row.repetition << ","
        << row.target_id << ","
        << row.x_mm << ","
        << row.y_mm << ","
        << row.z_mm << ","
        << row.qx << ","
        << row.qy << ","
        << row.qz << ","
        << row.qw << ","
        << row.start_time << ","
        << row.end_time << ","
        << row.motion_time_ms << "\n";
  }
  csv.close();

  std::ofstream acc_csv(accuracy_csv, std::ios::out | std::ios::trunc);
  if (!acc_csv.is_open())
  {
    RCLCPP_FATAL(LOGGER, "Could not open accuracy CSV: %s", accuracy_csv.c_str());
    rclcpp::shutdown();
    return 1;
  }

  acc_csv << "run_id,repetition,target_id,"
          << "world_dx_mm,world_dy_mm,world_dz_mm,world_dnorm_mm,world_dang_deg,"
          << "custom_dx_mm,custom_dy_mm,custom_dz_mm,custom_dnorm_mm,custom_dang_deg,"
          << "start_time,end_time,motion_time_ms\n";
  acc_csv << std::fixed << std::setprecision(6);
  for (const auto& row : accuracy_records)
  {
    acc_csv << row.run_id << ","
            << row.repetition << ","
            << row.target_id << ","
            << row.world_dx_mm << ","
            << row.world_dy_mm << ","
            << row.world_dz_mm << ","
            << row.world_dnorm_mm << ","
            << row.world_dang_deg << ","
            << row.custom_dx_mm << ","
            << row.custom_dy_mm << ","
            << row.custom_dz_mm << ","
            << row.custom_dnorm_mm << ","
            << row.custom_dang_deg << ","
            << row.start_time << ","
            << row.end_time << ","
            << row.motion_time_ms << "\n";
  }
  acc_csv.close();

  RCLCPP_INFO(LOGGER, "s8 single-run complete. Submission CSV: %s | Accuracy CSV: %s",
              output_csv.c_str(), accuracy_csv.c_str());

  rclcpp::shutdown();
  return 0;
}
