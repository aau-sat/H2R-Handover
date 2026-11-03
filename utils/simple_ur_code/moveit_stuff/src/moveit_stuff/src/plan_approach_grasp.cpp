
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/bool.hpp>
#include <control_msgs/msg/joint_jog.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
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
#include <atomic>
#include <deque>
#include <chrono>
#include <future>
#include <iomanip>
#include <limits>
#include <vector>
#include <string>
#include <fstream>
#include <optional>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cctype>
#include <functional>

#include "../../real_time_filter/src/OneEuroFilter.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <moveit_msgs/srv/get_motion_plan.hpp>
#include <moveit/collision_detection/collision_matrix.h>
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_state/robot_state.h>

#include <control_msgs/action/gripper_command.hpp>
#include <control_msgs/msg/dynamic_joint_state.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
// #include <rclcpp_components/register_node_macro.hpp>
#include <grasp_generation/action/generate_grasp.hpp>
#include <tactile_detection/msg/tactile_detection.hpp>
#include <visual_properties/msg/dimensions.hpp>
#include <std_srvs/srv/trigger.hpp>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("plan_approach_node");


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

static void waitForGripperCompletion(
    const std::atomic<int>& latest_gripper_status,
    const std::atomic<double>& latest_finger_joint_position,
    double target_position,
    double epsilon,
    std::chrono::milliseconds timeout)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  // bool saw_moving_status = false;

  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline)
  {
    const int status = latest_gripper_status.load();
    const double finger_joint_position = latest_finger_joint_position.load();

    if (std::isfinite(finger_joint_position) && std::abs(finger_joint_position - target_position) <= epsilon)
    {
      RCLCPP_INFO(LOGGER, "Gripper reached target position: current=%.4f, target=%.4f", finger_joint_position, target_position);
      return;
    }

    // if (status == 0)
    // {
    //   saw_moving_status = true;
    // }
    // else if (saw_moving_status && status != -1)
    if (status != 0 && status != -1)
    {
      RCLCPP_INFO(LOGGER, "Gripper action completed with status: %s (%d)", objectDetectionStatusToString(status).c_str(), status);
      return;
    }

    std::this_thread::sleep_for(10ms);
  }

  return;
}
}  // namespace

// Configuration parameters for handover reaction
struct HandoverConfig
{
  float approach_distance;        // 20 cm from object for initial approach
  float tracking_distance;        // 10 cm from object for servo tracking
  int velocity_estimation_frames;     // Number of frames to estimate velocity
  int prediction_frames;              // Predict object pose for N frames ahead
  float grasp_velocity_threshold; // Velocity below this triggers grasp generation request
  float velocity_threshold;       // Velocity below this allows executing the generated grasp
  float grasp_pose_tolerance;
  float grasp_pose_obj_tolerance; // 2.5 cm - distance between current grasp pose and object pose at which we consider the grasp pose close enough to the object to stop tracking
  float place_tolerance;        // 2 cm - distance to goal pose at which we consider the object placed and can stop tracking
  float touchdown_tolerance;    // distance to original object position at which we consider a touchdown detection valid to avoid false positives during placement approach
};

// TODO: parameter-based values initialization
// struct PIDGains
// {
//   double x_proportional_gain = 6.5;
//   double y_proportional_gain = 6.5;
//   double z_proportional_gain = 6.5;
//   double x_integral_gain = 0.1;
//   double y_integral_gain = 0.1;
//   double z_integral_gain = 0.1;
//   double x_derivative_gain = 0.5;
//   double y_derivative_gain = 0.5;
//   double z_derivative_gain = 0.5;
//   double angular_proportional_gain = 5.5;
//   double angular_integral_gain = 0.01;
//   double angular_derivative_gain = 0.2;
// };

enum class DirectionalManipulabilityMode
{
  TRANSLATIONAL,
  ROTATIONAL,
  BOTH
};

struct SubmissionReporting
{
  std::string config_id;

  // Robot initial pose in base/world frame [mm] and quaternion [x, y, z, w].
  double robot_initial_pose_x_mm = 0.0;
  double robot_initial_pose_y_mm = 0.0;
  double robot_initial_pose_z_mm = 0.0;
  double robot_initial_pose_q1 = 0.0;
  double robot_initial_pose_q2 = 0.0;
  double robot_initial_pose_q3 = 0.0;
  double robot_initial_pose_q4 = 1.0;

  // End-effector command frame position [mm].
  double ee_command_pose_x_mm = 0.0;
  double ee_command_pose_y_mm = 0.0;
  double ee_command_pose_z_mm = 0.0;

  // Contact timestamps [ms] from process start.
  std::optional<int64_t> t_human_last_contact_ms;
  std::optional<int64_t> t_robot_first_contact_ms;
  std::optional<int64_t> t_robot_last_contact_ms;
};

static geometry_msgs::msg::Quaternion normalizedQuaternion(geometry_msgs::msg::Quaternion q)
{
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (norm > 1e-9)
  {
    q.x /= norm;
    q.y /= norm;
    q.z /= norm;
    q.w /= norm;
  }
  else
  {
    q.x = 0.0;
    q.y = 0.0;
    q.z = 0.0;
    q.w = 1.0;
  }
  return q;
}

static geometry_msgs::msg::Pose interpolatePoseLinearSlerp(const geometry_msgs::msg::Pose& start_pose,
                                                           const geometry_msgs::msg::Pose& goal_pose,
                                                           double alpha)
{
  alpha = std::clamp(alpha, 0.0, 1.0);

  geometry_msgs::msg::Pose out;
  out.position.x = start_pose.position.x + alpha * (goal_pose.position.x - start_pose.position.x);
  out.position.y = start_pose.position.y + alpha * (goal_pose.position.y - start_pose.position.y);
  out.position.z = start_pose.position.z + alpha * (goal_pose.position.z - start_pose.position.z);

  // Keep a fixed orientation during the interpolation pre-ramp.
  out.orientation = goal_pose.orientation;
  return out;
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

  tf2::Quaternion q_start;
  tf2::Quaternion q_goal;
  tf2::fromMsg(start_pose.orientation, q_start);
  tf2::fromMsg(goal_pose.orientation, q_goal);
  q_start.normalize();
  q_goal.normalize();

  const tf2::Quaternion q_interp = q_start.slerp(q_goal, alpha);
  out.orientation = tf2::toMsg(q_interp);
  return out;
}

// static std::string toLowerAscii(std::string value)
// {
//   std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
//     return static_cast<char>(std::tolower(c));
//   });
//   return value;
// }

// static std::string normalizeLabelForClassification(std::string value)
// {
//   std::string normalized;
//   normalized.reserve(value.size());

//   bool previous_was_space = true;
//   for (unsigned char c : value)
//   {
//     if (std::isalnum(c))
//     {
//       normalized.push_back(static_cast<char>(std::tolower(c)));
//       previous_was_space = false;
//     }
//     else if (!previous_was_space)
//     {
//       normalized.push_back(' ');
//       previous_was_space = true;
//     }
//   }

//   if (!normalized.empty() && normalized.back() == ' ')
//     normalized.pop_back();

//   return normalized;
// }

static std::vector<float> build_gripper_joint_angle_range(float grasp_joint_angle,
                                                      bool use_joint_angle_interval,
                                                      float delta = 0.25f,
                                                      int num_steps = 100,
                                                      float min_joint_angle = 0.0f,
                                                      float max_joint_angle = 0.69f,
                                                      double non_linear_exponent = 10.0)
{
  std::vector<float> joint_angle_range;
  if (!use_joint_angle_interval)
  {
    joint_angle_range.push_back(std::clamp(grasp_joint_angle, min_joint_angle, max_joint_angle));
    return joint_angle_range;
  }

  // const float lower_bound = std::max(min_joint_angle, grasp_joint_angle - delta / 2.0f);
  const float lower_bound = std::max(min_joint_angle, grasp_joint_angle - delta / 4.0f);
  //const float lower_bound = grasp_joint_angle; 
  const float upper_bound = std::min(max_joint_angle, grasp_joint_angle + delta / 2.0f);
  const float actual_delta = std::max(0.0f, upper_bound - lower_bound);

  if (num_steps <= 1 || actual_delta <= 0.0f)
  {
    joint_angle_range.push_back(std::clamp(grasp_joint_angle, min_joint_angle, max_joint_angle));
    return joint_angle_range;
  }

  joint_angle_range.reserve(static_cast<size_t>(num_steps));
  for (int i = 0; i < num_steps; ++i)
  {
    const double alpha = static_cast<double>(i) / static_cast<double>(num_steps - 1);
    const double eased_alpha = 1.0 - std::pow(1.0 - alpha, non_linear_exponent);
    const float joint_angle = lower_bound + static_cast<float>(eased_alpha * actual_delta);
    joint_angle_range.push_back(std::clamp(joint_angle, min_joint_angle, max_joint_angle));
  }

  return joint_angle_range;
}

static double applyInterpolationProfile(double alpha_linear, const std::string& profile)
{
  const double a = std::clamp(alpha_linear, 0.0, 1.0);
  const std::string p = profile; //toLowerAscii(profile);

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

static bool writeSubmissionReportJson(const std::string& file_path, const SubmissionReporting& report)
{
  std::ofstream out(file_path, std::ios::out | std::ios::trunc);
  if (!out.is_open())
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Failed to open report file: " << file_path);
    return false;
  }

  out << "{\n"
      << "  \"config_id\": " << std::quoted(report.config_id) << ",\n"
      << "  \"robot_initial_pose_x_mm\": " << report.robot_initial_pose_x_mm << ",\n"
      << "  \"robot_initial_pose_y_mm\": " << report.robot_initial_pose_y_mm << ",\n"
      << "  \"robot_initial_pose_z_mm\": " << report.robot_initial_pose_z_mm << ",\n"
      << "  \"robot_initial_pose_q1\": " << report.robot_initial_pose_q1 << ",\n"
      << "  \"robot_initial_pose_q2\": " << report.robot_initial_pose_q2 << ",\n"
      << "  \"robot_initial_pose_q3\": " << report.robot_initial_pose_q3 << ",\n"
      << "  \"robot_initial_pose_q4\": " << report.robot_initial_pose_q4 << ",\n"
      << "  \"ee_command_pose_x_mm\": " << report.ee_command_pose_x_mm << ",\n"
      << "  \"ee_command_pose_y_mm\": " << report.ee_command_pose_y_mm << ",\n"
      << "  \"ee_command_pose_z_mm\": " << report.ee_command_pose_z_mm << ",\n"
      << "  \"t_human_last_contact_ms\": " << report.t_human_last_contact_ms.value_or(0) << ",\n"
      << "  \"t_robot_first_contact_ms\": " << report.t_robot_first_contact_ms.value_or(0) << ",\n"
      << "  \"t_robot_last_contact_ms\": " << report.t_robot_last_contact_ms.value_or(0) << "\n"
      << "}\n";

  out.close();
  return true;
}

bool evaluate_pose_anisotropy(const moveit::planning_interface::MoveGroupInterface& move_group,
                                  const geometry_msgs::msg::PoseStamped& target_pose,
                                  double& condition_number,
                                  double& min_singular_value)
{
  condition_number = std::numeric_limits<double>::infinity();
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
    RCLCPP_WARN_STREAM(LOGGER, "[PoseQuality] No IK solution found for target pose");
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

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd singular_values = svd.singularValues();
  if (singular_values.size() == 0)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[PoseQuality] No singular values available for Jacobian");
    return false;
  }

  min_singular_value = singular_values.minCoeff();
  if (min_singular_value < 1e-6)
  {
    condition_number = std::numeric_limits<double>::infinity();
  }
  else
  {
    condition_number = singular_values.maxCoeff() / min_singular_value;
  }

  return true;
}

bool evaluate_pose_manipulability(const moveit::planning_interface::MoveGroupInterface& move_group,
                                  const geometry_msgs::msg::PoseStamped& target_pose,
                                  double& manipulability,
                                  double& min_singular_value)
{
  manipulability = 0.0;
  min_singular_value = 0.0;

  // RCLCPP_INFO_STREAM(LOGGER, "Evaluating manipulability for target pose: [" 
  //                 << target_pose.pose.position.x << ", "
  //                 << target_pose.pose.position.y << ", "
  //                 << target_pose.pose.position.z << "]");

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
    RCLCPP_WARN_STREAM(LOGGER, "[PoseQuality] No IK solution found for target pose");
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

bool evaluate_pose_directional_manipulability(
    const moveit::planning_interface::MoveGroupInterface& move_group,
    const geometry_msgs::msg::PoseStamped& target_pose,
    const Eigen::Vector3d& translational_direction,
    const Eigen::Vector3d& rotational_direction,
    DirectionalManipulabilityMode mode,
    double& directional_manipulability,
    double& min_singular_value)
{
  directional_manipulability = 0.0;
  min_singular_value = 0.0;

  auto current_state_ptr = move_group.getCurrentState(1.0);
  if (!current_state_ptr)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[DirectionalManipulability] Could not get current robot state");
    return false;
  }

  moveit::core::RobotState ik_state(*current_state_ptr);
  const moveit::core::JointModelGroup* jmg = ik_state.getJointModelGroup(move_group.getName());
  if (!jmg)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[DirectionalManipulability] Joint model group not found: " << move_group.getName());
    return false;
  }

  std::string eef_link = move_group.getEndEffectorLink();
  if (eef_link.empty())
  {
    const auto& link_names = jmg->getLinkModelNames();
    if (link_names.empty())
    {
      RCLCPP_WARN_STREAM(LOGGER, "[DirectionalManipulability] No link names found in joint model group");
      return false;
    }
    eef_link = link_names.back();
  }

  const bool ik_found = ik_state.setFromIK(jmg, target_pose.pose, eef_link, 0.05);
  if (!ik_found)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[DirectionalManipulability] No IK solution found for target pose");
    return false;
  }

  const auto* eef_link_model = ik_state.getLinkModel(eef_link);
  if (!eef_link_model)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[DirectionalManipulability] Could not get link model for end effector: " << eef_link);
    return false;
  }

  const Eigen::Vector3d reference_point_position = Eigen::Vector3d::Zero();
  Eigen::MatrixXd jacobian;
  ik_state.getJacobian(jmg, eef_link_model, reference_point_position, jacobian);

  if (jacobian.rows() != 6 || jacobian.cols() == 0)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[DirectionalManipulability] Invalid Jacobian shape: "
                                  << jacobian.rows() << "x" << jacobian.cols());
    return false;
  }

  Eigen::MatrixXd selected_jacobian;
  Eigen::RowVectorXd projected_row;

  if (mode == DirectionalManipulabilityMode::TRANSLATIONAL)
  {
    const double dir_norm = translational_direction.norm();
    if (dir_norm < 1e-9)
    {
      RCLCPP_WARN_STREAM(LOGGER, "[DirectionalManipulability] Translational direction norm is near zero");
      return false;
    }

    const Eigen::Vector3d t = translational_direction / dir_norm;
    selected_jacobian = jacobian.topRows(3);
    projected_row = t.transpose() * selected_jacobian;
  }
  else if (mode == DirectionalManipulabilityMode::ROTATIONAL)
  {
    const double axis_norm = rotational_direction.norm();
    if (axis_norm < 1e-9)
    {
      RCLCPP_WARN_STREAM(LOGGER, "[DirectionalManipulability] Rotational direction norm is near zero");
      return false;
    }

    const Eigen::Vector3d r = rotational_direction / axis_norm;
    selected_jacobian = jacobian.bottomRows(3);
    projected_row = r.transpose() * selected_jacobian;
  }
  else
  {
    const double t_norm = translational_direction.norm();
    const double r_norm = rotational_direction.norm();
    if (t_norm < 1e-9 && r_norm < 1e-9)
    {
      RCLCPP_WARN_STREAM(LOGGER, "[DirectionalManipulability] Both translational and rotational directions are near zero");
      return false;
    }

    Eigen::Matrix<double, 6, 1> combined_direction = Eigen::Matrix<double, 6, 1>::Zero();
    if (t_norm >= 1e-9)
      combined_direction.head<3>() = translational_direction / t_norm;
    if (r_norm >= 1e-9)
      combined_direction.tail<3>() = rotational_direction / r_norm;

    combined_direction.normalize();
    selected_jacobian = jacobian;
    projected_row = combined_direction.transpose() * selected_jacobian;
  }

  directional_manipulability = projected_row.norm();

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(selected_jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
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

float getGripperAngleFromWidth(float width, float grip_offset = 0.0f)
{
  constexpr float max_stroke = 0.140;
  constexpr float max_joint_limit = 0.67; // radians, finger_joint limits are 0 to 0.69 but fingers can't fully close

  if (width - grip_offset < 0.0f)
  {
    width = grip_offset; // Set to grip_offset to achieve fully closed position
  }
  width -= grip_offset;
  float target_width = std::min(std::max(width, 0.0f), max_stroke);
  float joint_angle = max_joint_limit * (1.0 - target_width / max_stroke);
  joint_angle = std::min(std::max(joint_angle, 0.0f), max_joint_limit);
  return joint_angle;
}

float getRobotiqVerticalDisplacement(float joint_angle)
{
  constexpr float max_height = 0.0235; // maximum vertical displacement when fully closed (m)
  constexpr float max_joint_limit = 0.7; // radians

  float displacement = max_height * (joint_angle / max_joint_limit);
  return displacement;
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

  moveit_servo::StatusCode getStatus()
  {
    return status_;
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
      // geometry_msgs::msg::PoseStamped current_pose,
      const std::shared_ptr<tf2_ros::Buffer>& tf_buffer,
      const HandoverConfig& config,
      bool evaluate_grasp_poses = true,
      const std::string& base_frame = "base",
      const std::string& base_link_frame = "base_link",
      const std::string& command_frame = "tool_changer_tool0",
      const std::string& grasp_frame = "grasp_center",
      const std::string& goal_topic = "/goal_pose",
      const std::string& handover_topic = "/handover_detected",
      const std::string& dimensions_topic = "/estimated_dimensions",
      const std::string& replanned_grasps_topic = "/replanned_grasps"
      // const std::string& gripper_pose_topic = "gripper_goal_pose"
    ) : node_(node)
  {
    received_pose_ = false;
    start_pose_initialized_ = false;
    received_gripper_pose_ = false;
    handover_detected_ = false;
    tf_buffer_ = tf_buffer;
    base_frame_ = base_frame;
    base_link_frame_ = base_link_frame;
    command_frame_ = command_frame;
    grasp_frame_ = grasp_frame;
    config_ = config;
    start_approach_ = false;
    evaluate_grasp_poses_ = evaluate_grasp_poses;

    // debug - remove later
    // debug_publisher_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>("goal_pose_filtered", 10);

    // Initialize action client for grasp generation
    grasp_action_client_ = rclcpp_action::create_client<grasp_generation::action::GenerateGrasp>(
        node, "generate_grasp");
    grasp_generation_in_progress_ = false;
    last_grasp_request_time_ = node_->now();
    peak_object_speed_ = 0.0;
    last_object_speed_ = -1.0;
    deceleration_samples_ = 0;

    touchdown_service_client_ = node_->create_client<std_srvs::srv::Trigger>("detect_touchdown");
    touchdown_status_sub_ = node_->create_subscription<std_msgs::msg::Bool>(
      "/touchdown_status", rclcpp::SystemDefaultsQoS(),
      std::bind(&GoalPoseSubscriber::touchdownStatusCallback, this, std::placeholders::_1));
    touchdown_detection_in_progress_ = false;
    touchdown_detected_ = false;
    giovanni_event_.disconnected = true;

    // Real-time Filtering configuration
    node_->declare_parameter("enable_real_time_filtering", true);
    enable_real_time_filtering_ = node_->get_parameter("enable_real_time_filtering").as_bool();
    node_->declare_parameter("rt_filter_frequency", 120.0);
    frequency_ = node_->get_parameter("rt_filter_frequency").as_double();
    node_->declare_parameter("rt_filter_mincutoff", 0.5);
    mincutoff_ = node_->get_parameter("rt_filter_mincutoff").as_double();
    node_->declare_parameter("rt_filter_beta", 0.4);
    beta_ = node_->get_parameter("rt_filter_beta").as_double();
    node_->declare_parameter("rt_filter_dcutoff", 1.0);
    dcutoff_ = node_->get_parameter("rt_filter_dcutoff").as_double();
    node_->declare_parameter("rt_filter_max_linear_speed", 1.0);
    max_linear_speed_ = node_->get_parameter("rt_filter_max_linear_speed").as_double();
    node_->declare_parameter("rt_filter_max_consecutive_outliers", 5);
    max_consecutive_outliers_ = node_->get_parameter("rt_filter_max_consecutive_outliers").as_int();

    if (enable_real_time_filtering_) {
      filter_x_ = std::make_unique<OneEuroFilter>(frequency_, mincutoff_, beta_, dcutoff_);
      filter_y_ = std::make_unique<OneEuroFilter>(frequency_, mincutoff_, beta_, dcutoff_);
      filter_z_ = std::make_unique<OneEuroFilter>(frequency_, mincutoff_, beta_, dcutoff_);
      RCLCPP_INFO(LOGGER, "Real-time One-Euro filtering enabled for goal pose.");
    }

    // Create gripper action client and tactile/object-detection subscriptions
    gripper_action_client_ = rclcpp_action::create_client<GripperCommand>(node_, "/gripper_position_controller/gripper_cmd");

    dynamic_joint_state_sub_ = node_->create_subscription<control_msgs::msg::DynamicJointState>(
        "/dynamic_joint_states", rclcpp::SensorDataQoS(),
        [this](const control_msgs::msg::DynamicJointState::SharedPtr msg) {
          for (size_t joint_idx = 0; joint_idx < msg->joint_names.size() && joint_idx < msg->interface_values.size(); ++joint_idx)
          {
            const auto& joint_name = msg->joint_names[joint_idx];
            const auto& interface_value = msg->interface_values[joint_idx];
            for (size_t i = 0; i < interface_value.interface_names.size() && i < interface_value.values.size(); ++i)
            {
              const auto& interface_name = interface_value.interface_names[i];

              if (joint_name == "finger_joint" && interface_name == "position")
              {
                latest_finger_joint_position_.store(interface_value.values[i]);
                continue;
              }

              if (interface_name != "object_detection_status")
                continue;
              const int status = static_cast<int>(interface_value.values[i]);
              const int previous_status = latest_gripper_status_.exchange(status);
              // if (status != previous_status)
              //   RCLCPP_INFO(LOGGER, "[%s] object_detection_status = %s (%d)", joint_name.c_str(), objectDetectionStatusToString(status).c_str(), status);
              if (status == 2)
                object_detected_during_grasp_.store(true);
            }
          }
        });

    alfredo_sub_ = node_->create_subscription<tactile_detection::msg::TactileDetection>(
        "/ALFREDO/Contact", 10, //rclcpp::SensorDataQoS(),
        [this](const tactile_detection::msg::TactileDetection::SharedPtr msg) {
          alfredo_event_.disconnected.store(msg->disconnected);
          if (!start_grasping_.load())
            return;
          if (msg->detected)
            alfredo_event_.detected.store(msg->detected);
          if (msg->detected || msg->disconnected)
            alfredo_sub_.reset();
        });

    giovanni_sub_ = node_->create_subscription<tactile_detection::msg::TactileDetection>(
        "/GIOVANNI/Contact", 10, //rclcpp::SensorDataQoS(),
        [this](const tactile_detection::msg::TactileDetection::SharedPtr msg) {
          giovanni_event_.disconnected.store(msg->disconnected);
          if (!start_grasping_.load())
            return;
          if (msg->detected)
            giovanni_event_.detected.store(msg->detected);
          if (msg->detected || msg->disconnected)
            giovanni_sub_.reset();
        });

    handover_sub_ = node->create_subscription<std_msgs::msg::Bool>(handover_topic, rclcpp::SystemDefaultsQoS(),
        [this](const std_msgs::msg::Bool::ConstSharedPtr& msg) {
          return handoverCB(msg);
        });
    dim_sub_ = node->create_subscription<visual_properties::msg::Dimensions>(dimensions_topic, rclcpp::SystemDefaultsQoS(),
        [this](const visual_properties::msg::Dimensions::ConstSharedPtr& msg) {
          return dimensionsCB(msg);
        });
    pose_sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        goal_topic, rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg) { return goalPoseCB(msg); });

    replanned_grasps_sub_ = node->create_subscription<grasp_generation::action::GenerateGrasp_Result>(
        replanned_grasps_topic, rclcpp::SystemDefaultsQoS(),
        [this](const grasp_generation::action::GenerateGrasp_Result::ConstSharedPtr& msg) {
          {
            std::lock_guard<std::mutex> lock(mutex_);
            tmp_gripper_poses_ = msg->grasp_goals;
            tmp_grasp_types_ = msg->grasp_labels;
            tmp_grasp_widths_ = msg->grasp_widths;
            received_gripper_pose_ = true;
            regrasp_requested_ = true;
          }

          replanned_grasps_sub_.reset();
          RCLCPP_INFO_STREAM(LOGGER,
                             "Received replanned grasps from /replanned_grasps with "
                                 << msg->grasp_goals.poses.size() << " candidates");
        });

    getTF();
  }

  // Internal method: Estimate velocity from stored pose history (assumes mutex is already locked)
  bool estimateVelocityInternal(Eigen::Vector3d& velocity)
  {
    if (pose_history_.size() < config_.velocity_estimation_frames)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Not enough pose history for velocity estimation: " << pose_history_.size() << " frames");
      return false;
    }

    // Get vectors at the end of the history
    const auto& oldest_pose = pose_history_.front();
    const auto& newest_pose = pose_history_.back();
    
    Eigen::Vector3d position_diff = 
        Eigen::Vector3d(newest_pose.pose.position.x - oldest_pose.pose.position.x,
                       newest_pose.pose.position.y - oldest_pose.pose.position.y,
                       newest_pose.pose.position.z - oldest_pose.pose.position.z);
    
    const double oldest_stamp = rclcpp::Time(oldest_pose.header.stamp).seconds();
    const double newest_stamp = rclcpp::Time(newest_pose.header.stamp).seconds();
    const double time_diff = newest_stamp - oldest_stamp;
    if (time_diff < 1e-3)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Velocity estimation skipped due to small/non-positive stamp delta: " << time_diff << " s");
      return false;
    }
    
    velocity = position_diff / time_diff;
    // RCLCPP_INFO_STREAM(LOGGER, "Estimated velocity: [" << velocity.x() << ", " << velocity.y() << ", " << velocity.z()
    //                       << "] m/s based on " << pose_history_.size() << " frames, dt=" << time_diff << " s");
    return true;
  }

  void setTableVerticalOffset(float offset)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    table_z_offset_ = offset;
  }

  void setPlacementStrategy(const std::string& strategy)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    placement_strategy_ = strategy;
  }

  void setPredefinedPlacementCoordinates(float x, float y)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    predefined_placement_x_ = x;
    predefined_placement_y_ = y;
  }

  void setStartRobotPose(const geometry_msgs::msg::PoseStamped& pose)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tool_start_pose_ = Eigen::Vector3d(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
    start_pose_orientation_ = Eigen::Quaterniond(pose.pose.orientation.w,
                                                 pose.pose.orientation.x,
                                                 pose.pose.orientation.y,
                                                 pose.pose.orientation.z);
    start_pose_orientation_.normalize();
    start_pose_initialized_ = true;
  }

  void setWorkspaceLimits(float x_min, float x_max, float y_min, float y_max)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    workspace_x_min_ = x_min;
    workspace_x_max_ = x_max;
    workspace_y_min_ = y_min;
    workspace_y_max_ = y_max;
  }

  void adjustTouchdownToleranceForGeometry(float& tolerance)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Adjust touchdown tolerance to be at least table_z_offset + object_height
    float z_min = table_z_offset_ + latest_dimensions_.height;
    tolerance = std::max(tolerance, z_min);
  }

  bool canStopGraspApproach(float threshold, float obj_threshold, const geometry_msgs::msg::PoseStamped& current_grasping_pose, const geometry_msgs::msg::PoseStamped& current_tool_pose, geometry_msgs::msg::PoseStamped& gripper_pose, const bool motion_compensate = false)
  {
    constexpr float compensation_threshold = 0.1; // m, only apply compensation if object is far to avoid unnecessary compensation for small errors that are within tolerance
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    const Eigen::Vector3d current_grasp_pos(
      current_grasping_pose.pose.position.x,
      current_grasping_pose.pose.position.y,
      current_grasping_pose.pose.position.z);

    const Eigen::Vector3d latest_object_pos(
      latest_pose_.pose.position.x,
      latest_pose_.pose.position.y,
      latest_pose_.pose.position.z);
    const double distance_to_object = (current_grasp_pos - latest_object_pos).norm();

    if (motion_compensate && distance_to_object > compensation_threshold)
      applyObjectMotionCompensationToPoseInternal(gripper_pose, motion_compensate, true);
    // else
    //   RCLCPP_INFO_STREAM(LOGGER, "Skipping motion compensation for grasp approach since object is within compensation threshold: " << distance_to_object << " m");

    const Eigen::Vector3d current_tool_pos(
      current_tool_pose.pose.position.x,
      current_tool_pose.pose.position.y,
      current_tool_pose.pose.position.z);
    const Eigen::Vector3d goal_gripper_pos(
      gripper_pose.pose.position.x,
      gripper_pose.pose.position.y,
      gripper_pose.pose.position.z);
    
    const double distance_to_goal_grasp = (goal_gripper_pos - current_tool_pos).norm();

    obj_threshold = obj_threshold + robotiq_vertical_displacement_;
    const bool goal_reached = distance_to_goal_grasp < threshold;
    const bool object_contact = distance_to_object < obj_threshold;

    RCLCPP_INFO_STREAM(LOGGER,
               "[CanStopTracking] grasp->goal: " << distance_to_goal_grasp << " m (" 
               << (goal_reached ? "PASS" : "FAIL") << ")"
               << " | grasp->object: " << distance_to_object << " m ("
               << (object_contact ? "PASS" : "FAIL") << ")"
               << " | threshold: " << threshold << " m");

    // return goal_reached || object_contact; // competition
    return goal_reached;
  }

  // Internal method: Predict future object pose based on current velocity (assumes mutex is already locked)
  geometry_msgs::msg::PoseStamped predictFutureObjectPoseInternal(const geometry_msgs::msg::PoseStamped& current_object_pose)
  {
    Eigen::Vector3d velocity;
    if (!estimateVelocityInternal(velocity))
    {
      RCLCPP_WARN_STREAM(LOGGER, "Failed to estimate velocity for future pose prediction.");
      return current_object_pose;
    }
    double prediction_time = config_.prediction_frames * frame_time_;
    
    geometry_msgs::msg::PoseStamped predicted_pose = current_object_pose;
    predicted_pose.pose.position.x += velocity.x() * prediction_time;
    predicted_pose.pose.position.y += velocity.y() * prediction_time;
    predicted_pose.pose.position.z += velocity.z() * prediction_time;
    
    return predicted_pose;
  }

  // Run the closed-loop gripper control using MoveIt current-state feedback.
  void runClosedLoopGripperControl(moveit::planning_interface::MoveGroupInterface&,
                                   const std::vector<float>&);
  void runFastClosedLoopGripperControl(const std::vector<float>&);

  // Lightweight speed estimate from the latest two samples (no logging). TODO: REMOVE later
  // bool estimateCurrentSpeedInternal(double& speed)
  // {
  //   speed = 0.0;
  //   if (pose_history_.size() < 2)
  //   {
  //     return false;
  //   }

  //   const auto& prev_pose = pose_history_[pose_history_.size() - 2];
  //   const auto& curr_pose = pose_history_.back();

  //   const Eigen::Vector3d p_prev(prev_pose.pose.position.x,
  //                                prev_pose.pose.position.y,
  //                                prev_pose.pose.position.z);
  //   const Eigen::Vector3d p_curr(curr_pose.pose.position.x,
  //                                curr_pose.pose.position.y,
  //                                curr_pose.pose.position.z);

  //   const double dt = rclcpp::Time(curr_pose.header.stamp).seconds() -
  //                     rclcpp::Time(prev_pose.header.stamp).seconds();
  //   if (dt <= 1e-4)
  //   {
  //     return false;
  //   }

  //   speed = (p_curr - p_prev).norm() / dt;
  //   return true;
  // }

  // Shrink tracking distance from approach_distance toward tracking_distance.
  // High object speed => faster shrink (intercept behavior), low speed => very slow or no shrink.
  double updateDynamicTrackingDistanceInternal(double current_speed)
  {
    if (dynamic_tracking_distance_ < 0.0)
    {
      dynamic_tracking_distance_ = config_.approach_distance;
    }

    const double dt = (frame_time_ > 1e-4 && frame_time_ < 0.2) ? frame_time_ : 0.01;

    // Tune these if needed; max speed around 0.5 m/s maps to fastest shrink.
    const double v_ref = 0.5;
    const double min_shrink_rate = 0.05;   // m/s distance change when object is almost static
    const double max_shrink_rate = 0.5;  // m/s distance change when object is fast

    const double speed_ratio = std::min(1.0, std::max(0.0, current_speed / v_ref));
    const double shrink_rate = min_shrink_rate + (max_shrink_rate - min_shrink_rate) * speed_ratio;

    dynamic_tracking_distance_ -= shrink_rate * dt;
    dynamic_tracking_distance_ = std::max(static_cast<double>(config_.tracking_distance), dynamic_tracking_distance_);

    return dynamic_tracking_distance_;
  }

  // shifts the goal pose back, goal is in tool0 frame
  void getApproachPose(geometry_msgs::msg::PoseStamped& goal, 
                    const geometry_msgs::msg::PoseStamped& current_pose, 
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
        double offset_from_target = approach_distance; //+ grasp_center_shift_.norm();
        
        Eigen::Vector3d T_flange_target = T_goal - (unit_ray * offset_from_target);

        goal.pose.position.x = T_flange_target.x();
        goal.pose.position.y = T_flange_target.y();
        goal.pose.position.z = T_flange_target.z();
    }
  }

  void getTF()
  {
    try
    {
      T_grasp_to_tool0_msg_ = 
        tf_buffer_->lookupTransform(grasp_frame_, command_frame_, rclcpp::Time(0));
      grasp_center_shift_ = transformToMatrix(T_grasp_to_tool0_msg_.transform).translation();

      // Get transform from base to base_link (dynamic transform)
      T_baselink_to_base_msg_ = tf_buffer_->lookupTransform(base_link_frame_, base_frame_, rclcpp::Time(0));
      T_baselink_to_base_ = transformToMatrix(T_baselink_to_base_msg_.transform);
      tf_initialized_ = true;
    }
    catch (const tf2::TransformException& ex)
    {
      RCLCPP_WARN_STREAM(LOGGER, "TF lookup failed during initialization: " << ex.what());
    }
  }

  bool getApproachGoalPose(geometry_msgs::msg::PoseStamped& pose, 
                          geometry_msgs::msg::PoseStamped& current_pose)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!start_approach_) {
      RCLCPP_INFO_STREAM(LOGGER, "Handover not detected or frame time not estimated yet. Cannot compute approach goal pose.");
      return false;
    }

        // Predict future object pose based on velocity estimation (use internal method - mutex already locked)
        geometry_msgs::msg::PoseStamped predicted_object_pose = predictFutureObjectPoseInternal(latest_pose_);

        // Convert back to pose, output in world frame
        pose.header.frame_id = base_link_frame_;
        pose.header.stamp = predicted_object_pose.header.stamp;
        pose.pose = predicted_object_pose.pose;
        
        pose.pose.orientation = current_pose.pose.orientation;  // Keep original orientation
        
        // Modify pose to be the approach pose (40 cm away, facing object)
        getApproachPose(pose, current_pose, config_.approach_distance); 
        
        // Orient gripper to face the predicted object
        orientGripperTowardObject(pose, predicted_object_pose);

        // correct the orientation to match current end-effector Y-axis up/down direction (flip 180 around Z if opposite)
        correctEEOrientation(pose, current_pose);

        RCLCPP_INFO_STREAM(LOGGER, "Computed approach goal pose based on predicted object pose and velocity estimation.");
        
        return true;
  }

  bool getTrackingGoalPose(geometry_msgs::msg::PoseStamped& pose, 
                          geometry_msgs::msg::PoseStamped& current_pose, 
                          bool predict_future = false,
                          bool reorient_gripper = false)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!received_pose_)
      return false;

    geometry_msgs::msg::PoseStamped reference_object_pose;

    if (predict_future)
    {
      // Predict future object pose based on velocity estimation (use internal method - mutex already locked)
      reference_object_pose = predictFutureObjectPoseInternal(latest_pose_);
    }
    else
    {
      reference_object_pose = latest_pose_;
    }

    pose = reference_object_pose;
        
    pose.pose.orientation = current_pose.pose.orientation;  // Keep original orientation
        
    // Modify pose to be the tracking pose with dynamic intercept distance.
    // double current_speed = 0.0;
    Eigen::Vector3d velocity;
    if (!estimateVelocityInternal(velocity))
    {
      velocity.setZero();
    }
    double current_speed = velocity.norm();
    const double dynamic_tracking_distance = updateDynamicTrackingDistanceInternal(current_speed);
    getApproachPose(pose, current_pose, dynamic_tracking_distance);

    // Smooth commanded position to avoid jerky back-and-forth target jumps.
    const Eigen::Vector3d raw_cmd(pose.pose.position.x,
                                  pose.pose.position.y,
                                  pose.pose.position.z);
    Eigen::Vector3d filtered_cmd = raw_cmd;

    const double speed_ratio = std::min(1.0, std::max(0.0, current_speed / 0.5));
    const double alpha = 0.20 + 0.50 * speed_ratio;  // faster updates when object is faster

    const Eigen::Vector3d object_pos(reference_object_pose.pose.position.x,
                                     reference_object_pose.pose.position.y,
                                     reference_object_pose.pose.position.z);

    if (tracking_target_initialized_)
    {
      const Eigen::Vector3d prev_cmd(tracking_target_pose_.pose.position.x,
                                     tracking_target_pose_.pose.position.y,
                                     tracking_target_pose_.pose.position.z);

      filtered_cmd = prev_cmd + alpha * (raw_cmd - prev_cmd);

      // Prevent retreat from object due to noisy updates: allow only tiny increases in object distance.
      const double prev_obj_dist = (prev_cmd - object_pos).norm();
      double new_obj_dist = (filtered_cmd - object_pos).norm();
      const double kMaxRetreatPerUpdate = 0.002;  // 2 mm

      if (new_obj_dist > prev_obj_dist + kMaxRetreatPerUpdate)
      {
        Eigen::Vector3d dir = filtered_cmd - object_pos;
        if (dir.norm() > 1e-9)
        {
          dir.normalize();
          filtered_cmd = object_pos + dir * (prev_obj_dist + kMaxRetreatPerUpdate);
          new_obj_dist = (filtered_cmd - object_pos).norm();
        }
      }

      (void)new_obj_dist;
    }

    pose.pose.position.x = filtered_cmd.x();
    pose.pose.position.y = filtered_cmd.y();
    pose.pose.position.z = filtered_cmd.z();

    tracking_target_pose_ = pose;
    tracking_target_initialized_ = true;

    RCLCPP_INFO_STREAM_THROTTLE(LOGGER,
                                *node_->get_clock(),
                                500,
                                "Dynamic tracking distance=" << dynamic_tracking_distance
                                                              << " m, object speed=" << current_speed << " m/s");

    
    if (reorient_gripper)
    {
      orientGripperTowardObject(pose, reference_object_pose);
    }
    // correct the orientation to match current end-effector Y-axis up/down direction (flip 180 around Z if opposite)
    correctEEOrientation(pose, current_pose);
        
    return true;
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

  bool replanningNeeded() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return regrasp_requested_;
  }

  geometry_msgs::msg::PoseStamped getInitialObjectPose() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return initial_object_pose_;
  }

  bool getRetreatPoseTowardStart(geometry_msgs::msg::PoseStamped& pose,
                                 const geometry_msgs::msg::PoseStamped& current_pose,
                                 double retreat_distance = 0.1) const
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!start_pose_initialized_)
    {
      return false;
    }

    const Eigen::Vector3d current_pos(current_pose.pose.position.x,
                                      current_pose.pose.position.y,
                                      current_pose.pose.position.z);
    const Eigen::Vector3d start_pos = tool_start_pose_;
    Eigen::Vector3d retreat_vector = start_pos - current_pos;
    const double distance_to_start = retreat_vector.norm();

    if (distance_to_start < 1e-9)
    {
      return false;
    }

    retreat_distance = std::min(retreat_distance, distance_to_start);
    retreat_vector.normalize();

    pose = current_pose;
    pose.pose.position.x += retreat_vector.x() * retreat_distance;
    pose.pose.position.y += retreat_vector.y() * retreat_distance;
    pose.pose.position.z += retreat_vector.z() * retreat_distance;

    Eigen::Quaterniond q_current(current_pose.pose.orientation.w,
                                 current_pose.pose.orientation.x,
                                 current_pose.pose.orientation.y,
                                 current_pose.pose.orientation.z);
    q_current.normalize();

    const double retreat_ratio = retreat_distance / distance_to_start;
    const Eigen::Quaterniond q_target = q_current.slerp(retreat_ratio, start_pose_orientation_);

    pose.pose.orientation.w = q_target.w();
    pose.pose.orientation.x = q_target.x();
    pose.pose.orientation.y = q_target.y();
    pose.pose.orientation.z = q_target.z();
    return true;
  }

  void unsubscribeFromObjectPose()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pose_sub_.reset();
  }

  // Build a placement command in tool0 frame such that grasp_center aligns with initial object center.
  void getPlacementObjectPoseShifted(geometry_msgs::msg::PoseStamped& obj_initial_pose, const geometry_msgs::msg::Quaternion& desired_tool0_orientation) const
  {
    std::lock_guard<std::mutex> lock(mutex_);

    obj_initial_pose.header.stamp = node_->now();
    obj_initial_pose.header.frame_id = base_link_frame_;

    if (placement_strategy_ == "initial") {
      obj_initial_pose.pose = grasp_pose_.pose;
      return;
    }

    RCLCPP_INFO(LOGGER, "initial grasp position: x: %f, y: %f, z: %f", grasp_pose_.pose.position.x, grasp_pose_.pose.position.y, grasp_pose_.pose.position.z);

    Eigen::Isometry3d T_base_grasp_des = Eigen::Isometry3d::Identity();
    Eigen::Vector3d grasp_position;
 
    grasp_position.z() = 0.0;
    grasp_position.x() = predefined_placement_x_;
    grasp_position.y() = predefined_placement_y_;
    
    T_base_grasp_des.translation() = grasp_position;
    T_base_grasp_des.linear() = Eigen::Quaterniond(desired_tool0_orientation.w,
                                                   desired_tool0_orientation.x,
                                                   desired_tool0_orientation.y,
                                                   desired_tool0_orientation.z)
                                  .toRotationMatrix();

    // Convert desired grasp_center pose to tool0 command pose: T_base_tool0 = T_base_grasp * T_grasp_tool0
    const Eigen::Isometry3d T_grasp_to_tool0 = transformToMatrix(T_grasp_to_tool0_msg_.transform);
    const Eigen::Isometry3d T_base_tool0_des = T_base_grasp_des * T_grasp_to_tool0;

    obj_initial_pose.pose = matrixToPose(T_base_tool0_des);

    // Determine z-coordinate adjustment based on grasp type and object geometry
    float z_ratio = 0.4f; // default to mid grasp
    if (selected_grasp_type_ == kTopGraspType)
    {
      z_ratio = 0.75f; // 90% for top grasp
      RCLCPP_INFO_STREAM(LOGGER, "Using top grasp strategy with z_ratio=" << z_ratio);
    }
    else if (selected_grasp_type_ == kBottomGraspType)
    {
      z_ratio = 0.1f; // 20% for bottom grasp
      RCLCPP_INFO_STREAM(LOGGER, "Using bottom grasp strategy with z_ratio=" << z_ratio);
    }
    
    // Compute minimum z coordinate: table_z_offset + ratio * object_height
    float z_min = table_z_offset_ + z_ratio * latest_dimensions_.height;

    // Ensure the placement z coordinate is not below this minimum
    obj_initial_pose.pose.position.z = std::max(grasp_pose_.pose.position.z, static_cast<double>(table_z_offset_));//, static_cast<double>(z_min));
    RCLCPP_INFO(LOGGER, "computed placement position: x: %f, y: %f, z: %f", obj_initial_pose.pose.position.x, obj_initial_pose.pose.position.y, obj_initial_pose.pose.position.z);
  }

    // grasping pose is already in base_link frame, just need to apply predicted object centroid shift and orientation correction
  bool getGraspingPose(geometry_msgs::msg::PoseStamped& pose, const geometry_msgs::msg::PoseStamped& current_pose)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!received_gripper_pose_)
    {
      RCLCPP_WARN_STREAM(LOGGER, "Gripper pose not received yet");
      return false;
    }

    // Convert back to pose, output in base_link frame
    pose.header.frame_id = base_link_frame_;
    pose.header.stamp = grasp_pose_.header.stamp;
    pose.pose = grasp_pose_.pose;

    // Compare current and goal end-effector Y-axis up/down direction in the same frame.
    // If one points up (+Z) and the other down (-Z), flip goal by 180 degrees around Z.
    correctEEOrientation(pose, current_pose);

    return true;
  }

  // Fast grasp evaluation: hard-filter by min singular value, then rank by condition number.
  // Each candidate computes Jacobian/SVD once via evaluate_pose_anisotropy.
  void evaluateGraspPosesTwoStage(const moveit::planning_interface::MoveGroupInterface& move_group,
                                  geometry_msgs::msg::PoseStamped& pose,
                                  const geometry_msgs::msg::PoseStamped& current_pose,
                                  const bool update_motion_compensate = false)
  {
    regrasp_requested_ = false;
    constexpr double kMinSingularValueThreshold = 0.1 ; // If sigma_min is near zero, the robot is mathematically at a singularity
    constexpr double kConditionTieEpsilon = 1e-9;

    bool found_any_ik = false;
    bool found_filtered_candidate = false;
    geometry_msgs::msg::PoseStamped best_pose;
    double best_condition_number = std::numeric_limits<double>::infinity();
    double best_min_singular_value = -1.0;
    int best_candidate_index = -1;

    // Fallback (if no candidate passes min singular value threshold):
    geometry_msgs::msg::PoseStamped best_fallback_pose;
    double best_fallback_condition_number = std::numeric_limits<double>::infinity();
    double best_fallback_min_singular_value = -1.0;
    int best_fallback_candidate_index = -1;

    for (std::size_t i = 0; i < tmp_gripper_poses_.poses.size(); ++i)
    {
      const auto& tmp_pose = tmp_gripper_poses_.poses[i];
      geometry_msgs::msg::PoseStamped candidate_pose;
      candidate_pose.header = tmp_gripper_poses_.header;
      candidate_pose.pose = tmp_pose;

      if (!transformGraspingPose(candidate_pose, current_pose))
      {
        RCLCPP_ERROR_STREAM(LOGGER, "Failed to transform candidate grasp pose to base_link frame. Can't continue.");
        rclcpp::shutdown();
        exit(EXIT_FAILURE);
      }

      double condition_number = std::numeric_limits<double>::infinity();
      double min_singular_value = 0.0;

      if (!evaluate_pose_anisotropy(move_group, candidate_pose, condition_number, min_singular_value))
      {
        continue;
      }

      found_any_ik = true;

      // Track best fallback among all IK-valid candidates by condition number.
      if ((condition_number < best_fallback_condition_number - kConditionTieEpsilon) ||
          (std::abs(condition_number - best_fallback_condition_number) <= kConditionTieEpsilon &&
           min_singular_value > best_fallback_min_singular_value))
      {
        best_fallback_condition_number = condition_number;
        best_fallback_min_singular_value = min_singular_value;
        best_fallback_pose = candidate_pose;
        best_fallback_candidate_index = static_cast<int>(i);
      }

      // Hard filter.
      if (min_singular_value < kMinSingularValueThreshold)
      {
        continue;
      }

      // Rank filtered set by condition number, break ties by larger min singular value.
      if ((condition_number < best_condition_number - kConditionTieEpsilon) ||
          (std::abs(condition_number - best_condition_number) <= kConditionTieEpsilon &&
           min_singular_value > best_min_singular_value))
      {
        found_filtered_candidate = true;
        best_condition_number = condition_number;
        best_min_singular_value = min_singular_value;
        best_pose = candidate_pose;
        best_candidate_index = static_cast<int>(i);
      }
    }

    if (!found_any_ik)
    {
      RCLCPP_WARN_STREAM(LOGGER, "[TwoStageGrasps] No IK-valid candidate grasp poses found.");
      exit(EXIT_FAILURE); // debug
    }

    if (found_filtered_candidate)
    {
      pose = best_pose;
      grasp_pose_ = best_pose;
      if (best_candidate_index >= 0)
      {
        applyGraspTypeSelectionForCandidate(static_cast<std::size_t>(best_candidate_index));
      }
      applyObjectMotionCompensationToPose(pose, update_motion_compensate, true);
      RCLCPP_INFO_STREAM(LOGGER,
                         "[TwoStageGrasps] Selected grasp with hard min singular value filter (>= "
                             << kMinSingularValueThreshold << ")"
                             << ", min_sv=" << best_min_singular_value
                             << ", condition_number=" << best_condition_number);
      return;
    }

    // No candidate survived hard filter: fallback to best condition number among IK-valid candidates.
    pose = best_fallback_pose;
    grasp_pose_ = best_fallback_pose;
    if (best_fallback_candidate_index >= 0)
    {
      applyGraspTypeSelectionForCandidate(static_cast<std::size_t>(best_fallback_candidate_index));
    }
    applyObjectMotionCompensationToPose(pose, update_motion_compensate, true);
    RCLCPP_WARN_STREAM(LOGGER,
                       "[TwoStageGrasps] No candidate passed min singular value threshold ("
                           << kMinSingularValueThreshold
                           << "). Falling back to best-condition candidate: min_sv="
                           << best_fallback_min_singular_value
                           << ", condition_number=" << best_fallback_condition_number);
  }

  bool gotGraspingGoal() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_gripper_pose_;
  }

  bool shouldStartReach(const geometry_msgs::msg::Pose& current_pose, int stable_samples_required = 3, bool wait_for_grasp_evaluation = false)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!received_gripper_pose_)
    {
      peak_object_speed_ = 0.0;
      last_object_speed_ = -1.0;
      deceleration_samples_ = 0;
      stable_velocity_samples_ = 0;
      return false;
    }

    // If grasp evaluation was requested, wait for it to complete before exiting tracking
    if (wait_for_grasp_evaluation && (!grasp_evaluation_complete_ || grasp_evaluation_in_progress_))
    {
      return false;
    }

    // Check that robot is within tracking distance of object before reaching
    const Eigen::Vector3d object_pos(latest_pose_.pose.position.x,
                                      latest_pose_.pose.position.y,
                                      latest_pose_.pose.position.z);
    // Note: We use grasp_pose_ as proxy for current tool position since this is called during tracking
    const Eigen::Vector3d tool_pos(current_pose.position.x,
                                    current_pose.position.y,
                                    current_pose.position.z);
    const double distance_to_object = (tool_pos - object_pos).norm();

    if (distance_to_object > static_cast<double>(config_.approach_distance))
    {
      RCLCPP_INFO_STREAM_THROTTLE(
          LOGGER,
          *node_->get_clock(),
          1000,
          "Distance to object: " << distance_to_object << " m, still above tracking distance threshold of "
                                  << config_.tracking_distance << " m. Continuing to track.");
      return false;  // Still too far from object, continue tracking
    }
    // return true;

    Eigen::Vector3d velocity;
    if (!estimateVelocityInternal(velocity))
    {
      return false;
    }

    const double speed = velocity.norm();

    const bool dynamic_ready = speed >= static_cast<double>(config_.velocity_threshold);

    // Keep a simple low-speed counter as fallback if no clear descending branch is observed.
    if (speed >= static_cast<double>(config_.velocity_threshold))
    {
      ++stable_velocity_samples_;
    }
    else
    {
      stable_velocity_samples_ = 0;
    }

    const bool stable_ready = stable_velocity_samples_ >= stable_samples_required;

    return dynamic_ready && stable_ready;
  }

  bool isTouchdownDetected() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return touchdown_detected_;
  }

  // Check velocity and initiate grasp generation if below threshold
  void checkVelocityAndGenerateGrasps()
  {
    bool should_send_request = false;

    {
      std::lock_guard<std::mutex> lock(mutex_);

      // If grasping goal already arrived, nothing to do.
      if (received_gripper_pose_)
        return;

      // If a request is in progress for too long, release the state machine and retry.
      if (grasp_generation_in_progress_)
      {
        const double elapsed = (node_->now() - last_grasp_request_time_).seconds();
        if (elapsed <= grasp_request_timeout_sec_)
        {
          return;
        }

        RCLCPP_WARN_STREAM(LOGGER,
                           "Grasp generation request timed out after " << elapsed
                           << " s. Resetting request state and retrying.");
        grasp_generation_in_progress_ = false;
      }

      if (!received_pose_)
      {
        return;
      }

      // Estimate current velocity
      // Eigen::Vector3d velocity;
      // if (!estimateVelocityInternal(velocity))
      // {
      //   RCLCPP_WARN_STREAM(LOGGER, "Failed to estimate velocity for grasp generation.");
      //   return;
      // }

      // const double speed = velocity.norm();
      // // RCLCPP_INFO_STREAM_THROTTLE(
      // //     LOGGER,
      // //     *node_->get_clock(),
      // //     1000,
      // //     "Current object speed: " << speed << " m/s based on velocity vector ["
      // //                             << velocity.x() << ", " << velocity.y() << ", " << velocity.z() << "]");
      // if (speed >= static_cast<double>(config_.grasp_velocity_threshold))
      // {
        grasp_generation_in_progress_ = true;
        last_grasp_request_time_ = node_->now();
        should_send_request = true;
        RCLCPP_INFO_STREAM(LOGGER,
                           "Triggering grasp generation");
        // RCLCPP_INFO_STREAM(LOGGER,
        //                    "Triggering grasp generation using simple speed threshold: speed="
        //                        << speed << " m/s, threshold=" << config_.grasp_velocity_threshold << " m/s");
      // }
    }

    if (should_send_request)
    {
      // config_.tracking_distance = 10.0; // allow to close in on the object
      sendGraspGenerationRequest();
    }
  }

  // brings gripper poses back to the object initial pose (grasp generation request time)
  void resetMotionCompensation()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    object_pose_at_grasp_request_ = initial_object_pose_;
  }

  void sendTouchdownDetectionRequest()
  {
    if (!touchdown_service_client_)
    {
      RCLCPP_ERROR_STREAM(LOGGER, "Touchdown detection service client not initialized");
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      touchdown_detected_ = false;
      touchdown_detection_in_progress_ = false;
    }

    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    touchdown_service_client_->async_send_request(
        request,
        [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
          try
          {
            const auto response = future.get();
            if (!response)
            {
              RCLCPP_WARN_STREAM(LOGGER, "Touchdown detection service returned an empty response");
              return;
            }

            if (!response->success)
            {
              RCLCPP_WARN_STREAM(LOGGER, "Touchdown detection service request was rejected: " << response->message);
              return;
            }

            {
              std::lock_guard<std::mutex> lock(mutex_);
              touchdown_detection_in_progress_ = true;
            }

            RCLCPP_INFO_STREAM(LOGGER, "Touchdown detection service accepted: " << response->message);
          }
          catch (const std::exception& error)
          {
            RCLCPP_WARN_STREAM(LOGGER, "Touchdown detection service request failed: " << error.what());
          }
        });
    RCLCPP_INFO_STREAM(LOGGER, "Sent touchdown detection request to service");
  }

  void touchdownStatusCallback(const std_msgs::msg::Bool::ConstSharedPtr& msg)
  {
    if (!msg->data)
    {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!touchdown_detection_in_progress_)
    {
      return;
    }

    if (touchdown_detected_)
    {
      return;
    }
    touchdown_detected_ = true;
    touchdown_detection_in_progress_ = false;
    touchdown_status_sub_.reset();
    RCLCPP_INFO_STREAM(LOGGER, "Touchdown detected via /touchdown_status");
  }

  // Send request to grasp generation action server
  void sendGraspGenerationRequest()
  {
    using GoalHandleGenerateGrasp = rclcpp_action::ClientGoalHandle<grasp_generation::action::GenerateGrasp>;

    if (!grasp_action_client_)
    {
      RCLCPP_ERROR_STREAM(LOGGER, "Grasp action client not initialized");
      std::lock_guard<std::mutex> lock(mutex_);
      grasp_generation_in_progress_ = false;
      return;
    }

    // Wait for server to be available
    // TODO: invoke fallback if not available
    if (!grasp_action_client_->wait_for_action_server(0.5s))
    {
      RCLCPP_WARN_STREAM(LOGGER, "Grasp generation action server not available");
      std::lock_guard<std::mutex> lock(mutex_);
      grasp_generation_in_progress_ = false;
      return;
    }

    // Create goal (empty for this action)
    auto goal_msg = grasp_generation::action::GenerateGrasp::Goal();

    // // Save object pose at request time to compensate grasp translation after async generation.
    // {
    //   std::lock_guard<std::mutex> lock(mutex_);
    //   object_pose_at_grasp_request_ = latest_pose_;
    // }

    // Send goal with callbacks
    auto send_goal_options = rclcpp_action::Client<grasp_generation::action::GenerateGrasp>::SendGoalOptions();

    send_goal_options.goal_response_callback =
        [this](GoalHandleGenerateGrasp::SharedPtr goal_handle) {
          if (!goal_handle)
          {
            RCLCPP_WARN_STREAM(LOGGER, "Grasp generation goal was rejected by action server");
            std::lock_guard<std::mutex> lock(mutex_);
            grasp_generation_in_progress_ = false;
          }
          else
          {
            RCLCPP_INFO_STREAM(LOGGER, "Grasp generation goal accepted by action server");
          }
        };
    
    send_goal_options.result_callback =
        [this](const GoalHandleGenerateGrasp::WrappedResult& result) {
          graspGenerationResultCallback(result);
        };

    grasp_action_client_->async_send_goal(goal_msg, send_goal_options);
    RCLCPP_INFO_STREAM(LOGGER, "Sent grasp generation request to action server");

    // Save object pose at request time to compensate grasp translation after async generation.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      object_pose_at_grasp_request_ = initial_object_pose_;
    }
  }

  // Callback for grasp generation result
  void graspGenerationResultCallback(
      const rclcpp_action::ClientGoalHandle<grasp_generation::action::GenerateGrasp>::WrappedResult& result)
  {
    switch (result.code)
    {
      case rclcpp_action::ResultCode::SUCCEEDED:
      {
        if (!result.result)
        {
          RCLCPP_WARN_STREAM(LOGGER, "Grasp generation action succeeded but result is empty");
          std::lock_guard<std::mutex> lock(mutex_);
          grasp_generation_in_progress_ = false;
          break;
        }

        if (!result.result->status)
        {
          RCLCPP_WARN_STREAM(LOGGER, "Grasp generation action returned status=false");
          std::lock_guard<std::mutex> lock(mutex_);
          grasp_generation_in_progress_ = false;
          break;
        }

        // if (result.result->grasp_goals.poses.empty())
        // {
        //   RCLCPP_WARN_STREAM(LOGGER, "Grasp generation action returned no grasp poses");
        //   std::lock_guard<std::mutex> lock(mutex_);
        //   grasp_generation_in_progress_ = false;
        //   break;
        // }

        RCLCPP_INFO_STREAM(LOGGER, "Grasp generation action succeeded with "
                                   << result.result->grasp_goals.poses.size() << " grasp poses");

        bool apply_first_grasp_type = false;

        {
          std::lock_guard<std::mutex> lock(mutex_);
          tmp_grasp_types_ = result.result->grasp_labels;
          tmp_grasp_widths_ = result.result->grasp_widths;

          if (evaluate_grasp_poses_)
          {
            tmp_gripper_poses_ = result.result->grasp_goals;
          }
          else
          {
            grasp_pose_.header = result.result->grasp_goals.header;
            grasp_pose_.pose = result.result->grasp_goals.poses.front();
            transformToWorldFrame(grasp_pose_);
            apply_first_grasp_type = true;
          }

          grasp_generation_in_progress_ = false;
          received_gripper_pose_ = true;
        }

        if (apply_first_grasp_type)
        {
          applyGraspTypeSelectionForCandidate(0);
        }
        break;
      }
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_WARN_STREAM(LOGGER, "Grasp generation action was aborted");
        {
          std::lock_guard<std::mutex> lock(mutex_);
          grasp_generation_in_progress_ = false;
        }
        break;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_WARN_STREAM(LOGGER, "Grasp generation action was canceled");
        {
          std::lock_guard<std::mutex> lock(mutex_);
          grasp_generation_in_progress_ = false;
        }
        break;
      default:
        RCLCPP_WARN_STREAM(LOGGER, "Grasp generation action returned unknown code");
        {
          std::lock_guard<std::mutex> lock(mutex_);
          grasp_generation_in_progress_ = false;
        }
        break;
    }
  }

  float getGraspJointAngle() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return grasp_joint_angle_;
  }

  void applyObjectMotionCompensationToPose(geometry_msgs::msg::PoseStamped& pose, const bool update_for_next_compensation = false, const bool predict_future = false)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    geometry_msgs::msg::PoseStamped reference_object_pose;

    if (predict_future)
    {
      // Predict future object pose based on velocity estimation (use internal method - mutex already locked)
      reference_object_pose = predictFutureObjectPoseInternal(latest_pose_);
    }
    else
    {
      reference_object_pose = latest_pose_;
    }

    const Eigen::Vector3d request_pos(
        object_pose_at_grasp_request_.pose.position.x,
        object_pose_at_grasp_request_.pose.position.y,
        object_pose_at_grasp_request_.pose.position.z);
    const Eigen::Vector3d latest_pos(
        reference_object_pose.pose.position.x,
        reference_object_pose.pose.position.y,
        reference_object_pose.pose.position.z);
    const Eigen::Vector3d delta = latest_pos - request_pos;

    pose.pose.position.x += delta.x();
    pose.pose.position.y += delta.y();
    pose.pose.position.z += delta.z();

    if (update_for_next_compensation)
      object_pose_at_grasp_request_ = reference_object_pose;//latest_pose_; // update for next compensation

    // RCLCPP_INFO_STREAM(LOGGER,
    //                    "Applied grasp translation compensation by object delta ["
    //                        << delta.x() << ", " << delta.y() << ", " << delta.z() << "] m");
  }

  void applyObjectMotionCompensationToPoseInternal(geometry_msgs::msg::PoseStamped& pose, const bool update_for_next_compensation = false, const bool predict_future = false)
  {
    geometry_msgs::msg::PoseStamped reference_object_pose;

    if (predict_future)
    {
      // Predict future object pose based on velocity estimation (use internal method - mutex already locked)
      reference_object_pose = predictFutureObjectPoseInternal(latest_pose_);
    }
    else
    {
      reference_object_pose = latest_pose_;
    }
    const Eigen::Vector3d request_pos(
        object_pose_at_grasp_request_.pose.position.x,
        object_pose_at_grasp_request_.pose.position.y,
        object_pose_at_grasp_request_.pose.position.z);
    const Eigen::Vector3d latest_pos(
        reference_object_pose.pose.position.x,
        reference_object_pose.pose.position.y,
        reference_object_pose.pose.position.z);
    const Eigen::Vector3d delta = latest_pos - request_pos;

    pose.pose.position.x += delta.x();
    pose.pose.position.y += delta.y();
    pose.pose.position.z += delta.z();

    if (update_for_next_compensation)
      object_pose_at_grasp_request_ = reference_object_pose;//latest_pose_; // update for next compensation
  }

private:

  static constexpr int kBottomGraspType = 0;
  static constexpr int kMidGraspType = 1;
  static constexpr int kTopGraspType = 2;
  static constexpr float kDefaultGripOffset = 0.009f; // m, empirically determined
  static constexpr float kHeavyGripOffset = 0.017f; //0.013f // m, for heavy objects requiring deeper grasp
  static constexpr float kDelicateGripOffset = 0.0001f; // m, for light objects where a shallower grasp is sufficient

  void determineGripOffset()
  {
    if (received_dimensions_)
    {
      const std::string label = latest_dimensions_.label; //normalizeLabelForClassification(latest_dimensions_.label);

      static constexpr const char* kAsymmetricContainerPriors[] = {
        "box", "food box", "bag","food bag", "food container", "food package", "rectangle", "square"};
      static constexpr const char* kSymmetricContainerPriors[] = {"cup", "glass", "bottle", "mug", "can", "cylinder"};

      const auto containsAny = [&label](const char* const* labels, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i)
        {
          if (label.find(labels[i]) != std::string::npos)
            return true;
        }
        return false;
      };

      if (containsAny(kAsymmetricContainerPriors,
                      sizeof(kAsymmetricContainerPriors) / sizeof(kAsymmetricContainerPriors[0])))
      {
        RCLCPP_DEBUG(LOGGER, "Asymmetric container detected based on label.");
        grip_offset_ = kHeavyGripOffset;
      }
      else if (label.find("glass") != std::string::npos)
      {
        grip_offset_ = kDelicateGripOffset;
        is_delicate_ = true;
      }
      else
      {
        RCLCPP_DEBUG(LOGGER, "Container type is set to default.");
        grip_offset_ = kDefaultGripOffset;
      }
    }
    else
    {
      RCLCPP_WARN_STREAM(LOGGER,
                         "Dimensions not received. Using default grip offset of " << kDefaultGripOffset
                                                                                      << " m for grasp width to joint angle conversion.");
      grip_offset_ = kDefaultGripOffset;
    }

  }

  void applyGraspTypeSelectionForCandidate(std::size_t candidate_index)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (candidate_index >= tmp_grasp_types_.size())
    {
      RCLCPP_WARN_STREAM(LOGGER,
                         "Grasp type labels missing for selected candidate index " << candidate_index
                                                                                  << ". Keeping previous gripper settings.");
      return;
    }

    float target_width = tmp_grasp_widths_[candidate_index];
    if (target_width < 0.0f || std::isnan(target_width) || std::isinf(target_width) || (received_dimensions_ && target_width > std::max(latest_dimensions_.top_w, latest_dimensions_.bot_w)))
    {
      RCLCPP_WARN_STREAM(LOGGER,
                         "Invalid grasp width " << target_width << " for selected candidate index " << candidate_index
                                                 << ". Keeping previous gripper settings.");
    
      const int selected_grasp_type = tmp_grasp_types_[candidate_index];
      selected_grasp_type_ = selected_grasp_type;

      if (selected_grasp_type == kMidGraspType)
      {
        // RCLCPP_INFO_STREAM(LOGGER, "Selected mid grasp type: " << 0.5 * (latest_dimensions_.top_w + latest_dimensions_.bot_w) << " m width");
        // Mid grasp keeps current opening/displacement calibration unchanged.
        return;
      }

      if (!received_dimensions_)
      {
        // RCLCPP_WARN_STREAM(LOGGER,
        //                    "Selected grasp type requires width adaptation, but dimensions are unavailable. "
        //                    "Keeping previous gripper settings.");
        return;
      }

      if (selected_grasp_type == kTopGraspType)
      {
        target_width = latest_dimensions_.top_w;
        // RCLCPP_INFO_STREAM(LOGGER, "Selected top grasp type: " << latest_dimensions_.top_w << " m width");
      }
      else if (selected_grasp_type == kBottomGraspType)
      {
        target_width = latest_dimensions_.bot_w;
        // RCLCPP_INFO_STREAM(LOGGER, "Selected bottom grasp type: " << latest_dimensions_.bot_w << " m width");
      }
    }

    grasp_joint_angle_ = getGripperAngleFromWidth(target_width);
    RCLCPP_INFO_STREAM(LOGGER, "Selected target width " << target_width
                                                      << " m, corresponding to gripper joint angle " << grasp_joint_angle_ << " rad");
    robotiq_vertical_displacement_ = getRobotiqVerticalDisplacement(grasp_joint_angle_);
  }

  // geometry_msgs::msg::Pose getGraspCenterShiftedPose(const geometry_msgs::msg::PoseStamped& grasp_pose, const Eigen::Vector3d& grasp_center_shift)
  // {
  //   geometry_msgs::msg::Pose shifted_pose;
  //   // grasp_center and tool0 have same orientation
  //   // shift with both in base_link frame
  //   shifted_pose.position.x = grasp_pose.pose.position.x + grasp_center_shift.x();
  //   shifted_pose.position.y = grasp_pose.pose.position.y + grasp_center_shift.y();
  //   shifted_pose.position.z = grasp_pose.pose.position.z + grasp_center_shift.z();
  //   shifted_pose.orientation = grasp_pose.pose.orientation;
  //   return shifted_pose;
  // }

  // transforms grasps to base_link frame and applies any necessary corrections (e.g. orientation flip) for fair manipulability evaluation, without modifying the original gripper_pose_ member variable
  bool transformGraspingPose(geometry_msgs::msg::PoseStamped& pose, const geometry_msgs::msg::PoseStamped& current_pose)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!received_gripper_pose_)
    {
      RCLCPP_WARN_STREAM(LOGGER, "Gripper pose not received yet");
      return false;
    }

      Eigen::Isometry3d T_base = poseToMatrix(pose.pose);
        
      if (pose.header.frame_id != base_frame_)
      {
        RCLCPP_ERROR_STREAM(LOGGER, "Received pose IS NOT in base frame as expected: " << pose.header.frame_id);
        return false;
      }

      // Then: apply base->base_link transform: T_out = T_baselink_to_base @ T_base
      Eigen::Isometry3d T_out = T_baselink_to_base_ * T_base;

      // Convert back to pose, output in base_link frame
      pose.header.frame_id = base_link_frame_;
      pose.pose = matrixToPose(T_out);

      // Compare current and goal end-effector Y-axis up/down direction in the same frame.
      // If one points up (+Z) and the other down (-Z), flip goal by 180 degrees around Z.
      correctEEOrientation(pose, current_pose);

      return true;
  }

  void correctEEOrientation(geometry_msgs::msg::PoseStamped& pose, const geometry_msgs::msg::PoseStamped& current_pose)
  {
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
        // RCLCPP_INFO_STREAM(LOGGER, "Flipping goal pose by 180 degrees around Z-axis to match Y-axis up/down direction");
        const Eigen::AngleAxisd flip_rotation(M_PI, Eigen::Vector3d::UnitZ());
        const Eigen::Quaterniond q_flip(flip_rotation);
        const Eigen::Quaterniond q_flipped = q_goal * q_flip;

        pose.pose.orientation.w = q_flipped.w();
        pose.pose.orientation.x = q_flipped.x();
        pose.pose.orientation.y = q_flipped.y();
        pose.pose.orientation.z = q_flipped.z();
      }
  }

  void handoverCB(const std_msgs::msg::Bool::ConstSharedPtr& msg)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (msg->data)
      {
        RCLCPP_INFO_STREAM(LOGGER, "Handover detected! Starting approach to goal pose...");
        handover_detected_ = true;
        dynamic_tracking_distance_ = config_.approach_distance;
        tracking_target_initialized_ = false;
        handover_sub_.reset();  // No longer need to listen for handover after it's detected
      }
      else
      {
        handover_detected_ = false;
        tracking_target_initialized_ = false;
      }
    }
  }

  void dimensionsCB(const visual_properties::msg::Dimensions::ConstSharedPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    latest_dimensions_ = *msg;
    received_dimensions_ = true;

    // Default width before grasp type is selected: interpolate between top and bottom widths.
    float object_width = 0.5 * (latest_dimensions_.top_w + latest_dimensions_.bot_w);
    grasp_joint_angle_ = getGripperAngleFromWidth(object_width);
    robotiq_vertical_displacement_ = getRobotiqVerticalDisplacement(grasp_joint_angle_);
    dim_sub_.reset(); // No longer need to listen for dimensions after receiving them once
    determineGripOffset();
    // RCLCPP_INFO_STREAM(LOGGER, "Updated grasp joint angle from perception: " << grasp_joint_angle_ << " rad");
    RCLCPP_INFO_STREAM(LOGGER, "Updated object width from perception: " << object_width << " rad");
  }

  void transformToWorldFrame(geometry_msgs::msg::PoseStamped& pose)
  {
    // Convert predicted pose to transformation matrix
    Eigen::Isometry3d T_base = poseToMatrix(pose.pose);

    // Apply base->base_link transform
    Eigen::Isometry3d T_out = T_baselink_to_base_ * T_base;

    // Convert back to pose, output in world frame
    pose.header.frame_id = base_link_frame_;
    pose.header.stamp = pose.header.stamp;
    pose.pose = matrixToPose(T_out);
  }

  inline bool checkPoseInWorkspace(float x, float y) const
  {
    return (x >= workspace_x_min_ && x <= workspace_x_max_ &&
            y >= workspace_y_min_ && y <= workspace_y_max_);
  }

  void goalPoseCB(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!tf_initialized_)
    {
      RCLCPP_WARN_STREAM(LOGGER, "TF not initialized yet. Ignoring received pose.");
      getTF();
      return;
    }

    geometry_msgs::msg::PoseStamped transformed_msg = *msg;
    transformToWorldFrame(transformed_msg); // transform incoming pose to world frame for workspace check and storage
    if (!checkPoseInWorkspace(transformed_msg.pose.position.x, transformed_msg.pose.position.y))
    {
      RCLCPP_WARN_STREAM(LOGGER, "Received pose is out of workspace bounds. Ignoring. POse (x, y): " << transformed_msg.pose.position.x << ',' << transformed_msg.pose.position.y << " min x: " << workspace_x_min_ << ", max x: " << workspace_x_max_ << ", min y: " << workspace_y_min_ << ", max y: " << workspace_y_max_);
      return;
    }

    if (!received_pose_)
    {
      initial_object_pose_ = transformed_msg;
      // transformToWorldFrame(initial_object_pose_);
      received_pose_ = true;
      // update config_.aproach_distance based on initial euclidian distance from the object
      if (start_pose_initialized_)
      {
        const Eigen::Vector3d object_pos(initial_object_pose_.pose.position.x,
                                        initial_object_pose_.pose.position.y,
                                        initial_object_pose_.pose.position.z);
        config_.approach_distance = (object_pos - tool_start_pose_).norm();
      }
      RCLCPP_INFO_STREAM(LOGGER, "Received first goal pose in frame: " << msg->header.frame_id);
    }

    if (enable_real_time_filtering_)
    {
      double timestamp = transformed_msg.header.stamp.sec + transformed_msg.header.stamp.nanosec * 1e-9;
      geometry_msgs::msg::Pose target_pose = transformed_msg.pose;

      if (!has_first_filter_pose_)
      {
        last_valid_filter_pose_ = target_pose;
        last_filter_timestamp_ = timestamp;
        has_first_filter_pose_ = true;
      }
      else
      {
        double dt = timestamp - last_filter_timestamp_;
        if (dt > 0.0)
        {
          double dx = target_pose.position.x - last_valid_filter_pose_.position.x;
          double dy = target_pose.position.y - last_valid_filter_pose_.position.y;
          double dz = target_pose.position.z - last_valid_filter_pose_.position.z;
          double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

          if (distance / dt > max_linear_speed_)
          {
            filter_outlier_count_++;
            if (filter_outlier_count_ < max_consecutive_outliers_)
            {
              // Jump detected, use last valid pose for filtering state
              target_pose = last_valid_filter_pose_;
            }
            else
            {
              filter_outlier_count_ = 0;
              // TODO: can interpolate between last valid pose and current target pose instead of jumping directly to target pose after consecutive outliers
              last_valid_filter_pose_ = target_pose;
            }
          }
          else
          {
            filter_outlier_count_ = 0;
            last_valid_filter_pose_ = target_pose;
          }
        }
        last_filter_timestamp_ = timestamp;
      }

      double filtered_x = filter_x_->filter(target_pose.position.x, timestamp);
      double filtered_y = filter_y_->filter(target_pose.position.y, timestamp);
      double filtered_z = filter_z_->filter(target_pose.position.z, timestamp);
      
      transformed_msg.pose.position.x = filtered_x;
      transformed_msg.pose.position.y = filtered_y;
      transformed_msg.pose.position.z = filtered_z;
      transformed_msg.pose.orientation = target_pose.orientation;

      // debug_publisher_->publish(transformed_msg);
    }

    latest_pose_ = transformed_msg;
    // latest_pose_ = *msg;
    // transformToWorldFrame(latest_pose_);
    
    // Maintain a history of poses for velocity estimation
    pose_history_.push_back(latest_pose_);
    if (pose_history_.size() > static_cast<size_t>(config_.velocity_estimation_frames))
    {
      pose_history_.pop_front();
    }

    if (pose_history_.size() >= 2)
    {
      const auto& prev_msg = pose_history_[pose_history_.size() - 2];
      const auto& curr_msg = pose_history_.back();

      const double prev_stamp = rclcpp::Time(prev_msg.header.stamp).seconds();
      const double curr_stamp = rclcpp::Time(curr_msg.header.stamp).seconds();
      const double dt = curr_stamp - prev_stamp;

      if (dt > 1e-4 && dt < 1.0)
      {
        if (frame_time_ == 0.0)
        {
          frame_time_ = dt;
        }
        else
        {
          frame_time_ = 0.8 * frame_time_ + 0.2 * dt;
        }
      }
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
  float workspace_x_min_ = -1.0;
  float workspace_x_max_ = 1.0;
  float workspace_y_min_ = -1.0;
  float workspace_y_max_ = 1.0;
  geometry_msgs::msg::TransformStamped T_grasp_to_tool0_msg_; 
  Eigen::Vector3d grasp_center_shift_;
  geometry_msgs::msg::TransformStamped T_baselink_to_base_msg_;
  Eigen::Isometry3d T_baselink_to_base_;
  bool tf_initialized_ = false;
  geometry_msgs::msg::PoseStamped initial_object_pose_;
  Eigen::Vector3d tool_start_pose_;
  Eigen::Quaterniond start_pose_orientation_{1.0, 0.0, 0.0, 0.0};
  bool start_pose_initialized_ = false;
  std::string placement_strategy_ = "initial";
  float predefined_placement_x_ = 0.0f;
  float predefined_placement_y_ = 0.0f;
  float table_z_offset_ = 0.0f;
  float grip_offset_ = kDefaultGripOffset;
  bool is_delicate_ = false;
  float grasp_joint_angle_ = 0.45; // default, will be updated by perception
  float robotiq_vertical_displacement_ = 0.0;
  int selected_grasp_type_ = kMidGraspType;
  bool received_dimensions_ = false;
  visual_properties::msg::Dimensions latest_dimensions_;
  geometry_msgs::msg::PoseStamped latest_pose_; // object centroid pose
  geometry_msgs::msg::PoseStamped grasp_pose_;
  geometry_msgs::msg::PoseArray tmp_gripper_poses_;
  std::vector<int32_t> tmp_grasp_types_;
  std::vector<float> tmp_grasp_widths_;
  bool received_pose_ = false;
  bool received_gripper_pose_ = false;
  bool handover_detected_ = false;
  bool start_approach_ = false;
  std::deque<geometry_msgs::msg::PoseStamped> pose_history_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr handover_sub_;
  rclcpp::Subscription<visual_properties::msg::Dimensions>::SharedPtr dim_sub_;
  rclcpp::Subscription<grasp_generation::action::GenerateGrasp_Result>::SharedPtr replanned_grasps_sub_;
  rclcpp_action::Client<GripperCommand>::SharedPtr gripper_action_client_;
  rclcpp::Subscription<control_msgs::msg::DynamicJointState>::SharedPtr dynamic_joint_state_sub_;
  rclcpp::Subscription<tactile_detection::msg::TactileDetection>::SharedPtr alfredo_sub_;
  rclcpp::Subscription<tactile_detection::msg::TactileDetection>::SharedPtr giovanni_sub_;

  struct TactileEvent
  {
    std::atomic<bool> detected{false};
    std::atomic<bool> disconnected{false};
  };

  std::atomic<bool> object_detected_during_grasp_{false};
  TactileEvent alfredo_event_;
  TactileEvent giovanni_event_;
  bool regrasp_requested_ = false;
  std::atomic<bool> start_grasping_{false};
  std::atomic<int> latest_gripper_status_{-1};
  std::atomic<double> latest_finger_joint_position_{std::numeric_limits<double>::quiet_NaN()};
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::string base_frame_;
  std::string base_link_frame_;
  std::string command_frame_;
  std::string grasp_frame_;
  HandoverConfig config_;
  double frame_time_ = 0.0;
  bool evaluate_grasp_poses_;
  rclcpp::Node::SharedPtr node_;
  rclcpp_action::Client<grasp_generation::action::GenerateGrasp>::SharedPtr grasp_action_client_;
  bool grasp_generation_in_progress_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr touchdown_service_client_;
  bool touchdown_detection_in_progress_;
  bool touchdown_detected_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr touchdown_status_sub_;
  rclcpp::Time last_grasp_request_time_;
  double grasp_request_timeout_sec_ = 5.0;
  double peak_object_speed_ = 0.0;
  double last_object_speed_ = -1.0;
  int deceleration_samples_ = 0;
  int stable_velocity_samples_ = 0;
  geometry_msgs::msg::PoseStamped object_pose_at_grasp_request_;
  double dynamic_tracking_distance_ = -1.0;
  bool tracking_target_initialized_ = false;
  geometry_msgs::msg::PoseStamped tracking_target_pose_;

  // Real-time filtering variables
  bool enable_real_time_filtering_ = false;
  double frequency_ = 120.0;
  double mincutoff_ = 0.5;
  double beta_ = 0.4;
  double dcutoff_ = 1.0;
  double max_linear_speed_ = 1.0;
  int max_consecutive_outliers_ = 5;

  std::unique_ptr<OneEuroFilter> filter_x_;
  std::unique_ptr<OneEuroFilter> filter_y_;
  std::unique_ptr<OneEuroFilter> filter_z_;
  bool has_first_filter_pose_ = false;
  geometry_msgs::msg::Pose last_valid_filter_pose_;
  double last_filter_timestamp_ = 0.0;
  int filter_outlier_count_ = 0;

  // rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr debug_publisher_;

public:
  // Background grasp evaluation state (accessed from main thread)
  std::atomic<bool> grasp_evaluation_complete_{false};
  std::atomic<bool> grasp_evaluation_in_progress_{false};
  geometry_msgs::msg::PoseStamped evaluated_gripper_pose_;
  std::mutex grasp_evaluation_mutex_;

  bool isDelicate() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_delicate_;
  }

  // Getter for latest object pose (thread-safe access)
  geometry_msgs::msg::PoseStamped getLatestObjectPose() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_pose_;
  }

  // Check if grasp evaluation is complete
  bool isGraspEvaluationComplete() const
  {
    return grasp_evaluation_complete_;
  }
};


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
    request->activate_controllers = {"forward_position_controller"};
  } else {
    request->deactivate_controllers = {"forward_position_controller"};
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

// doesn't monitor the gripper position feedback during the grasping loop
void GoalPoseSubscriber::runFastClosedLoopGripperControl(const std::vector<float>& joint_angle_range)
{
  if (!gripper_action_client_)
  {
    RCLCPP_WARN_STREAM(LOGGER, "Gripper action client not initialized; skipping closed-loop control.");
    return;
  }

  if (!gripper_action_client_->wait_for_action_server(1s))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Gripper action server not available after waiting");
    return;
  }

  // RCLCPP_INFO_STREAM(LOGGER, "Starting closed-loop grasping with " << joint_angle_range.size() << " discrete steps");

  object_detected_during_grasp_.store(false);
  alfredo_event_.detected.store(false);
  giovanni_event_.detected.store(false);
  bool tactile_detected = false;
  bool object_detected = false;
  bool visual_prior_forced = false;
  float last_commanded_angle = 0.0f;
  constexpr double kFingerJointEpsilon = 0.01;
  constexpr auto kStepTimeout = 500ms;

  auto goal_msg = GripperCommand::Goal();
  start_grasping_.store(true);

  for (size_t step = 0; step < joint_angle_range.size(); ++step)
  {
    const float current_angle = joint_angle_range[step];
    last_commanded_angle = current_angle;

    goal_msg.command.position = current_angle;
    goal_msg.command.max_effort = 0.1;

    auto goal_handle_future = gripper_action_client_->async_send_goal(goal_msg);
    auto goal_handle = goal_handle_future.get();
    if (!goal_handle)
    {
      RCLCPP_ERROR(LOGGER, "Failed to send gripper goal at step %zu", step + 1);
      break;
    }

    auto result_future = gripper_action_client_->async_get_result(goal_handle);
    bool cancelled = false;
    while (rclcpp::ok())
    {
      const bool alfredo_detected = alfredo_event_.detected.load();
      const bool alfredo_disconnected = alfredo_event_.disconnected.load();
      const bool giovanni_detected = giovanni_event_.detected.load();
      const bool giovanni_disconnected = giovanni_event_.disconnected.load();

      tactile_detected = (alfredo_detected && giovanni_detected);
      if (tactile_detected)
      {
        (void)gripper_action_client_->async_cancel_goal(goal_handle);
        cancelled = true;
        break;
      }
      else if ((alfredo_disconnected && giovanni_detected) || (giovanni_disconnected && alfredo_detected))
      {
        (void)gripper_action_client_->async_cancel_goal(goal_handle);
        cancelled = true;
        tactile_detected = true;
        break;
      }
      else if (alfredo_disconnected && giovanni_disconnected)
      {
        (void)gripper_action_client_->async_cancel_goal(goal_handle);
        goal_msg.command.position = grasp_joint_angle_;
        goal_msg.command.max_effort = 0.1;
        (void)gripper_action_client_->async_send_goal(goal_msg);
        last_commanded_angle = grasp_joint_angle_;
        cancelled = true;
        visual_prior_forced = true;
        break;
      }

      if (object_detected_during_grasp_.load())
      {
        (void)gripper_action_client_->async_cancel_goal(goal_handle);
        cancelled = true;
        object_detected = true;
        break;
      }

      if (result_future.wait_for(2ms) == std::future_status::ready)
      {
        break;
      }
    }

    if (!cancelled)
    {
      (void)result_future.get();
    }

    if (tactile_detected)
    {
      // RCLCPP_INFO(LOGGER, "Tactile detection at step %zu; stopping grasp loop.", step + 1);
      break;
    }

    if (object_detected)
    {
      // RCLCPP_INFO(LOGGER, "Object detection at step %zu; stopping grasp loop.", step + 1);
      break;
    }
  }

  const double latest_joint_position = latest_finger_joint_position_.load();
  const double achieved_joint_angle = visual_prior_forced
                                       ? grasp_joint_angle_
                                       :
      std::isfinite(latest_joint_position) ? latest_joint_position : last_commanded_angle;

  if (!object_detected)
  {
    // Reliability squeeze: if tactile contact was detected, close a tiny bit more asynchronously.
    constexpr float kMinJointAngle = 0.0f;
    constexpr float kMaxJointAngle = 0.69f;
    constexpr float kMaxStroke = 0.140f;
    constexpr float kMaxJointAngleForConversion = 0.67f;

    const float squeeze_angle_increment =
      std::clamp(grip_offset_, 0.0f, kMaxStroke) * (kMaxJointAngleForConversion / kMaxStroke);
    const float squeeze_angle = std::clamp(static_cast<float>(achieved_joint_angle) + squeeze_angle_increment,
                       kMinJointAngle,
                       kMaxJointAngle);

    goal_msg.command.position = squeeze_angle;
    goal_msg.command.max_effort = 0.1;

    (void)gripper_action_client_->async_send_goal(goal_msg);

    waitForGripperCompletion(latest_gripper_status_,
                            latest_finger_joint_position_,
                            squeeze_angle,
                            kFingerJointEpsilon,
                            kStepTimeout);
  }

  RCLCPP_INFO(LOGGER, "Closed-loop grasping finished. Object secured: %s, Tactile contact: %s",
              object_detected ? "yes" : "no",
              tactile_detected ? "yes" : "no");
}


// Member implementation: run closed-loop gripper control using GoalPoseSubscriber's clients/subscriptions
void GoalPoseSubscriber::runClosedLoopGripperControl(moveit::planning_interface::MoveGroupInterface& gripper_interface,
                                                     const std::vector<float>& joint_angle_range)
{
  if (!gripper_action_client_)
  {
    RCLCPP_WARN_STREAM(LOGGER, "Gripper action client not initialized; skipping closed-loop control.");
    return;
  }

  if (!gripper_action_client_->wait_for_action_server(1s))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Gripper action server not available after waiting");
    return;
  }

  // RCLCPP_INFO_STREAM(LOGGER, "Starting closed-loop grasping with " << joint_angle_range.size() << " discrete steps");

  object_detected_during_grasp_.store(false);
  alfredo_event_.detected.store(false);
  giovanni_event_.detected.store(false);
  bool tactile_detected = false;
  bool object_detected = false;
  bool visual_prior_forced = false;
  float last_commanded_angle = 0.0f;
  constexpr double kFingerJointEpsilon = 0.02;
  constexpr auto kStepTimeout = 500ms;

  const auto wait_for_joint_angle = [this](double target_position,
                                           double epsilon,
                                           std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;

    while (!alfredo_event_.detected.load() && !giovanni_event_.detected.load() && rclcpp::ok() && std::chrono::steady_clock::now() < deadline)
    {
      const double current_position = latest_finger_joint_position_.load();
      if (std::isfinite(current_position) && std::abs(current_position - target_position) <= epsilon)
        return true;

      std::this_thread::sleep_for(10ms);
    }

    return false;
  };

  auto goal_msg = GripperCommand::Goal();
  start_grasping_.store(true);

  for (size_t step = 0; step < joint_angle_range.size(); ++step)
  {
    const float current_angle = joint_angle_range[step];
    last_commanded_angle = current_angle;
    // RCLCPP_INFO(LOGGER, "Grasping step %zu/%zu: sending joint angle = %.4f", step + 1, joint_angle_range.size(), current_angle);

    goal_msg.command.position = current_angle;
    goal_msg.command.max_effort = 0.1;

    auto goal_handle_future = gripper_action_client_->async_send_goal(goal_msg);
    auto goal_handle = goal_handle_future.get();
    if (!goal_handle)
    {
      RCLCPP_ERROR(LOGGER, "Failed to send gripper goal at step %zu", step + 1);
      break;
    }

    const auto step_deadline = std::chrono::steady_clock::now() + kStepTimeout;
    bool reached_target = false;
    bool cancelled = false;
    while (rclcpp::ok() && std::chrono::steady_clock::now() < step_deadline)
    {
      const bool alfredo_detected = alfredo_event_.detected.load();
      const bool alfredo_disconnected = alfredo_event_.disconnected.load();
      const bool giovanni_detected = giovanni_event_.detected.load();
      const bool giovanni_disconnected = giovanni_event_.disconnected.load();

      tactile_detected = (alfredo_detected && giovanni_detected);
      if (tactile_detected)
      {
        (void)gripper_action_client_->async_cancel_goal(goal_handle);
        cancelled = true;
        break;
      }
      else if ((alfredo_disconnected && giovanni_detected) || (giovanni_disconnected && alfredo_detected))
      {
        (void)gripper_action_client_->async_cancel_goal(goal_handle);
        cancelled = true;
        tactile_detected = true;
        RCLCPP_INFO(LOGGER, "Only one tactile sensor connected and detected contact.");
        break;
      }
      else if (alfredo_disconnected && giovanni_disconnected)
      {
        (void)gripper_action_client_->async_cancel_goal(goal_handle);
        goal_msg.command.position = grasp_joint_angle_ + 0.2;
        goal_msg.command.max_effort = 0.1;
        (void)gripper_action_client_->async_send_goal(goal_msg);

        if (wait_for_joint_angle(grasp_joint_angle_, kFingerJointEpsilon, 100ms))
          reached_target = true;

        last_commanded_angle = grasp_joint_angle_;
        cancelled = true;
        visual_prior_forced = true;
        RCLCPP_INFO(LOGGER, "Visual prior forced due to tactile sensor disconnection");
        break;
      }

      if (object_detected_during_grasp_.load())
      {
        (void)gripper_action_client_->async_cancel_goal(goal_handle);
        cancelled = true;
        object_detected = true;
        break;
      }

      if (wait_for_joint_angle(current_angle, kFingerJointEpsilon, 10ms))
      {
        reached_target = true;
        break;
      }
    }

    if (cancelled)
    {
      RCLCPP_INFO(LOGGER, "Gripper goal cancelled at step %zu", step + 1);
    }
    else if (!reached_target)
    {
      RCLCPP_WARN(LOGGER,
                  "Timed out waiting for finger_joint to reach %.4f at step %zu (epsilon %.4f)",
                  current_angle,
                  step + 1,
                  kFingerJointEpsilon);
    }

    if (tactile_detected || object_detected || visual_prior_forced)
    {
      break;
    }
  }

  // if (!object_detected && !tactile_detected && !visual_prior_forced)
  // {
  //   waitForGripperCompletion(latest_gripper_status_,
  //                           latest_finger_joint_position_,
  //                           last_commanded_angle,
  //                           kFingerJointEpsilon,
  //                           kStepTimeout);
  //   return;
  // }

  if (!object_detected)
  {
    // Reliability squeeze: if tactile contact was detected, close a tiny bit more asynchronously.
    constexpr float kMinJointAngle = 0.0f;
    constexpr float kMaxJointAngle = 0.69f;
    constexpr float kMaxStroke = 0.140f;
    constexpr float kMaxJointAngleForConversion = 0.67f;

    const double latest_joint_position = latest_finger_joint_position_.load();
    const double achieved_joint_angle = visual_prior_forced
                                            ? static_cast<double>(grasp_joint_angle_)
                                            : (std::isfinite(latest_joint_position)
                                                   ? latest_joint_position
                                                   : last_commanded_angle);
    const float squeeze_angle_increment = std::clamp(grip_offset_, 0.0f, kMaxStroke) * (kMaxJointAngleForConversion / kMaxStroke);
    const float squeeze_angle = std::clamp(static_cast<float>(achieved_joint_angle) + squeeze_angle_increment,
                                           kMinJointAngle,
                                           kMaxJointAngle);

    goal_msg.command.position = squeeze_angle;
    goal_msg.command.max_effort = 0.1;

    (void)gripper_action_client_->async_send_goal(goal_msg);
    last_commanded_angle = squeeze_angle;

    waitForGripperCompletion(latest_gripper_status_,
                            latest_finger_joint_position_,
                            last_commanded_angle,
                            kFingerJointEpsilon,
                            kStepTimeout);
  }

  RCLCPP_INFO(LOGGER, "Closed-loop grasping finished. Object secured: %s, Tactile contact: %s, Visual prior forced: %s",
              object_detected ? "yes" : "no",
              tactile_detected ? "yes" : "no",
              visual_prior_forced ? "yes" : "no");
  // RCLCPP_INFO(LOGGER, "grasp_joint_angle_: %.4f, grip_offset_: %.4f", grasp_joint_angle_, grip_offset_);
}

// competition
/* void add_collision_objects(
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr scene_pub,
  const planning_scene_monitor::PlanningSceneMonitorPtr& planning_scene_monitor,
  float& table_z_coordinate)
{
    const float base_y_shift = 0.0;
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = "base_link";
    collision_object.id = "base_box";

    // Make a box and put it in the way
    shape_msgs::msg::SolidPrimitive base;
    base.type = base.BOX;
    base.dimensions = { 0.21, 0.21, 0.025 };
    geometry_msgs::msg::Pose base_pose;
    base_pose.position.x = 0.0;
    base_pose.position.y = 0.0;
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
    // base.dimensions = { 1.08, 1.085, 0.025 };
    // base_pose.position.x = 0.0;
    // base_pose.position.y = 0.0 + base_y_shift;
    // base_pose.position.z = -0.0375;
    // collision_object.primitives.push_back(base);
    // collision_object.primitive_poses.push_back(base_pose);

    // third base
    // base.dimensions = { 1.08, 0.775, 0.86 };
    // base_pose.position.x = 0.0;
    // base_pose.position.y = 0.0 + base_y_shift;
    // base_pose.position.z = -0.48;
    // collision_object.primitives.push_back(base);
    // collision_object.primitive_poses.push_back(base_pose);

    shape_msgs::msg::SolidPrimitive table;
    table.type = table.BOX;
    table.dimensions = { 1.2, 1.085, 0.708 };
    geometry_msgs::msg::Pose table_pose;
    table_pose.position.x = 0.0;
    table_pose.position.y = 0.5425 + base_y_shift;
    table_pose.position.z = -0.354 - 0.025 - 0.005;
    table_pose.orientation.w = 1.0;
    table_pose.orientation.x = 0.0;
    table_pose.orientation.y = 0.0;
    table_pose.orientation.z = 0.0;

    // used for placement height adjustment in place execution
    table_z_coordinate = table_pose.position.z + 0.5 * table.dimensions[2];

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
} */
void add_collision_objects(
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr scene_pub,
  const planning_scene_monitor::PlanningSceneMonitorPtr& planning_scene_monitor,
  float& table_z_coordinate)
{
    const float base_y_shift = 0.0; //-0.17;
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = "base_link";
    collision_object.id = "base_box";

    // Make a box and put it in the way
    shape_msgs::msg::SolidPrimitive base;
    base.type = base.BOX;
    base.dimensions = { 0.21, 0.21, 0.025 };
    geometry_msgs::msg::Pose base_pose;
    base_pose.position.x = 0.0;
    base_pose.position.y = 0.0;
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
    base.dimensions = { 1.08, 0.6, 0.025 };
    base_pose.position.x = 0.0;
    base_pose.position.y = 0.0 + base_y_shift;
    base_pose.position.z = -0.0375;
    collision_object.primitives.push_back(base);
    collision_object.primitive_poses.push_back(base_pose);

    // third base
    base.dimensions = { 1.08, 0.775, 0.86 };
    base_pose.position.x = 0.0;
    base_pose.position.y = 0.0 + base_y_shift;
    base_pose.position.z = -0.48;
    collision_object.primitives.push_back(base);
    collision_object.primitive_poses.push_back(base_pose);

    shape_msgs::msg::SolidPrimitive table;
    table.type = table.BOX;
    table.dimensions = { 1.75, 1.0, 0.14 };
    geometry_msgs::msg::Pose table_pose;
    table_pose.position.x = 0.0;
    table_pose.position.y = 0.6799999999999999 + base_y_shift;
    table_pose.position.z = -0.285; // -0.29
    table_pose.orientation.w = 1.0;
    table_pose.orientation.x = 0.0;
    table_pose.orientation.y = 0.0;
    table_pose.orientation.z = 0.0;

    // used for placement height adjustment in place execution
    table_z_coordinate = table_pose.position.z + 0.5 * table.dimensions[2];

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

void log_pose_quality(const geometry_msgs::msg::PoseStamped& pose_msg, 
                      moveit::planning_interface::MoveGroupInterface& move_group)
{
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
}

// Visualize an approximate UR reach envelope as radial rays from the visualization frame.
void publishReachabilityEnvelope(moveit_visual_tools::MoveItVisualTools& visual_tools,
                                 std::string& ur_type,
                                 int azimuth_samples = 24,
                                 int elevation_samples = 24)
{
  namespace rvt = rviz_visual_tools;

  double tool_length = 0.28;
  double reach_radius_m = (ur_type == "ur3e") ? 0.573+tool_length : 0.946+tool_length; // UR3e or UR7e max reach

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

void send_tactile_calibration_request(rclcpp::Node::SharedPtr node)
{
  try
  {
    auto giovanni_client = node->create_client<std_srvs::srv::Trigger>("/GIOVANNI/calibrate_contact");
    auto alfredo_client = node->create_client<std_srvs::srv::Trigger>("/ALFREDO/calibrate_contact");
    const auto wait_timeout = std::chrono::seconds(5);

    if (giovanni_client->wait_for_service(wait_timeout))
    {
      auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
      auto fut = giovanni_client->async_send_request(req);
      if (fut.wait_for(std::chrono::seconds(3)) == std::future_status::ready)
      {
        const auto response = fut.get();
        if (response->success)
          RCLCPP_INFO(LOGGER, "GIOVANNI tactile calibration succeeded: %s", response->message.c_str());
        else
          RCLCPP_WARN(LOGGER, "GIOVANNI tactile calibration responded with failure: %s", response->message.c_str());
      }
      else
      {
        RCLCPP_WARN(LOGGER, "GIOVANNI tactile calibration request timed out");
      }
    }
    else
    {
      RCLCPP_WARN(LOGGER, "GIOVANNI tactile calibration service not available");
    }

    if (alfredo_client->wait_for_service(wait_timeout))
    {
      auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
      auto fut = alfredo_client->async_send_request(req);
      if (fut.wait_for(std::chrono::seconds(3)) == std::future_status::ready)
      {
        const auto response = fut.get();
        if (response->success)
          RCLCPP_INFO(LOGGER, "ALFREDO tactile calibration succeeded: %s", response->message.c_str());
        else
          RCLCPP_WARN(LOGGER, "ALFREDO tactile calibration responded with failure: %s", response->message.c_str());
      }
      else
      {
        RCLCPP_WARN(LOGGER, "ALFREDO tactile calibration request timed out");
      }
    }
    else
    {
      RCLCPP_WARN(LOGGER, "ALFREDO tactile calibration service not available");
    }
  }
  catch (const std::exception& e)
  {
    RCLCPP_WARN_STREAM(LOGGER, "Exception while calling tactile calibration services: " << e.what());
  }
}

// void set_PID_gains_from_parameters(std::shared_ptr<const moveit_servo::ServoParameters>& servo_parameters, PIDGains& servo_gains)
// {
//   servo_gains.x_proportional_gain = servo_parameters->x_proportional_gain;
//   servo_gains.y_proportional_gain = servo_parameters->y_proportional_gain;
//   servo_gains.z_proportional_gain = servo_parameters->z_proportional_gain;
//   servo_gains.x_integral_gain = servo_parameters->x_integral_gain;
//   servo_gains.y_integral_gain = servo_parameters->y_integral_gain;
//   servo_gains.z_integral_gain = servo_parameters->z_integral_gain;
//   servo_gains.x_derivative_gain = servo_parameters->x_derivative_gain;
//   servo_gains.y_derivative_gain = servo_parameters->y_derivative_gain;
//   servo_gains.z_derivative_gain = servo_parameters->z_derivative_gain;
//   servo_gains.angular_proportional_gain = servo_parameters->angular_proportional_gain;
//   servo_gains.angular_integral_gain = servo_parameters->angular_integral_gain;
//   servo_gains.angular_derivative_gain = servo_parameters->angular_derivative_gain;
// }

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.use_intra_process_comms(false);
  rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("handover_execution_node", node_options);

  rcl_interfaces::msg::ParameterDescriptor config_id_descriptor;
  config_id_descriptor.dynamic_typing = true;
  node->declare_parameter("config_id", rclcpp::ParameterValue("0"), config_id_descriptor);
  rclcpp::Parameter config_id_param;
  node->get_parameter("config_id", config_id_param);
  std::string config_id = "0";
  if (config_id_param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
  {
    config_id = std::to_string(config_id_param.as_int());
  }
  else if (config_id_param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
  {
    config_id = config_id_param.as_string();
  }
  const std::string submission_report_json = "submission_report_" + config_id + ".json";
  SubmissionReporting submission_reporting;
  submission_reporting.config_id = config_id;

  node->declare_parameter("evaluate_grasp_poses", true);
  bool evaluate_grasp_poses = node->get_parameter("evaluate_grasp_poses").as_bool();
  node->declare_parameter("ur_type", "ur7e");
  std::string ur_type = node->get_parameter("ur_type").as_string();
  node->declare_parameter("enable_placement_pose_interpolation", false);
  const bool enable_placement_pose_interpolation =
    node->get_parameter("enable_placement_pose_interpolation").as_bool();
  node->declare_parameter("placement_pose_interpolation_steps", 1);
  const int placement_pose_interpolation_steps = std::max(
      0,
    static_cast<int>(node->get_parameter("placement_pose_interpolation_steps").as_int()));
  node->declare_parameter("enable_grasp_pose_interpolation", false);
  const bool enable_grasp_pose_interpolation =
    node->get_parameter("enable_grasp_pose_interpolation").as_bool();
  node->declare_parameter("grasp_pose_interpolation_steps", 1);
  const int grasp_pose_interpolation_steps = std::max(
    0,
    static_cast<int>(node->get_parameter("grasp_pose_interpolation_steps").as_int()));
  node->declare_parameter("grasp_pose_interpolation_profile", "linear");
  const std::string grasp_pose_interpolation_profile =
    node->get_parameter("grasp_pose_interpolation_profile").as_string();
  node->declare_parameter("placement_pose_interpolation_profile", "linear");
  const std::string placement_pose_interpolation_profile =
    node->get_parameter("placement_pose_interpolation_profile").as_string();
  node->declare_parameter("continuous_motion_compensation_enabled", true); // compensate for object motion during grasping
  const bool continuous_motion_compensation_enabled = node->
    get_parameter("continuous_motion_compensation_enabled").as_bool();
  node->declare_parameter("gripper_use_joint_angle_interval", true);
  const bool gripper_use_joint_angle_interval =
    node->get_parameter("gripper_use_joint_angle_interval").as_bool();
  node->declare_parameter("gripper_joint_angle_interval_delta", 0.35); // 0.5
  const float gripper_joint_angle_interval_delta =
    static_cast<float>(node->get_parameter("gripper_joint_angle_interval_delta").as_double());
  node->declare_parameter("gripper_joint_angle_interval_steps", 30);
  const int gripper_joint_angle_interval_steps = std::max(
    1,
    static_cast<int>(node->get_parameter("gripper_joint_angle_interval_steps").as_int()));
  node->declare_parameter("gripper_joint_angle_interval_exponent", 5.0);
  const double gripper_joint_angle_interval_exponent =
    node->get_parameter("gripper_joint_angle_interval_exponent").as_double();
  node->declare_parameter("placement_strategy", "initial"); // "initial" or "predefined"
  std::string placement_strategy = node->get_parameter("placement_strategy").as_string();
  node->declare_parameter("predefined_placement_x", 0.0f);
  const float predefined_placement_x = node->get_parameter("predefined_placement_x").as_double();
  node->declare_parameter("predefined_placement_y", 0.0f);
  const float predefined_placement_y = node->get_parameter("predefined_placement_y").as_double();
  node->declare_parameter("workspace_limit_x", 1.0f);
  const float workspace_limit_x = node->get_parameter("workspace_limit_x").as_double();
  const float tool_length = 0.28;
  node->declare_parameter("workspace_limit_y", 1.0f);
  const float workspace_limit_y = node->get_parameter("workspace_limit_y").as_double(); //- tool_length;

  // Configuration
  HandoverConfig config;
  config.approach_distance = 0.35;        // for Piltz approach
  config.tracking_distance = 0.20; //0.32        // for servo tracking
  // More responsive defaults for fast-moving objects.
  // Smaller history reduces lag in velocity estimation; smaller lookahead reduces overshoot.
  config.velocity_estimation_frames = 3;  // Estimate velocity from 3 frames
  config.prediction_frames = 4;           // Predict 4 frames ahead
  config.grasp_velocity_threshold = 0.005; //0.2; // m/s - trigger grasp generation while object is still moving
  config.velocity_threshold = 0.01;   // 0.15   // m/s - threshold to execute the grasp once motion is stable
  config.grasp_pose_tolerance = 0.057; // 0.02      // tolerance for matching grasp pose to current object pose
  config.grasp_pose_obj_tolerance = 0.005; // 0.02
  config.place_tolerance = 0.001; // 0.0055           // tolerance for placing object at goal pose
  config.touchdown_tolerance = 0.05;         // tolerance for detecting touchdown

  // PIDGains servo_gains;

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
  moveit_visual_tools::MoveItVisualTools reach_visual_tools(node, "shoulder_link", "reachability_marker",
                                                            move_group.getRobotModel());
  visual_tools.deleteAllMarkers();
  reach_visual_tools.deleteAllMarkers();
  // publishReachabilityEnvelope(reach_visual_tools, ur_type); // UR3e max reach + tool length

  auto servo_parameters = moveit_servo::ServoParameters::makeServoParameters(node);

  if (servo_parameters == nullptr)
  {
    RCLCPP_FATAL(LOGGER, "Could not get servo parameters!");
    exit(EXIT_FAILURE);
  }

  // get pid gains for servo from parameters
  // RCLCPP_INFO_STREAM(LOGGER, "Servo PID gains:");
  // set_PID_gains_from_parameters(servo_parameters, servo_gains);

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

  float table_z_coordinate = 0.0f;
  auto scene_pub = node->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 10);
  add_collision_objects(scene_pub, planning_scene_monitor, table_z_coordinate);

  // test: visualize table pose
  // geometry_msgs::msg::PoseStamped table_pose;
  // table_pose.header.frame_id = "base_link";
  // table_pose.pose.position.x = 0.0;
  // table_pose.pose.position.y = 0.68;
  // table_pose.pose.position.z = table_z_coordinate;
  // table_pose.pose.orientation.w = 1.0;
  // visual_tools.publishAxisLabeled(table_pose.pose, "table");
  // visual_tools.trigger();

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

  {
    const geometry_msgs::msg::Quaternion q_normalized = normalizedQuaternion(current_pose.pose.orientation);
    submission_reporting.robot_initial_pose_x_mm = current_pose.pose.position.x * 1000.0;
    submission_reporting.robot_initial_pose_y_mm = current_pose.pose.position.y * 1000.0;
    submission_reporting.robot_initial_pose_z_mm = current_pose.pose.position.z * 1000.0;
    submission_reporting.robot_initial_pose_q1 = q_normalized.x;
    submission_reporting.robot_initial_pose_q2 = q_normalized.y;
    submission_reporting.robot_initial_pose_q3 = q_normalized.z;
    submission_reporting.robot_initial_pose_q4 = q_normalized.w;
    RCLCPP_INFO_STREAM(LOGGER, "Current EE position: [" 
                          << current_pose.pose.position.x << ", "
                          << current_pose.pose.position.y << ", "
                          << current_pose.pose.position.z << "]");
    RCLCPP_INFO_STREAM(LOGGER, "Current EE orientation (quaternion): [" 
                          << current_pose.pose.orientation.x << ", "
                          << current_pose.pose.orientation.y << ", "
                          << current_pose.pose.orientation.z << ", "
                          << current_pose.pose.orientation.w << "]");
  }

  // Subscribe to goal_pose topic and object poses
  GoalPoseSubscriber goal_pose_subscriber(node, tf_buffer, config, evaluate_grasp_poses);
  goal_pose_subscriber.setTableVerticalOffset(table_z_coordinate);
  goal_pose_subscriber.setPlacementStrategy(placement_strategy);
  goal_pose_subscriber.setPredefinedPlacementCoordinates(predefined_placement_x, predefined_placement_y);
  goal_pose_subscriber.setStartRobotPose(current_pose);
  goal_pose_subscriber.setWorkspaceLimits(-workspace_limit_x, workspace_limit_x, 0.0f, workspace_limit_y);

  move_group.setStartStateToCurrentState();

  // Prepare servo tracker
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
    auto weight_estimation_status_pub =
      node->create_publisher<std_msgs::msg::Bool>("/weight_estimation_status", 10);

  // Subscribe to servo status
  StatusMonitor status_monitor(node, servo_parameters->status_topic);

  // Get current EE transform for servo
  tracker.getCommandFrameTransform(current_ee_tf);
  current_pose.header.frame_id = current_ee_tf.header.frame_id;  
  current_pose.pose.position.x = current_ee_tf.transform.translation.x;
  current_pose.pose.position.y = current_ee_tf.transform.translation.y;
  current_pose.pose.position.z = current_ee_tf.transform.translation.z;
  current_pose.pose.orientation = current_ee_tf.transform.rotation;
  submission_reporting.ee_command_pose_x_mm = current_ee_tf.transform.translation.x * 1000.0;
  submission_reporting.ee_command_pose_y_mm = current_ee_tf.transform.translation.y * 1000.0;
  submission_reporting.ee_command_pose_z_mm = current_ee_tf.transform.translation.z * 1000.0;

  // Tolerance for servo tracking
  Eigen::Vector3d lin_tol{ 0.0025, 0.0025, 0.0025 };  // 1 mm tolerance
  double rot_tol = 0.025;  // 0.57 degree tolerance

  // Send calibration requests to tactile sensors before starting handover checks
  send_tactile_calibration_request(node);

  // Wait for handover detection and first goal pose
  RCLCPP_INFO_STREAM(LOGGER, "Waiting for handover detection...");
  rclcpp::WallRate wait_rate(10);

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
  target_pose_pub->publish(target_pose);

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
  std::thread grasp_eval_thread;

  // Main servo tracking loop
  while (rclcpp::ok())
  {
    // Check object velocity and initiate grasp generation while preserving the original
    // peak/deceleration-based velocity tracking logic.
    goal_pose_subscriber.checkVelocityAndGenerateGrasps();

    // Launch grasp evaluation in background thread as soon as grasps are received
    if (evaluate_grasp_poses && !goal_pose_subscriber.grasp_evaluation_in_progress_ &&
        !goal_pose_subscriber.grasp_evaluation_complete_ && goal_pose_subscriber.gotGraspingGoal())
    {
      goal_pose_subscriber.grasp_evaluation_in_progress_ = true;
      geometry_msgs::msg::PoseStamped eval_current_pose = current_pose;  // Capture by value
      grasp_eval_thread = std::thread([&goal_pose_subscriber, &move_group, eval_current_pose, continuous_motion_compensation_enabled]() {
        geometry_msgs::msg::PoseStamped gripper_pose;
        goal_pose_subscriber.evaluateGraspPosesTwoStage(move_group, gripper_pose, eval_current_pose,
                                                        continuous_motion_compensation_enabled);
        {
          std::lock_guard<std::mutex> lock(goal_pose_subscriber.grasp_evaluation_mutex_);
          goal_pose_subscriber.evaluated_gripper_pose_ = gripper_pose;
        }
        goal_pose_subscriber.grasp_evaluation_in_progress_ = false;
        goal_pose_subscriber.grasp_evaluation_complete_ = true;
      });
    }

    // Only leave tracking once the grasp request has succeeded and the object is stable
    // below the reach threshold for a few consecutive samples.
    // If grasp evaluation is enabled, also wait for it to complete.
    current_pose = move_group.getCurrentPose();
    if (goal_pose_subscriber.shouldStartReach(current_pose.pose, 2, evaluate_grasp_poses))
    {
      RCLCPP_INFO_STREAM(LOGGER,
                         "Grasp request succeeded and object motion is stable; transitioning to reach phase.");
      break;
    }

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
    if (goal_pose_subscriber.getTrackingGoalPose(new_goal_pose, current_pose, true /* predict object motion */))
    {
        current_goal_pose = new_goal_pose;
        // current_goal_pose.header.stamp = node->now();

        // Visualize updated goal
        visual_tools.deleteAllMarkers();
        visual_tools.trigger();
        pose_name = "tracking_pose_" + std::to_string(pose_index);
        visual_tools.publishAxisLabeled(current_goal_pose.pose, pose_name);
        visual_tools.publishText(current_goal_pose.pose, "Tracking_Goal", rvt::WHITE, rvt::XLARGE);
        visual_tools.trigger();

        pose_index++;
    }

    // Publish current tracking goal
    current_goal_pose.header.stamp = node->now();
    target_pose_pub->publish(current_goal_pose);

    loop_rate.sleep();
  }

  // Wait for background grasp evaluation to complete (if still running)
  if (grasp_eval_thread.joinable())
  {
    RCLCPP_INFO(LOGGER, "Waiting for grasp evaluation to complete...");
    grasp_eval_thread.join();
  }

  geometry_msgs::msg::PoseStamped gripper_pose;
  current_pose = move_group.getCurrentPose();

  if (evaluate_grasp_poses)
  {
    std::lock_guard<std::mutex> lock(goal_pose_subscriber.grasp_evaluation_mutex_);
    gripper_pose = goal_pose_subscriber.evaluated_gripper_pose_;

    // Apply motion compensation to account for object motion during evaluation
    if (continuous_motion_compensation_enabled)
    {
      goal_pose_subscriber.applyObjectMotionCompensationToPoseInternal(gripper_pose, true, true);
      RCLCPP_INFO(LOGGER, "Applied motion compensation to evaluated grasp pose.");
    }
  }
  else if (!goal_pose_subscriber.getGraspingPose(gripper_pose, current_pose))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Failed to get grasping pose");
    return EXIT_FAILURE;
  }

  // log_pose_quality(gripper_pose, move_group);

  // tracker.resetTargetPose();
  // servo_gains.x_proportional_gain = servo_gains.y_proportional_gain = servo_gains.z_proportional_gain = 8.5; 
  // servo_gains.angular_proportional_gain = 8.0;
  // servo_gains.angular_derivative_gain = 1.0;
  // servo_gains.angular_integral_gain = 0.0;
  // // servo_gains.x_integral_gain = servo_gains.y_integral_gain = servo_gains.z_integral_gain = 0.05;
  // // servo_gains.x_derivative_gain = servo_gains.y_derivative_gain = servo_gains.z_derivative_gain = 0.8;
  // tracker.updatePIDConfig(servo_gains.x_proportional_gain, servo_gains.x_integral_gain, servo_gains.x_derivative_gain,
  //                         servo_gains.y_proportional_gain, servo_gains.y_integral_gain, servo_gains.y_derivative_gain,
  //                         servo_gains.z_proportional_gain, servo_gains.z_integral_gain, servo_gains.z_derivative_gain,
  //                         servo_gains.angular_proportional_gain, servo_gains.angular_integral_gain, servo_gains.angular_derivative_gain);

  // visualize the target pose in RViz
  // visual_tools.deleteAllMarkers();
  pose_name = "gripper_pose";
  visual_tools.publishAxisLabeled(gripper_pose.pose, pose_name);
  visual_tools.publishText(gripper_pose.pose, "Gripper_Goal", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();

  if (tracking_status != moveit_servo::PoseTrackingStatusCode::INVALID)
  {
    RCLCPP_INFO_STREAM(LOGGER, "Tracking thread exited, restarting...");
    tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
    
    if (move_to_pose_thread.joinable())
    {
      move_to_pose_thread.join();
    }
    
    // Restart tracker
    move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
      tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
      RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
                                     << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
    });
  }

  // Smooth servo-driven transition from tracking to grasp pose (maintains 100 Hz servo rate)
  // if (enable_grasp_pose_interpolation && grasp_pose_interpolation_steps > 0)
  // {
  //   geometry_msgs::msg::PoseStamped grasp_start_pose = move_group.getCurrentPose();
  //   if (grasp_start_pose.header.frame_id.empty())
  //   {
  //     grasp_start_pose.header.frame_id = gripper_pose.header.frame_id;
  //   }

  //   rclcpp::Time transition_start = node->now();
  //   double transition_duration_sec = (grasp_pose_interpolation_steps + 1) * (1.0 / 100.0);  // Time for all steps at 100 Hz

  //   while (rclcpp::ok())
  //   {
  //     rclcpp::Time now = node->now();
  //     double elapsed_sec = (now - transition_start).seconds();

  //     if (elapsed_sec >= transition_duration_sec)
  //     {
  //       // Transition complete, publish final grasp pose once more
  //       gripper_pose.header.stamp = node->now();
  //       target_pose_pub->publish(gripper_pose);
  //       break;
  //     }

  //     // Time-based linear interpolation with profile
  //     double alpha_linear = elapsed_sec / transition_duration_sec;
  //     double alpha = applyInterpolationProfile(alpha_linear, grasp_pose_interpolation_profile);

  //     geometry_msgs::msg::PoseStamped interpolated_pose;
  //     interpolated_pose.header.frame_id = gripper_pose.header.frame_id;
  //     interpolated_pose.pose = interpolatePoseLinearWithSlerp(grasp_start_pose.pose, gripper_pose.pose, alpha);

  //     // Apply motion compensation during transition if enabled
  //     if (continuous_motion_compensation_enabled)
  //     {
  //       goal_pose_subscriber.applyObjectMotionCompensationToPoseInternal(interpolated_pose, true, true);
  //     }

  //     interpolated_pose.header.stamp = node->now();
  //     target_pose_pub->publish(interpolated_pose);

  //     loop_rate.sleep();  // Maintain 100 Hz servo rate
  //   }
  // }
  // else
  // {
  //   gripper_pose.header.stamp = node->now();
  //   target_pose_pub->publish(gripper_pose);
  // }

  if (tracking_status != moveit_servo::PoseTrackingStatusCode::INVALID)
  {
    RCLCPP_INFO_STREAM(LOGGER, "Tracking thread exited, restarting...");
    tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
    
    if (move_to_pose_thread.joinable())
    {
      move_to_pose_thread.join();
    }
    
    // Restart tracker
    move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
      tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
      RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
                                     << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
    });
  }

  // Grasping loop
  geometry_msgs::msg::PoseStamped current_grasping_pose;
  float grasp_diff_offset = 0.0f;
  while (rclcpp::ok())
  {
    current_pose = move_group.getCurrentPose();
    current_grasping_pose = move_group.getCurrentPose("grasp_center");
    bool stop_tracking = goal_pose_subscriber.canStopGraspApproach(config.grasp_pose_tolerance, config.grasp_pose_obj_tolerance, current_grasping_pose, current_pose, gripper_pose, continuous_motion_compensation_enabled);
    
    if (stop_tracking)
    {
      grasp_diff_offset = current_pose.pose.position.z - gripper_pose.pose.position.z;

      goal_pose_subscriber.unsubscribeFromObjectPose();
      const float grasp_joint_angle = goal_pose_subscriber.getGraspJointAngle();
      //gripper_interface.setJointValueTarget("finger_joint", grasp_joint_angle);
      //gripper_interface.move();
      // closed loop tactile grasping
      const auto joint_angle_range = build_gripper_joint_angle_range(
        grasp_joint_angle,
        gripper_use_joint_angle_interval,
        gripper_joint_angle_interval_delta,
        gripper_joint_angle_interval_steps,
        0.0f,
        0.69f,
        gripper_joint_angle_interval_exponent);


      goal_pose_subscriber.runClosedLoopGripperControl(gripper_interface, joint_angle_range);
      
      RCLCPP_INFO_STREAM(LOGGER, "Reached grasping pose within 2.5 cm, stopping servo tracking...");
      break;
    }
    if (goal_pose_subscriber.replanningNeeded())
    {
      RCLCPP_INFO_STREAM(LOGGER, "Replanning needed during grasp approach, stopping servo tracking to replan...");
      
      tracker.stopMotion();
      tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
    
      if (move_to_pose_thread.joinable())
      {
        move_to_pose_thread.join();
      }
      tracker.resetTargetPose();

      goal_pose_subscriber.resetMotionCompensation();

      current_pose = move_group.getCurrentPose();

      // geometry_msgs::msg::PoseStamped retreat_pose;
      // if (goal_pose_subscriber.getRetreatPoseTowardStart(retreat_pose, current_pose))
      // {
      //   retreat_pose.header.stamp = node->now();
      //   target_pose_pub->publish(retreat_pose);

      //   move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
      //     tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
      //     RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
      //                                 << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
      //   });

      //   if (move_to_pose_thread.joinable())
      //   {
      //     move_to_pose_thread.join();
      //   }

      //   tracker.resetTargetPose();
      //   current_pose = move_group.getCurrentPose();
      // }

      if (evaluate_grasp_poses)
      {
        goal_pose_subscriber.evaluateGraspPosesTwoStage(move_group, gripper_pose, current_pose, continuous_motion_compensation_enabled);
      }
      else {
        if (!goal_pose_subscriber.getGraspingPose(gripper_pose, current_pose))
        {
          RCLCPP_ERROR_STREAM(LOGGER, "Failed to get grasping pose");
          return EXIT_FAILURE;
        }
      }

      // debug
      pose_name = "replanned_pose";
      visual_tools.deleteAllMarkers();
      visual_tools.publishAxisLabeled(gripper_pose.pose, pose_name);
      visual_tools.publishText(gripper_pose.pose, "Gripper_Goal", rvt::WHITE, rvt::XLARGE);
      visual_tools.trigger();

      tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
      gripper_pose.header.stamp = node->now();
      target_pose_pub->publish(gripper_pose);
    
      // Restart tracker
      move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
        tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
        RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
                                    << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
      });
    }
    if (tracking_status != moveit_servo::PoseTrackingStatusCode::INVALID)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Tracking thread exited, restarting...");
      tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;

      if (move_to_pose_thread.joinable())
      {
        move_to_pose_thread.join();
      }
      // tracker.resetTargetPose();

      // Restart tracker
      move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
        tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
        RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
                                      << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
      });
    }
    
    gripper_pose.header.stamp = node->now();
    target_pose_pub->publish(gripper_pose);
    
    loop_rate.sleep();
  }
    
  {
    const auto absolute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch())
        .count();
    submission_reporting.t_human_last_contact_ms = absolute_ms;
    submission_reporting.t_robot_first_contact_ms = absolute_ms;
  }

  // return object to original position
  move_group.setStartStateToCurrentState();
  // keep object-center target in grasp_center frame for convergence checks
  auto obj_orig_goal_pose = goal_pose_subscriber.getInitialObjectPose();

  // get placement goal command in tool0 frame with tracking-like orientation logic
  current_pose = move_group.getCurrentPose();
  geometry_msgs::msg::PoseStamped obj_goal_pose;
  goal_pose_subscriber.getPlacementObjectPoseShifted(obj_goal_pose, current_pose.pose.orientation);
 
  // bias obj a bit downward to enable touchdown detection and improve placement stability
  float downward_bias_z = goal_pose_subscriber.isDelicate() ? -0.0035 : 0.005; // 1 mm downward bias
  RCLCPP_INFO_STREAM(LOGGER, "Applying downward bias of " << -downward_bias_z << " m.");
  obj_goal_pose.pose.position.z = obj_goal_pose.pose.position.z - downward_bias_z; // + grasp_diff_offset;

  // debug: visualize placement goal
  visual_tools.publishAxisLabeled(obj_goal_pose.pose, "placement_goal");
  visual_tools.publishText(obj_goal_pose.pose, "Placement_Goal", rvt::WHITE, rvt::XLARGE);
  visual_tools.publishAxisLabeled(obj_orig_goal_pose.pose, "original_object_pose");
  visual_tools.publishText(obj_orig_goal_pose.pose, "Original_Object_Pose", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();
  
  // resume tracking thread
  if (tracking_status != moveit_servo::PoseTrackingStatusCode::INVALID)
  {
    RCLCPP_INFO_STREAM(LOGGER, "Tracking thread exited, restarting...");
    tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
    
    if (move_to_pose_thread.joinable())
    {
      move_to_pose_thread.join();
    }
    // tracker.resetTargetPose();
    
    // Restart tracker
    move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
      tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
      RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
                                    << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
    });
  }

  // Adjust touchdown tolerance based on object geometry before requesting touchdown detection
  goal_pose_subscriber.adjustTouchdownToleranceForGeometry(config.touchdown_tolerance);
  
  // send touchdown request
  goal_pose_subscriber.sendTouchdownDetectionRequest();
  moveit_servo::StatusCode servo_status;
  tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;

  if (enable_placement_pose_interpolation && placement_pose_interpolation_steps > 0)
  {
    tracker.getCommandFrameTransform(current_ee_tf);
    geometry_msgs::msg::PoseStamped placement_start_pose;
    placement_start_pose.header.frame_id = current_ee_tf.header.frame_id;
    placement_start_pose.pose.position.x = current_ee_tf.transform.translation.x;
    placement_start_pose.pose.position.y = current_ee_tf.transform.translation.y;
    placement_start_pose.pose.position.z = current_ee_tf.transform.translation.z;
    placement_start_pose.pose.orientation = current_ee_tf.transform.rotation;

    if (obj_goal_pose.header.frame_id.empty())
    {
      obj_goal_pose.header.frame_id = placement_start_pose.header.frame_id;
    }

    for (int step = 1; step <= placement_pose_interpolation_steps; ++step)
    {
      const double alpha_linear = static_cast<double>(step) /
                                  static_cast<double>(placement_pose_interpolation_steps + 1);
      const double alpha = applyInterpolationProfile(alpha_linear, placement_pose_interpolation_profile);
      geometry_msgs::msg::PoseStamped interpolated_pose;
      interpolated_pose.header.frame_id = obj_goal_pose.header.frame_id;
      interpolated_pose.pose = interpolatePoseLinearSlerp(placement_start_pose.pose, obj_goal_pose.pose, alpha);
      interpolated_pose.header.stamp = node->now();
      target_pose_pub->publish(interpolated_pose);
      loop_rate.sleep();
    }
    obj_goal_pose.header.stamp = node->now();
    target_pose_pub->publish(obj_goal_pose);
  }
  else
  {
    obj_goal_pose.header.stamp = node->now();
    target_pose_pub->publish(obj_goal_pose);
  }

  // send mass estimation request
  // std_msgs::msg::Bool status_msg;
  // status_msg.data = true;
  // for (int i = 0; i < 5; ++i)
  // {
  //   weight_estimation_status_pub->publish(status_msg);
  //   std::this_thread::sleep_for(std::chrono::milliseconds(5));
  // }

  // resume tracking thread
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
    tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
  }

  obj_goal_pose.header.stamp = node->now();
  target_pose_pub->publish(obj_goal_pose);
  
  while (rclcpp::ok())
  {
    current_grasping_pose = move_group.getCurrentPose();
    double position_diff = sqrt(pow(obj_goal_pose.pose.position.x - current_grasping_pose.pose.position.x, 2) +
                                  pow(obj_goal_pose.pose.position.y - current_grasping_pose.pose.position.y, 2) +
                                  pow(obj_goal_pose.pose.position.z - current_grasping_pose.pose.position.z, 2));
                                  // pow(obj_goal_pose.pose.position.z - current_grasping_pose.pose.position.z + downward_bias_z, 2));
    
    // only check touchdown if we're close to the original object position to avoid false positives during placement approach
    if (position_diff <= config.touchdown_tolerance && goal_pose_subscriber.isTouchdownDetected())
    {
      tracker.stopMotion(); // TODO: remove
      RCLCPP_INFO_STREAM(LOGGER, "Touchdown detected.");
      break;
    }
    // RCLCPP_INFO_STREAM_THROTTLE(LOGGER, *node->get_clock(), 2000, "Current position difference to original object pose: " << position_diff << " m");

    if (position_diff <= config.place_tolerance)  // if position is within the configured tolerance of the goal, assume we've reached the target pose and can stop tracking
    {
      RCLCPP_INFO_STREAM(LOGGER, "Returned object to original position within " << config.place_tolerance << " m, stopping servo tracking...");
      break;
    }
    servo_status = status_monitor.getStatus();

    if (servo_status == moveit_servo::StatusCode::DECELERATE_FOR_COLLISION || servo_status == moveit_servo::StatusCode::HALT_FOR_COLLISION)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Close to collision, stopping");
      break;
    }
    if (tracking_status != moveit_servo::PoseTrackingStatusCode::INVALID)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Grasping thread exited, stopping...");
      break;
    }
    if (servo_status == moveit_servo::StatusCode::HALT_FOR_SINGULARITY)
    {
      RCLCPP_WARN_STREAM(LOGGER, "Reached singularity");
      break;
    }
    
    obj_goal_pose.header.stamp = node->now();
    target_pose_pub->publish(obj_goal_pose);
    
    loop_rate.sleep();
  }

  tracker.stopMotion();

  gripper_interface.setNamedTarget("open");
  gripper_interface.move();

  RCLCPP_INFO_STREAM(LOGGER, "Current placement position: x: "
                                    << current_grasping_pose.pose.position.x << ", y: " << current_grasping_pose.pose.position.y << ", z: " << current_grasping_pose.pose.position.z);
  
  {
    const auto absolute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch())
        .count();
    submission_reporting.t_robot_last_contact_ms = absolute_ms;
  }

  // stop mass estimation request
  // status_msg.data = false;
  // for (int i = 0; i < 5; ++i)
  // {
  //   weight_estimation_status_pub->publish(status_msg);
  //   std::this_thread::sleep_for(std::chrono::milliseconds(5));
  // }

  move_to_pose_thread.join();

  // wait for gripper to open before shutting down
  std::this_thread::sleep_for(std::chrono::seconds(1));

  if (!switch_controllers(node, true))
  {
    RCLCPP_ERROR(LOGGER, "Aborting execution because controller switch failed.");
    executor.cancel();
    executor_thread.join();
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }

  executor.cancel();
  executor_thread.join();

  const bool report_written = writeSubmissionReportJson(submission_report_json, submission_reporting);
  if (report_written)
  {
    RCLCPP_INFO_STREAM(LOGGER, "Submission report written to: " << submission_report_json);
  }

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
