
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
#include <moveit_servo/pose_tracking_thesis.h>
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

#include <dqrobotics/DQ.h>
#include <dqrobotics/robot_modeling/DQ_Kinematics.h>
#include <dqrobotics/robot_modeling/DQ_SerialManipulator.h>
#include <dqrobotics/utils/DQ_Geometry.h>

#include "real_time_filter/OneEuroFilter.h"

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
#include <grasp_generation/action/generate_grasp.hpp>
#include <grasp_generation/msg/grasp_cluster.hpp>
#include <tactile_detection/msg/tactile_detection.hpp>
#include <visual_properties/msg/dimensions.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <omp.h>

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
  // float grasp_velocity_threshold; // Velocity below this triggers grasp generation request
  float velocity_threshold;       // Velocity below this allows executing the generated grasp
  float grasp_pose_tolerance;
  float carry_tolerance;
  float place_tolerance;        // 2 cm - distance to goal pose at which we consider the object placed and can stop tracking
  float touchdown_tolerance;    // distance to original object position at which we consider a touchdown detection valid to avoid false positives during placement approach
  float stable_velocity_samples;  // number of consecutive samples below threshold to trigger grasp
  // optimization
  struct OptimizationWeights
  {
    double alpha; // Weight for RRMC utility function
    double manipulability_w; // Weight for manipulability in utility function
    double manipulability_grad_w; // Weight for manipulability in RRMC utility gradient
    double condition_number_w; // Weight for condition number in utility function
    double condition_number_grad_w; // Weight for condition number in RRMC utility gradient
    double dir_manipulability_w; // Weight for directional manipulability in utility function
    double dir_manipulability_grad_w; // Weight for directional manipulability in RRMC utility gradient
    double joint_limit_w; // Weight for joint limit proximity in utility function
    double joint_limit_buffer_fraction; // Fraction of joint range to consider as buffer for joint limit proximity
    double joint_limit_grad_w; // Weight for joint limit proximity in RRMC utility gradient
    double pitch_w; // Weight for pitch angle in utility function
  } optimization_weights;
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

struct GraspPoseQualitySample
{
  geometry_msgs::msg::PoseStamped pose;
  DQ_robotics::DQ dual_quaternion;
  bool ik_found = false;
  double condition_number = std::numeric_limits<double>::infinity();
  double min_singular_value = 0.0;
  double manipulability = 0.0;
};

static DQ_robotics::DQ poseToUnitDualQuaternion(const geometry_msgs::msg::Pose& pose)
{
  const geometry_msgs::msg::Quaternion q_normalized = normalizedQuaternion(pose.orientation);
  const DQ_robotics::DQ dq(q_normalized.w,
                           q_normalized.x,
                           q_normalized.y,
                           q_normalized.z,
                           0.0,
                           pose.position.x * 0.5,
                           pose.position.y * 0.5,
                           pose.position.z * 0.5);
  return DQ_robotics::normalize(dq);
}

static Eigen::MatrixXd projectDualQuaternionsToPcaCoordinates(const std::vector<DQ_robotics::DQ>& unit_dual_quaternions,
                                                               const int component_count)
{
  if (unit_dual_quaternions.empty() || component_count <= 0)
  {
    return Eigen::MatrixXd();
  }

  const int requested_components = std::min<int>(component_count, 3);
  Eigen::MatrixXd features(static_cast<Eigen::Index>(unit_dual_quaternions.size()), 8);
  for (std::size_t i = 0; i < unit_dual_quaternions.size(); ++i)
  {
    const Eigen::Matrix<double, 8, 1> feature = DQ_robotics::vec8(DQ_robotics::log(unit_dual_quaternions[i]));
    features.row(static_cast<Eigen::Index>(i)) = feature.transpose();
  }

  if (features.rows() < 2)
  {
    return Eigen::MatrixXd::Zero(features.rows(), requested_components);
  }

  const Eigen::RowVectorXd mean = features.colwise().mean();
  const Eigen::MatrixXd centered = features.rowwise() - mean;

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinV);
  const Eigen::MatrixXd principal_axes = svd.matrixV().leftCols(requested_components);
  return centered * principal_axes;
}

static std::vector<double> normalizePlotQuality(const std::vector<GraspPoseQualitySample>& samples,
                                                const bool use_condition_number)
{
  std::vector<double> qualities(samples.size(), 0.0);
  std::vector<double> successful_values;
  successful_values.reserve(samples.size());

  for (const auto& sample : samples)
  {
    if (!sample.ik_found)
    {
      continue;
    }

    const double value = use_condition_number ? sample.condition_number : sample.min_singular_value;
    if (std::isfinite(value))
    {
      successful_values.push_back(value);
    }
  }

  if (successful_values.empty())
  {
    return qualities;
  }

  const auto [min_it, max_it] = std::minmax_element(successful_values.begin(), successful_values.end());
  const double min_value = *min_it;
  const double max_value = *max_it;
  const double value_range = max_value - min_value;

  if (std::abs(value_range) < 1e-12)
  {
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
      if (samples[i].ik_found && std::isfinite(use_condition_number ? samples[i].condition_number : samples[i].min_singular_value))
      {
        qualities[i] = 1.0;
      }
    }
    return qualities;
  }

  for (std::size_t i = 0; i < samples.size(); ++i)
  {
    const auto& sample = samples[i];
    if (!sample.ik_found)
    {
      qualities[i] = 0.0;
      continue;
    }

    const double value = use_condition_number ? sample.condition_number : sample.min_singular_value;
    if (!std::isfinite(value))
    {
      qualities[i] = 0.0;
      continue;
    }

    if (use_condition_number)
    {
      qualities[i] = (max_value - value) / value_range;
    }
    else
    {
      qualities[i] = (value - min_value) / value_range;
    }
  }

  return qualities;
}

static void exportGraspQualityDataToCSV(const std::string& file_name,
                                         const std::vector<GraspPoseQualitySample>& samples)
{
  if (samples.empty())
  {
    RCLCPP_WARN_STREAM(LOGGER, "[PoseQualityCSV] No grasp poses available for CSV export: " << file_name);
    return;
  }

  std::ofstream out(file_name, std::ios::out | std::ios::trunc);
  if (!out.is_open())
  {
    RCLCPP_ERROR_STREAM(LOGGER, "[PoseQualityCSV] Failed to open CSV file for writing: " << file_name);
    return;
  }

  out << "pose_index,ik_found,condition_number,min_singular_value,manipulability,x,y,z,qw,qx,qy,qz\n";
  for (std::size_t i = 0; i < samples.size(); ++i)
  {
    const auto& sample = samples[i];
    out << i << ","
        << (sample.ik_found ? "1" : "0") << ","
        << sample.condition_number << ","
        << sample.min_singular_value << ","
        << sample.manipulability << "," 
        << sample.pose.pose.position.x << ","
        << sample.pose.pose.position.y << ","
        << sample.pose.pose.position.z << ","
        << sample.pose.pose.orientation.w << ","
        << sample.pose.pose.orientation.x << ","
        << sample.pose.pose.orientation.y << ","
        << sample.pose.pose.orientation.z << "\n";
  }

  out.close();
}

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

double computeJointLimitPenalty(
    const moveit::core::RobotState& state,
    const moveit::core::JointModelGroup* jmg,
    double buffer_fraction = 0.10)
{
  Eigen::VectorXd q;
  state.copyJointGroupPositions(jmg, q);
  const auto& joint_models = jmg->getActiveJointModels();

  double total_penalty = 0.0;

  for (std::size_t i = 0; i < joint_models.size(); ++i)
  {
    const auto* joint = joint_models[i];
    const auto& bounds_vec = joint->getVariableBounds();

    if (bounds_vec.empty() || !bounds_vec[0].position_bounded_)
      continue;

    const double q_min = bounds_vec[0].min_position_;
    const double q_max = bounds_vec[0].max_position_;
    const double range = q_max - q_min;

    if (range <= 1e-6)
      continue;

    const double q_max_thr = q_max - (buffer_fraction * range);
    const double q_min_thr = q_min + (buffer_fraction * range);
    const double q_i = q[i];

    if (q_i > q_max_thr)
      total_penalty += std::pow((q_i - q_max_thr) / (q_max - q_max_thr), 2);
    else if (q_i < q_min_thr)
      total_penalty += std::pow((q_i - q_min_thr) / (q_min - q_min_thr), 2);
  }

  return total_penalty;
}

/**
 * \brief Evaluates the combined scalar utility U for a target end-effector pose.
 * 
 * \param move_group MoveGroup interface to access state and kinematics.
 * \param target_pose End-effector target pose to evaluate.
 * \param weights Weighting factors for utility components.
 * \param translational_direction Task-space direction vector for directional manipulability.
 * \param mode Directional manipulability mode (TRANSLATIONAL, ROTATIONAL, BOTH).
 * \param[out] utility Combined objective function score U.
 * \param[out] condition_number Output raw condition number (kappa).
 * \param[out] dir_manipulability Output raw directional manipulability.
 * \return true if IK solution exists and evaluation succeeded.
 */
bool utility_function(
    const moveit::planning_interface::MoveGroupInterface& move_group,
    const geometry_msgs::msg::PoseStamped& target_pose,
    const HandoverConfig::OptimizationWeights& weights,
    const Eigen::Vector3d& translational_direction,
    double& utility,
    double pitch_deg,
    int cluster, // DEBUG
    int id // DEBUG
    )
{
  // Initialize outputs
  utility = 0.0;

  // 1. Fetch current robot state
  auto current_state_ptr = move_group.getCurrentState(1.0);
  if (!current_state_ptr)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[PoseUtility] Could not get current robot state");
    return false;
  }

  moveit::core::RobotState ik_state(*current_state_ptr);
  const moveit::core::JointModelGroup* jmg = ik_state.getJointModelGroup(move_group.getName());
  if (!jmg)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[PoseUtility] Joint model group not found: " << move_group.getName());
    return false;
  }

  std::string eef_link = move_group.getEndEffectorLink();
  if (eef_link.empty())
  {
    const auto& link_names = jmg->getLinkModelNames();
    if (link_names.empty()) return false;
    eef_link = link_names.back();
  }

  // 2. Compute IK for target pose
  const bool ik_found = ik_state.setFromIK(jmg, target_pose.pose, eef_link, 0.05);
  if (!ik_found)
  {
    RCLCPP_WARN_STREAM(LOGGER, "[PoseUtility] No IK solution found for target pose");
    return false;
  }

  // Determine active components using weight thresholding
  const bool eval_cond = weights.condition_number_w > 1e-6;
  const bool eval_manip = weights.manipulability_w > 1e-6;
  const bool eval_dir  = weights.dir_manipulability_w > 1e-6;
  const bool eval_jl   = weights.joint_limit_w > 1e-6;
  const bool eval_pitch = weights.pitch_w > 1e-6;

  double condition_number = std::numeric_limits<double>::infinity();
  double manipulability = 0.0;
  double dir_manipulability = 0.0;
  double inv_condition_number = 0.0;
  double joint_limit_penalty = 0.0;
  double pitch_penalty = 0.0;

  // 3. Compute Jacobian and Singular Values if velocity metrics are needed
  if (eval_cond || eval_dir || eval_manip)
  {
    const auto* eef_link_model = ik_state.getLinkModel(eef_link);
    if (!eef_link_model) return false;

    Eigen::MatrixXd jacobian;
    ik_state.getJacobian(jmg, eef_link_model, Eigen::Vector3d::Zero(), jacobian);

    if (jacobian.rows() != 6 || jacobian.cols() == 0)
    {
      RCLCPP_WARN_STREAM(LOGGER, "[PoseUtility] Invalid Jacobian size: " << jacobian.rows() << "x" << jacobian.cols());
      return false;
    }

    // --- Condition Number (1 / kappa) ---
    if (eval_cond)
    {
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
      const auto& singular_values = svd.singularValues();
      const double min_sv = singular_values.minCoeff();

      if (min_sv > 1e-6)
      {
        const double max_sv = singular_values.maxCoeff();
        condition_number = max_sv / min_sv;
        inv_condition_number = min_sv / max_sv; // Scaled to (0, 1]
      }
      else
      {
        condition_number = std::numeric_limits<double>::infinity();
        inv_condition_number = 0.0;
      }
    }

    // Rotational manipulability
    if (eval_manip)
    {
      manipulability = std::sqrt((jacobian.bottomRows(3) * jacobian.bottomRows(3).transpose()).determinant());
    }

    // --- Directional Manipulability ---
    if (eval_dir)
    {
      const double norm = translational_direction.norm();
      if (norm > 1e-9)
      {
        const Eigen::Vector3d u = translational_direction / norm;
        const Eigen::Matrix3d JJT = jacobian.topRows(3) * jacobian.topRows(3).transpose();

        // Solve (J*J^T) * x = u  => x = (J*J^T)^-1 * u
        // Using LDLT decomposition for speed and stability on 3x3 SPD matrix
        const Eigen::Vector3d x = JJT.ldlt().solve(u);

        const double quad_form = u.dot(x); // u^T * (J*J^T)^-1 * u
        if (quad_form > 1e-12)
          dir_manipulability = 1.0 / std::sqrt(quad_form);
        else
          dir_manipulability = 0.0; // Near singularity
      }
    }
  }

  // 4. Compute Joint Limit Penalty
  if (eval_jl)
    joint_limit_penalty = computeJointLimitPenalty(ik_state, jmg);
  
  if (eval_pitch)
  {
    // RCLCPP_INFO_STREAM(LOGGER, "[PoseUtility] Cluster " << cluster << " ID " << id << ": Pitch (degrees): " << pitch_deg);
    pitch_penalty = std::max(0.0, pitch_deg);
  }
  
  // RCLCPP_INFO_STREAM(LOGGER, "[PoseUtility] Cluster " << cluster << " ID " << id << ": Inv Condition Number: " << inv_condition_number
  //                     << ", Manipulability: " << manipulability
  //                     << ", Directional Manipulability: " << dir_manipulability
  //                     << ", Joint Limit Penalty: " << joint_limit_penalty
  //                     << ", Pitch Penalty: " << pitch_penalty);

  // 5. Combine weighted components into total utility U
  utility = (weights.condition_number_w * inv_condition_number)
          + (weights.dir_manipulability_w * dir_manipulability)
          + (weights.manipulability_w * manipulability)
          - (weights.joint_limit_w * joint_limit_penalty)
          - (weights.pitch_w * pitch_penalty);
  
  // RCLCPP_INFO_STREAM(LOGGER, "[PoseUtility] Cluster " << cluster << " ID " << id << ". CN: "
  //                     << weights.condition_number_w * inv_condition_number << ", DM: " << weights.dir_manipulability_w * dir_manipulability << ", JL: " << weights.joint_limit_w * joint_limit_penalty << ", P: " << weights.pitch_w * pitch_penalty);
  // RCLCPP_INFO_STREAM(LOGGER, "[PoseUtility] Cluster " << cluster << " ID " << id << ": Total Utility: " << utility);

  return true;
}

bool evaluate_pose_anisotropy(const moveit::planning_interface::MoveGroupInterface& move_group,
                                  const geometry_msgs::msg::PoseStamped& target_pose,
                                  double& condition_number,
                                  double& min_singular_value,
                                  double& manipulability)
{
  condition_number = std::numeric_limits<double>::infinity();
  min_singular_value = 0.0;
  manipulability = 0.0;

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

  const Eigen::MatrixXd jj_t = jacobian * jacobian.transpose();
  const double det = std::max(0.0, jj_t.determinant());
  manipulability = std::sqrt(det);

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
      const std::shared_ptr<tf2_ros::Buffer>& tf_buffer,
      const HandoverConfig& config,
      const std::string& base_frame = "base",
      const std::string& base_link_frame = "base_link",
      const std::string& command_frame = "grasp_center",
      const std::string& grasp_frame = "tool_changer_tool0",
      const std::string& goal_topic = "/goal_pose",
      const std::string& handover_topic = "/handover_detected",
      const std::string& dimensions_topic = "/estimated_dimensions"
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

    // debug - remove later
    // debug_publisher_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>("goal_pose_filtered", 10);

    // Initialize action client for grasp generation
    // grasp_action_client_ = rclcpp_action::create_client<grasp_generation::action::GenerateGrasp>(
    //     node, "generate_grasp");
    // grasp_generation_in_progress_ = false;
    // last_grasp_request_time_ = node_->now();
    peak_object_speed_ = 0.0;
    last_object_speed_ = -1.0;
    deceleration_samples_ = 0;

    touchdown_service_client_ = node_->create_client<std_srvs::srv::Trigger>("detect_touchdown");
    touchdown_status_sub_ = node_->create_subscription<std_msgs::msg::Bool>(
      "/touchdown_status", rclcpp::SystemDefaultsQoS(),
      std::bind(&GoalPoseSubscriber::touchdownStatusCallback, this, std::placeholders::_1));
    touchdown_detected_ = false;
    touchdown_detection_in_progress_ = false;
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

    grasps_sub_ = node->create_subscription<grasp_generation::action::GenerateGrasp_Result>(
        "/grasp_poses", rclcpp::SystemDefaultsQoS(),
        [this](const grasp_generation::action::GenerateGrasp_Result::ConstSharedPtr& msg) {
          return graspsCB(msg);
        });

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

  void setBaseTableSafeDistance(float distance)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    base_safe_y_distance_ = distance;
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

  bool canStopGraspApproach(float threshold, const geometry_msgs::msg::PoseStamped& current_pose, geometry_msgs::msg::PoseStamped& gripper_pose, const bool motion_compensate = false)
  {
    constexpr float compensation_threshold = 0.1; // m, only apply compensation if object is far to avoid unnecessary compensation for small errors that are within tolerance
    
    std::lock_guard<std::mutex> lock(mutex_);

    const Eigen::Vector3d current_pos(
      current_pose.pose.position.x,
      current_pose.pose.position.y,
      current_pose.pose.position.z);
    const Eigen::Vector3d goal_gripper_pos(
      gripper_pose.pose.position.x,
      gripper_pose.pose.position.y,
      gripper_pose.pose.position.z);
    
    const double distance_to_goal_grasp = (goal_gripper_pos - current_pos).norm();
    const bool goal_reached = distance_to_goal_grasp < threshold;

    // RCLCPP_INFO_STREAM(LOGGER,
    //            "[CanStopTracking] grasp->goal: " << distance_to_goal_grasp << " m (" 
    //            << (goal_reached ? "PASS" : "FAIL") << ")"
    //           //  << " | grasp->object: " << distance_to_object << " m ("
    //           //  << (object_contact ? "PASS" : "FAIL") << ")"
    //            << " | threshold: " << threshold << " m");

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

  // Shrink tracking distance from approach_distance toward tracking_distance.
  // High object speed => faster shrink (intercept behavior), low speed => very slow or no shrink.
  double updateDynamicTrackingDistanceInternal(double current_speed)
  {
    if (dynamic_tracking_distance_ < 0.0)
      dynamic_tracking_distance_ = config_.approach_distance;

    const double dt = (frame_time_ > 1e-4 && frame_time_ < 0.2) ? frame_time_ : 0.01;

    // Tune these if needed; max speed around 0.5 m/s maps to fastest shrink.
    const double v_ref = 0.7;//0.5;
    const double min_shrink_rate = 0.05;   // m/s distance change when object is almost static
    const double max_shrink_rate = 0.5;  // m/s distance change when object is fast

    const double speed_ratio = std::min(1.0, std::max(0.0, current_speed / v_ref));
    const double shrink_rate = min_shrink_rate + (max_shrink_rate - min_shrink_rate) * speed_ratio;

    dynamic_tracking_distance_ -= shrink_rate * dt;
    dynamic_tracking_distance_ = std::max(static_cast<double>(config_.tracking_distance), dynamic_tracking_distance_);

    // RCLCPP_INFO_STREAM(LOGGER, "Dynamic tracking distance updated: " << dynamic_tracking_distance_ << " m (current speed: " << current_speed << " m/s)");

    return dynamic_tracking_distance_;
  }

  // void getApproachPose(geometry_msgs::msg::PoseStamped& goal, 
  //                    const geometry_msgs::msg::PoseStamped& current_pose, 
  //                    double approach_distance)
  // {
  //   const double alpha = 0.1; // interpolation factor (0.0 = current z, 1.0 = target z)
  //   // 1. Convert the current pose to an Eigen Affine3d matrix to easily extract orientation
  //   Eigen::Vector3d T_goal = poseToMatrix(goal.pose).translation();
  //   Eigen::Vector3d ray;

  //   // keep above the base table
  //   // if (current_pose.pose.position.y <= base_safe_y_distance_)
  //   // {
  //   //   // using general "diagonal-like" approach vector
  //   //   Eigen::Vector3d T_current = poseToMatrix(current_pose.pose).translation();
  //   //   ray = T_goal - T_current;
  //   // }
  //   // else
  //   // {
  //     // 1. Convert the current pose to an Eigen Affine3d matrix to easily extract orientation
  //     Eigen::Affine3d transform_current = poseToMatrix(current_pose.pose);
  //     // 2. Extract the Z-axis vector of the tool0 frame in the base coordinate system.
  //     // In Eigen, the columns of the rotation matrix represent the X, Y, and Z axes.
  //     // column(0) = X-axis, column(1) = Y-axis, column(2) = Z-axis
  //     ray = transform_current.rotation().col(2);
  //   // }

  //   // 3. Normalize just to be perfectly safe (it should already be unit length)
  //   ray.normalize();

  //   // 4. Calculate the approach position by backing off along the tool's Z-axis.
  //   // Since the Z-axis points OUTWARD toward the object, subtracting backs it away.
  //   Eigen::Vector3d T_flange_target = T_goal - (ray * approach_distance);

  //   // 5. Update the goal pose position
  //   goal.pose.position.x = T_flange_target.x();
  //   goal.pose.position.y = T_flange_target.y();
  //   goal.pose.position.z = T_flange_target.z();

  //   // interpolate z coordinate between current z and target z to reduce the rate of descent
  //   if (current_pose.pose.position.y <= base_safe_y_distance_)
  //     goal.pose.position.z = alpha * goal.pose.position.z + (1.0 - alpha) * current_pose.pose.position.z;
  // }

  void getApproachPose(geometry_msgs::msg::PoseStamped& goal, 
                     const geometry_msgs::msg::PoseStamped& current_pose, 
                     double approach_distance)
  {
    Eigen::Vector3d T_goal = poseToMatrix(goal.pose).translation();
    Eigen::Vector3d T_current = poseToMatrix(current_pose.pose).translation();
    Eigen::Affine3d transform_current = poseToMatrix(current_pose.pose);

    // 1. Pure EE Z-axis directional vector (where the tool points)
    Eigen::Vector3d z_ray = transform_current.rotation().col(2).normalized();

    // 2. Direct spatial ray from current pose to goal (for diagonal transit)
    Eigen::Vector3d spatial_ray = T_goal - T_current;
    double dist_to_goal = spatial_ray.norm();

    Eigen::Vector3d approach_direction;

    if (dist_to_goal <= config_.tracking_distance) 
      approach_direction = z_ray;
    else 
    {
      spatial_ray.normalize();

      // 3. Calculate blend factor 't' based on increasing Y coordinate progress
      // t = 0.0 -> Near starting platform (Y ~ 0.0, pure diagonal transit)
      // t = 1.0 -> At/past safe distance (Y >= 0.5, pure tool Z-axis tracking)

      // Set a transition window (e.g. 0.15m before reaching safe Y)
      // Blending starts at Y = 0.35m and completes fully at Y = 0.50m
      const double transition_window = 0.15; 
      double start_blend_y = base_safe_y_distance_ - transition_window; // e.g., 0.35m

      double progress = (current_pose.pose.position.y - start_blend_y) / transition_window;

      // Clamp progress between 0.0 and 1.0
      double t = std::clamp(progress, 0.0, 1.0); 

      // 4. Smoothly blend the directional vector from direct diagonal to tool Z-axis
      approach_direction = (1.0 - t) * spatial_ray + t * z_ray;
      approach_direction.normalize();
    }

    // 5. Compute target approach pose
    Eigen::Vector3d T_flange_target = T_goal - (approach_direction * approach_distance);

    goal.pose.position.x = T_flange_target.x();
    goal.pose.position.y = T_flange_target.y();
    goal.pose.position.z = T_flange_target.z();
  }

  void getTF()
  {
    geometry_msgs::msg::TransformStamped T_grasp_to_cmd_msg; 
    geometry_msgs::msg::TransformStamped T_baselink_to_base_msg;
    try
    {
      T_grasp_to_cmd_msg = tf_buffer_->lookupTransform(grasp_frame_, command_frame_, rclcpp::Time(0));
      T_grasp_to_cmd_ = transformToMatrix(T_grasp_to_cmd_msg.transform);
      // Get transform from base to base_link (dynamic transform)
      T_baselink_to_base_msg = tf_buffer_->lookupTransform(base_link_frame_, base_frame_, rclcpp::Time(0));
      T_baselink_to_base_ = transformToMatrix(T_baselink_to_base_msg.transform);
      tf_initialized_ = true;
    }
    catch (const tf2::TransformException& ex)
    {
      RCLCPP_WARN_STREAM(LOGGER, "TF lookup failed during initialization: " << ex.what());
    }
  }

  bool getTrackingGoalPose(geometry_msgs::msg::PoseStamped& pose, 
                          geometry_msgs::msg::PoseStamped& current_pose, 
                          bool predict_future = false,
                          bool reorient_gripper = false)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!received_pose_)
      return false;
    
    if (!tracking_target_initialized_)
    {
      // Initialize tracking target pose to the current pose for smooth start
      tracking_target_pose_ = current_pose;
      tracking_target_initialized_ = true;
      return true;
    }

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

    // RCLCPP_INFO_STREAM_THROTTLE(LOGGER,
    //                             *node_->get_clock(),
    //                             500,
    //                             "Dynamic tracking distance=" << dynamic_tracking_distance
    //                                                           << " m, object speed=" << current_speed << " m/s");

    
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

  // Build a placement command in grasp_center frame such that grasp_center aligns with initial object center.
  double getPlacementObjectPose(geometry_msgs::msg::PoseStamped& obj_placement_pose, const geometry_msgs::msg::Quaternion& desired_orientation, bool verbose = false) const
  {
    std::lock_guard<std::mutex> lock(mutex_);

    obj_placement_pose.header.stamp = node_->now();
    obj_placement_pose.header.frame_id = base_link_frame_;

    if (placement_strategy_ == "initial") {
      obj_placement_pose.pose = evaluated_gripper_pose_.pose;
      return 0.0;
    }

    Eigen::Isometry3d T_base_grasp_des = Eigen::Isometry3d::Identity();
    Eigen::Vector3d grasp_position;
 
    grasp_position.z() = 0.0;
    grasp_position.x() = predefined_placement_x_;
    grasp_position.y() = predefined_placement_y_;
    
    T_base_grasp_des.translation() = grasp_position;
    T_base_grasp_des.linear() = Eigen::Quaterniond(desired_orientation.w,
                                                   desired_orientation.x,
                                                   desired_orientation.y,
                                                   desired_orientation.z)
                                  .toRotationMatrix();
    
    // 1. Extract local Z axis (approach) in World Frame coordinates
    Eigen::Vector3d z_approach = T_base_grasp_des.linear().col(2);

    // 2. Compute pitch angle relative to the ground plane (World XY)
    //    z_approach.z() > 0 means pointing UP towards sky
    //    z_approach.z() < 0 means pointing DOWN towards ground
    const double z_clamped = std::clamp(z_approach.z(), -1.0, 1.0);
    const double pitch = std::asin(z_clamped);
    const double pitch_deg = pitch * 180.0 / M_PI;

    // 3. Flatten pitch if pointing UP into the sky (pitch_deg > 0)
    if (pitch_deg > 0.0) {
      if (verbose) {
        RCLCPP_INFO_STREAM(LOGGER, "Applying correction. Gripper pitch angle from world XY plane: " << pitch_deg << " degrees. Applying level correction.");
      }
      // Horizontal projection of approach vector in World XY
      Eigen::Vector3d z_horiz(z_approach.x(), z_approach.y(), 0.0);

      // Guard against singularity (pointing 100% straight up)
      if (z_horiz.norm() > 1e-4) {
        z_horiz.normalize();

        // World UP vector (0, 0, 1)
        Eigen::Vector3d world_up(0.0, 0.0, 1.0);

        // Compute a WORLD-SPACE pitch axis: orthogonal to approach and world UP
        // This axis points horizontally to the left of the approach direction
        Eigen::Vector3d pitch_axis_world = world_up.cross(z_horiz).normalized();

        // Rotate by -pitch around this world pitch axis to level the gripper to horizontal
        Eigen::AngleAxisd level_correction(pitch, pitch_axis_world);
        
        // Apply correction in WORLD frame (pre-multiply)
        T_base_grasp_des.linear() = level_correction.toRotationMatrix() * T_base_grasp_des.linear();
      }
    }
    
    obj_placement_pose.pose = matrixToPose(T_base_grasp_des);

    // Determine z-coordinate adjustment
    float z_ratio = 0.4f;
    if (selected_grasp_type_ == kTopGraspType) {
      z_ratio = 0.75f;
    } else if (selected_grasp_type_ == kBottomGraspType) {
      z_ratio = 0.1f;
    }
    
    obj_placement_pose.pose.position.z = std::max(evaluated_gripper_pose_.pose.position.z, static_cast<double>(table_z_offset_ + 0.05));

    return pitch_deg;
  }
  // double getPlacementObjectPose(geometry_msgs::msg::PoseStamped& obj_placement_pose, const geometry_msgs::msg::Quaternion& desired_orientation) const
  // {
  //   std::lock_guard<std::mutex> lock(mutex_);

  //   obj_placement_pose.header.stamp = node_->now();
  //   obj_placement_pose.header.frame_id = base_link_frame_;

  //   if (placement_strategy_ == "initial") {
  //     obj_placement_pose.pose = evaluated_gripper_pose_.pose;
  //     return 0.0;
  //   }

  //   // RCLCPP_INFO(LOGGER, "initial grasp position: x: %f, y: %f, z: %f", evaluated_gripper_pose_.pose.position.x, evaluated_gripper_pose_.pose.position.y, evaluated_gripper_pose_.pose.position.z);

  //   Eigen::Isometry3d T_base_grasp_des = Eigen::Isometry3d::Identity();
  //   Eigen::Vector3d grasp_position;
 
  //   grasp_position.z() = 0.0;
  //   grasp_position.x() = predefined_placement_x_;
  //   grasp_position.y() = predefined_placement_y_;
    
  //   T_base_grasp_des.translation() = grasp_position;
  //   T_base_grasp_des.linear() = Eigen::Quaterniond(desired_orientation.w,
  //                                                  desired_orientation.x,
  //                                                  desired_orientation.y,
  //                                                  desired_orientation.z)
  //                                 .toRotationMatrix();
    
  //    // check if end effector is pointing up, forming an angle between EE's z axis and world y axis
  //   Eigen::Vector3d z = T_base_grasp_des.linear().col(2);
  //   const double pitch = std::atan2(z.z(), std::sqrt(z.x()*z.x() + z.y()*z.y()));
  //   const double pitch_deg = pitch * 180.0 / M_PI;
  //   // RCLCPP_INFO_STREAM(LOGGER, "EE z-axis pitch angle from world y-axis: " << pitch_deg << " degrees");
  //     // Elliminate the tilt angle relative to the world y-axis if it's positive
  //   if (pitch_deg > 0.0) {
  //     Eigen::Vector3d pitch_axis = T_base_grasp_des.linear().col(0);   // local X expressed in world
  //     Eigen::AngleAxisd correction(pitch, pitch_axis);
  //     T_base_grasp_des.linear() = correction.toRotationMatrix() * T_base_grasp_des.linear();
  //   }
    
  //   obj_placement_pose.pose = matrixToPose(T_base_grasp_des);

  //   // Determine z-coordinate adjustment based on grasp type and object geometry
  //   float z_ratio = 0.4f; // default to mid grasp
  //   if (selected_grasp_type_ == kTopGraspType)
  //   {
  //     z_ratio = 0.75f; // 90% for top grasp
  //     // RCLCPP_INFO_STREAM(LOGGER, "Using top grasp strategy with z_ratio=" << z_ratio);
  //   }
  //   else if (selected_grasp_type_ == kBottomGraspType)
  //   {
  //     z_ratio = 0.1f; // 20% for bottom grasp
  //     // RCLCPP_INFO_STREAM(LOGGER, "Using bottom grasp strategy with z_ratio=" << z_ratio);
  //   }
    
  //   // Compute minimum z coordinate: table_z_offset + ratio * object_height
  //   float z_min = table_z_offset_ + z_ratio * latest_dimensions_.height;

  //   // Ensure the placement z coordinate is not below this minimum
  //   obj_placement_pose.pose.position.z = std::max(evaluated_gripper_pose_.pose.position.z, static_cast<double>(table_z_offset_ + 0.05));//, static_cast<double>(z_min));
  //   // RCLCPP_INFO(LOGGER, "computed placement position: x: %f, y: %f, z: %f", obj_placement_pose.pose.position.x, obj_placement_pose.pose.position.y, obj_placement_pose.pose.position.z);
  //   return pitch_deg;
  // }

  struct FlatGraspMapping
  {
    std::size_t cluster_idx;
    std::size_t local_idx;
    geometry_msgs::msg::Pose pose;
  };


  void exportGraspMetricsTwoStage(const moveit::planning_interface::MoveGroupInterface& move_group,
                                      const geometry_msgs::msg::PoseStamped& current_pose)
  {
    constexpr double kMinSingularValueThreshold = 0.1;
    constexpr double kConditionTieEpsilon = 1e-9;

    if (tmp_grasp_clusters_.empty())
    {
      RCLCPP_WARN_STREAM(LOGGER, "[TwoStageGrasps] Received empty grasp cluster array.");
      exit(EXIT_FAILURE);
    }

    // 1. Flatten the clustered poses to preserve the exact old flat loop behavior
    std::vector<FlatGraspMapping> flat_grasps;
    for (std::size_t c = 0; c < tmp_grasp_clusters_.size(); ++c)
    {
      const auto& cluster = tmp_grasp_clusters_[c];
      for (std::size_t j = 0; j < cluster.grasp_goals.poses.size(); ++j)
      {
        FlatGraspMapping mapping;
        mapping.cluster_idx = c;
        mapping.local_idx = j;
        mapping.pose = cluster.grasp_goals.poses[j];
        flat_grasps.push_back(mapping);
      }
    }

    bool found_any_ik = false;
    bool found_filtered_candidate = false;
    // geometry_msgs::msg::PoseStamped best_pose;
    double best_condition_number = std::numeric_limits<double>::infinity();
    double best_min_singular_value = -1.0;
    int best_candidate_index = -1;

    std::vector<GraspPoseQualitySample> plot_samples;
    plot_samples.reserve(flat_grasps.size());

    // 2. Exact same loop as before, now iterating over our flattened mapping
    for (std::size_t i = 0; i < flat_grasps.size(); ++i)
    {
      const auto& flat_g = flat_grasps[i];
      geometry_msgs::msg::PoseStamped candidate_pose;
      
      // Grab header from the parent cluster
      candidate_pose.header = tmp_grasp_clusters_[flat_g.cluster_idx].grasp_goals.header;
      candidate_pose.pose = flat_g.pose;

      if (!transformGraspingPose(candidate_pose, current_pose))
      {
        RCLCPP_ERROR_STREAM(LOGGER, "Failed to transform candidate grasp pose to base_link frame. Can't continue.");
        rclcpp::shutdown();
        exit(EXIT_FAILURE);
      }

      GraspPoseQualitySample sample;
      sample.pose.pose = flat_g.pose; // not EE-adjusted, for plotting

      double condition_number = std::numeric_limits<double>::infinity();
      double min_singular_value = 0.0;
      double manipulability = 0.0;

      if (!evaluate_pose_anisotropy(move_group, candidate_pose, condition_number, min_singular_value, manipulability))
      {
        plot_samples.push_back(sample);
        continue;
      }

      sample.ik_found = true;
      sample.condition_number = condition_number;
      sample.min_singular_value = min_singular_value;
      sample.manipulability = manipulability;
      plot_samples.push_back(sample);

      found_any_ik = true;

      if (min_singular_value < kMinSingularValueThreshold)
      {
        continue;
      }

      if ((condition_number < best_condition_number - kConditionTieEpsilon) ||
          (std::abs(condition_number - best_condition_number) <= kConditionTieEpsilon &&
           min_singular_value > best_min_singular_value))
      {
        found_filtered_candidate = true;
        best_condition_number = condition_number;
        best_min_singular_value = min_singular_value;
        // best_pose = candidate_pose;
        best_candidate_index = static_cast<int>(i);
      }
    }

    std::vector<DQ_robotics::DQ> unit_dual_quaternions;
    unit_dual_quaternions.reserve(plot_samples.size());
    for (auto& sample : plot_samples)
    {
      sample.dual_quaternion = poseToUnitDualQuaternion(sample.pose.pose);
      unit_dual_quaternions.push_back(sample.dual_quaternion);
    }

    // Export grasp quality data to CSV
    exportGraspQualityDataToCSV("grasp_quality_data.csv", plot_samples);
    RCLCPP_INFO_STREAM(LOGGER, "[AMONG ALL GRASPS] Best candidate index: " << best_candidate_index << ", condition number: " << best_condition_number << ", min singular value: " << best_min_singular_value);
  }

  void getValidGripperPose(geometry_msgs::msg::PoseStamped& pose) //const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pose = evaluated_gripper_pose_;
    applyObjectMotionCompensationToPoseInternal(pose, true);
    pose.header.stamp = node_->now();
    pose.header.frame_id = base_link_frame_;
  }

  void reevaluateGraspPoses(const moveit::planning_interface::MoveGroupInterface& move_group, moveit_visual_tools::MoveItVisualTools& visual_tools)
  {
    // 1: check if object moved significantly since last evaluation
    // 2: if so, translate all grasp poses by the delta 
    // 3: re-evaluate all grasp medoids and update best cluster index and best grasp pose
    constexpr double reevaluation_threshold = 0.05; // meters
    geometry_msgs::msg::PoseStamped gripper_pose;
    Eigen::Vector3d delta = Eigen::Vector3d::Zero(); 
    Eigen::Vector3d old_delta = delta;

    while (!grasp_evaluation_complete_)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    // RCLCPP_INFO_STREAM(LOGGER, "[Reevaluation] Grasp evaluation complete. Starting reevaluation thread.");

    // loop this thread at a fixed rate to check for object motion and re-evaluate grasp poses if necessary
    while (rclcpp::ok() && !stop_reevaluation_thread_.load())
    {
      delta = getLatestMotionCompensation(true);
      if ((delta - old_delta).norm() > reevaluation_threshold)
      {
        // RCLCPP_INFO_STREAM(LOGGER, "[Reevaluation] Object moved by " << (delta - old_delta).norm() << " m since last evaluation. Re-evaluating grasp poses.  ");
        evaluateGraspPosesTwoStageParallel(move_group);
        old_delta = delta;
      }
      // Sleep for a short duration before checking again
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  void transformGraspPoses(const geometry_msgs::msg::PoseStamped& ref_grasp_pose)
  {
    for (auto& cluster : tmp_grasp_clusters_)
    {
      for (auto& grasp_pose : cluster.grasp_goals.poses)
      {
        geometry_msgs::msg::PoseStamped grasp_pose_stamped;
        grasp_pose_stamped.header = cluster.grasp_goals.header;
        grasp_pose_stamped.pose = grasp_pose;

        if (!transformGraspingPose(grasp_pose_stamped, ref_grasp_pose))
        {
          RCLCPP_ERROR_STREAM(LOGGER, "Failed to transform grasp poses.");
          continue;
        }
        grasp_pose = grasp_pose_stamped.pose; // Update the original pose with the transformed one
      }
      cluster.grasp_goals.header.frame_id = base_link_frame_; // Update the header frame_id to match the current pose
    }
  }

  void evaluateGraspPosesTwoStageParallel(const moveit::planning_interface::MoveGroupInterface& move_group)
{
  if (tmp_grasp_clusters_.empty())
  {
    RCLCPP_WARN_STREAM(LOGGER, "[TwoStageGrasps] Received empty grasp cluster array.");
    return;
  }

  const std::size_t num_clusters = tmp_grasp_clusters_.size();
  const HandoverConfig::OptimizationWeights& weights = config_.optimization_weights;

  // --- PHASE 1: COARSE SELECTION (Top 3 Medoids) ---

  // Tracking containers for all medoid evaluations
  std::vector<geometry_msgs::msg::PoseStamped> transformed_medoids(num_clusters);
  std::vector<double> medoid_utilities(num_clusters, -std::numeric_limits<double>::infinity());
  std::vector<bool> medoid_valid(num_clusters, false);

  const Eigen::Vector3d delta = getLatestMotionCompensation(true);

  #pragma omp parallel
  {
    #pragma omp for nowait
    for (std::size_t c = 0; c < num_clusters; ++c)
    {
      const auto& cluster = tmp_grasp_clusters_[c];
      if (cluster.grasp_goals.poses.empty()) continue;

      // Medoid is at index 0
      geometry_msgs::msg::PoseStamped medoid_pose;
      medoid_pose.header = cluster.grasp_goals.header;
      medoid_pose.pose = cluster.grasp_goals.poses[0];

      // medoid_pose.pose.position.x += delta.x();
      // medoid_pose.pose.position.y += delta.y();
      // medoid_pose.pose.position.z += delta.z();

      transformed_medoids[c] = medoid_pose;

      // Calculate directional vector u
      geometry_msgs::msg::PoseStamped obj_goal_pose;
      double pitch_deg = getPlacementObjectPose(obj_goal_pose, medoid_pose.pose.orientation);

      Eigen::Vector3d diff{
        obj_goal_pose.pose.position.x - medoid_pose.pose.position.x,
        obj_goal_pose.pose.position.y - medoid_pose.pose.position.y,
        obj_goal_pose.pose.position.z - medoid_pose.pose.position.z
      };

      if (diff.squaredNorm() < 1e-12) continue;
      const Eigen::Vector3d u = diff.normalized();

      double utility = 0.0;
      if (utility_function(move_group, medoid_pose, weights, u, utility, pitch_deg, c, 0))
      {
          medoid_utilities[c] = utility;
          medoid_valid[c] = true;
      }
    }
  }

  // Extract Top-3 valid cluster indices using std::partial_sort
  std::vector<std::size_t> cluster_indices(num_clusters);
  std::iota(cluster_indices.begin(), cluster_indices.end(), 0);

  // Sort top 3 clusters in descending order of utility
  std::size_t top_k = std::min<std::size_t>(3, num_clusters);
  std::partial_sort(
      cluster_indices.begin(), 
      cluster_indices.begin() + top_k, 
      cluster_indices.end(),
      [&medoid_utilities](std::size_t a, std::size_t b) {
        return medoid_utilities[a] > medoid_utilities[b];
      });

  // Filter out invalid clusters from top candidates
  std::vector<std::size_t> top_clusters;
  for (std::size_t i = 0; i < top_k; ++i)
  {
    std::size_t idx = cluster_indices[i];
    if (medoid_valid[idx])
    {
      top_clusters.push_back(idx);
    }
  }

  if (top_clusters.empty())
  {
    RCLCPP_ERROR_STREAM(LOGGER, "[TwoStageGrasps] No valid IK or non-zero utility found for any cluster medoids.");
    return;
  }

  RCLCPP_INFO_STREAM(LOGGER, "[TwoStageGrasps] Evaluating Top " << top_clusters.size() << " clusters in Phase 2.");

  // --- PHASE 2: FINE-GRAINED EVALUATION OVER TOP 3 CLUSTERS ---

  int global_best_cluster_idx = -1;
  int global_best_local_candidate_idx = -1;
  double global_best_utility = -std::numeric_limits<double>::infinity();
  geometry_msgs::msg::PoseStamped global_best_pose;

  // Initialize global best with the absolute best medoid among top clusters
  {
    std::size_t best_medoid_cluster = top_clusters[0];
    global_best_cluster_idx = static_cast<int>(best_medoid_cluster);
    global_best_local_candidate_idx = 0; // Medoid local index
    global_best_utility = medoid_utilities[best_medoid_cluster];
    global_best_pose = transformed_medoids[best_medoid_cluster];
  }

  // Evaluate candidate poses across top clusters (excluding index 0)
  for (const std::size_t cluster_idx : top_clusters)
  {
    const auto& target_cluster = tmp_grasp_clusters_[cluster_idx];
    const std::size_t num_poses = target_cluster.grasp_goals.poses.size();

    if (num_poses <= 1) continue; // Skip if cluster only has a medoid

    #pragma omp parallel
    {
      double thread_best_utility = -std::numeric_limits<double>::infinity();
      int thread_best_idx = -1;
      geometry_msgs::msg::PoseStamped thread_best_pose;

      #pragma omp for nowait
      for (std::size_t i = 1; i < num_poses; ++i) // Ignore index 0 (medoid)
      {
        geometry_msgs::msg::PoseStamped candidate_pose;
        candidate_pose.header = target_cluster.grasp_goals.header;
        candidate_pose.pose = target_cluster.grasp_goals.poses[i];

        // candidate_pose.pose.position.x += delta.x();
        // candidate_pose.pose.position.y += delta.y();
        // candidate_pose.pose.position.z += delta.z();

        geometry_msgs::msg::PoseStamped obj_goal_pose;
        double pitch_deg = getPlacementObjectPose(obj_goal_pose, candidate_pose.pose.orientation);

        Eigen::Vector3d diff{
          obj_goal_pose.pose.position.x - candidate_pose.pose.position.x,
          obj_goal_pose.pose.position.y - candidate_pose.pose.position.y,
          obj_goal_pose.pose.position.z - candidate_pose.pose.position.z
        };

        if (diff.squaredNorm() < 1e-12) continue;
        const Eigen::Vector3d u = diff.normalized();

        double utility = 0.0;
        if (!utility_function(move_group, candidate_pose, weights, u, utility, pitch_deg, cluster_idx, i))
          continue;

        if (utility > thread_best_utility)
        {
          thread_best_utility = utility;
          thread_best_idx = static_cast<int>(i);
          thread_best_pose = candidate_pose;
        }
      }

      // Single thread-safe update per openmp thread block
      #pragma omp critical
      {
        if (thread_best_idx != -1 && thread_best_utility > global_best_utility)
        {
          global_best_utility = thread_best_utility;
          global_best_cluster_idx = static_cast<int>(cluster_idx);
          global_best_local_candidate_idx = thread_best_idx;
          global_best_pose = thread_best_pose;
        }
      }
    }
  }

  // --- PHASE 3: WRITE OUT SELECTION ---
  {
    std::lock_guard<std::mutex> lock(mutex_);
    selected_cluster_ = global_best_cluster_idx;
    best_candidate_index_ = global_best_local_candidate_idx;

    evaluated_gripper_pose_.pose = tmp_grasp_clusters_[selected_cluster_].grasp_goals.poses[static_cast<std::size_t>(best_candidate_index_)];
    evaluated_gripper_pose_.header = tmp_grasp_clusters_[selected_cluster_].grasp_goals.header;
    tmp_grasp_types_ = tmp_grasp_clusters_[selected_cluster_].grasp_labels;
    tmp_grasp_widths_ = tmp_grasp_clusters_[selected_cluster_].grasp_widths;

    applyGraspTypeSelectionForCandidateInternal(static_cast<std::size_t>(best_candidate_index_));
  }

  // RCLCPP_INFO_STREAM(LOGGER,
  //                    "[TwoStageGrasps] Successful Multi-Cluster Utility Search! "
  //                    << "Selected Cluster: " << selected_cluster_
  //                    << ", Local Candidate Index: " << best_candidate_index_
  //                    << ", Utility Score: " << global_best_utility);
}

  // void evaluateGraspPosesTwoStageParallel(const moveit::planning_interface::MoveGroupInterface& move_group)
  // {
  //   constexpr double kMinSingularValueThreshold = 0.1;
  //   constexpr double kConditionTieEpsilon = 1e-9;

  //   if (tmp_grasp_clusters_.empty())
  //   {
  //     RCLCPP_WARN_STREAM(LOGGER, "[TwoStageGrasps] Received empty grasp cluster array.");
  //     exit(EXIT_FAILURE);
  //   }

  //   // --- PHASE 1: COARSE SELECTION (Evaluate Medoids - Parallelized) ---
  //   int best_cluster_idx = -1;
  //   double best_medoid_condition = std::numeric_limits<double>::infinity();
  //   double best_medoid_min_sv = -1.0;

  //   const std::size_t num_clusters = tmp_grasp_clusters_.size();

  //   // Pre-allocated tracking vectors (Direct-indexed safe writes)
  //   std::vector<geometry_msgs::msg::PoseStamped> transformed_medoids(num_clusters);
  //   std::vector<bool> medoid_ik_valid(num_clusters, false);
  //   std::vector<double> medoid_conditions(num_clusters, std::numeric_limits<double>::infinity());
  //   std::vector<double> medoid_min_svs(num_clusters, 0.0);
  //   std::vector<double> medoid_manipulabilities(num_clusters, 0.0);
  //   Eigen::Vector3d delta = getLatestMotionCompensation(true);

  //   #pragma omp parallel
  //   {
  //     // Thread-local variables for selection reduction
  //     int thread_best_cluster_idx = -1;
  //     double thread_best_condition = std::numeric_limits<double>::infinity();
  //     double thread_best_min_sv = -1.0;

  //     #pragma omp for nowait
  //     for (std::size_t c = 0; c < num_clusters; ++c)
  //     {
  //       const auto& cluster = tmp_grasp_clusters_[c];
  //       if (cluster.grasp_goals.poses.empty()) continue;

  //       // The medoid is guaranteed to be at index 0
  //       geometry_msgs::msg::PoseStamped medoid_pose;
  //       medoid_pose.header = cluster.grasp_goals.header;
  //       medoid_pose.pose = cluster.grasp_goals.poses[0];

  //       medoid_pose.pose.position.x += delta.x();
  //       medoid_pose.pose.position.y += delta.y();
  //       medoid_pose.pose.position.z += delta.z();

  //       // Safe direct indexing write (independent indices across threads)
  //       transformed_medoids[c] = medoid_pose;

  //       double condition_number = std::numeric_limits<double>::infinity();
  //       double min_singular_value = 0.0;
  //       double manipulability = 0.0;

  //       if (evaluate_pose_anisotropy(move_group, medoid_pose, condition_number, min_singular_value, manipulability))
  //       {
  //         // Safe direct indexing writes
  //         medoid_ik_valid[c] = true;
  //         medoid_conditions[c] = condition_number;
  //         medoid_min_svs[c] = min_singular_value;
  //         medoid_manipulabilities[c] = manipulability;

  //         // Standard checks (respects hard singular value threshold)
  //         if (min_singular_value >= kMinSingularValueThreshold)
  //         {
  //           if ((condition_number < thread_best_condition - kConditionTieEpsilon) ||
  //               (std::abs(condition_number - thread_best_condition) <= kConditionTieEpsilon &&
  //                min_singular_value > thread_best_min_sv))
  //           {
  //             thread_best_condition = condition_number;
  //             thread_best_min_sv = min_singular_value;
  //             thread_best_cluster_idx = static_cast<int>(c);
  //           }
  //         }
  //       }
  //     }

  //     // Merge thread-local winners into global variables
  //     #pragma omp critical
  //     {
  //       if (thread_best_cluster_idx != -1)
  //       {
  //         if ((thread_best_condition < best_medoid_condition - kConditionTieEpsilon) ||
  //             (std::abs(thread_best_condition - best_medoid_condition) <= kConditionTieEpsilon &&
  //              thread_best_min_sv > best_medoid_min_sv))
  //         {
  //           best_medoid_condition = thread_best_condition;
  //           best_medoid_min_sv = thread_best_min_sv;
  //           best_cluster_idx = thread_best_cluster_idx;
  //         }
  //       }
  //     }
  //   }

  //   // Determine target cluster for Phase 2 fine search
  //   if (best_cluster_idx >= 0)
  //   {
  //     selected_cluster_ = best_cluster_idx;
  //     RCLCPP_INFO_STREAM(LOGGER, "[TwoStageGrasps] Selected cluster " << selected_cluster_);
  //   }
  //   else
  //   {
  //     RCLCPP_ERROR_STREAM(LOGGER, "[TwoStageGrasps] Absolutely no valid IK solutions found for any cluster medoids.");
  //     // keep the previous selected cluster if available, otherwise return
  //     return;
  //   }

  //   // --- PHASE 2: FINE-GRAINED EVALUATION (Parallelized) ---
  //   const auto& target_cluster = tmp_grasp_clusters_[selected_cluster_];
  //   const std::size_t num_poses = target_cluster.grasp_goals.poses.size();

  //   // Initialize global variables to the current best (medoid results)
  //   geometry_msgs::msg::PoseStamped best_pose = transformed_medoids[selected_cluster_];
  //   double best_condition_number = best_medoid_condition;
  //   double best_min_singular_value = best_medoid_min_sv;
  //   int best_candidate_local_index = 0;
  //   delta = getLatestMotionCompensation(true);

  //   #pragma omp parallel
  //   {
  //     // Thread-local storage to prevent cache-line bouncing (false sharing)
  //     geometry_msgs::msg::PoseStamped thread_best_pose;
  //     double thread_best_condition = std::numeric_limits<double>::infinity();
  //     double thread_best_min_sv = -1.0;
  //     int thread_best_idx = -1;

  //     #pragma omp for nowait
  //     for (std::size_t i = 1; i < num_poses; ++i)
  //     {
  //       geometry_msgs::msg::PoseStamped candidate_pose;
  //       candidate_pose.header = target_cluster.grasp_goals.header;
  //       candidate_pose.pose = target_cluster.grasp_goals.poses[i];

  //       candidate_pose.pose.position.x += delta.x();
  //       candidate_pose.pose.position.y += delta.y();
  //       candidate_pose.pose.position.z += delta.z();

  //       double condition_number = std::numeric_limits<double>::infinity();
  //       double min_singular_value = 0.0;
  //       double manipulability = 0.0;

  //       // Note: Ensure evaluate_pose_anisotropy implementation is thread-safe!
  //       if (!evaluate_pose_anisotropy(move_group, candidate_pose, condition_number, min_singular_value, manipulability))
  //         continue;

  //       // Hard filter threshold checking
  //       if (min_singular_value < kMinSingularValueThreshold)
  //         continue;

  //       // Evaluate against thread-local best candidate
  //       if ((condition_number < thread_best_condition - kConditionTieEpsilon) ||
  //           (std::abs(condition_number - thread_best_condition) <= kConditionTieEpsilon &&
  //            min_singular_value > thread_best_min_sv))
  //       {
  //         thread_best_condition = condition_number;
  //         thread_best_min_sv = min_singular_value;
  //         thread_best_pose = candidate_pose;
  //         thread_best_idx = static_cast<int>(i);
  //       }
  //     }

  //     // Merge thread-local results into the global result within a critical block
  //     #pragma omp critical
  //     {
  //       if (thread_best_idx != -1)
  //       {
  //         if ((thread_best_condition < best_condition_number - kConditionTieEpsilon) ||
  //             (std::abs(thread_best_condition - best_condition_number) <= kConditionTieEpsilon &&
  //              thread_best_min_sv > best_min_singular_value))
  //         {
  //           best_condition_number = thread_best_condition;
  //           best_min_singular_value = thread_best_min_sv;
  //           best_pose = thread_best_pose;
  //           best_candidate_local_index = thread_best_idx;
  //         }
  //       }
  //     }
  //   }

  //   // Assign state variables after parallel block
  //   {
  //     std::lock_guard<std::mutex> lock(mutex_);
  //     best_candidate_index_ = best_candidate_local_index;
  //     evaluated_gripper_pose_.pose = tmp_grasp_clusters_[selected_cluster_].grasp_goals.poses[static_cast<std::size_t>(best_candidate_index_)]; // original pose
  //     evaluated_gripper_pose_.header = tmp_grasp_clusters_[selected_cluster_].grasp_goals.header;
  //     tmp_grasp_types_ = tmp_grasp_clusters_[selected_cluster_].grasp_labels;
  //     tmp_grasp_widths_ = tmp_grasp_clusters_[selected_cluster_].grasp_widths;
  //     applyGraspTypeSelectionForCandidateInternal(static_cast<std::size_t>(best_candidate_index_));
  //   }
         
  //   RCLCPP_INFO_STREAM(LOGGER,
  //                      "[TwoStageGrasps] Successful Coarse-to-Fine Search! "
  //                      << "Selected Cluster: " << selected_cluster_
  //                      << ", local_idx=" << best_candidate_index_
  //                      << ", min_sv=" << best_min_singular_value
  //                      << ", condition_number=" << best_condition_number);
  // }

  // void evaluatePlotGraspPosesTwoStage(const moveit::planning_interface::MoveGroupInterface& move_group,
  //                                     geometry_msgs::msg::PoseStamped& pose,
  //                                     const geometry_msgs::msg::PoseStamped& current_pose,
  //                                     const bool update_motion_compensate = false)
  // {
  //   constexpr double kMinSingularValueThreshold = 0.1;
  //   constexpr double kConditionTieEpsilon = 1e-9;

  //   if (tmp_grasp_clusters_.empty())
  //   {
  //     RCLCPP_WARN_STREAM(LOGGER, "[TwoStageGrasps] Received empty grasp cluster array.");
  //     exit(EXIT_FAILURE);
  //   }

  //   // 1. Flatten the clustered poses to preserve the exact old flat loop behavior
  //   std::vector<FlatGraspMapping> flat_grasps;
  //   for (std::size_t c = 0; c < tmp_grasp_clusters_.size(); ++c)
  //   {
  //     const auto& cluster = tmp_grasp_clusters_[c];
  //     for (std::size_t j = 0; j < cluster.grasp_goals.poses.size(); ++j)
  //     {
  //       FlatGraspMapping mapping;
  //       mapping.cluster_idx = c;
  //       mapping.local_idx = j;
  //       mapping.pose = cluster.grasp_goals.poses[j];
  //       flat_grasps.push_back(mapping);
  //     }
  //   }

  //   bool found_any_ik = false;
  //   bool found_filtered_candidate = false;
  //   geometry_msgs::msg::PoseStamped best_pose;
  //   double best_condition_number = std::numeric_limits<double>::infinity();
  //   double best_min_singular_value = -1.0;
  //   int best_candidate_index = -1;

  //   geometry_msgs::msg::PoseStamped best_fallback_pose;
  //   double best_fallback_condition_number = std::numeric_limits<double>::infinity();
  //   double best_fallback_min_singular_value = -1.0;
  //   int best_fallback_candidate_index = -1;

  //   std::vector<GraspPoseQualitySample> plot_samples;
  //   plot_samples.reserve(flat_grasps.size());

  //   // identity
  //   Eigen::Vector3d delta = Eigen::Vector3d::Zero();

  //   // 2. Exact same loop as before, now iterating over our flattened mapping
  //   for (std::size_t i = 0; i < flat_grasps.size(); ++i)
  //   {
  //     const auto& flat_g = flat_grasps[i];
  //     geometry_msgs::msg::PoseStamped candidate_pose;
      
  //     // Grab header from the parent cluster
  //     candidate_pose.header = tmp_grasp_clusters_[flat_g.cluster_idx].grasp_goals.header;
  //     candidate_pose.pose = flat_g.pose;

  //     if (!transformGraspingPose(candidate_pose, current_pose, delta))
  //     {
  //       RCLCPP_ERROR_STREAM(LOGGER, "Failed to transform candidate grasp pose to base_link frame. Can't continue.");
  //       rclcpp::shutdown();
  //       exit(EXIT_FAILURE);
  //     }

  //     GraspPoseQualitySample sample;
  //     sample.pose.pose = flat_g.pose; // not EE-adjusted, for plotting

  //     double condition_number = std::numeric_limits<double>::infinity();
  //     double min_singular_value = 0.0;
  //     double manipulability = 0.0;

  //     if (!evaluate_pose_anisotropy(move_group, candidate_pose, condition_number, min_singular_value, manipulability))
  //     {
  //       plot_samples.push_back(sample);
  //       continue;
  //     }

  //     sample.ik_found = true;
  //     sample.condition_number = condition_number;
  //     sample.min_singular_value = min_singular_value;
  //     sample.manipulability = manipulability;
  //     plot_samples.push_back(sample);

  //     found_any_ik = true;

  //     if ((condition_number < best_fallback_condition_number - kConditionTieEpsilon) ||
  //         (std::abs(condition_number - best_fallback_condition_number) <= kConditionTieEpsilon &&
  //          min_singular_value > best_fallback_min_singular_value))
  //     {
  //       best_fallback_condition_number = condition_number;
  //       best_fallback_min_singular_value = min_singular_value;
  //       best_fallback_pose = candidate_pose;
  //       best_fallback_candidate_index = static_cast<int>(i);
  //     }

  //     if (min_singular_value < kMinSingularValueThreshold)
  //     {
  //       continue;
  //     }

  //     if ((condition_number < best_condition_number - kConditionTieEpsilon) ||
  //         (std::abs(condition_number - best_condition_number) <= kConditionTieEpsilon &&
  //          min_singular_value > best_min_singular_value))
  //     {
  //       found_filtered_candidate = true;
  //       best_condition_number = condition_number;
  //       best_min_singular_value = min_singular_value;
  //       best_pose = candidate_pose;
  //       best_candidate_index = static_cast<int>(i);
  //     }
  //   }

  //   std::vector<DQ_robotics::DQ> unit_dual_quaternions;
  //   unit_dual_quaternions.reserve(plot_samples.size());
  //   for (auto& sample : plot_samples)
  //   {
  //     sample.dual_quaternion = poseToUnitDualQuaternion(sample.pose.pose);
  //     unit_dual_quaternions.push_back(sample.dual_quaternion);
  //   }

  //   // Export grasp quality data to CSV
  //   exportGraspQualityDataToCSV("grasp_quality_data.csv", plot_samples);

  //   if (!found_any_ik)
  //   {
  //     RCLCPP_WARN_STREAM(LOGGER, "[TwoStageGrasps] No IK-valid candidate grasp poses found.");
  //     exit(EXIT_FAILURE);
  //   }

  //   if (found_filtered_candidate)
  //   {
  //     pose = best_pose;
  //     if (best_candidate_index >= 0)
  //     {
  //       // Update selected cluster tracking to match your new class structure
  //       selected_cluster_ = static_cast<int>(flat_grasps[best_candidate_index].cluster_idx);
  //       {
  //         std::lock_guard<std::mutex> lock(mutex_);
  //         applyGraspTypeSelectionForCandidateInternal(static_cast<std::size_t>(best_candidate_index));
  //         applyObjectMotionCompensationToPoseInternal(pose, true);
  //       }
  //     }
  //     RCLCPP_INFO_STREAM(LOGGER,
  //                        "[TwoStageGrasps] Selected grasp with hard min singular value filter (>= "
  //                            << kMinSingularValueThreshold << ")"
  //                            << ", min_sv=" << best_min_singular_value
  //                            << ", condition_number=" << best_condition_number
  //                            << ", chosen_cluster=" << selected_cluster_);
  //     return;
  //   }

  //   pose = best_fallback_pose;
  //   if (best_fallback_candidate_index >= 0)
  //   {
  //     // Update selected cluster tracking for fallback selection
  //     selected_cluster_ = static_cast<int>(flat_grasps[best_fallback_candidate_index].cluster_idx);
  //     {
  //       std::lock_guard<std::mutex> lock(mutex_);
  //       applyGraspTypeSelectionForCandidateInternal(static_cast<std::size_t>(best_fallback_candidate_index));
  //       applyObjectMotionCompensationToPoseInternal(pose, true);
  //     }
  //   }
  //   RCLCPP_WARN_STREAM(LOGGER,
  //                      "[TwoStageGrasps] No candidate passed min singular value threshold ("
  //                          << kMinSingularValueThreshold
  //                          << "). Falling back to best-condition candidate: min_sv="
  //                          << best_fallback_min_singular_value
  //                          << ", condition_number=" << best_fallback_condition_number
  //                          << ", fallback_cluster=" << selected_cluster_);
  // }

  bool gotGraspingGoal() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_gripper_pose_;
  }

  // bool isGraspGenerationInProgress() const
  // {
  //   std::lock_guard<std::mutex> lock(mutex_);
  //   return grasp_generation_in_progress_;
  // }

  bool shouldStartReach(const geometry_msgs::msg::Pose& current_pose, int stable_samples_required = 3)
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
    if (!grasp_evaluation_complete_ || grasp_evaluation_in_progress_)
    {
      return false;
    }

    if (best_candidate_index_ < 0)
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
    double distance_to_object = (tool_pos - object_pos).norm();
    // floor the distance to 2 decimal places to avoid oscillation around the threshold
    distance_to_object = std::floor(distance_to_object * 100.0) / 100.0;

    // if (distance_to_object > static_cast<double>(config_.tracking_distance))
    // {
    //   // RCLCPP_INFO_STREAM_THROTTLE(
    //   //     LOGGER,
    //   //     *node_->get_clock(),
    //   //     1000,
    //   //     "Distance to object: " << distance_to_object << " m, still above tracking distance threshold of "
    //   //                             << config_.tracking_distance << " m. Continuing to track.");
    //   return false;  // Still too far from object, continue tracking
    // }

    Eigen::Vector3d velocity;
    if (!estimateVelocityInternal(velocity))
    {
      // RCLCPP_INFO_STREAM(LOGGER, "estimateVelocityInternal");
      return false;
    }

    const double speed = velocity.norm();

    const bool dynamic_ready = speed <= static_cast<double>(config_.velocity_threshold);

    // Keep a simple low-speed counter as fallback if no clear descending branch is observed.
    if (dynamic_ready)
    {
      ++stable_velocity_samples_;
    }
    else
    {
      stable_velocity_samples_ = 0;
      // RCLCPP_INFO_STREAM(LOGGER, "dynamic_ready");
    }

    const bool stable_ready = stable_velocity_samples_ >= stable_samples_required;

    // if (!stable_ready)
    //   RCLCPP_INFO_STREAM(LOGGER, "stable_ready");

    return dynamic_ready && stable_ready;
  }

  bool isTouchdownDetected() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return touchdown_detected_;
  }

  // Check velocity and initiate grasp generation if below threshold
  // void checkVelocityAndGenerateGrasps()
  // {
  //   bool should_send_request = false;

  //   {
  //     std::lock_guard<std::mutex> lock(mutex_);

  //     // If grasping goal already arrived, nothing to do.
  //     if (received_gripper_pose_)
  //       return;

  //     // If a request is in progress for too long, release the state machine and retry.
  //     if (grasp_generation_in_progress_)
  //     {
  //       const double elapsed = (node_->now() - last_grasp_request_time_).seconds();
  //       if (elapsed <= grasp_request_timeout_sec_)
  //       {
  //         return;
  //       }

  //       RCLCPP_WARN_STREAM(LOGGER,
  //                          "Grasp generation request timed out after " << elapsed
  //                          << " s. Resetting request state and retrying.");
  //       grasp_generation_in_progress_ = false;
  //     }

  //     grasp_generation_in_progress_ = true;
  //     last_grasp_request_time_ = node_->now();
  //     should_send_request = true;
  //     RCLCPP_INFO_STREAM(LOGGER,
  //                          "Triggering grasp generation");
  //   }

  //   if (should_send_request)
  //   {
  //     sendGraspGenerationRequest();
  //   }
  // }

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

  void graspsCB(const grasp_generation::action::GenerateGrasp_Result::ConstSharedPtr& result)
  {
    if (!result->status)
    {
      RCLCPP_ERROR_STREAM(LOGGER, "Grasp generation action returned status=false");
      exit(EXIT_FAILURE);
    }

    RCLCPP_INFO_STREAM(LOGGER, "Grasp generation action succeeded with " << result->clusters.size() << " grasp clusters");
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tmp_grasp_clusters_ = result->clusters;
      received_gripper_pose_ = true;
    }
    
    grasps_sub_.reset(); 
  }

  float getGraspJointAngle() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return grasp_joint_angle_;
  }

  const Eigen::Vector3d getLatestMotionCompensation(const bool predict_future = false)
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
        initial_object_pose_.pose.position.x,
        initial_object_pose_.pose.position.y,
        initial_object_pose_.pose.position.z);
    const Eigen::Vector3d latest_pos(
        reference_object_pose.pose.position.x,
        reference_object_pose.pose.position.y,
        reference_object_pose.pose.position.z);
    const Eigen::Vector3d delta = latest_pos - request_pos;

    return delta;
  }

  const Eigen::Vector3d getLatestMotionCompensationInternal(const bool predict_future = false)
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
        initial_object_pose_.pose.position.x,
        initial_object_pose_.pose.position.y,
        initial_object_pose_.pose.position.z);
    const Eigen::Vector3d latest_pos(
        reference_object_pose.pose.position.x,
        reference_object_pose.pose.position.y,
        reference_object_pose.pose.position.z);
    const Eigen::Vector3d delta = latest_pos - request_pos;

    return delta;
  }

  const Eigen::Vector3d applyObjectMotionCompensationToPoseInternal(geometry_msgs::msg::PoseStamped& pose, const bool predict_future = false)
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
        initial_object_pose_.pose.position.x,
        initial_object_pose_.pose.position.y,
        initial_object_pose_.pose.position.z);
    const Eigen::Vector3d latest_pos(
        reference_object_pose.pose.position.x,
        reference_object_pose.pose.position.y,
        reference_object_pose.pose.position.z);
    const Eigen::Vector3d delta = latest_pos - request_pos;

    pose.pose.position.x += delta.x();
    pose.pose.position.y += delta.y();
    pose.pose.position.z += delta.z();

    return delta;
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

  void applyGraspTypeSelectionForCandidateInternal(std::size_t candidate_index)
  {
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
        return;
      }

      if (!received_dimensions_)
      {
        return;
      }

      if (selected_grasp_type == kTopGraspType)
      {
        target_width = latest_dimensions_.top_w;
      }
      else if (selected_grasp_type == kBottomGraspType)
      {
        target_width = latest_dimensions_.bot_w;
      }
    }

    grasp_joint_angle_ = getGripperAngleFromWidth(target_width);
    RCLCPP_INFO_STREAM(LOGGER, "Selected target width " << target_width
                                                      << " m, corresponding to gripper joint angle " << grasp_joint_angle_ << " rad");
    robotiq_vertical_displacement_ = getRobotiqVerticalDisplacement(grasp_joint_angle_);
  }

  // transforms grasps to base_link frame and applies any necessary corrections (e.g. orientation flip) for fair manipulability evaluation, without modifying the original gripper_pose_ member variable
  bool transformGraspingPose(geometry_msgs::msg::PoseStamped& pose, const geometry_msgs::msg::PoseStamped& ref_pose, const Eigen::Vector3d& translation = Eigen::Vector3d::Zero())
  {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!received_gripper_pose_)
    {
      RCLCPP_WARN_STREAM(LOGGER, "Gripper pose not received yet");
      return false;
    }

      Eigen::Isometry3d T_base = poseToMatrix(pose.pose);
      Eigen::Isometry3d T_out;
      if (pose.header.frame_id == base_frame_)
      {
        // Then: apply base->base_link transform: T_out = T_baselink_to_base @ T_base
        T_out = T_baselink_to_base_ * T_base;
        RCLCPP_WARN_STREAM_THROTTLE(LOGGER, *node_->get_clock(), 1000, "Received grasp pose in base frame, transforming to base_link frame");
      }
      else if (pose.header.frame_id == base_link_frame_)
        T_out = T_base;
      else
      {
        RCLCPP_ERROR_STREAM(LOGGER, "Received pose IS NOT in base or base_link frame as expected: " << pose.header.frame_id);
        return false;
      }

      // Multiply on the RIGHT with the INVERSE to apply in tool0's local frame
      T_out = T_out * T_grasp_to_cmd_;

      // Convert back to pose, output in base_link frame
      pose.header.frame_id = base_link_frame_;
      pose.pose = matrixToPose(T_out);

      // Compare current and goal end-effector Y-axis up/down direction in the same frame.
      // If one points up (+Z) and the other down (-Z), flip goal by 180 degrees around Z.
      correctEEOrientation(pose, ref_pose);

      // Apply the translation correction
      pose.pose.position.x += translation.x();
      pose.pose.position.y += translation.y();
      pose.pose.position.z += translation.z();

      return true;
  }

  // void correctEEOrientation(geometry_msgs::msg::PoseStamped& pose, const geometry_msgs::msg::PoseStamped& ref_pose)
  // {
  //     Eigen::Quaterniond q_current(ref_pose.pose.orientation.w,
  //                                  ref_pose.pose.orientation.x,
  //                                  ref_pose.pose.orientation.y,
  //                                  ref_pose.pose.orientation.z);
  //     Eigen::Quaterniond q_goal(pose.pose.orientation.w,
  //                               pose.pose.orientation.x,
  //                               pose.pose.orientation.y,
  //                               pose.pose.orientation.z);
  //     q_current.normalize();
  //     q_goal.normalize();

  //     const Eigen::Vector3d current_y_axis = q_current * Eigen::Vector3d::UnitY();
  //     const Eigen::Vector3d goal_y_axis = q_goal * Eigen::Vector3d::UnitY();

  //     const double current_y_z = current_y_axis.dot(Eigen::Vector3d::UnitZ());
  //     const double goal_y_z = goal_y_axis.dot(Eigen::Vector3d::UnitZ());
  //     const double updown_eps = 1e-4;

  //     // Strictly opposite up/down direction: one positive and one negative along world Z.
  //     if ((std::abs(current_y_z) > updown_eps) &&
  //         (std::abs(goal_y_z) > updown_eps) &&
  //         ((current_y_z > 0.0 && goal_y_z < 0.0) || (current_y_z < 0.0 && goal_y_z > 0.0)))
  //     {
  //       // RCLCPP_INFO_STREAM(LOGGER, "Flipping goal pose by 180 degrees around Z-axis to match Y-axis up/down direction");
  //       const Eigen::AngleAxisd flip_rotation(M_PI, Eigen::Vector3d::UnitZ());
  //       const Eigen::Quaterniond q_flip(flip_rotation);
  //       const Eigen::Quaterniond q_flipped = q_goal * q_flip;

  //       pose.pose.orientation.w = q_flipped.w();
  //       pose.pose.orientation.x = q_flipped.x();
  //       pose.pose.orientation.y = q_flipped.y();
  //       pose.pose.orientation.z = q_flipped.z();
  //     }
  // }


  void correctEEOrientation(geometry_msgs::msg::PoseStamped& pose, const geometry_msgs::msg::PoseStamped& ref_pose)
{
    Eigen::Quaterniond q_current(ref_pose.pose.orientation.w,
                                 ref_pose.pose.orientation.x,
                                 ref_pose.pose.orientation.y,
                                 ref_pose.pose.orientation.z);
    Eigen::Quaterniond q_goal(pose.pose.orientation.w,
                              pose.pose.orientation.x,
                              pose.pose.orientation.y,
                              pose.pose.orientation.z);
    q_current.normalize();
    q_goal.normalize();

    // Local Y-axis vectors in base_link frame
    const Eigen::Vector3d current_y_axis = q_current * Eigen::Vector3d::UnitY();
    const Eigen::Vector3d goal_y_axis = q_goal * Eigen::Vector3d::UnitY();

    // Check 3D alignment of Y axes. 
    // If dot product is negative, Y axes point in opposite directions (>90 degrees apart).
    // Flipping 180 deg around local Z aligns the goal Y axis with the current Y axis.
    if (current_y_axis.dot(goal_y_axis) < 0.0)
    {
      const Eigen::AngleAxisd flip_rotation(M_PI, Eigen::Vector3d::UnitZ());
      const Eigen::Quaterniond q_flipped = q_goal * flip_rotation;

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
    if (transformed_msg.header.frame_id != base_link_frame_)
      transformToWorldFrame(transformed_msg); // transform incoming pose to world frame for workspace check and storage
    
    if (!checkPoseInWorkspace(transformed_msg.pose.position.x, transformed_msg.pose.position.y))
    {
      RCLCPP_WARN_STREAM(LOGGER, "Received pose is out of workspace bounds. Ignoring. POse (x, y): " << transformed_msg.pose.position.x << ',' << transformed_msg.pose.position.y << " min x: " << workspace_x_min_ << ", max x: " << workspace_x_max_ << ", min y: " << workspace_y_min_ << ", max y: " << workspace_y_max_);
      return;
    }

    if (!received_pose_)
    {
      initial_object_pose_ = transformed_msg;
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
  Eigen::Isometry3d T_grasp_to_cmd_;
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
  float base_safe_y_distance_ = 0.0f;
  float grip_offset_ = kDefaultGripOffset;
  bool is_delicate_ = false;
  float grasp_joint_angle_ = 0.45; // default, will be updated by perception
  float robotiq_vertical_displacement_ = 0.0;
  int selected_grasp_type_ = kMidGraspType;
  bool received_dimensions_ = false;
  visual_properties::msg::Dimensions latest_dimensions_;
  geometry_msgs::msg::PoseStamped latest_pose_; // object centroid pose
  std::vector<grasp_generation::msg::GraspCluster> tmp_grasp_clusters_;
  int selected_cluster_ = -1;
  int best_candidate_index_ = -1;
  // geometry_msgs::msg::PoseArray tmp_gripper_poses_;
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
  rclcpp::Subscription<grasp_generation::action::GenerateGrasp_Result>::SharedPtr grasps_sub_;
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
  rclcpp::Node::SharedPtr node_;
  // rclcpp_action::Client<grasp_generation::action::GenerateGrasp>::SharedPtr grasp_action_client_;
  // bool grasp_generation_in_progress_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr touchdown_service_client_;
  bool touchdown_detection_in_progress_;
  bool touchdown_detected_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr touchdown_status_sub_;
  // rclcpp::Time last_grasp_request_time_;
  // double grasp_request_timeout_sec_ = 5.0;
  double peak_object_speed_ = 0.0;
  double last_object_speed_ = -1.0;
  int deceleration_samples_ = 0;
  int stable_velocity_samples_ = 0;
  double dynamic_tracking_distance_ = -1.0;
  bool tracking_target_initialized_ = false;
  geometry_msgs::msg::PoseStamped tracking_target_pose_;

  // Real-time filtering variables
  bool enable_real_time_filtering_ = false;
  double frequency_ = 120.0;
  double mincutoff_ = 0.5;
  double beta_ = 0.5;
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
  std::atomic<bool> stop_reevaluation_thread_{false};
  geometry_msgs::msg::PoseStamped evaluated_gripper_pose_;

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
}

void add_collision_objects(
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr scene_pub,
  const planning_scene_monitor::PlanningSceneMonitorPtr& planning_scene_monitor,
  float& table_z_coordinate,
  float& base_table_y_shift)
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

    base_table_y_shift = base_y_shift + base.dimensions[1] * 0.5;

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
  config.approach_distance = 0.40;        // for Piltz approach
  config.tracking_distance = 0.15;        // for servo tracking
  // More responsive defaults for fast-moving objects.
  // Smaller history reduces lag in velocity estimation; smaller lookahead reduces overshoot.
  config.velocity_estimation_frames = 3;  // Estimate velocity from n frames
  config.prediction_frames = 5; //10           // Predict n frames ahead
  config.velocity_threshold = 0.015;   // m/s - threshold to execute the grasp once motion is stable
  config.stable_velocity_samples = 5; // 5      // number of consecutive samples below threshold to trigger grasp
  config.grasp_pose_tolerance = 0.1; // 0.025      // tolerance for grasp pose 
  config.carry_tolerance = 0.1;      // tolerance for matching carry pose to current object pose
  config.place_tolerance = 0.005;          // tolerance for placing object at goal pose
  config.touchdown_tolerance = 0.05;         // tolerance for detecting touchdown
  config.optimization_weights.alpha = 11.0;
  config.optimization_weights.condition_number_w = 200.0;//100.0;
  config.optimization_weights.condition_number_grad_w = 0.6;
  config.optimization_weights.manipulability_w = 1.0;
  config.optimization_weights.manipulability_grad_w = 0.4;
  config.optimization_weights.dir_manipulability_w = 2.0;
  config.optimization_weights.dir_manipulability_grad_w = 0.0;
  config.optimization_weights.joint_limit_w = 10.0;
  config.optimization_weights.joint_limit_grad_w = 2.0;
  config.optimization_weights.pitch_w = 0.02;

  RCLCPP_INFO(LOGGER, "OpenMP is enabled. Max threads available: %d", omp_get_max_threads());

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

  float table_z_coordinate = 0.0f;
  float base_table_y_shift = 0.0f;
  const float ee_to_joint_offset = 0.44f; // arm length
  auto scene_pub = node->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 10);
  add_collision_objects(scene_pub, planning_scene_monitor, table_z_coordinate, base_table_y_shift);

  // Open gripper in background
  std::thread([&gripper_interface]() {
        gripper_interface.setNamedTarget("open");
        gripper_interface.move();
      }).detach();

  // Create TF2 buffer and listener for dynamic frame transformations
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  move_group.setStartStateToCurrentState();

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
  GoalPoseSubscriber goal_pose_subscriber(node, tf_buffer, config);
  goal_pose_subscriber.setTableVerticalOffset(table_z_coordinate);
  goal_pose_subscriber.setBaseTableSafeDistance(base_table_y_shift + ee_to_joint_offset);
  goal_pose_subscriber.setPlacementStrategy(placement_strategy);
  goal_pose_subscriber.setPredefinedPlacementCoordinates(predefined_placement_x, predefined_placement_y);
  goal_pose_subscriber.setStartRobotPose(current_pose);
  goal_pose_subscriber.setWorkspaceLimits(-workspace_limit_x, workspace_limit_x, 0.0f, workspace_limit_y);

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
  // moveit_servo::RRMCConfig rrmc_config;
  // rrmc_config.alpha = config.optimization_weights.alpha;
  // rrmc_config.condition_number_w = config.optimization_weights.condition_number_grad_w;
  // rrmc_config.manipulability_w = config.optimization_weights.manipulability_grad_w;
  // rrmc_config.dir_manipulability_w = config.optimization_weights.dir_manipulability_grad_w;
  // rrmc_config.joint_limit_w = config.optimization_weights.joint_limit_grad_w;
  // rrmc_config.redundant_dims.x_rotation = true;
  // rrmc_config.redundant_dims.y_rotation = true;
  // rrmc_config.redundant_dims.z_rotation = true;
  // tracker.setRRMCConfig(rrmc_config);
  // tracker.setControlMode(moveit_servo::PoseTrackingControlMode::JOINT);

  // Make a publisher for sending pose commands to the tracker
  auto target_pose_pub =
      node->create_publisher<geometry_msgs::msg::PoseStamped>("target_pose", rclcpp::SystemDefaultsQoS());
  
  // Subscribe to servo status
  StatusMonitor status_monitor(node, servo_parameters->status_topic);

  // Tolerance for servo tracking
  Eigen::Vector3d lin_tol{ 0.0025, 0.0025, 0.0025 };  // 1 mm tolerance
  double rot_tol = 0.025;  // 0.57 degree tolerance

  // Send calibration requests to tactile sensors before starting handover checks
  send_tactile_calibration_request(node);

  // Get current EE transform for servo
  tracker.getCommandFrameTransform(current_ee_tf);
  current_pose = move_group.getCurrentPose();
  submission_reporting.ee_command_pose_x_mm = current_ee_tf.transform.translation.x * 1000.0;
  submission_reporting.ee_command_pose_y_mm = current_ee_tf.transform.translation.y * 1000.0;
  submission_reporting.ee_command_pose_z_mm = current_ee_tf.transform.translation.z * 1000.0;

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
    // RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
    //                                << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
  });

  rclcpp::WallRate loop_rate(100);  // 100 Hz tracking loop
  geometry_msgs::msg::PoseStamped current_goal_pose = target_pose;
  int pose_index = 0;
  std::thread grasp_eval_thread;
  std::thread grasp_replan_thread;

  // Main servo tracking loop
  while (rclcpp::ok())
  { 
    // Launch grasp evaluation in background thread as soon as grasps are received
    if (goal_pose_subscriber.gotGraspingGoal() && !goal_pose_subscriber.grasp_evaluation_in_progress_ &&
        !goal_pose_subscriber.grasp_evaluation_complete_)
    {
      goal_pose_subscriber.grasp_evaluation_in_progress_ = true;
      grasp_eval_thread = std::thread([&goal_pose_subscriber, &current_pose, &move_group]() {
        goal_pose_subscriber.transformGraspPoses(current_pose);
        goal_pose_subscriber.evaluateGraspPosesTwoStageParallel(move_group);
   
        goal_pose_subscriber.grasp_evaluation_in_progress_ = false;
        goal_pose_subscriber.grasp_evaluation_complete_ = true;
      });

      // launch a grasp replanning thread which activates after grasp_evaluation_complete_ = true
      grasp_replan_thread = std::thread([&goal_pose_subscriber, &move_group, &visual_tools]() {
        goal_pose_subscriber.reevaluateGraspPoses(move_group, visual_tools);
      });
    }

    // Only leave tracking once the grasp request has succeeded and the object is stable
    // below the reach threshold for a few consecutive samples.
    // If grasp evaluation is enabled, also wait for it to complete.
    current_pose = move_group.getCurrentPose();
    if (goal_pose_subscriber.shouldStartReach(current_pose.pose, config.stable_velocity_samples))
    {
      goal_pose_subscriber.stop_reevaluation_thread_ = true;
      RCLCPP_INFO_STREAM(LOGGER,
                         "Grasp request succeeded and object motion is stable; transitioning to reach phase.");
      break;
    }

    if (tracking_status != moveit_servo::PoseTrackingStatusCode::INVALID)
    {
      tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
      
      if (move_to_pose_thread.joinable())
      {
        move_to_pose_thread.join();
      }
      tracker.resetTargetPose();
      
      // Restart tracker
      move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // small delay to avoid rapid restart
        tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
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
        // visual_tools.deleteAllMarkers();
        // visual_tools.trigger();
        // pose_name = "tracking_pose"; //"tracking_pose_" + std::to_string(pose_index);
        // visual_tools.publishAxisLabeled(current_goal_pose.pose, pose_name);
        // visual_tools.publishText(current_goal_pose.pose, "Tracking_Goal", rvt::WHITE, rvt::XLARGE);
        // visual_tools.trigger();
        // pose_index++;
    }

    // Publish current tracking goal
    current_goal_pose.header.stamp = node->now();
    target_pose_pub->publish(current_goal_pose);

    loop_rate.sleep();
  }

  // tracker.setControlMode(moveit_servo::PoseTrackingControlMode::CARTESIAN);
  // rrmc_config.redundant_dims.x_rotation = false;
  // rrmc_config.redundant_dims.y_rotation = false;
  // rrmc_config.redundant_dims.z_rotation = false;
  // tracker.setRRMCConfig(rrmc_config);

  // Wait for background grasp evaluation to complete (if still running)
  if (grasp_eval_thread.joinable())
  {
    RCLCPP_INFO(LOGGER, "Waiting for grasp evaluation to complete...");
    grasp_eval_thread.join();
  }

  geometry_msgs::msg::PoseStamped gripper_pose;
  current_pose = move_group.getCurrentPose();

  goal_pose_subscriber.getValidGripperPose(gripper_pose);

  // visualize the target pose in RViz
  // visual_tools.deleteAllMarkers();
  pose_name = "gripper_pose";
  visual_tools.publishAxisLabeled(gripper_pose.pose, pose_name);
  visual_tools.publishText(gripper_pose.pose, "Gripper_Goal", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();

  if (tracking_status != moveit_servo::PoseTrackingStatusCode::INVALID)
  {
    // RCLCPP_INFO_STREAM(LOGGER, "Tracking thread exited, restarting...");
    tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
    
    if (move_to_pose_thread.joinable())
    {
      move_to_pose_thread.join();
    }
    
    // Restart tracker
    move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
      tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
      // RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
      //                                << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
    });
  }

  // Grasping loop
  float grasp_diff_offset = 0.0f;
  while (rclcpp::ok())
  {
    current_pose = move_group.getCurrentPose();
    // get updated gripper pose here: replanned and motion compensated
    goal_pose_subscriber.getValidGripperPose(gripper_pose);

    // bool stop_tracking = goal_pose_subscriber.canStopGraspApproach(config.grasp_pose_tolerance, config.grasp_pose_obj_tolerance, current_grasping_pose, current_pose, gripper_pose, true /* continuous_motion_compensation_enabled */);
    bool stop_tracking = goal_pose_subscriber.canStopGraspApproach(config.grasp_pose_tolerance, current_pose, gripper_pose);
    
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
      
      RCLCPP_INFO_STREAM(LOGGER, "Reached grasping pose, stopping servo tracking...");
      break;
    }
    if (tracking_status != moveit_servo::PoseTrackingStatusCode::INVALID)
    {
      // RCLCPP_INFO_STREAM(LOGGER, "Tracking thread exited, restarting...");
      tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;

      if (move_to_pose_thread.joinable())
      {
        move_to_pose_thread.join();
      }
      // tracker.resetTargetPose();

      // Restart tracker
      move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
        tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
        // RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
        //                               << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
      });
    }
    
    gripper_pose.header.stamp = node->now();
    target_pose_pub->publish(gripper_pose);
    
    loop_rate.sleep();
  }
    
  // {
  //   const auto absolute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
  //       std::chrono::system_clock::now().time_since_epoch())
  //       .count();
  //   submission_reporting.t_human_last_contact_ms = absolute_ms;
  //   submission_reporting.t_robot_first_contact_ms = absolute_ms;
  // }

  // return object to original position
  move_group.setStartStateToCurrentState();

  // get placement goal command in grasp_center frame with tracking-like orientation logic
  current_pose = move_group.getCurrentPose();
  geometry_msgs::msg::PoseStamped obj_goal_pose;
  goal_pose_subscriber.getPlacementObjectPose(obj_goal_pose, current_pose.pose.orientation, true);
  const geometry_msgs::msg::PoseStamped obj_goal_pose_orig = obj_goal_pose;

  // Adjust touchdown tolerance based on object geometry before requesting touchdown detection
  goal_pose_subscriber.adjustTouchdownToleranceForGeometry(config.touchdown_tolerance);

  double upward_bias_z = 0.05 + config.touchdown_tolerance; // 5 cm upward bias for carry stage
  obj_goal_pose.pose.position.z += upward_bias_z;
 
  if (obj_goal_pose.header.frame_id.empty())
    obj_goal_pose.header.frame_id = current_ee_tf.header.frame_id;

  // rrmc_config.alpha = 20.0;
  // rrmc_config.manipulability_w = 0.0;
  // rrmc_config.dir_manipulability_w = 0.6;
  // Eigen::Vector3d p = Eigen::Vector3d{
  //   obj_goal_pose.pose.position.x,
  //   obj_goal_pose.pose.position.y,
  //   obj_goal_pose.pose.position.z,
  // };
  // rrmc_config.p = p;
  // // 5dof nullspace control for placement
  // rrmc_config.redundant_dims.x_rotation = false;
  // rrmc_config.redundant_dims.y_rotation = false;
  // rrmc_config.redundant_dims.z_rotation = true;
  // tracker.setRRMCConfig(rrmc_config);
  // tracker.setControlMode(moveit_servo::PoseTrackingControlMode::JOINT);

  // debug: visualize placement goal
  // visual_tools.publishAxisLabeled(obj_goal_pose.pose, "placement_goal");
  // visual_tools.publishText(obj_goal_pose.pose, "Placement_Goal", rvt::WHITE, rvt::XLARGE);
  // visual_tools.publishAxisLabeled(obj_orig_goal_pose.pose, "original_object_pose");
  // visual_tools.publishText(obj_orig_goal_pose.pose, "Original_Object_Pose", rvt::WHITE, rvt::XLARGE);
  // visual_tools.trigger();

  // send touchdown request
  goal_pose_subscriber.sendTouchdownDetectionRequest();

  moveit_servo::StatusCode servo_status;
  tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;

  tracker.getCommandFrameTransform(current_ee_tf);
  // publish current pose to keep tracker alive
  geometry_msgs::msg::PoseStamped placement_start_pose;
  placement_start_pose.header.frame_id = current_ee_tf.header.frame_id;
  placement_start_pose.pose.position.x = current_ee_tf.transform.translation.x;
  placement_start_pose.pose.position.y = current_ee_tf.transform.translation.y;
  placement_start_pose.pose.position.z = current_ee_tf.transform.translation.z;
  placement_start_pose.pose.orientation = current_ee_tf.transform.rotation;
  target_pose_pub->publish(placement_start_pose);

    // resume tracking thread
  if (tracking_status != moveit_servo::PoseTrackingStatusCode::INVALID)
  {
    // RCLCPP_INFO_STREAM(LOGGER, "Tracking thread exited, restarting...");
    tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
    
    if (move_to_pose_thread.joinable())
    {
      move_to_pose_thread.join();
    }
    
    // Restart tracker
    move_to_pose_thread = std::thread([&tracker, &lin_tol, &rot_tol, &tracking_status]() {
      tracking_status = tracker.moveToPose(lin_tol, rot_tol, 0.1);
      // RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
      //                               << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
    });
  }

  if (enable_placement_pose_interpolation && placement_pose_interpolation_steps > 0)
  {
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
  }
  
  obj_goal_pose.header.stamp = node->now();
  target_pose_pub->publish(obj_goal_pose);

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
  
  // carry phase
  while (rclcpp::ok())
  {
    current_pose = move_group.getCurrentPose();
    double position_diff = sqrt(pow(obj_goal_pose.pose.position.x - current_pose.pose.position.x, 2) +
                                  pow(obj_goal_pose.pose.position.y - current_pose.pose.position.y, 2)); //+
                                  // pow(obj_goal_pose.pose.position.z - current_pose.pose.position.z, 2));
                                  // pow(obj_goal_pose.pose.position.z - current_pose.pose.position.z + downward_bias_z, 2));

    if (position_diff <= config.carry_tolerance)  // if position is within the configured tolerance of the goal, assume we've reached the target pose and can stop tracking
    {
      RCLCPP_INFO_STREAM(LOGGER, "Carried object to original position within " << config.carry_tolerance << " m, stopping servo tracking...");
      break;
    }
    servo_status = status_monitor.getStatus();

    if (servo_status == moveit_servo::StatusCode::HALT_FOR_COLLISION)
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

  // placement stage
  obj_goal_pose = obj_goal_pose_orig; 
  // bias obj a bit downward to enable touchdown detection and improve placement stability
  float downward_bias_z = goal_pose_subscriber.isDelicate() ? -0.0035 : 0.005;
  // RCLCPP_INFO_STREAM(LOGGER, "Applying downward bias of " << -downward_bias_z << " m.");
  obj_goal_pose.pose.position.z = obj_goal_pose.pose.position.z - downward_bias_z; // + grasp_diff_offset;

  visual_tools.publishAxisLabeled(obj_goal_pose.pose, "obj_goal_pose");
  visual_tools.publishText(obj_goal_pose.pose, "Goal_Placement_Pose", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();

  tracker.overrideSceneCollisionThreshold(0.01);

  obj_goal_pose.header.stamp = node->now();
  target_pose_pub->publish(obj_goal_pose);

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
      // RCLCPP_INFO_STREAM(LOGGER, "Pose tracker exited with status: "
      //                               << moveit_servo::POSE_TRACKING_STATUS_CODE_MAP.at(tracking_status));
    });
    tracking_status = moveit_servo::PoseTrackingStatusCode::INVALID;
  }

  while (rclcpp::ok())
  {
    current_pose = move_group.getCurrentPose();
    double position_diff = sqrt(pow(obj_goal_pose.pose.position.x - current_pose.pose.position.x, 2) +
                                  pow(obj_goal_pose.pose.position.y - current_pose.pose.position.y, 2) +
                                  pow(obj_goal_pose.pose.position.z - current_pose.pose.position.z, 2));
    
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

    if (servo_status == moveit_servo::StatusCode::HALT_FOR_COLLISION || servo_status == moveit_servo::StatusCode::DECELERATE_FOR_COLLISION)
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

  if (grasp_replan_thread.joinable())
  {
    grasp_replan_thread.join();
  }
  
  // {
  //   const auto absolute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
  //       std::chrono::system_clock::now().time_since_epoch())
  //       .count();
  //   submission_reporting.t_robot_last_contact_ms = absolute_ms;
  // }

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

  // const bool report_written = writeSubmissionReportJson(submission_report_json, submission_reporting);
  // if (report_written)
  // {
  //   RCLCPP_INFO_STREAM(LOGGER, "Submission report written to: " << submission_report_json);
  // }

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
