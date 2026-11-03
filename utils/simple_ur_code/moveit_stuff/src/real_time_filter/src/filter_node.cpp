#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cmath>
#include <algorithm>

#include "OneEuroFilter.h"

using namespace std::chrono_literals;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("filter_node");

struct FilterConfig
{
  double frequency = 120.0; // Hz
  double mincutoff = 0.2;   // Hz (Increase to reduce lag at low speeds)
  double beta = 0.4;        // Speed coefficient (Increase to reduce lag at high speeds)
  double dcutoff = 1.0;     // Cutoff frequency for the derivative

  // Outlier rejection parameters
  double max_linear_speed = 1.0;  // m/s
  int max_consecutive_outliers = 5; // Accept jump if it persists
};

class RealTimeFilter : public rclcpp::Node
{
  public:
    RealTimeFilter(
      const FilterConfig& config,
      const std::string& raw_goal_topic = "goal_pose",
      const std::string& filtered_goal_topic = "goal_pose_filtered")
    : Node("real_time_filter_node"), 
      config_(config),
      filter_x_(config.frequency, config.mincutoff, config.beta, config.dcutoff), 
      filter_y_(config.frequency, config.mincutoff, config.beta, config.dcutoff), 
      filter_z_(config.frequency, config.mincutoff, config.beta, config.dcutoff)
    {
        publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(filtered_goal_topic, 10);
        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
          raw_goal_topic, rclcpp::SystemDefaultsQoS(),
          [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg) { return raw_pose_callback(msg); });
    }

  private:
    void raw_pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg)
    {
      double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
      geometry_msgs::msg::Pose target_pose = msg->pose;

      if (!has_first_pose_)
      {
        last_valid_pose_ = msg->pose;
        last_timestamp_ = timestamp;
        has_first_pose_ = true;
      }
      else
      {
        double dt = timestamp - last_timestamp_;
        if (dt > 0.0)
        {
          double dx = msg->pose.position.x - last_valid_pose_.position.x;
          double dy = msg->pose.position.y - last_valid_pose_.position.y;
          double dz = msg->pose.position.z - last_valid_pose_.position.z;
          double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

          if (distance / dt > config_.max_linear_speed)
          {
            outlier_count_++;
            if (outlier_count_ < config_.max_consecutive_outliers)
            {
              RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Outlier jump detected. Rejecting pose.");
              target_pose = last_valid_pose_; // Feed last good pose to keep filter stable
            }
            else
            {
              // Jump has persisted long enough, accept it as the new true pose
              RCLCPP_INFO(this->get_logger(), "Jump persisted. Accepting new pose.");
              outlier_count_ = 0;
              last_valid_pose_ = target_pose;
            }
          }
          else
          {
            outlier_count_ = 0;
            last_valid_pose_ = target_pose;
          }
        }
        last_timestamp_ = timestamp;
      }

      double filtered_x = filter_x_.filter(target_pose.position.x, timestamp);
      double filtered_y = filter_y_.filter(target_pose.position.y, timestamp);
      double filtered_z = filter_z_.filter(target_pose.position.z, timestamp);
      
      auto filtered_msg = *msg;
      filtered_msg.pose.position.x = filtered_x;
      filtered_msg.pose.position.y = filtered_y;
      filtered_msg.pose.position.z = filtered_z;
      // Use the accepted orientation to avoid sending jumping orientations
      filtered_msg.pose.orientation = target_pose.orientation;
      
      publisher_->publish(filtered_msg);
    }

    FilterConfig config_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr publisher_;
    OneEuroFilter filter_x_;
    OneEuroFilter filter_y_;
    OneEuroFilter filter_z_;

    bool has_first_pose_ = false;
    geometry_msgs::msg::Pose last_valid_pose_;
    double last_timestamp_ = 0.0;
    int outlier_count_ = 0;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RealTimeFilter>(FilterConfig{}));
  rclcpp::shutdown();
  return 0;
}