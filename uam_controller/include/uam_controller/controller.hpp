#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <string>

#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <Eigen/Dense>

#include <uam_controller/debug_macros.hpp>
#include <uam_controller/log_publisher.hpp>
#include <uam_controller/uam_controllerConfig.h>

namespace uam_controller
{

struct AttitudeGains
{
  double kp_roll{6.0};
  double kp_pitch{6.0};
  double kp_yaw{3.0};
  double ki_roll{0.0};
  double ki_pitch{0.0};
  double ki_yaw{0.2};
  double kd_roll{0.15};
  double kd_pitch{0.15};
  double kd_yaw{0.1};
};

struct PositionGains
{
  double kp_xy{3.0};
  double kp_z{6.0};
  double kd_xy{3.5};
  double kd_z{4.5};
};

struct VehicleState
{
  ros::Time stamp;
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  Eigen::Vector3d filtered_velocity{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};
  tf2::Quaternion orientation{0.0, 0.0, 0.0, 1.0};
  bool valid{false};
};

struct Setpoint
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  bool has_yaw{false};
  ros::Time stamp;
  bool valid{false};
};

class ControllerNode
{
public:
  ControllerNode();
  void spin();

private:
  using ReconfigureServer = dynamic_reconfigure::Server<uam_controller::uam_controllerConfig>;

  void loadParameters();
  void setupSubscriptions();
  void setupMixer();

  void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);
  void setpointCallback(const geometry_msgs::PoseStamped::ConstPtr &msg);
  void disturbanceCallback(const geometry_msgs::Vector3Stamped::ConstPtr &msg);
  void reconfigureCallback(uam_controller::uam_controllerConfig &cfg, uint32_t level);
  void controlLoop(const ros::TimerEvent &event);

  Eigen::Vector3d computePositionControl(double dt, double &desired_yaw);
  void computeOrientationTargets(const Eigen::Vector3d &force_cmd,
                                 double yaw_des,
                                 tf2::Quaternion &q_des_out,
                                 double &thrust_out) const;
  Eigen::Vector4d mixWrenchToThrust(double thrust,
                                    const Eigen::Vector3d &torques) const;
  void publishMotorCommand(const Eigen::Vector4d &thrusts);
  void resetIntegrators();

  bool disturbanceFresh(const ros::Time &now) const;
  double clamp(double value, double min_val, double max_val) const;

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_{"~"};

  ros::Subscriber odom_sub_;
  ros::Subscriber setpoint_sub_;
  ros::Subscriber disturbance_sub_;
  ros::Publisher motor_pub_;

  ros::Timer control_timer_;

  std::unique_ptr<ReconfigureServer> reconfig_server_;

  std::string frame_id_{"world"};
  std::string odom_topic_{"/quad/odometry"};
  std::string setpoint_topic_{"/quad/position_setpoint"};
  std::string disturbance_topic_{"/quad/disturbance_estimate"};
  std::string motor_topic_{"/quad/motor_cmd"};

  double control_rate_hz_{200.0};
  double mass_{2.458};
  double gravity_{9.8066};
  double max_thrust_{50.0};
  double hover_throttle_{0.5};
  double arm_length_{0.225};
  double motor_force_scale_{12.5};
  double motor_constant_{6.450e-6};
  double moment_constant_{1.056e-7};
  Eigen::Vector3d inertia_diag_{0.01860, 0.01860, 0.03215};
  double motor_time_const_up_{0.0125};
  double motor_time_const_down_{0.025};
  std::array<double, 4> rotor_arm_length_{{0.225, 0.225, 0.225, 0.225}};
  std::array<double, 4> rotor_directions_{{1.0, -1.0, -1.0, 1.0}};   // 顺序: 前左, 前右, 后左, 后右
  std::array<double, 4> rotor_angles_{{0.7853981633974483, -0.7853981633974483, 2.356194490192345, -2.356194490192345}};
  std::array<double, 4> rotor_kappa_{{moment_constant_ / motor_constant_,
                                      moment_constant_ / motor_constant_,
                                      moment_constant_ / motor_constant_,
                                      moment_constant_ / motor_constant_}};
  std::array<Eigen::Vector2d, 4> rotor_xy_{{
      Eigen::Vector2d(arm_length_ / std::sqrt(2.0), arm_length_ / std::sqrt(2.0)),
      Eigen::Vector2d(arm_length_ / std::sqrt(2.0), -arm_length_ / std::sqrt(2.0)),
      Eigen::Vector2d(-arm_length_ / std::sqrt(2.0), arm_length_ / std::sqrt(2.0)),
      Eigen::Vector2d(-arm_length_ / std::sqrt(2.0), -arm_length_ / std::sqrt(2.0))}};
  std::array<int, 4> motor_output_order_{{0, 1, 2, 3}};  // motor1=前左, motor2=前右, motor3=后左, motor4=后右
  double velocity_filter_alpha_{0.35};
  double integral_limit_{0.2};
  double disturbance_timeout_{0.2};
  double motor_min_cmd_{0.0};
  double motor_max_cmd_{0.95};
  double max_tilt_deg_{35.0};
  double max_ascent_rate_{2.0};
  double max_descent_rate_{1.0};
  bool use_yaw_rate_stabilizer_{true};

  PositionGains pos_gains_;
  AttitudeGains att_gains_;

  VehicleState state_;
  Setpoint setpoint_;
  Eigen::Vector3d disturbance_force_{Eigen::Vector3d::Zero()};  // 最近一次扰动估计（机体系）
  ros::Time disturbance_stamp_;

  Eigen::Vector3d attitude_integral_{Eigen::Vector3d::Zero()};

  double target_yaw_{0.0};
  bool enable_takeoff_{false};
  bool enable_feedforward_{true};

  Eigen::Matrix4d mixer_matrix_;

  ros::Time last_control_time_;

#if UAM_DEBUG_LEVEL >= UAM_DEBUG_LEVEL_FLOW
  std::unique_ptr<LogPublisher> log_pub_;
#endif
};

}  // 命名空间 uam_controller 结束
