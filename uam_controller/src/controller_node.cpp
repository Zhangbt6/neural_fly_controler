#include "uam_controller/controller.hpp"

#include <algorithm>
#include <cmath>

#include <XmlRpcValue.h>
#include <boost/bind/bind.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <Eigen/QR>

using namespace boost::placeholders;

namespace uam_controller
{
namespace
{
constexpr double kDegToRad = M_PI / 180.0;
}

ControllerNode::ControllerNode()
{
  loadParameters();
  setupMixer();
  setupSubscriptions();

#if UAM_DEBUG_LEVEL >= UAM_DEBUG_LEVEL_DETAIL
  log_pub_ = std::make_unique<LogPublisher>(nh_);
#endif

  motor_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(motor_topic_, 10);

  reconfig_server_ = std::unique_ptr<ReconfigureServer>(new ReconfigureServer(pnh_));
  ReconfigureServer::CallbackType cb = boost::bind(&ControllerNode::reconfigureCallback, this, _1, _2);
  reconfig_server_->setCallback(cb);

  control_timer_ = nh_.createTimer(ros::Duration(1.0 / control_rate_hz_), &ControllerNode::controlLoop, this);

  UAM_DEBUG_FLOW("控制器初始化完成，已订阅里程计话题: " << odom_topic_);
}

void ControllerNode::spin()
{
  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
}

void ControllerNode::loadParameters()
{
  pnh_.param<std::string>("frame_id", frame_id_, frame_id_);
  pnh_.param<std::string>("odom_topic", odom_topic_, odom_topic_);
  pnh_.param<std::string>("setpoint_topic", setpoint_topic_, setpoint_topic_);
  pnh_.param<std::string>("disturbance_topic", disturbance_topic_, disturbance_topic_);
  pnh_.param<std::string>("motor_cmd_topic", motor_topic_, motor_topic_);

  pnh_.param("control_rate", control_rate_hz_, control_rate_hz_);
  pnh_.param("mass", mass_, mass_);
  pnh_.param("gravity", gravity_, gravity_);
  pnh_.param("max_thrust", max_thrust_, max_thrust_);
  pnh_.param("hover_throttle", hover_throttle_, hover_throttle_);
  pnh_.param("arm_length", arm_length_, arm_length_);
  pnh_.param("motor_force_scale", motor_force_scale_, motor_force_scale_);
  pnh_.param("motor_constant", motor_constant_, motor_constant_);
  pnh_.param("moment_constant", moment_constant_, moment_constant_);
  pnh_.param("motor_time_constant_up", motor_time_const_up_, motor_time_const_up_);
  pnh_.param("motor_time_constant_down", motor_time_const_down_, motor_time_const_down_);
  pnh_.param("inertia_xx", inertia_diag_[0], inertia_diag_[0]);
  pnh_.param("inertia_yy", inertia_diag_[1], inertia_diag_[1]);
  pnh_.param("inertia_zz", inertia_diag_[2], inertia_diag_[2]);
  pnh_.param("velocity_filter_alpha", velocity_filter_alpha_, velocity_filter_alpha_);
  pnh_.param("integral_limit", integral_limit_, integral_limit_);
  pnh_.param("feedforward_timeout", disturbance_timeout_, disturbance_timeout_);
  pnh_.param("motor_min_cmd", motor_min_cmd_, motor_min_cmd_);
  pnh_.param("motor_max_cmd", motor_max_cmd_, motor_max_cmd_);
  pnh_.param("max_tilt_deg", max_tilt_deg_, max_tilt_deg_);
  pnh_.param("max_ascent_rate", max_ascent_rate_, max_ascent_rate_);
  pnh_.param("max_descent_rate", max_descent_rate_, max_descent_rate_);
  pnh_.param("use_yaw_rate_stabilizer", use_yaw_rate_stabilizer_, use_yaw_rate_stabilizer_);

  double kp_xy = pos_gains_.kp_xy;
  double kp_z = pos_gains_.kp_z;
  double kd_xy = pos_gains_.kd_xy;
  double kd_z = pos_gains_.kd_z;
  pnh_.param("position_kp_xy", kp_xy, kp_xy);
  pnh_.param("position_kp_z", kp_z, kp_z);
  pnh_.param("velocity_kd_xy", kd_xy, kd_xy);
  pnh_.param("velocity_kd_z", kd_z, kd_z);
  pos_gains_.kp_xy = kp_xy;
  pos_gains_.kp_z = kp_z;
  pos_gains_.kd_xy = kd_xy;
  pos_gains_.kd_z = kd_z;

  double kp_roll = att_gains_.kp_roll;
  double kp_pitch = att_gains_.kp_pitch;
  double kp_yaw = att_gains_.kp_yaw;
  double ki_roll = att_gains_.ki_roll;
  double ki_pitch = att_gains_.ki_pitch;
  double ki_yaw = att_gains_.ki_yaw;
  double kd_roll = att_gains_.kd_roll;
  double kd_pitch = att_gains_.kd_pitch;
  double kd_yaw = att_gains_.kd_yaw;
  pnh_.param("attitude_kp_roll", kp_roll, kp_roll);
  pnh_.param("attitude_kp_pitch", kp_pitch, kp_pitch);
  pnh_.param("attitude_kp_yaw", kp_yaw, kp_yaw);
  pnh_.param("attitude_ki_roll", ki_roll, ki_roll);
  pnh_.param("attitude_ki_pitch", ki_pitch, ki_pitch);
  pnh_.param("attitude_ki_yaw", ki_yaw, ki_yaw);
  pnh_.param("attitude_kd_roll", kd_roll, kd_roll);
  pnh_.param("attitude_kd_pitch", kd_pitch, kd_pitch);
  pnh_.param("attitude_kd_yaw", kd_yaw, kd_yaw);
  att_gains_.kp_roll = kp_roll;
  att_gains_.kp_pitch = kp_pitch;
  att_gains_.kp_yaw = kp_yaw;
  att_gains_.ki_roll = ki_roll;
  att_gains_.ki_pitch = ki_pitch;
  att_gains_.ki_yaw = ki_yaw;
  att_gains_.kd_roll = kd_roll;
  att_gains_.kd_pitch = kd_pitch;
  att_gains_.kd_yaw = kd_yaw;

  // 默认按照四旋翼 X 型布置初始化电机参数
  const double default_kappa = (motor_constant_ > 1e-9) ? (moment_constant_ / motor_constant_) : 0.0;
  const double diag = arm_length_ / std::sqrt(2.0);
  rotor_arm_length_.fill(arm_length_);
  rotor_directions_ = {-1.0, 1.0, 1.0, -1.0};  // {前左, 前右, 后左, 后右}
  rotor_angles_ = {M_PI_4, -M_PI_4, 3 * M_PI_4, -3 * M_PI_4};
  rotor_kappa_.fill(default_kappa);
  rotor_xy_ = {Eigen::Vector2d(diag, diag), Eigen::Vector2d(diag, -diag),
               Eigen::Vector2d(-diag, diag), Eigen::Vector2d(-diag, -diag)};

  // 私有参数可覆写旋翼转向
  XmlRpc::XmlRpcValue rotor_dirs;
  if (pnh_.getParam("rotor_directions", rotor_dirs) && rotor_dirs.getType() == XmlRpc::XmlRpcValue::TypeArray)
  {
    if (rotor_dirs.size() == 4)
    {
      for (int i = 0; i < 4; ++i)
      {
        if (rotor_dirs[i].getType() == XmlRpc::XmlRpcValue::TypeDouble)
        {
          rotor_directions_[i] = static_cast<double>(rotor_dirs[i]);
        }
        else if (rotor_dirs[i].getType() == XmlRpc::XmlRpcValue::TypeInt)
        {
          rotor_directions_[i] = static_cast<int>(rotor_dirs[i]);
        }
      }
    }
  }

  auto quadrantIndex = [](double x, double y) -> int {
    const double tol = 1e-6;
    if (x >= -tol && y >= -tol)
    {
      return 0;  // 前左
    }
    if (x >= -tol && y <= tol)
    {
      return 1;  // 前右
    }
    if (x <= tol && y >= -tol)
    {
      return 2;  // 后左
    }
    return 3;      // 后右
  };

  auto assignRotor = [&](int idx, double angle, double arm_len, double direction,
                         double force_const, double moment_const) {
    rotor_angles_[idx] = angle;
    rotor_arm_length_[idx] = arm_len;
    rotor_directions_[idx] = direction;
    const double fx = arm_len * std::cos(angle);
    const double fy = arm_len * std::sin(angle);
    rotor_xy_[idx] = Eigen::Vector2d(fx, fy);
    if (force_const > 1e-9)
    {
      rotor_kappa_[idx] = moment_const / force_const;
    }
    else
    {
      rotor_kappa_[idx] = default_kappa;
    }
  };

  // 尝试从全局配置中加载电机参数
  ros::NodeHandle nh_global;
  XmlRpc::XmlRpcValue rotor_config;
  if (nh_global.getParam("/uam_config/rotor_configuration", rotor_config) &&
      rotor_config.getType() == XmlRpc::XmlRpcValue::TypeStruct)
  {
    for (int i = 0; i < 4; ++i)
    {
      const std::string key = std::to_string(i);
      if (!rotor_config.hasMember(key))
      {
        continue;
      }
      XmlRpc::XmlRpcValue &entry = rotor_config[key];
      if (entry.getType() != XmlRpc::XmlRpcValue::TypeStruct)
      {
        continue;
      }
      if (!entry.hasMember("angle"))
      {
        continue;
      }

      double angle = 0.0;
      if (entry["angle"].getType() == XmlRpc::XmlRpcValue::TypeDouble)
      {
        angle = static_cast<double>(entry["angle"]);
      }
      else if (entry["angle"].getType() == XmlRpc::XmlRpcValue::TypeInt)
      {
        angle = static_cast<int>(entry["angle"]);
      }
      else
      {
        continue;
      }

      double arm_len = arm_length_;
      double force_const = motor_constant_;
      double moment_const = moment_constant_;

      if (entry.hasMember("arm_length"))
      {
        if (entry["arm_length"].getType() == XmlRpc::XmlRpcValue::TypeDouble)
        {
          arm_len = static_cast<double>(entry["arm_length"]);
        }
        else if (entry["arm_length"].getType() == XmlRpc::XmlRpcValue::TypeInt)
        {
          arm_len = static_cast<int>(entry["arm_length"]);
        }
      }

      if (entry.hasMember("rotor_force_constant"))
      {
        if (entry["rotor_force_constant"].getType() == XmlRpc::XmlRpcValue::TypeDouble)
        {
          force_const = static_cast<double>(entry["rotor_force_constant"]);
        }
        else if (entry["rotor_force_constant"].getType() == XmlRpc::XmlRpcValue::TypeInt)
        {
          force_const = static_cast<int>(entry["rotor_force_constant"]);
        }
      }

      if (entry.hasMember("rotor_moment_constant"))
      {
        if (entry["rotor_moment_constant"].getType() == XmlRpc::XmlRpcValue::TypeDouble)
        {
          moment_const = static_cast<double>(entry["rotor_moment_constant"]);
        }
        else if (entry["rotor_moment_constant"].getType() == XmlRpc::XmlRpcValue::TypeInt)
        {
          moment_const = static_cast<int>(entry["rotor_moment_constant"]);
        }
      }

      const double x = arm_len * std::cos(angle);
      const double y = arm_len * std::sin(angle);
      const int idx = quadrantIndex(x, y);
      double direction = rotor_directions_[idx];
      if (entry.hasMember("direction"))
      {
        if (entry["direction"].getType() == XmlRpc::XmlRpcValue::TypeDouble)
        {
          direction = static_cast<double>(entry["direction"]);
        }
        else if (entry["direction"].getType() == XmlRpc::XmlRpcValue::TypeInt)
        {
          direction = static_cast<int>(entry["direction"]);
        }
      }
      assignRotor(idx, angle, arm_len, direction, force_const, moment_const);
    }
  }

  double inertia_xx = inertia_diag_[0];
  double inertia_yy = inertia_diag_[1];
  double inertia_zz = inertia_diag_[2];
  nh_global.param("/uam_config/inertia_quad_body/xx", inertia_xx, inertia_xx);
  nh_global.param("/uam_config/inertia_quad_body/yy", inertia_yy, inertia_yy);
  nh_global.param("/uam_config/inertia_quad_body/zz", inertia_zz, inertia_zz);
  inertia_diag_[0] = inertia_xx;
  inertia_diag_[1] = inertia_yy;
  inertia_diag_[2] = inertia_zz;
}

void ControllerNode::setupMixer()
{
  Eigen::Matrix4d allocation;
  allocation.setZero();
  for (int i = 0; i < 4; ++i)
  {
    const double x = rotor_xy_[i].x();
    const double y = rotor_xy_[i].y();
    allocation(0, i) = 1.0;                          // 贡献到总推力
    allocation(1, i) = y;                           // 贡献到滚转力矩 (τx = -y*T)
    allocation(2, i) = -x;                            // 贡献到俯仰力矩 (τy = x*T)
    allocation(3, i) = rotor_directions_[i] * rotor_kappa_[i];  // 贡献到偏航力矩
  }

  const double determinant = allocation.determinant();
  if (std::abs(determinant) < 1e-6)
  {
    Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix4d> cod(allocation);
    mixer_matrix_ = cod.pseudoInverse();
    ROS_WARN("电机分配矩阵逼近奇异，已使用伪逆进行混合。");
  }
  else
  {
    mixer_matrix_ = allocation.inverse();
  }
}

void ControllerNode::setupSubscriptions()
{
  odom_sub_ = nh_.subscribe(odom_topic_, 10, &ControllerNode::odomCallback, this);
  setpoint_sub_ = nh_.subscribe(setpoint_topic_, 10, &ControllerNode::setpointCallback, this);
  disturbance_sub_ = nh_.subscribe(disturbance_topic_, 10, &ControllerNode::disturbanceCallback, this);
}

void ControllerNode::odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
  state_.stamp = msg->header.stamp;
  state_.position = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
  state_.velocity = Eigen::Vector3d(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
  state_.angular_velocity = Eigen::Vector3d(msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z);
  tf2::fromMsg(msg->pose.pose.orientation, state_.orientation);

  state_.filtered_velocity = velocity_filter_alpha_ * state_.velocity + (1.0 - velocity_filter_alpha_) * state_.filtered_velocity;
  state_.valid = true;

  UAM_DEBUG_DETAIL("里程计位置: " << state_.position.transpose() << " 速度: " << state_.velocity.transpose());
  UAM_DEBUG_PUBLISH_VECTOR(log_pub_.get(), "/uam_controller/debug/position", state_.position);
  UAM_DEBUG_PUBLISH_VECTOR(log_pub_.get(), "/uam_controller/debug/velocity", state_.velocity);
  UAM_DEBUG_PUBLISH_VECTOR(log_pub_.get(), "/uam_controller/debug/angular_velocity", state_.angular_velocity);
}

void ControllerNode::setpointCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
  setpoint_.position = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  setpoint_.velocity.setZero();
  setpoint_.stamp = msg->header.stamp;

  tf2::Quaternion q;
  tf2::fromMsg(msg->pose.orientation, q);
  if (std::abs(q.length2() - 1.0) < 1e-3)
  {
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    setpoint_.yaw = yaw;
    setpoint_.has_yaw = true;
  }
  else
  {
    setpoint_.has_yaw = false;
  }
  setpoint_.valid = true;

  UAM_DEBUG_DETAIL("收到期望位置: " << setpoint_.position.transpose());
  UAM_DEBUG_PUBLISH_VECTOR(log_pub_.get(), "/uam_controller/debug/target_position", setpoint_.position);
}

void ControllerNode::disturbanceCallback(const geometry_msgs::Vector3Stamped::ConstPtr &msg)
{
  disturbance_force_ = Eigen::Vector3d(msg->vector.x, msg->vector.y, msg->vector.z);
  disturbance_stamp_ = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;

  UAM_DEBUG_DETAIL("更新扰动估计: " << disturbance_force_.transpose());
  UAM_DEBUG_PUBLISH_VECTOR(log_pub_.get(), "/uam_controller/debug/disturbance", disturbance_force_);
}

void ControllerNode::reconfigureCallback(uam_controller::uam_controllerConfig &cfg, uint32_t /*level*/)
{
  enable_takeoff_ = cfg.enable_takeoff;
  enable_feedforward_ = cfg.enable_feedforward;
  hover_throttle_ = cfg.hover_throttle;
  target_yaw_ = cfg.target_yaw_deg * kDegToRad;
  max_tilt_deg_ = cfg.max_tilt_deg;
  motor_min_cmd_ = cfg.motor_min_cmd;
  motor_max_cmd_ = cfg.motor_max_cmd;
  disturbance_timeout_ = cfg.disturbance_timeout;
  max_ascent_rate_ = cfg.max_ascent_rate;
  max_descent_rate_ = cfg.max_descent_rate;
  use_yaw_rate_stabilizer_ = cfg.use_yaw_rate_stabilizer;

  pos_gains_.kp_xy = cfg.position_kp_xy;
  pos_gains_.kp_z = cfg.position_kp_z;
  pos_gains_.kd_xy = cfg.velocity_kd_xy;
  pos_gains_.kd_z = cfg.velocity_kd_z;

  att_gains_.kp_roll = cfg.attitude_kp_roll;
  att_gains_.kp_pitch = cfg.attitude_kp_pitch;
  att_gains_.kp_yaw = cfg.attitude_kp_yaw;
  att_gains_.ki_roll = cfg.attitude_ki_roll;
  att_gains_.ki_pitch = cfg.attitude_ki_pitch;
  att_gains_.ki_yaw = cfg.attitude_ki_yaw;
  att_gains_.kd_roll = cfg.attitude_kd_roll;
  att_gains_.kd_pitch = cfg.attitude_kd_pitch;
  att_gains_.kd_yaw = cfg.attitude_kd_yaw;

  resetIntegrators();
}

void ControllerNode::controlLoop(const ros::TimerEvent &event)
{
  if (!state_.valid)
  {
    return;
  }

  const ros::Time now = ros::Time::now();
  double dt = 1.0 / control_rate_hz_;
  if (!last_control_time_.isZero())
  {
    dt = std::max(1e-4, (now - last_control_time_).toSec());
  }
  last_control_time_ = now;

  static bool takeoff_notice = false;

  if (!enable_takeoff_)
  {
    publishMotorCommand(Eigen::Vector4d::Zero());
    resetIntegrators();
    if (!takeoff_notice)
    {
      UAM_DEBUG_FLOW("控制未使能，输出零油门等待起飞指令。");
      takeoff_notice = true;
    }
    return;
  }

  if (takeoff_notice)
  {
    UAM_DEBUG_FLOW("控制器已使能，开始闭环控制。");
    takeoff_notice = false;
  }

  double desired_yaw = setpoint_.has_yaw ? setpoint_.yaw : target_yaw_;
  Eigen::Vector3d accel_cmd = computePositionControl(dt, desired_yaw);

  Eigen::Vector3d disturbance = Eigen::Vector3d::Zero();
  if (enable_feedforward_ && disturbanceFresh(now))
  {
    disturbance = disturbance_force_;
    UAM_DEBUG_DETAIL("应用扰动补偿: " << disturbance.transpose());
  }

  Eigen::Vector3d force_cmd = mass_ * accel_cmd;
  force_cmd += Eigen::Vector3d(0.0, 0.0, mass_ * gravity_);
  force_cmd -= disturbance;

  UAM_DEBUG_DETAIL("合力指令 (含重力/扰动补偿): " << force_cmd.transpose());

  tf2::Quaternion q_des;
  double thrust_cmd = 0.0;
  computeOrientationTargets(force_cmd, desired_yaw, q_des, thrust_cmd);

  tf2::Quaternion q_current = state_.orientation;
  tf2::Quaternion q_error = q_current.inverse() * q_des; 
  if (q_error.w() < 0.0)
  {
    q_error = tf2::Quaternion(-q_error.x(), -q_error.y(), -q_error.z(), -q_error.w());
  }

  tf2::Vector3 attitude_error(2.0 * q_error.x(), 2.0 * q_error.y(), 2.0 * q_error.z());
  Eigen::Vector3d attitude_error_e(attitude_error.x(), attitude_error.y(), attitude_error.z());

  attitude_integral_ += attitude_error_e * dt;
  for (int i = 0; i < 3; ++i)
  {
    attitude_integral_[i] = clamp(attitude_integral_[i], -integral_limit_, integral_limit_);
  }

  Eigen::Vector3d angular_rate(state_.angular_velocity.x(), state_.angular_velocity.y(), state_.angular_velocity.z());

  Eigen::Vector3d ang_acc_cmd;
  ang_acc_cmd.x() = att_gains_.kp_roll * attitude_error_e.x() + att_gains_.kd_roll * (-angular_rate.x()) + att_gains_.ki_roll * attitude_integral_.x();
  ang_acc_cmd.y() = att_gains_.kp_pitch * attitude_error_e.y() + att_gains_.kd_pitch * (-angular_rate.y()) + att_gains_.ki_pitch * attitude_integral_.y();
  double yaw_rate_error = use_yaw_rate_stabilizer_ ? (-angular_rate.z()) : 0.0;
  ang_acc_cmd.z() = att_gains_.kp_yaw * attitude_error_e.z() + att_gains_.kd_yaw * yaw_rate_error + att_gains_.ki_yaw * attitude_integral_.z();

  Eigen::Vector3d torques = inertia_diag_.cwiseProduct(ang_acc_cmd);
  UAM_DEBUG_DETAIL("姿态误差: " << attitude_error_e.transpose()
                   << " 角加速度指令: " << ang_acc_cmd.transpose()
                   << " 扭矩指令: " << torques.transpose());
  UAM_DEBUG_PUBLISH_VECTOR(log_pub_.get(), "/uam_controller/debug/attitude_error", attitude_error_e);
  UAM_DEBUG_PUBLISH_VECTOR(log_pub_.get(), "/uam_controller/debug/ang_acc_cmd", ang_acc_cmd);
  UAM_DEBUG_PUBLISH_VECTOR(log_pub_.get(), "/uam_controller/debug/torque_cmd", torques);

   //torques.x() = 0;
   //torques.y() = 0;
   //torques.z() = 0;

  torques.x() = 10*torques.x();
  torques.y() = 10*torques.y();
  torques.z() = 10*torques.z();
  
  Eigen::Vector4d rotor_forces = mixWrenchToThrust(thrust_cmd, torques);
  UAM_DEBUG_DETAIL("混合后的推力向量: " << rotor_forces.transpose());
  publishMotorCommand(rotor_forces);
}

Eigen::Vector3d ControllerNode::computePositionControl(double dt, double &desired_yaw)
{
  (void)dt;
  Eigen::Vector3d position_error = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity_error = -state_.filtered_velocity;

  if (setpoint_.valid)
  {
    position_error = setpoint_.position - state_.position;
  }

  Eigen::Vector2d vel_cmd_xy = pos_gains_.kp_xy * position_error.head<2>();
  Eigen::Vector2d vel_error_xy = vel_cmd_xy - state_.filtered_velocity.head<2>();

  Eigen::Vector3d accel_cmd = Eigen::Vector3d::Zero();
  accel_cmd.x() = pos_gains_.kd_xy * vel_error_xy.x();
  accel_cmd.y() = pos_gains_.kd_xy * vel_error_xy.y();

  double vel_cmd_z = pos_gains_.kp_z * position_error.z();
  vel_cmd_z = clamp(vel_cmd_z, -max_descent_rate_, max_ascent_rate_);
  double vel_error_z = vel_cmd_z - state_.filtered_velocity.z();
  accel_cmd.z() = pos_gains_.kd_z * vel_error_z;

  double max_tilt_rad = max_tilt_deg_ * kDegToRad;
  double max_lateral_accel = gravity_ * std::tan(max_tilt_rad);
  double accel_xy_norm = std::hypot(accel_cmd.x(), accel_cmd.y());
  if (accel_xy_norm > max_lateral_accel)
  {
    double scale = max_lateral_accel / accel_xy_norm;
    accel_cmd.x() *= scale;
    accel_cmd.y() *= scale;
  }

  if (!std::isfinite(desired_yaw))
  {
    desired_yaw = target_yaw_;
  }

  UAM_DEBUG_DETAIL("位置误差: " << position_error.transpose() << " 加速度指令: " << accel_cmd.transpose());
  UAM_DEBUG_PUBLISH_VECTOR(log_pub_.get(), "/uam_controller/debug/position_error", position_error);
  UAM_DEBUG_PUBLISH_VECTOR(log_pub_.get(), "/uam_controller/debug/accel_cmd", accel_cmd);

  return accel_cmd;
}

void ControllerNode::computeOrientationTargets(const Eigen::Vector3d &force_cmd,
                                                double yaw_des,
                                                tf2::Quaternion &q_des_out,
                                                double &thrust_out) const
{
  Eigen::Vector3d zb_des = force_cmd;
  const double min_force = 1e-3;
  double force_norm = zb_des.norm();

  if (force_norm < min_force)
  {
    zb_des = Eigen::Vector3d(0.0, 0.0, 1.0);
    force_norm = min_force;
  }
  else
  {
    zb_des.normalize();
  }

  Eigen::Vector3d xc(std::cos(yaw_des), std::sin(yaw_des), 0.0);
  Eigen::Vector3d yb_des = zb_des.cross(xc);
  if (yb_des.norm() < 1e-4)
  {
    xc = Eigen::Vector3d(1.0, 0.0, 0.0);
    yb_des = zb_des.cross(xc);
  }
  yb_des.normalize();
  Eigen::Vector3d xb_des = yb_des.cross(zb_des);
  xb_des.normalize();

  tf2::Matrix3x3 R_des(xb_des.x(), yb_des.x(), zb_des.x(),
                       xb_des.y(), yb_des.y(), zb_des.y(),
                       xb_des.z(), yb_des.z(), zb_des.z());

  R_des.getRotation(q_des_out);
  q_des_out.normalize();

  thrust_out = clamp(force_norm, 0.0, max_thrust_);
}

Eigen::Vector4d ControllerNode::mixWrenchToThrust(double thrust,
                                                   const Eigen::Vector3d &torques) const
{
  Eigen::Vector4d wrench;
  wrench << thrust, torques.x(), torques.y(), torques.z();
  Eigen::Vector4d rotor_forces = mixer_matrix_ * wrench;
  for (int i = 0; i < 4; ++i)
  {
    if (!std::isfinite(rotor_forces[i]) || rotor_forces[i] < 0.0)
    {
      rotor_forces[i] = 0.0;
    }
  }
  return rotor_forces;
}

void ControllerNode::publishMotorCommand(const Eigen::Vector4d &forces)
{
  std_msgs::Float64MultiArray cmd;
  cmd.data.resize(4);
  // forces 的顺序为: [前左, 前右, 后左, 后右]，直接按照 motor1~4 对应输出
  for (int i = 0; i < 4; ++i)
  {
    const int rotor_index = motor_output_order_[i];
    double throttle = forces[rotor_index] / motor_force_scale_;
    throttle = clamp(throttle, motor_min_cmd_, motor_max_cmd_);
    if (!std::isfinite(throttle))
    {
      throttle = motor_min_cmd_;
    }
    cmd.data[i] = throttle;
    UAM_DEBUG_PUBLISH_SCALAR(log_pub_.get(), "/uam_controller/debug/throttle_" + std::to_string(i), throttle);
  }
  motor_pub_.publish(cmd);
}

void ControllerNode::resetIntegrators()
{
  attitude_integral_.setZero();
}

bool ControllerNode::disturbanceFresh(const ros::Time &now) const
{
  if (disturbance_stamp_.isZero())
  {
    return false;
  }
  return (now - disturbance_stamp_).toSec() < disturbance_timeout_;
}

double ControllerNode::clamp(double value, double min_val, double max_val) const
{
  return std::max(min_val, std::min(max_val, value));
}

}  // 命名空间 uam_controller 结束

int main(int argc, char **argv)
{
  ros::init(argc, argv, "uam_controller_node");
  uam_controller::ControllerNode node;
  node.spin();
  return 0;
}
