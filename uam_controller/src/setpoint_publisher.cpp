#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>

namespace
{
constexpr double kTwoPi = 2.0 * M_PI;
}

// 轻量级参考轨迹发布器，支持悬停/圆形/直线任务
class SetpointPublisher
{
public:
  SetpointPublisher()
  {
    loadParameters();
    pub_ = nh_.advertise<geometry_msgs::PoseStamped>(topic_, 10, true);
    timer_ = nh_.createTimer(ros::Duration(1.0 / publish_rate_), &SetpointPublisher::timerCallback, this);
    start_time_ = ros::Time::now();
  }

  void spin()
  {
    ros::spin();
  }

private:
  // 读取轨迹类型与定时参数
  void loadParameters()
  {
    ros::NodeHandle pnh("~");
    pnh.param("publish_rate", publish_rate_, 20.0);
    pnh.param("start_delay", start_delay_, 1.0);
    pnh.param<std::string>("trajectory", trajectory_, std::string("hover"));
    pnh.param<std::string>("path_frame_id", frame_id_, std::string("world"));
    pnh.param<std::string>("setpoint_topic", topic_, std::string("/quad/position_setpoint"));

    std::vector<double> hover_point_vec;
    if (pnh.getParam("hover_point", hover_point_vec) && hover_point_vec.size() == 3)
    {
      hover_point_ = Eigen::Vector3d(hover_point_vec[0], hover_point_vec[1], hover_point_vec[2]);
    }
    else
    {
      hover_point_ = Eigen::Vector3d(0.0, 0.0, 1.0);
    }

    pnh.param("circle_radius", circle_radius_, 1.0);
    pnh.param("circle_period", circle_period_, 20.0);
    pnh.param("line_speed", line_speed_, 0.5);
    pnh.param("line_length", line_length_, 2.0);
  }

  // 在设定位置原地悬停
  Eigen::Vector3d hoverTrajectory(double /*t*/) const
  {
    return hover_point_;
  }

  // 围绕悬停点做水平圆轨迹
  Eigen::Vector3d circleTrajectory(double t) const
  {
    const double omega = kTwoPi / circle_period_;
    double x = hover_point_.x() + circle_radius_ * std::cos(omega * t);
    double y = hover_point_.y() + circle_radius_ * std::sin(omega * t);
    double z = hover_point_.z();
    return Eigen::Vector3d(x, y, z);
  }

  // 沿水平直线往返运动
  Eigen::Vector3d lineTrajectory(double t) const
  {
    double half_length = 0.5 * line_length_;
    double distance = std::fmod(line_speed_ * t, 2.0 * line_length_);
    double offset = 0.0;
    if (distance <= line_length_)
    {
      offset = distance - half_length;
    }
    else
    {
      offset = (2.0 * line_length_ - distance) - half_length;
    }
    double x = hover_point_.x() + offset;
    return Eigen::Vector3d(x, hover_point_.y(), hover_point_.z());
  }

  // 定时发布期望位姿
  void timerCallback(const ros::TimerEvent &event)
  {
    const ros::Time now = ros::Time::now();
    if ((now - start_time_).toSec() < start_delay_)
    {
      return;
    }

    const double t = (now - start_time_).toSec();
    Eigen::Vector3d pose;
    double yaw = 0.0;

    if (trajectory_ == "circle")
    {
      pose = circleTrajectory(t);
      const double omega = kTwoPi / circle_period_;
      double dx_dt = -circle_radius_ * omega * std::sin(omega * t);
      double dy_dt = circle_radius_ * omega * std::cos(omega * t);
      yaw = std::atan2(dy_dt, dx_dt);
    }
    else if (trajectory_ == "line")
    {
      pose = lineTrajectory(t);
      yaw = 0.0;
    }
    else
    {
      pose = hoverTrajectory(t);
      yaw = 0.0;
    }

    geometry_msgs::PoseStamped msg;
    msg.header.stamp = now;
    msg.header.frame_id = frame_id_;
    msg.pose.position.x = pose.x();
    msg.pose.position.y = pose.y();
    msg.pose.position.z = pose.z();

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    q.normalize();
    msg.pose.orientation.x = q.x();
    msg.pose.orientation.y = q.y();
    msg.pose.orientation.z = q.z();
    msg.pose.orientation.w = q.w();

    pub_.publish(msg);
  }

  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Timer timer_;

  double publish_rate_{20.0};
  double start_delay_{1.0};
  std::string trajectory_{"hover"};
  std::string frame_id_{"world"};
  std::string topic_{"/quad/position_setpoint"};
  Eigen::Vector3d hover_point_{0.0, 0.0, 1.0};
  double circle_radius_{1.0};
  double circle_period_{20.0};
  double line_speed_{0.5};
  double line_length_{2.0};
  ros::Time start_time_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "setpoint_publisher");
  SetpointPublisher node;
  node.spin();
  return 0;
}
