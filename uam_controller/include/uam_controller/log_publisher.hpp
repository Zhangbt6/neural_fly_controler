#pragma once

// 该日志发布器用于按需创建数值/向量话题，便于在 rqt_plot 等工具中调试查看
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Vector3.h>

#include <Eigen/Core>
#include <iostream>
#include <string>
#include <unordered_map>

namespace uam_controller
{

class LogPublisher
{
public:
  explicit LogPublisher(ros::NodeHandle &nh) : nh_(nh) {}

  void publishScalar(const std::string &topic_name, double data)
  {
    auto it = scalar_publishers_.find(topic_name);
    if (it == scalar_publishers_.end())
    {
      ros::Publisher pub = nh_.advertise<std_msgs::Float64>(topic_name, 10);
      scalar_publishers_.emplace(topic_name, pub);
      std::cout << "创建标量调试话题: " << topic_name << std::endl;
    }

    std_msgs::Float64 msg;
    msg.data = data;
    scalar_publishers_[topic_name].publish(msg);
  }

  void publishVector(const std::string &topic_name, const Eigen::Vector3d &vec)
  {
    publishVector(topic_name, vec.x(), vec.y(), vec.z());
  }

  void publishVector(const std::string &topic_name, double x, double y, double z)
  {
    auto it = vector_publishers_.find(topic_name);
    if (it == vector_publishers_.end())
    {
      ros::Publisher pub = nh_.advertise<geometry_msgs::Vector3>(topic_name, 10);
      vector_publishers_.emplace(topic_name, pub);
      std::cout << "创建向量调试话题: " << topic_name << std::endl;
    }

    geometry_msgs::Vector3 msg;
    msg.x = x;
    msg.y = y;
    msg.z = z;
    vector_publishers_[topic_name].publish(msg);
  }

private:
  ros::NodeHandle &nh_;
  std::unordered_map<std::string, ros::Publisher> scalar_publishers_;
  std::unordered_map<std::string, ros::Publisher> vector_publishers_;
};

}  // 命名空间 uam_controller 结束
