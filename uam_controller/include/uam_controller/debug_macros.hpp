#pragma once

// 需要 ROS 时间工具
#include <ros/ros.h>

#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

// 调试等级定义
#define UAM_DEBUG_LEVEL_NONE 0
#define UAM_DEBUG_LEVEL_FLOW 1
#define UAM_DEBUG_LEVEL_DETAIL 2

// catkin build uam_controller --cmake-args -DUAM_DEBUG_LEVEL=2
#define UAM_DEBUG_LEVEL UAM_DEBUG_LEVEL_DETAIL
#ifndef UAM_DEBUG_LEVEL
#define UAM_DEBUG_LEVEL UAM_DEBUG_LEVEL_NONE
#endif

#ifndef UAM_DEBUG_FLOW_THROTTLE_SEC
#define UAM_DEBUG_FLOW_THROTTLE_SEC 1.0
#endif

#ifndef UAM_DEBUG_DETAIL_THROTTLE_SEC
#define UAM_DEBUG_DETAIL_THROTTLE_SEC 1.0
#endif

namespace uam_controller
{
namespace debug
{
inline std::string makeKey(const char *file, int line, const char *tag)
{
  std::ostringstream oss;
  oss << tag << ':' << file << ':' << line;
  return oss.str();
}

inline bool shouldLog(const std::string &key, double throttle_sec)
{
  static std::unordered_map<std::string, ros::Time> last_emit;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  const ros::Time now = ros::Time::now();
  ros::Time &stamp = last_emit[key];
  if (stamp.isZero() || (now - stamp).toSec() >= throttle_sec)
  {
    stamp = now;
    return true;
  }
  return false;
}

inline void printLine(const std::string &text)
{
  std::cout << text << std::endl;
}
}  // namespace debug
}  // namespace uam_controller

#if UAM_DEBUG_LEVEL >= UAM_DEBUG_LEVEL_FLOW
#define UAM_DEBUG_FLOW(stream_expr)                                                                 \
  do                                                                                                \
  {                                                                                                 \
    const std::string _key = ::uam_controller::debug::makeKey(__FILE__, __LINE__, "FLOW");          \
    if (::uam_controller::debug::shouldLog(_key, UAM_DEBUG_FLOW_THROTTLE_SEC))                      \
    {                                                                                               \
      std::ostringstream _oss;                                                                      \
      _oss << "[FLOW] " << stream_expr;                                                             \
      ::uam_controller::debug::printLine(_oss.str());                                               \
    }                                                                                               \
  } while (0)
#else
#define UAM_DEBUG_FLOW(stream_expr) do {} while (0)
#endif

#if UAM_DEBUG_LEVEL >= UAM_DEBUG_LEVEL_DETAIL
#define UAM_DEBUG_DETAIL(stream_expr)                                                               \
  do                                                                                                \
  {                                                                                                 \
    const std::string _key = ::uam_controller::debug::makeKey(__FILE__, __LINE__, "DETAIL");        \
    if (::uam_controller::debug::shouldLog(_key, UAM_DEBUG_DETAIL_THROTTLE_SEC))                    \
    {                                                                                               \
      std::ostringstream _oss;                                                                      \
      _oss << "[DETAIL] " << stream_expr;                                                           \
      ::uam_controller::debug::printLine(_oss.str());                                               \
    }                                                                                               \
  } while (0)
#else
#define UAM_DEBUG_DETAIL(stream_expr) do {} while (0)
#endif

// 调试数据发布（仅细节级别时启用）
#if UAM_DEBUG_LEVEL >= UAM_DEBUG_LEVEL_DETAIL
#define UAM_DEBUG_PUBLISH_SCALAR(logger_ptr, topic, value)                           \
  do                                                                                 \
  {                                                                                  \
    if ((logger_ptr) != nullptr)                                                     \
    {                                                                                \
      (logger_ptr)->publishScalar(topic, value);                                     \
    }                                                                                \
  } while (0)

#define UAM_DEBUG_PUBLISH_VECTOR(logger_ptr, topic, vec)                             \
  do                                                                                 \
  {                                                                                  \
    if ((logger_ptr) != nullptr)                                                     \
    {                                                                                \
      (logger_ptr)->publishVector(topic, vec);                                       \
    }                                                                                \
  } while (0)
#else
#define UAM_DEBUG_PUBLISH_SCALAR(logger_ptr, topic, value) do {} while (0)
#define UAM_DEBUG_PUBLISH_VECTOR(logger_ptr, topic, vec) do {} while (0)
#endif
