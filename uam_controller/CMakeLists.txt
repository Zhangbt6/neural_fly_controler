cmake_minimum_required(VERSION 3.0.2)
project(uam_controller)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  geometry_msgs
  nav_msgs
  roscpp
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

find_package(Eigen3 REQUIRED)

generate_dynamic_reconfigure_options(
  cfg/ControllerConfig.cfg
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS dynamic_reconfigure geometry_msgs nav_msgs roscpp std_msgs tf2 tf2_geometry_msgs tf2_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(controller_node src/controller_node.cpp)
add_dependencies(controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(controller_node ${catkin_LIBRARIES})

add_executable(setpoint_publisher src/setpoint_publisher.cpp)
add_dependencies(setpoint_publisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(setpoint_publisher ${catkin_LIBRARIES})

catkin_install_python(PROGRAMS
  scripts/control_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS controller_node setpoint_publisher
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
