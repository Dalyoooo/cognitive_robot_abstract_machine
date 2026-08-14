cmake_minimum_required(VERSION 3.8)
project(thesis_demo_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)

# Generates the Goal, Result and Feedback types for every listed interface, in
# each supported language. This is what makes
# `from thesis_demo_msgs.action import ExecutePlan` resolve at runtime.
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ExecutePlan.action"
  DEPENDENCIES action_msgs
)

ament_package()
