cmake_minimum_required(VERSION 3.8)
project(moveit_stuff)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(moveit_servo REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(grasp_generation REQUIRED)
find_package(visual_properties REQUIRED)
# find_package(touchdown_detection REQUIRED)
find_package(tactile_detection REQUIRED)

# add_executable(move_group_interface_test src/move_group_interface_test.cpp)
# add_executable(orient_gripper_test src/orient_gripper_test.cpp)
# add_executable(servo_test src/servo_test.cpp)
# add_executable(pose_tracking_test src/pose_tracking_test.cpp)
# add_executable(estimated_goal_tracking src/estimated_goal_tracking.cpp)
# add_executable(generated_goal_tracking src/generated_goal_tracking.cpp)
# add_executable(plan_approach src/plan_approach.cpp)
add_executable(plan_approach_grasp 
  src/plan_approach_grasp.cpp
  ../real_time_filter/src/OneEuroFilter.cpp
)
# add_executable(gripper_set_width_test src/gripper_set_width_test.cpp)
add_executable(end_effector_s8 src/end_effector_s8.cpp)
# add_executable(gripper_closed_loop_control src/gripper_closed_loop_control.cpp)

# target_include_directories(move_group_interface_test PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
# target_include_directories(orient_gripper_test PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
# target_include_directories(servo_test PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
# target_include_directories(estimated_goal_tracking PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
# target_include_directories(pose_tracking_test PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
# target_include_directories(generated_goal_tracking PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
# target_include_directories(plan_approach PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
target_include_directories(plan_approach_grasp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# target_include_directories(gripper_set_width_test PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
target_include_directories(end_effector_s8 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# target_include_directories(gripper_closed_loop_control PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)

# target_compile_features(move_group_interface_test PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
# target_compile_features(orient_gripper_test PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
# target_compile_features(servo_test PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
# target_compile_features(estimated_goal_tracking PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
# target_compile_features(pose_tracking_test PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
# target_compile_features(generated_goal_tracking PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
# target_compile_features(plan_approach PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_features(plan_approach_grasp PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
# target_compile_features(gripper_set_width_test PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_features(end_effector_s8 PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
# target_compile_features(gripper_closed_loop_control PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# ament_target_dependencies(
#   move_group_interface_test
#   "rclcpp"
#   "moveit_ros_planning_interface"
#   "geometry_msgs"
#   "moveit_visual_tools"
#   ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
# )
# ament_target_dependencies(
#   orient_gripper_test
#   "rclcpp"
#   "moveit_ros_planning_interface"
#   "geometry_msgs"
#   "moveit_visual_tools"
#   ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
# )
# ament_target_dependencies(
#   servo_test
#   "rclcpp"
#   "moveit_ros_planning_interface"
#   "geometry_msgs"
#   "moveit_visual_tools"
#   "moveit_servo"
#   ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
# )
# ament_target_dependencies(
#   pose_tracking_test
#   "rclcpp"
#   "moveit_ros_planning_interface"
#   "geometry_msgs"
#   "moveit_visual_tools"
#   "controller_manager_msgs"
#   "moveit_servo"
#   ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
# )
# ament_target_dependencies(
#   estimated_goal_tracking
#   "rclcpp"
#   "moveit_ros_planning_interface"
#   "geometry_msgs"
#   "moveit_visual_tools"
#   "moveit_servo"
#   "controller_manager_msgs"
#   ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
# )
# ament_target_dependencies(
#   generated_goal_tracking
#   "rclcpp"
#   "moveit_ros_planning_interface"
#   "geometry_msgs"
#   "moveit_visual_tools"
#   "moveit_servo"
#   "controller_manager_msgs"
#   ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
# )
# ament_target_dependencies(
#   plan_approach
#   "rclcpp"
#   "moveit_ros_planning_interface"
#   "geometry_msgs"
#   "moveit_visual_tools"
#   "moveit_servo"
#   "controller_manager_msgs"
#   ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
# )
ament_target_dependencies(
  plan_approach_grasp
  "rclcpp"
  "moveit_ros_planning_interface"
  "geometry_msgs"
  "moveit_visual_tools"
  "moveit_servo"
  "controller_manager_msgs"
  "grasp_generation"
  # "touchdown_detection"
  "visual_properties"
  "tactile_detection"
  "rclcpp"
  "rclcpp_action"
  "rclcpp_components"
  ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
)
# ament_target_dependencies(
#   gripper_set_width_test
#   "rclcpp"
#   "moveit_ros_planning_interface"
#   "geometry_msgs"
#   "moveit_visual_tools"
#   ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
# )
ament_target_dependencies(
  end_effector_s8
  "rclcpp"
  "moveit_ros_planning_interface"
  "geometry_msgs"
  "moveit_visual_tools"
  "moveit_servo"
  "controller_manager_msgs"
  ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
)

# ament_target_dependencies(
#   gripper_closed_loop_control
#   "rclcpp"
#   "moveit_ros_planning_interface"
#   "geometry_msgs"
#   "moveit_visual_tools"
#   "moveit_servo"
#   "controller_manager_msgs"
#   ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
# )

install(TARGETS plan_approach_grasp end_effector_s8 #gripper_closed_loop_control
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
