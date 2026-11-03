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
find_package(OpenMP REQUIRED)

# add_executable(plan_approach_grasp 
#   src/plan_approach_grasp.cpp
#   ../real_time_filter/src/OneEuroFilter.cpp
# )
add_executable(end_effector_s8 src/end_effector_s8.cpp)
add_executable(h2r_handover 
  src/h2r_thesis.cpp
  src/real_time_filter/OneEuroFilter.cpp
)

# target_include_directories(plan_approach_grasp PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
target_include_directories(end_effector_s8 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# target_include_directories(gripper_closed_loop_control PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
target_include_directories(h2r_handover PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# target_compile_features(plan_approach_grasp PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_features(end_effector_s8 PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_features(h2r_handover PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# ament_target_dependencies(
#   plan_approach_grasp
#   "rclcpp"
#   "moveit_ros_planning_interface"
#   "geometry_msgs"
#   "moveit_visual_tools"
#   "moveit_servo"
#   "controller_manager_msgs"
#   "grasp_generation"
#   # "touchdown_detection"
#   "visual_properties"
#   "tactile_detection"
#   "rclcpp"
#   "rclcpp_action"
#   "rclcpp_components"
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

ament_target_dependencies(
  h2r_handover
  "rclcpp"
  "moveit_ros_planning_interface"
  "geometry_msgs"
  "moveit_visual_tools"
  "moveit_servo"
  "controller_manager_msgs"
  "grasp_generation"
  "visual_properties"
  "tactile_detection"
  "rclcpp"
  "rclcpp_action"
  "rclcpp_components"
  ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
)

target_link_libraries(h2r_handover dqrobotics OpenMP::OpenMP_CXX)

install(TARGETS end_effector_s8 h2r_handover #plan_approach_grasp
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
