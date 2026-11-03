#!/bin/bash

SESSION="RGMC_PHYSICAL"
START_DELAY=5

# Vorhandene Session beenden
tmux kill-session -t $SESSION 2>/dev/null

# ==============================================================================
# WINDOW 0: CORE VISION (YOLO + SAM)
# ==============================================================================
tmux new-session -d -s $SESSION -n "Core_Physical"

tmux split-window -v -t $SESSION:0
tmux split-window -h -t $SESSION:0.0
tmux split-window -h -t $SESSION:0.2

# Pane 0: Gripper Connection
tmux send-keys -t $SESSION:0.0 "source /home/christian/rgmc-2026/utils/ur/install/setup.bash" C-m
tmux send-keys -t $SESSION:0.0 "ros2 run ur_robot_driver tool_communication.py --ros-args -p robot_ip:=192.168.2.11 -p device_name:=/tmp/ttyUR" C-m
sleep $START_DELAY

# Pane 1: UR ROS2 Control
tmux send-keys -t $SESSION:0.1 "source /home/christian/rgmc-2026/utils/ur/install/setup.bash" C-m
tmux send-keys -t $SESSION:0.1 "ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur7e robot_ip:=192.168.2.11 use_tool_communication:=false   launch_rviz:=false description_file:=/home/christian/rgmc-2026/utils/ur/src/ur_custom/urdf/ur_custom_gripper.urdf.xacro runtime_config_package:=ur_custom  controllers_file:=ur_custom_controllers_gripper.yaml kinematics_params_file:=/home/christian/rgmc-2026/utils/ur/src/ur_custom_gripper_moveit_config/calibration/ur7e.yaml headless_mode:=true" C-m
sleep $START_DELAY

# Pane 2: Motion Planning with Moveit2
tmux send-keys -t $SESSION:0.2 "source /home/christian/rgmc-2026/utils/ur/install/setup.bash" C-m
tmux send-keys -t $SESSION:0.2 "ros2 launch ur_custom spawn_ur_with_gripper.launch.py use_sim_time:=false use_fake_hardware:=false fake_sensor_commands:=false" C-m 
# sleep $START_DELAY

# Pane 3: Tactile Sensing
tmux send-keys -t $SESSION:0.3 "source /home/christian/rgmc-2026/utils/ros2_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION:0.3 "cd /home/christian/rgmc-2026/utils/tactile_ws" C-m
tmux send-keys -t $SESSION:0.3 "source install/setup.bash" C-m
tmux send-keys -t $SESSION:0.3 "ros2 launch tactile_nodes tactile_sensor_launch.py" C-m

# ==============================================================================
# WINDOW 1: Pose estimation (2x2 GRID)
# ==============================================================================
# Pane 1: Pose Estimation
tmux new-window -t $SESSION:1 -n "Cameras_Setup"

tmux split-window -v -t $SESSION:1   # Split top/bottom
tmux split-window -h -t $SESSION:1.0 # Split top half
tmux split-window -h -t $SESSION:1.2 # Split bottom half

# tmux send-keys -t $SESSION:1.0 "ros2 launch realsense2_camera rs_launch.py enable_rgbd:=false enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 serial_no:=_346522071495 camera_name:=left pointcloud.enable:=false temporal_filter.enable:=true json_file_path:=/home/christian/rgmc-2026/utils/realsense/presets/HighDensityPreset.json depth_module.enable_auto_exposure:=false depth_module.exposure:=5000" C-m
# sleep $START_DELAY

# tmux send-keys -t $SESSION:1.1 "ros2 launch realsense2_camera rs_launch.py enable_rgbd:=false enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 serial_no:=_913522071225 camera_name:=right pointcloud.enable:=false temporal_filter.enable:=true depth_module.enable_auto_exposure:=false depth_module.exposure:=5000 json_file_path:=/home/christian/rgmc-2026/utils/realsense/presets/HighDensityPreset.json" C-m

# back to first window
tmux select-window -t $SESSION:0
tmux select-pane -t $SESSION:0.0

if [ -z "$TMUX" ]; then
    tmux select-window -t $SESSION:0
    tmux attach-session -t $SESSION
else
    echo "Session $SESSION wurde im Hintergrund erstellt. Nutze 'tmux switch-client -t $SESSION' zum Wechseln."
fi
