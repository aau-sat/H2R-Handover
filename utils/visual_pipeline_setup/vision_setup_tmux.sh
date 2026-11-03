#!/usr/bin/env bash

# run: sh vision_setup_tmux.sh "LOG_DIR" --depth_clipping_right --depth_clipping_left
log_folder=${1:-"/home/$USER/.ros/log"}
# --depth_clipping_right
right_depth_clipping_flag=${2:-""}
# --depth_clipping_left
left_depth_clipping_flag=${3:-""}

# Debug: Print the flags
echo "DEBUG: right_depth_clipping_flag='$right_depth_clipping_flag'"
echo "DEBUG: left_depth_clipping_flag='$left_depth_clipping_flag'"

#mkdir -p "$log_folder"
export ROS_LOG_DIR="$log_folder"
SESSION="RGMC_VISION"
START_DELAY=2

notify_popup() {
  local title="$1"
  local body="$2"
  local notify_display="$DISPLAY"
  local notify_bus="$DBUS_SESSION_BUS_ADDRESS"

  # Fallbacks for scripts started outside a desktop terminal
  if [ -z "$notify_display" ]; then
    notify_display=":0"
  fi
  if [ -z "$notify_bus" ] && [ -S "/run/user/$(id -u)/bus" ]; then
    notify_bus="unix:path=/run/user/$(id -u)/bus"
  fi

  if command -v notify-send >/dev/null 2>&1; then
    DISPLAY="$notify_display" DBUS_SESSION_BUS_ADDRESS="$notify_bus" notify-send "$title" "$body"
  fi
}

# Vorhandene Session beenden
tmux kill-session -t $SESSION 2>/dev/null

# ==============================================================================
# WINDOW 0: CORE VISION (YOLO + SAM)
# ==============================================================================
tmux new-session -d -s $SESSION -n "Core_Vision"

tmux split-window -v -t $SESSION:0

tmux split-window -h -t $SESSION:0.0
tmux split-window -h -t $SESSION:0.2

# Pane 0: Handover Detector + Hand tracker
tmux send-keys -t $SESSION:0.1 "conda deactivate" C-m
#tmux send-keys -t $SESSION:0.1 "export ROS_LOG_DIR="$log_folder"" C-m
tmux send-keys -t $SESSION:0.1 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:0.1 "source ~/rgmc-2026/utils/ros2_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION:0.1 "cd ~/rgmc-2026/utils/object_tracker/ && source install/setup.bash" C-m #&&  colcon build
tmux send-keys -t $SESSION:0.1 "ros2 run object_tracker hand_tracker_node_new" C-m # remove _depth to run the RGB version


# Pane 1: Yolo e Segmentation
tmux send-keys -t $SESSION:0.2 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.2 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:0.2 "cd ~/rgmc-2026/utils/yolo-world" C-m #&&  colcon build
tmux send-keys -t $SESSION:0.2 "source install/setup.bash" C-m
tmux send-keys -t $SESSION:0.2 "ros2 run yolo_world yoloe_tracker_dual" C-m # remove _depth to run the RGB version

# # Pane 1: Yoloe/RFDETR Segmentation
# tmux send-keys -t $SESSION:0.2 "conda deactivate" C-m
# tmux send-keys -t $SESSION:0.2 "source ~/env_seg/bin/activate" C-m
# tmux send-keys -t $SESSION:0.2 "cd ~/rgmc-2026/utils/yolo-world" C-m #&&  colcon build
# tmux send-keys -t $SESSION:0.2 "source install/setup.bash" C-m
# tmux send-keys -t $SESSION:0.2 "ros2 run yolo_world yoloerfdetr_tracker_dual" C-m # remove _depth to run the RGB version

# ==============================================================================
# WINDOW 1: Pose estimation (2x2 GRID)
# ==============================================================================
# Pane 2: Pose Estimation
tmux new-window -t $SESSION:1 -n "Vision_Workspace_1"

tmux split-window -v -t $SESSION:1   # Split top/bottom
tmux split-window -h -t $SESSION:1.0 # Split top half
tmux split-window -h -t $SESSION:1.2 # Split bottom half

tmux send-keys -t $SESSION:1.0 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.0 "cd ~/rgmc-2026/utils/pose_tracking/depth_segmentation_tracking" C-m
tmux send-keys -t $SESSION:1.0 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:1.0 "source ~/rgmc-2026/utils/ros2_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION:1.0 "python3 goal_pose_tracking_nearest_hand_open_changes.py --ros-args \
  -p target_frame:=base \
  -p pose_topic:=/goal_pose \
  -p tf_static_topic:=/tf_static" C-m  # also runs without any parameters: python3 segmentation_tracking.py


# Pane 3: publish tf of cameras
tmux send-keys -t $SESSION:1.1 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.1 "cd ~/rgmc-2026/utils/easy_handeye2_ws" C-m
tmux send-keys -t $SESSION:1.1 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:1.1 "source install/setup.bash" C-m
tmux send-keys -t $SESSION:1.1 "python3 publish_cams_transforms_new.py --refined_first_to_base" C-m

# Pane 4: Grasp Generator
tmux send-keys -t $SESSION:1.2 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.2 "export ROS_LOG_DIR="$log_folder"" C-m
tmux send-keys -t $SESSION:1.2 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:1.2 "source ~/rgmc-2026/utils/ros2_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION:1.2 "cd ~/rgmc-2026/utils/visual_properties_detection" C-m
tmux send-keys -t $SESSION:1.2 "python dimensions_estimator_dual_new.py" C-m


# ==============================================================================
# TASK 1: Call Dimensions Estimator
# ==============================================================================
# sleep $START_DELAY
# Pane 5: dimensions estimator
tmux send-keys -t $SESSION:1.3 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.3 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:1.3 "source ~/rgmc-2026/utils/ros2_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION:1.3 "cd ~/rgmc-2026/utils/grasp_generation" C-m
tmux send-keys -t $SESSION:1.3 "python grasp_generator_three_stage.py" C-m


tmux new-window -t $SESSION:2 -n "Others"
tmux split-window -v -t $SESSION:2
tmux split-window -h -t $SESSION:2.0
tmux split-window -h -t $SESSION:2.2
tmux select-layout -t $SESSION:2 tiled

# Pane 0: depth clipping left
if [ "$left_depth_clipping_flag" = "--depth_clipping_left" ]; then
    tmux send-keys -t $SESSION:2.0 "conda deactivate" C-m
    tmux send-keys -t $SESSION:2.0 "conda activate object_tracker_2" C-m
    tmux send-keys -t $SESSION:2.0 "cd ~/rgmc-2026/utils/object_tracker" C-m
    tmux send-keys -t $SESSION:2.0 "source install/setup.bash" C-m 
    tmux send-keys -t $SESSION:2.0 "ros2 run object_tracker depth_clipper_node --ros-args -p depth_clipped_topic:='/camera/left/color/depth_clipped' -p camera_topic_rgb:='/camera/left/color/image_raw' -p camera_topic_d:='/camera/left/aligned_depth_to_color/image_raw'" C-m
fi

# Pane 1: depth clipping right
if [ "$right_depth_clipping_flag" = "--depth_clipping_right" ]; then
    tmux send-keys -t $SESSION:2.1 "conda deactivate" C-m
    tmux send-keys -t $SESSION:2.1 "conda activate object_tracker_2" C-m
    tmux send-keys -t $SESSION:2.1 "cd ~/rgmc-2026/utils/object_tracker" C-m
    tmux send-keys -t $SESSION:2.1 "source install/setup.bash" C-m 
    tmux send-keys -t $SESSION:2.1 "ros2 run object_tracker depth_clipper_node --ros-args -p depth_clipped_topic:='/camera/right/color/depth_clipped' -p camera_topic_rgb:='/camera/right/color/image_raw' -p camera_topic_d:='/camera/right/aligned_depth_to_color/image_raw'" C-m
fi

tmux select-window -t $SESSION:1
#tmux select-pane -t $SESSION:1:0
#notify_popup "RGMC Vision" "System is ready. Starting dimensions estimator."

if [ -z "$TMUX" ]; then
    tmux select-window -t $SESSION:1
    tmux attach-session -t $SESSION
else
    echo "Session $SESSION wurde im Hintergrund erstellt. Nutze 'tmux switch-client -t $SESSION' zum Wechseln."
fi
