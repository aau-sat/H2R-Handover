#!/bin/bash

log_folder=${1:-"/home/$USER/.ros/log"}
transparent_flag=${2:-""}
mkdir -p "$log_folder"
export ROS_LOG_DIR="$log_folder"
SESSION="RGMC_VISION"
START_DELAY=5

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

# Pane 0: Handover Detector 
tmux send-keys -t $SESSION:0.0 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.0 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:0.0 "cd ~/rgmc-2026/utils/handover_detection/  && source install/setup.bash" C-m #&&  colcon build
tmux send-keys -t $SESSION:0.0 "ros2 run handover_detection handover_detector" C-m 
# sleep $START_DELAY

# Pane 1: Hand tracker
tmux send-keys -t $SESSION:0.1 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.1 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:0.1 "cd ~/rgmc-2026/utils/object_tracker/ && source install/setup.bash" C-m #&&  colcon build
tmux send-keys -t $SESSION:0.1 "ros2 run object_tracker hand_tracker_node" C-m
# sleep $START_DELAY

# Pane 2: Sam Service
tmux send-keys -t $SESSION:0.2 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.2 "conda activate sam3" C-m
tmux send-keys -t $SESSION:0.2 "cd ~/rgmc-2026/utils/segmentation" C-m
tmux send-keys -t $SESSION:0.2 "python segmentation_service_multiprompt.py" C-m 
sleep $START_DELAY


# Pane 3: Yolo e Segmentation
tmux send-keys -t $SESSION:0.3 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.3 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:0.3 "cd ~/rgmc-2026/utils/yolo-world" C-m #&&  colcon build
tmux send-keys -t $SESSION:0.3 "source install/setup.bash" C-m
tmux send-keys -t $SESSION:0.3 "ros2 run yolo_world yoloe_tracker_dual" C-m
# sleep $START_DELAY

# ==============================================================================
# WINDOW 1: Pose estimation (2x2 GRID)
# ==============================================================================
# Pane 1: Pose Estimation
tmux new-window -t $SESSION:1 -n "Vision_Workspace_1"

tmux split-window -v -t $SESSION:1   # Split top/bottom
tmux split-window -h -t $SESSION:1.0 # Split top half
tmux split-window -h -t $SESSION:1.2 # Split bottom half

tmux send-keys -t $SESSION:1.0 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.0 "cd ~/rgmc-2026/utils/pose_tracking/depth_segmentation_tracking" C-m
tmux send-keys -t $SESSION:1.0 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:1.0 "python3 segmentation_tracking_dual_fusion.py --ros-args \
  -p target_frame:=base \
  -p pose_topic:=/goal_pose \
  -p tf_static_topic:=/tf_static" C-m  # also runs without any parameters: python3 segmentation_tracking.py
# sleep $START_DELAY

# Pane 2: publish tf of cameras
tmux send-keys -t $SESSION:1.1 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.1 "cd ~/rgmc-2026/utils/easy_handeye2_ws" C-m
tmux send-keys -t $SESSION:1.1 "source install/setup.bash" C-m
tmux send-keys -t $SESSION:1.1 "python3 publish_cams_transforms_new.py --refined_second_to_base --refined_first_to_base" C-m
# tmux send-keys -t $SESSION:1.1 "python3 publish_cams_transforms.py --refined_second_to_base" C-m
# sleep $START_DELAY

# ==============================================================================
# WINDOW 2: Grasp Gen
# ==============================================================================
tmux new-window -t $SESSION:2 -n "Grasp Gen"
tmux split-window -v -t $SESSION:2
tmux split-window -h -t $SESSION:2.0
tmux split-window -h -t $SESSION:2.2
tmux select-layout -t $SESSION:2 tiled

# Pane 1: start meshcat for grasp gen visualization
tmux send-keys -t $SESSION:2.0 "conda deactivate" C-m
tmux send-keys -t $SESSION:2.0 "conda activate graspgen" C-m
tmux send-keys -t $SESSION:2.0 "cd ~/packages/GraspGen/GraspGen" C-m
tmux send-keys -t $SESSION:2.0 "meshcat-server" C-m # should run before  grasp gen scripts
sleep $START_DELAY

# Pane 2: Grasp Generator Server
tmux send-keys -t $SESSION:2.1 "conda deactivate" C-m
tmux send-keys -t $SESSION:2.1 "conda activate graspgen" C-m
tmux send-keys -t $SESSION:2.1 "cd ~/packages/GraspGen/GraspGen" C-m
tmux send-keys -t $SESSION:2.1 "PYTHONPATH=. python scripts/grasp_service_batched_new.py --orient $transparent_flag --gripper_config models/checkpoints/graspgen_robotiq_2f_140.yml --visualize" C-m
# sleep $START_DELAY

# Pane 3: Grasp Generator
tmux send-keys -t $SESSION:2.2 "conda deactivate" C-m
tmux send-keys -t $SESSION:2.2 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:2.2 "source ~/rgmc-2026/utils/ros2_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION:2.2 "cd ~/rgmc-2026/utils/grasp_generation" C-m
tmux send-keys -t $SESSION:2.2 "python grasp_generator_enhancer.py" C-m

# ==============================================================================
# WINDOW 1: Call Dimensions Estimator and Start with the handover time after some delay
# ==============================================================================
#sleep $START_DELAY

# Pane 4: dimensions estimator
tmux send-keys -t $SESSION:1.3 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.3 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:1.3 "source ~/rgmc-2026/utils/ros2_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION:1.3 "cd ~/rgmc-2026/utils/visual_properties_detection" C-m

tmux send-keys -t $SESSION:1.3 "python dimensions_estimator_single_optimized.py" C-m




# Pane 4: GPU Monitor
# tmux send-keys -t $SESSION:2.3 "watch -n 1 nvidia-smi" C-m

# back to first window
tmux select-window -t $SESSION:2
#tmux select-pane -t $SESSION:2:0
#notify_popup "RGMC Vision" "System is ready. Starting dimensions estimator."



if [ -z "$TMUX" ]; then
    tmux select-window -t $SESSION:2
    tmux attach-session -t $SESSION
else
    echo "Session $SESSION wurde im Hintergrund erstellt. Nutze 'tmux switch-client -t $SESSION' zum Wechseln."
fi
