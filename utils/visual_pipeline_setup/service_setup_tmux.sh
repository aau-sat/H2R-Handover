#!/bin/bash

# run:
# sh service_setup_tmux.sh --reconstruct --visualize
reconstruct_flag=${1:-""}
visualize_flag=${2:-""}
SESSION="RGMC_SERVICES"
START_DELAY=1
START_DELAY_cams=8


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
tmux new-session -d -s $SESSION -n "Vision Services"

tmux split-window -v -t $SESSION:0
tmux split-window -h -t $SESSION:0.0
tmux split-window -h -t $SESSION:0.2

# Sam3 Service
tmux send-keys -t $SESSION:0.0 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.0 "conda activate sam3" C-m
tmux send-keys -t $SESSION:0.0 "cd ~/rgmc-2026/utils/segmentation" C-m
tmux send-keys -t $SESSION:0.0 "python segmentation_service_multiprompt.py" C-m 

# meshcat for grasp gen visualization
tmux send-keys -t $SESSION:0.1 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.1 "conda activate graspgen" C-m
tmux send-keys -t $SESSION:0.1 "cd ~/packages/GraspGen/GraspGen" C-m
tmux send-keys -t $SESSION:0.1 "meshcat-server" C-m # should run before  grasp gen scripts
sleep $START_DELAY

# GraspGen Server
tmux send-keys -t $SESSION:0.2 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.2 "conda activate graspgen" C-m
tmux send-keys -t $SESSION:0.2 "cd ~/packages/GraspGen/GraspGen" C-m
# tmux send-keys -t $SESSION:0.2 "PYTHONPATH=. python scripts/grasp_service_batched_new.py --orient $reconstruct_flag --gripper_config models/checkpoints/graspgen_robotiq_2f_140.yml --visualize" C-m
tmux send-keys -t $SESSION:0.2 "PYTHONPATH=. python scripts/grasp_service_three_stage.py --orient $reconstruct_flag $visualize_flag --gripper_config models/checkpoints/graspgen_robotiq_2f_140.yml" C-m

# LLM Service
tmux send-keys -t $SESSION:0.3 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.3 "source ~/llm/.venv/bin/activate" C-m
tmux send-keys -t $SESSION:0.3 "cd ~/rgmc-2026/utils/vlm" C-m
tmux send-keys -t $SESSION:0.3 "python server.py" C-m


tmux new-window -t $SESSION:1 -n "Cams"
tmux split-window -v -t $SESSION:1
tmux split-window -h -t $SESSION:1.0
tmux split-window -h -t $SESSION:1.2
tmux select-layout -t $SESSION:1 tiled

# Cam left
tmux send-keys -t $SESSION:1.0 "ros2 launch realsense2_camera rs_launch.py enable_rgbd:=false enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 serial_no:=_346522071495 camera_name:=left pointcloud.enable:=false temporal_filter.enable:=true json_file_path:=/home/student-team/Downloads/HighDensityPreset.json depth_module.enable_auto_exposure:=false depth_module.exposure:=5019 initial_reset:=true rgb_camera.enable_auto_exposure:=true" C-m

# cam right
tmux send-keys -t $SESSION:1.1 "ros2 launch realsense2_camera rs_launch.py enable_rgbd:=false enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 serial_no:=_913522071225 camera_name:=right pointcloud.enable:=false temporal_filter.enable:=true depth_module.enable_auto_exposure:=false depth_module.exposure:=5019 json_file_path:=/home/student-team/Downloads/HighDensityPreset.json initial_reset:=true rgb_camera.enable_auto_exposure:=true" C-m
sleep $START_DELAY_cams

# change settings
tmux send-keys -t $SESSION:1.2 "ros2 param set camera/left depth_module.laser_power 240.0 " C-m
tmux send-keys -t $SESSION:1.2 "ros2 param set camera/right depth_module.laser_power 240.0 " C-m


# back to first window
tmux select-window -t $SESSION:0

if [ -z "$TMUX" ]; then
    tmux select-window -t $SESSION:0
    tmux attach-session -t $SESSION
else
    echo "Session $SESSION wurde im Hintergrund erstellt. Nutze 'tmux switch-client -t $SESSION' zum Wechseln."
fi
