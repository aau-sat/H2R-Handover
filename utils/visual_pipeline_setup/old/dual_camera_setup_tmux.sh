#!/bin/bash

SESSION="RGMC_VISION"

# Vorhandene Session beenden
tmux kill-session -t $SESSION 2>/dev/null

# ==============================================================================
# WINDOW 0: CORE VISION (YOLO + SAM)
# ==============================================================================
tmux new-session -d -s $SESSION -n "Core_Vision"

tmux split-window -v -t $SESSION:0
tmux split-window -h -t $SESSION:0.0
tmux split-window -h -t $SESSION:0.2

# Pane 0: Object Detection (YOLO)
tmux send-keys -t $SESSION:0.0 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.0 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:0.0 "cd ~/rgmc-2026/utils/object_tracker/ && colcon build && source install/setup.bash" C-m
tmux send-keys -t $SESSION:0.0 "ros2 run object_tracker object_tracker_node" C-m

# Pane 1: Camera
tmux send-keys -t $SESSION:0.1 "ros2 topic hz /camera/left/color/image_raw" C-m
#tmux send-keys -t $SESSION:0.1 "./~/rgmc-2026/utils/easy_handeye2_ws/run_cameras.bash" C-m needs to be done outside

# Pane 2: SAM Server or Sam Service
tmux send-keys -t $SESSION:0.2 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.2 "conda activate sam3" C-m
# unconmment for running SAM  stream bridge (for testing)
tmux send-keys -t $SESSION:0.2 "cd ~/rgmc-2026/utils/segmentation/dual_setup/" C-m
tmux send-keys -t $SESSION:0.2 "python sam_stream_server_dual.py" C-m # for sam video, for sam tracking: python sam_stream_server_dual.py tracker


# Pane 3: SAM Bridge (continuous stream or single shot)
tmux send-keys -t $SESSION:0.3 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.3 "source ~/env_isaaclab/bin/activate" C-m
# uncomment for running SAM stream bridge (for testing)
tmux send-keys -t $SESSION:0.3 "cd ~/rgmc-2026/utils/segmentation/dual_setup/" C-m
tmux send-keys -t $SESSION:0.3 "python3 sam_stream_bridge_dual.py" C-m






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
tmux send-keys -t $SESSION:1.0 "python3 segmentation_tracking.py" C-m

# Pane 2: publish tf of cameras
tmux send-keys -t $SESSION:1.1 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.1 "cd ~/rgmc-2026/utils/easy_handeye2_ws" C-m
tmux send-keys -t $SESSION:1.1 "source install/setup.bash" C-m
tmux send-keys -t $SESSION:1.1 "ros2 launch publish.launch.xml" C-m

# Pane 3: compute goal pose
tmux send-keys -t $SESSION:1.2 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.2 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:1.2 "cd ~/rgmc-2026/utils/easy_handeye2_ws" C-m
tmux send-keys -t $SESSION:1.2 "source install/setup.bash" C-m
tmux send-keys -t $SESSION:1.2 "python3 compute_goal_pose.py --camera-frame left_color_optical_frame" C-m  # change the camera frame if needed, all available camera transforms can be seen with the command below


# pane 4: tf monitor
tmux send-keys -t $SESSION:1.3 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.3 "ros2 run tf2_ros tf2_monitor" C-m
tmux send-keys -t $SESSION:1.3 "ros2 run tf2_ros tf2_monitor" C-m





# ==============================================================================
# WINDOW 2: OTHER TASKS (2x2 GRID)
# ==============================================================================
tmux new-window -t $SESSION:2 -n "Other_Tasks"
tmux split-window -v -t $SESSION:2
tmux split-window -h -t $SESSION:2.0
tmux split-window -h -t $SESSION:2.2
tmux select-layout -t $SESSION:2 tiled

# Default commands for the utility window
tmux send-keys -t $SESSION:2.0 "ros2 topic list" C-m
tmux send-keys -t $SESSION:2.1 "ros2 node list" C-m
tmux send-keys -t $SESSION:2.2 "watch -n 1 nvidia-smi" C-m
tmux send-keys -t $SESSION:2.3 "top" C-m

# Zurück zum Hauptfenster und Fokus auf YOLO
tmux select-window -t $SESSION:0
tmux select

# ... (dein restliches Skript oben)

# Falls wir bereits in tmux sind, erstelle die Session nur. 
# Falls nicht, attache sie am Ende.
if [ -z "$TMUX" ]; then
    tmux select-window -t $SESSION:0
    tmux attach-session -t $SESSION
else
    echo "Session $SESSION wurde im Hintergrund erstellt. Nutze 'tmux switch-client -t $SESSION' zum Wechseln."
fi
