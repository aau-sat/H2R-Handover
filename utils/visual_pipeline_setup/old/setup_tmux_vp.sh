#!/bin/bash

SESSION="RGMC"

tmux kill-session -t $SESSION 2>/dev/null


tmux new-session -d -s $SESSION -n "Vision Workspace_0"

# 2x2 grid

tmux split-window -v -t $SESSION:0   # Split top/bottom
tmux split-window -h -t $SESSION:0.0 # Split top half
tmux split-window -h -t $SESSION:0.2 # Split bottom half
tmux select-layout -t $SESSION:0 tiled

# FoundationPose
tmux send-keys -t $SESSION:0.0 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.0 "cd ~/packages/FoundationPose &&

conda activate foundationpose_ros" C-m

tmux send-keys -t $SESSION:0.0 "source realsense_ws/install/setup.bash &&
source ros2_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION:0.0 "python pose_tracker_node_scaling.py" C-m


# Object Detection Node (top right)
tmux send-keys -t $SESSION:0.1 "conda deactivate" C-m

tmux send-keys -t $SESSION:0.1 "cd ~/rgmc-2026" C-m

tmux send-keys -t $SESSION:0.1 "source ~/env_isaaclab/bin/activate &&
cd utils/handover_detection/ &&
source install/setup.bash" C-m

tmux send-keys -t $SESSION:0.1 "ros2 run handover_detection test_yolo_rgb" C-m




# Sam3 Segmentation Service (bottom left)
tmux send-keys -t $SESSION:0.2 "cd ~/rgmc-2026/utils/segmentation" C-m
tmux send-keys -t $SESSION:0.2 "conda deactivate" C-m
tmux send-keys -t $SESSION:0.2 "conda activate sam3" C-m
tmux send-keys -t $SESSION:0.2 "python segmentation_service.py" C-m


# Mesh Generation Node (bottom right)
tmux send-keys -t $SESSION:0.3 "conda deactivate" C-m

tmux send-keys -t $SESSION:0.3 "cd ~/rgmc-2026/utils/mesh_generation/instantmesh" C-m
tmux send-keys -t $SESSION:0.3 "conda activate instantmesh" C-m

tmux send-keys -t $SESSION:0.3 "source ros2_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION:0.3 "python mesh_generator_node.py" C-m



# FoundatoinPose Container Window
tmux new-window -t $SESSION:1 -n "Vision_Workspace_1"

tmux split-window -v -t $SESSION:1   # Split top/bottom
tmux split-window -h -t $SESSION:1.0 # Split top half
tmux split-window -h -t $SESSION:1.2 # Split bottom half
tmux send-keys -t $SESSION:1.0 "conda deactivate" C-m

tmux send-keys -t $SESSION:1.0 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:1.0 "cd ~/rgmc-2026/utils/handover_detection/" C-m 
tmux send-keys -t $SESSION:1.0 "source install/setup.bash" C-m
tmux send-keys -t $SESSION:1.0 "cd ~/rgmc-2026/utils/handover_detection/src/handover_detection/handover_detection" C-m
tmux send-keys -t $SESSION:1.0 "python3 stop_movement_detection_node.py" C-m

tmux send-keys -t $SESSION:1.1 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.1 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:1.1 "cd ~/rgmc-2026/utils/handover_detection/" C-m 
tmux send-keys -t $SESSION:1.1 "source install/setup.bash" C-m
tmux send-keys -t $SESSION:1.1 "cd ~/rgmc-2026/utils/handover_detection/src/handover_detection/handover_detection" C-m
tmux send-keys -t $SESSION:1.1 "python3 grasp_coordinator_node_mirror.py" C-m


tmux send-keys -t $SESSION:1.2 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.2 "source ~/env_isaaclab/bin/activate" C-m
tmux send-keys -t $SESSION:1.2 "cd ~/rgmc-2026/utils/segmentation/test_scripts/" C-m
tmux send-keys -t $SESSION:1.2 "python3 sam_stream_bridge.py" C-m


tmux send-keys -t $SESSION:1.3 "conda deactivate" C-m
tmux send-keys -t $SESSION:1.3 "conda activate sam3" C-m
tmux send-keys -t $SESSION:1.3 "cd ~/rgmc-2026/utils/segmentation/test_scripts/" C-m
tmux send-keys -t $SESSION:1.3 "python sam_stream_server.py" C-m


# Topic Listener Window
tmux new-window -t $SESSION:2 -n "Topic Listener"
tmux split-window -v -t $SESSION:2   # Split top/bottom
tmux split-window -h -t $SESSION:2.0 # Split top half
tmux split-window -h -t $SESSION:2.2 # Split bottom half

tmux send-keys -t $SESSION:2.0 "ros2 topic echo /detected_objects" C-m
tmux send-keys -t $SESSION:2.1 "ros2 topic echo /goal_pose" C-m
tmux send-keys -t $SESSION:2.2 "ros2 topic echo /object_ready_to_grasp" C-m
tmux send-keys -t $SESSION:2.3 "ros2 topic list" C-m


tmux new-window -t $SESSION:3 -n "utils"
tmux split-window -v -t $SESSION:3   # Split top/bottom
tmux split-window -h -t $SESSION:3.0 # Split top half
tmux split-window -h -t $SESSION:3.2 # Split bottom half
tmux send-keys -t $SESSION:3.0 "ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true pointcloud.enable:=true depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30" C-m
tmux send-keys -t $SESSION:3.1 "code ." C-m
tmux send-keys -t $SESSION:3.2 "watch -n 1 nvidia-smi" C-m



# 6. Set focus to the main workspace and attach
tmux select-window -t $SESSION:0
tmux select-pane -t $SESSION:0.0
tmux attach-session -t $SESSION
