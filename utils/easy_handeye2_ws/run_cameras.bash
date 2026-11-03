#!/bin/bash

SESSION_NAME="rs_cameras"

# 1. Create a new session, name it, and run the first camera in the first window
tmux new-session -d -s $SESSION_NAME "ros2 launch realsense2_camera rs_launch.py \
  camera_name:='right' \
  serial_no:=\"'913522071225'\" \
  enable_rgbd:=true \
  enable_sync:=true \
  align_depth.enable:=true \
  enable_color:=true \
  enable_depth:=true \
  pointcloud.enable:=true \
  depth_module.depth_profile:=640x480x30 \
  rgb_camera.color_profile:=640x480x30"

tmux split-window -h -t $SESSION_NAME "ros2 launch realsense2_camera rs_launch.py \
  camera_name:='left' \
  serial_no:=\"'346522071495'\" \
  enable_rgbd:=true \
  enable_sync:=true \
  align_depth.enable:=true \
  enable_color:=true \
  enable_depth:=true \
  pointcloud.enable:=true \
  depth_module.depth_profile:=640x480x30 \
  rgb_camera.color_profile:=640x480x30"

tmux attach-session -t $SESSION_NAME