#!/bin/bash

# Configuration
SESSION_NAME="ft_sensor_session"
WS_PATH="ros2_ws"

# Create a new tmux session (detached)
tmux new-session -d -s $SESSION_NAME -n "Publisher"

# Window 1: FT Sensor Publisher
tmux send-keys -t $SESSION_NAME:0 "source $WS_PATH/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0 "ros2 run ft_sensor publisher" C-m

# Window 2: Mass Estimation
tmux new-window -t $SESSION_NAME -n "Mass_Estimation"
tmux send-keys -t $SESSION_NAME:1 "source $WS_PATH/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:1 "python3 mass.py" C-m

# Window 3: Touch Down Detection
tmux new-window -t $SESSION_NAME -n "Touch_Down"
tmux send-keys -t $SESSION_NAME:2 "source $WS_PATH/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:2 "python3 touch_down_detection.py" C-m

# Attach to the session
tmux attach-session -t $SESSION_NAME