#!/bin/bash

mode=${1:-"sim"}
SESSION="RGMC_PHYSICAL_EXECUTION"
START_DELAY=5

# Vorhandene Session beenden
tmux kill-session -t $SESSION 2>/dev/null

# ==============================================================================
# WINDOW 0: CORE VISION (YOLO + SAM)
# ==============================================================================
tmux new-session -d -s $SESSION -n "Core_Physical_Execution"

tmux split-window -v -t $SESSION:0
tmux split-window -h -t $SESSION:0.0
tmux split-window -h -t $SESSION:0.2

# Pane 0: Moveit2 Setup in Simulation
if [ "$mode" = "sim" ]; then
    tmux send-keys -t $SESSION:0.0 "cd /home/christian/rgmc-2026/utils/ur" C-m
    tmux send-keys -t $SESSION:0.0 "source install/setup.bash" C-m
    tmux send-keys -t $SESSION:0.0 "ros2 launch ur_custom spawn_ur_with_gripper_thesis.launch.py" C-m
    sleep $START_DELAY
fi

# Pane 1: Touchdown Detection
tmux send-keys -t $SESSION:0.1 "source /home/christian/rgmc-2026/utils/ros2_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION:0.1 "cd /home/christian/rgmc-2026/utils/ft_sensor" C-m
# tmux send-keys -t $SESSION:0.1 "python3 touchdown_detector_internal_improved.py" C-m
tmux send-keys -t $SESSION:0.1 "python3 touchdown_detector_service.py" C-m

# Pane 2: Motion Planning
tmux send-keys -t $SESSION:0.2 "source /home/christian/rgmc-2026/utils/ur/install/setup.bash" C-m
tmux send-keys -t $SESSION:0.2 "source /home/christian/rgmc-2026/utils/ros2_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION:0.2 "cd /home/christian/rgmc-2026/utils/simple_ur_code/moveit_stuff" C-m
tmux send-keys -t $SESSION:0.2 "source install/setup.bash" C-m
tmux send-keys -t $SESSION:0.2 "ros2 launch moveit_stuff h2r_thesis.launch.py placement_pose_interpolation_steps:=10 placement_pose_interpolation_profile:=ease_in_quad placement_strategy:=predefined use_sim_time:=false use_fake_hardware:=false fake_sensor_commands:=false config_id:=666 enable_real_time_filtering:=true" C-m

# back to first window
tmux select-window -t $SESSION:0
tmux select-pane -t $SESSION:0.0

if [ -z "$TMUX" ]; then
    tmux select-window -t $SESSION:0
    tmux attach-session -t $SESSION
else
    echo "Session $SESSION wurde im Hintergrund erstellt. Nutze 'tmux switch-client -t $SESSION' zum Wechseln."
fi
