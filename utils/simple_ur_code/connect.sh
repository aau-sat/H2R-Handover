#!/usr/bin/env bash

SESSION=ur_session
MODE="sim"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --real) MODE="real" ;;
    --sim)  MODE="sim" ;;
    *) echo "Unknown option: $1" ;;
  esac
  shift
done

tmux kill-session -t $SESSION 2>/dev/null

tmux new-session -d -s $SESSION

if [[ "$MODE" == "real" ]]; then
  tmux send-keys -t $SESSION:0.0 \
    "export LC_NUMERIC=en_US.UTF-8; ros2 launch ur_robot_driver ur_control.launch.py headless_mode:=True ur_type:=ur3e robot_ip:=192.168.2.12 launch_rviz:=false ; exit" C-m
else
  tmux send-keys -t $SESSION:0.0 \
    "export LC_NUMERIC=en_US.UTF-8; ros2 launch ur_robot_driver ur_control.launch.py   ur_type:=ur3e   robot_ip:=127.0.0.1   use_fake_hardware:=true   launch_rviz:=false; exit" C-m
fi

tmux split-window -h -t $SESSION

tmux send-keys -t $SESSION:0.1 \
"export LC_NUMERIC=en_US.UTF-8; ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur3e ; exit" C-m

tmux attach -t $SESSION