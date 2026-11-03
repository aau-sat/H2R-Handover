#!/bin/bash

# Default to 'sam' if no argument is provided
SERVER_TYPE=${1:-sam}

echo "[BASH] Starting dual servers of type: $SERVER_TYPE"

trap 'echo -e "\n[BASH] Terminating both servers..."; kill $PID1 $PID2 2>/dev/null; exit 0' SIGINT SIGTERM

python3 run_single_server.py --pull "tcp://*:5555" --pub "tcp://*:5556" --type "$SERVER_TYPE" &
PID1=$!

python3 run_single_server.py --pull "tcp://*:5565" --pub "tcp://*:5566" --type "$SERVER_TYPE" &
PID2=$!

echo "[BASH] Left server PID: $PID1"
echo "[BASH] Right server PID: $PID2"
echo "[BASH] Press Ctrl+C to stop both servers."

wait $PID1
wait $PID2

#./start_servers.sh tracker or sam