# Simple UR Code
This folder contains some simple code for connecting to the UR and for simulating movements on the UR.

## SAFETY INFORMATION
When sending position goals to the UR, make sure to add the obstacles first and also check if the obstacles in RVIz are the same as in real life.
Reduce the max speed so you have more time to stop if an error occures. Make sure to stay close to the red stop button in case the robot is about to crash.
When only reading from the UR, you do not need to enter the remote mode of the UR, you can simply use the local mode, this can be an additional security measure.

## Connecting to the Simulation
To connect to the simulation please run ```bash ./connect.sh --sim```, for this to run tmux is required (sudo apt install tmux).

## Connecting to the Robot
To connect to the real robot please run ```bash ./connect.sh --real```, for this to run tmux is required (sudo apt install tmux).

## Disconnecting
To disconnect from the simulation or the real robot, just press ctrl + c in the window where you ran the connect.sh

## Obstacles
For the planner to safely work the objects need to be defined, running ```python3 ./add_obstacles.py``` adds the obstacles to Moveit.

The definitions of the objects can be found in the ```obstacles``` folder.
They should be configured correctly already, but if the gripper is changed it's dimenstions need to be adjusted, the same is true for the table.
To make calibration of the table easyer, the z-Dimension of the table can be calibrated by running  ```python3 ./calibrate_table.py```.

## Reading Positions
The files ```read_joints.py``` and ```read_pose.py``` contain some basic code outlining how reading the position can be done.

## Sending Position Goals
The files ```send_goal_pose.py``` and ```send_goal_pose2.py``` contain some basic code outlining how the robot can be told to move to a certain position.

## Additional Info
The connection can also be established without tmux, but this requires a few more manual steps, for this to work see the ```old_start_scripts``` folder.