# Testing Motion Planning for Simulated UR Robot

This package contains test scripts to perform:
* Global planning to a pose goal (useful to set the initial orientation or approach position)
* Real-time servoing to a pose goal (useful for tracking a moving target or performing fine adjustments)

## Before Running the Tests
1. Make sure the `ur` packages are built and sourced:
```bash
cd ~/rgmc-2026/utils/ur
colcon build
source install/setup.bash
```
2. And the `moveit_stuff` package is built and sourced:
```bash
cd ~/rgmc-2026/utils/simple_ur_code/moveit_stuff
colcon build
source install/setup.bash
```
3. Also, the robot and `moveit` config is spawned (either via `ur` setup `tmux` scripts or manually):
```bash
ros2 launch ur_custom spawn_ur_with_gripper.launch.py
```

## Global Planning to a Pose Goal
The currently defined default planner is `pilz_industrial_motion_planner`, which is defined in ['ur_custom` package](../../ur/src/ur_custom/launch/spawn_ur_with_gripper.launch.py). You can change this to any other planner that is configured in `moveit` [setup package](../../ur/src/ur_custom_gripper_moveit_config), and also in `rviz` GUI.

## Servoing Testing with JointJog and Twist Commands
To run `moveit_servo`, we need to switch the default controller from `scaled_joint_trajectory_controller` to `forward_position_controller`, which is a simple controller that accepts position commands and executes them as fast as possible. This is necessary because `moveit_servo` sends velocity commands, and the default controller doesn't support velocity control.(TODO: check this info).

Switch controllers in a new terminal:
```bash
ros2 control switch_controllers --deactivate scaled_joint_trajectory_controller --activate forward_position_controller
```

You can inspect which controllers are active by running:
```bash
ros2 control list_controllers
```

## Real-time Servoing with Pose Tracking

Switch controllers in a new terminal:
```bash
ros2 control switch_controllers --deactivate scaled_joint_trajectory_controller --activate forward_position_controller
```

Then run the servoing test:
```bash
ros2 launch moveit_stuff pose_tracking_test.launch.py
```
UPDATE: this script was outdated and move to `old` folder.

## Debugging
If you encounter any issues, you can check the following:
1. Check that all relevant `move_group` parameters are defined:
```bash
ros2 param list /move_group
```
2. Check specific parameter values, for example the default planner:
```bash
ros2 param get /move_group pilz_industrial_motion_planner.default_planner_config
```

## H2R Handover System Execution

Before running the system, ensure that `cycloneDDS` is used as the middleware. To run on a real robot, the UR has to be connected to the **Motion PC** over Ethernet. At the same time, the **Blackwell/Visual PC** also needs to be connected to the **Motion PC** over Ethernet. Refer to the root README.md for the details and for the link to the technical report which explains the motion state machine and the handover system in detail.

The H2R handover system is controlled by a state machine that is defined in the `moveit_stuff` package. The main execution script is `plan_approach_grasp.cpp`.
It's important to start both the **Visual Pipeline** and **Motion Control** at around the same time, because the **Motion Control** depends on the data from the **Visual Pipeline**: it receives dimensions, handover status etc. This relation is unidirectional only.
Don't be scared by the length of the script - not all of it is relevant to the handover system. Keep in mind that the state machine is pretty simple: 
1. Initialization: robot services are initialized, moveit2 is configured, and pose tracking `moveit_servo` thread is spawned.
2. Handover anticipation: the system is waiting on the handover detection status. Before it is detected, the robot doesn't move. The visual object properties and other info is received from the visual pipeline in the background.
3. Object tracking: once the handover is detected, the robot starts tracking the object and moves to a pre-defined approach position once the object is ready to be grasped.
4. Grasping: the blocking grasping loop runs. It monitors gripper position feedback (although note that it may not work in a stable manner due to the drivers limitations) and tactile feedback if available.
5. Placement: once the grasp position is reached, the robot moves to a pre-defined placement position and releases the object. The release may happen prematurely if force-based touchdown feedback is received.

I suggest to just look at the `main` and scroll through it to see the code progression.

The parameters which control the handover execution are:
```bash
struct HandoverConfig
{
  float approach_distance;          // Max distance from command frame to the object
  float tracking_distance;          // Continuous object tracking distance
  int velocity_estimation_frames;   // Number of frames to estimate velocity (constant velocity model)
  int prediction_frames;            // Predict object pose for N frames ahead
  float velocity_threshold;         // Velocity below this allows executing the generated grasp
  float grasp_pose_tolerance;       // Distance to goal pose at which we consider the object grasped and can stop tracking
  float place_tolerance;            // Distance to goal pose at which we consider the object placed and can release the grasp
  float touchdown_tolerance;        // Distance to original object position at which we consider a touchdown detection valid to avoid false positives during placement approach
};
```

Try changing these parameters to the how the system responds. I suggest to record the rosbag files with necessary data to execute the full handover without the human involvement for this.

### MoveIt Config
Look up the UR description and MoveIt config in `utils/ur/src/ur_custom` and `utils/ur/src/ur_custom_gripper_moveit_config`. The most important files are:
* [PID config](../../ur/src/ur_custom/config/pose_tracking_settings_s8.yaml)
* [MoveIt Servo config](../../ur/src/ur_custom/config/servo_config.yaml)
* [MoveIt SRDF](../../ur/src/ur_custom_gripper_moveit_config/config/ur.srdf)
* [UR URDF and XACRO files](../../ur/src/ur_custom/urdf/ur_custom_gripper.urdf.xacro)
* [Initial joint positions](../../ur/src/ur_custom/config/initial_positions.yaml)

For example, if you want to use the internal force-torque sensor instead of external one for touchdown detection, besides changing the touchdown script name in the startup file you should also modify the URDF to remove the extra length of the tool (with the FT sensor model). You may also need to modify the SRDF to comment out the collision links from there but only if errors appear.

Note that this branch uses Robotiq's gripper base_link as the MoveIt command frame, since it's a frame that grasp generation model uses as well. 