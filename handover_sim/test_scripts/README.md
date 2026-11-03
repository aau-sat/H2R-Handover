# Handover Stage Testing
This directory contains scripts to test hand motion simulations in a predefined stage environment. 

The main script, [`test_stage.py`](./test_stage.py), sets up the full handover stage, including the robot and hand models, loads specified motion data, and visualizes the setting. It also publishes RGB and depth images to ROS2 topics from wrist-mounted camera.

[`test_new_hand.py`](./test_new_hand.py) isolates handover motion testing by setting up the simulation environment, loading the hand and object motion data, and visualizing the results.

Before running the scripts, don't forget to source the IsaacLab environment:
```bash
source ~/IsaacLab/env_isaaclab.sh
```

## Configuration
The configuration for the scripts is located in [`config_hand_motion.py`](./config_motion/config_hand_motion.py). This file specifies various constant parameters for the hand motion simulations, including paths to USD files, joint names, and tuneable parameters like physical properties.

### Setting the Experience
Make sure to set the appropriate Isaac Experience in the configuration file. For example, for optimal rendering speed for multiple cameras, use the following [setting](https://github.com/isaac-sim/IsaacLab/blob/main/apps/isaacsim_4_5/isaaclab.python.rendering.kit):
```python
ISAAC_EXPERIENCE = "isaaclab.python.rendering.kit"
```
Other options include [`"isaacsim.exp.full.kit"`](https://github.com/isaac-sim/isaacsim-app-template?tab=readme-ov-file#-pre-built-isaac-sim-applications) for full IsaacSim features, which renders a higher contrast image but may be slower. If something doesn't work as expected, try switching between these experiences.
A more lightweight experience like `"isaacsim.exp.base.kit"` can also be used, but I couldn't see the difference. If using `"isaacsim.exp.full.kit"` results in `segmentation fault`, and you don't like `"isaaclab.python.rendering.kit"` for some reason, you can try loading the script until it works (happened to me many times) or try `"isaacsim.exp.base.kit"`.

A very promising experience for headless rendering is [`"isaaclab.python.headless.rendering.kit"`](https://github.com/isaac-sim/IsaacLab/blob/main/apps/isaacsim_4_5/isaaclab.python.headless.rendering.kit), but I couldn't get it to work with the current scripts. Maybe updating IsaacLab to version `2.3.0` will make it work. If **you** manage to make it work, please share the solution!

## Test Hand Motions
The hand model used is the MANO hand model, which can be found in the [`assets/usd/hand/mano_realistic`](../assets/usd/hand/mano_realistic) directory. The motion data is sourced from the GENH2R dataset, located in [`assets/motions/genh2r`](../assets/motions/genh2r).

How to run:

```bash
~/IsaacLab/isaaclab.sh -p test_new_hand.py
```
Or, in case something goes wrong with dependencies, you can also run it directly with Python:
```bash
python test_new_hand.py
```

## Test Handover Stage
Currently, only UR10 model is available. A realistic room stage can be loaded by specifying 'stage_room.usd' in the configuration file, or a more lightweight version 'stage2.usd' for better performance:

```python
ROBOT_STAGE_USD_FILE = ROBOT_STAGE_DIR / 'stage_room.usd' # or more lightweight 'stage2.usd'
``` 

How to run:
```bash
~/IsaacLab/isaaclab.sh -p test_stage.py
```
Or:
```bash
python test_stage.py
```

## Current issues & limitations
1. Due to incorrect density settings of the left hand model, the left hand may fall through the table during simulation. This issue is being investigated.
2. The stage is under development, especially the robot model, sensors and motion planning aspects.
3. Handling fine objects, like thin mug handle, may result in unnatural hand grasping due to limitations in the current physics simulation. To preserve realism, change Mano hand model in `config_hand_motion.py` to 'mano_coarse.usd' which disables finger collisions. More optimal collision masks option is being explored.
4. Current motion trajectories are not limited in length, and handover ends soon after execution. In real simulation, hand motion needs to be paused & adjusted so that robot can react to the environment and the object being handed over.
5. I disabled IMU sensor inside the Realsense camera because its' driver doesn't work with `torch` backend. That's why running IsaacLab scripts produced annoying errors of device incompatibility. If we want to add it back, we need to either run in Isaac Sim only (without IsaacLab) or create a [sensor](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/ros2_tutorials/tutorial_ros2_python.html) in IsaacLab script.