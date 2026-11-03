# UR Custom with Gripper
Custom package to simulate UR robot with Robotiq gripper.
Based on multiple videos from https://www.youtube.com/@learn-robotics-with-ros:

## Setup
It's quite easy to set up the things on your machine.

1. **Git**
    This package contains the UR definitions as well as the Robotiq gripper definitions as **submodules**!

    For that reason, the required packages are by default not checked out. You need to run:
    ```sh
    git submodule update --init --recursive
    ```

    There shouldn't be any need to update the references from the submodule, but if so, please look up online guides and commit the changes in the repo! (or ask Markus)

2. **Conda/pip**
    Install the `requirements.txt` files. If you get errors, there might be some ros2 humble system packages missing. Make sure you have them installed and install missing ones if necessary (also make sure to document them here).

## Running the launch file
Before running the launch file, you need to compile the package:
```sh
colcon build
source install/setup.bash
```

In case of errors there are likely system packages missing.

Now you can run:
```sh
ros2 launch ur_custom spawn_ur_with_gripper.launch.py
```

For different launch files just replace the name accordingly.

## Notes

In the current version, Gazebo is not necessary to run the simulation. However, the URDF files are prepared to run a Gazebo simulation (with some limitations like a spam of errors). See older commits for more information.

There are a few switches in the URDF file of interest:
* `sim_gazebo`: set to `true` to run a gazebo simulation
* `sim_ignition`: set to `true` for ignition simulation
* `use_fake_hardware`: this gotta be `true` when running a simulation
