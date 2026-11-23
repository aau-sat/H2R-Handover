# H2R Handover System - Setup & Deployment Guide

This document provides complete, step-by-step instructions for installing dependencies, configuring external workspaces, compiling packages, and launching the **Human-to-Robot (H2R) Handover System** on physical hardware or in simulation.

---

## Table of Contents
- [H2R Handover System - Setup \& Deployment Guide](#h2r-handover-system---setup--deployment-guide)
  - [Table of Contents](#table-of-contents)
  - [0. Repository Structure](#0-repository-structure)
  - [1. System Prerequisites \& Infrastructure](#1-system-prerequisites--infrastructure)
    - [Operating System \& ROS 2](#operating-system--ros-2)
    - [Cyclone DDS Configuration](#cyclone-dds-configuration)
    - [Two-Workstation Network Architecture](#two-workstation-network-architecture)
  - [2. Getting the Code](#2-getting-the-code)
  - [3. External Dependencies Setup](#3-external-dependencies-setup)
    - [GraspGen Workspace Setup](#graspgen-workspace-setup)
    - [MoveIt 2 External Workspace Setup](#moveit-2-external-workspace-setup)
    - [SAM 3 Credentials](#sam-3-credentials)
    - [Realsense ROS2 Driver](#realsense-ros2-driver)
  - [4. Building ROS 2 Workspace Packages](#4-building-ros-2-workspace-packages)
    - [Hardware \& Acceleration Requirements](#hardware--acceleration-requirements)
  - [5. Camera Calibration (`easy_handeye2`)](#5-camera-calibration-easy_handeye2)
  - [6. Running on Physical Hardware (Tmux Workflow)](#6-running-on-physical-hardware-tmux-workflow)
    - [Step A: Vision Pipeline Workstation](#step-a-vision-pipeline-workstation)
    - [Step B: Motion Planning Workstation](#step-b-motion-planning-workstation)
  - [7. Running in Simulation](#7-running-in-simulation)
    - [Option 1: MoveIt 2 Mock Simulation](#option-1-moveit-2-mock-simulation)
    - [Option 2: Isaac Sim 4.5 Digital Twin (`handover_sim`)](#option-2-isaac-sim-45-digital-twin-handover_sim)
  - [8. Troubleshooting \& Known Issues](#8-troubleshooting--known-issues)

---

## 0. Repository Structure

```shell
├── handover_sim/                        # Simulation environment for testing handovers (Isaac Sim 4.5 / Isaac Lab)
├── utils/                               # Core modules, workspaces, and drivers
│   ├── config.py                        # Global configuration variables
│   ├── easy_handeye2_ws/                # Eye-on-base calibration workspace
│   ├── ft_sensor/                       # Force/torque sensor drivers and processing
│   ├── grasp_generation/                # Grasp pose synthesis 
│   ├── handover_detection/              # legacy: Algorithms for detecting human handover intent
│   ├── mesh_generation/                 # legacy: 3D object mesh processing
│   ├── object_tracker/                  # Handover detection with subsequent hand tracking
│   ├── pose_tracking/                   # Object pose tracking
│   ├── realsense/                       # Intel RealSense camera utilities
│   ├── ros2_ws/                         # Custom ROS2 messages and interfaces
│   ├── segmentation/                    # Image segmentation models
│   ├── simple_ur_code/                  # Universal Robots control scripts
│   │       ├── moveit_stuff/            # Handover execution and MoveIt2 code
│   ├── tactile_ws/                      # Tactile sensing workspace
│   ├── ur/                              # Universal Robots and gripper URDF and MoveIt2 configurations
│   ├── visual_pipeline_setup/           # System startup scripts for vision pipeline
│   ├── visual_properties_detection/     # Estimating object physical properties
│   ├── vlm/                             # Vision-Language Model integrations
│   └── yolo-world/                      # Real-time object segmentation and detection (YOLO-E)
└── README.md
```

---

## 1. System Prerequisites & Infrastructure

### Operating System & ROS 2
* **OS:** Ubuntu Linux 22.04 LTS (Jammy Jellyfish)
* **ROS Distribution:** ROS 2 Humble Hawksbill
* **Python Version:** Python 3.10

### Cyclone DDS Configuration
To guarantee real-time, low-latency communication across workstations without message drops, **Eclipse Cyclone DDS** is used:

```bash
sudo apt update
sudo apt install ros-humble-rmw-cyclonedds-cpp
```

Add the following to your `~/.bashrc` on **both** workstations:
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### Two-Workstation Network Architecture
To guarantee strict real-time execution, the computational workload is bifurcated across two workstations connected via a dedicated 2.5 Gbps Ethernet link:
* **Vision Pipeline Workstation:** Powered by an AMD Ryzen 9 9950X 16-Core Processor and an NVIDIA RTX PRO 6000 Blackwell GPU (96 GB VRAM), executing object detection (YOLOE), SAM 3 segmentation, MediaPipe hand tracking, 3D object tracking, physical property estimation, and GraspGen synthesis.
* **Motion Planning Workstation:** Powered by an AMD Ryzen 9 5950X 16-Core Processor, exclusively reserved for real-time trajectory generation, MoveIt 2 Servo control, and direct hardware communication with the UR7e controller (1 Gbps link).

---

## 2. Getting the Code

Large files (meshes, simulation assets, model weights) are tracked with [Git LFS](https://git-lfs.com), and third-party packages are included as Git submodules. 

Install Git LFS before cloning:
```bash
sudo apt install git git-lfs
git lfs install
```

Clone the repository with submodules to `~/rgmc-2026`:
```bash
git clone --recurse-submodules https://github.com/aau-sat/RGMC-2026.git ~/rgmc-2026
```

> **Note:** The setup scripts expect the repository at `~/rgmc-2026`. If you cloned without `--recurse-submodules`, run:
> ```bash
> cd ~/rgmc-2026
> git submodule update --init --recursive
> ```

---

## 3. External Dependencies Setup

### GraspGen Workspace Setup
The grasp generation module utilizes [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) for 6-DoF grasp pose synthesis. Due to its dedicated PyTorch environment and dependencies, GraspGen is expected as an external workspace at `~/packages/GraspGen/GraspGen`.

1. **Clone GraspGen:**
   ```bash
   mkdir -p ~/packages/GraspGen
   git clone https://github.com/NVlabs/GraspGen.git ~/packages/GraspGen/GraspGen
   ```
2. **Set up the Conda environment:**
   Follow the installation guide in [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) to create and activate the `graspgen` conda environment, and download the model checkpoints (including the Robotiq 2F-140 configuration).
3. **Deploy our custom three-stage grasp service script:**
   Copy the three-stage service script from this repository into the GraspGen `scripts/` directory:
   ```bash
   cp ~/rgmc-2026/utils/grasp_generation/scripts/grasp_service_three_stage.py ~/packages/GraspGen/GraspGen/scripts/
   ```
   Ensure the checkpoint file `models/checkpoints/graspgen_robotiq_2f_140.yml` is present in your GraspGen workspace.

Once configured, the grasp service and Meshcat visualization server are launched automatically by `utils/visual_pipeline_setup/service_setup_tmux.sh`.

### MoveIt 2 External Workspace Setup
The motion control module relies on **MoveIt 2** built and sourced from an external workspace:
1. Build MoveIt 2 from source in a separate ROS 2 workspace with the `humble` branch selected.
2. Source the MoveIt 2 workspace prior to executing robot motion scripts:
   ```bash
   source <path_to_moveit2_ws>/install/setup.bash
   ```
3. **Thesis Branch Compatibility (`nad_thesis`):** For the thesis extensions on the `nad_thesis` branch, the MoveIt 2 version corresponds exactly to the validated release tag:
   * [`thesis-ur7e-validated`](https://github.com/Incomprehensible/moveit2/releases/tag/thesis-ur7e-validated)

### SAM 3 Credentials
Segment Anything Model 3 requires Hugging Face credentials for downloading model checkpoints:
1. Create a Hugging Face account and request access to the [Meta SAM 3 model](https://github.com/facebookresearch/sam3).
2. Log in using `huggingface-cli login` in your SAM 3 environment.

### Realsense ROS2 Driver
Make sure to install the `realsense2_camera` package via instructions on [realsense-ros](https://github.com/realsenseai/realsense-ros).

---

## 4. Building ROS 2 Workspace Packages

The ROS 2 workspace packages in `utils/` are built using `colcon`.

The system-active packages that require building are contained in:
* `utils/ur`
* `utils/object_tracker`
* `utils/tactile_ws`
* `utils/yolo-world`
* `utils/easy_handeye2_ws`
* `utils/ros2_ws`
* `utils/simple_ur_code/moveit_stuff`

Source ROS 2 and build with colcon:
```bash
source /opt/ros/humble/setup.bash
cd ~/rgmc-2026/utils
colcon build --symlink-install
```

### Hardware & Acceleration Requirements
* **GPU Compute:** Perception modules (SAM 3, YOLO-E, VLM, GraspGen) require an NVIDIA GPU with CUDA support. Ensure your NVIDIA driver and CUDA version match your PyTorch installation.
* **Conda Environments:** Various vision services utilize dedicated Conda / Python virtual environments (e.g., `sam3`, `graspgen`, `llm`) to prevent dependency conflicts, as launched in `service_setup_tmux.sh`.

---

## 5. Camera Calibration (`easy_handeye2`)

For eye-on-base calibration, `easy_handeye2` is used to register both static RealSense D435i cameras to the robot base frame:
1. Attach the calibration board to the UR7e end-effector.
2. Refer to [`utils/easy_handeye2_ws`](./utils/easy_handeye2_ws) and the [easy_handeye2 repository](https://github.com/marcoesposito1988/easy_handeye2) for launch scripts.
3. Sample 20–30 robot poses covering the workspace to achieve registration accuracy of approximately 0.5–2 cm.

---

## 6. Running on Physical Hardware (Tmux Workflow)

The system setup uses `tmux` sessions to organize processes. Familiarize yourself with `tmux` shortcuts (e.g., `Ctrl+b` then arrow keys to navigate panes).

### Step A: Vision Pipeline Workstation

1. **Start Persistent Services (once per session):**
   Starts SAM 3, GraspGen service, Meshcat server, VLM server, and RealSense cameras. Make sure to specify the paths correctly, including the path to Realsense preset. This session remains active and does **not** need to be restarted between handovers:
   ```bash
   cd ~/rgmc-2026/utils/visual_pipeline_setup
   sh service_setup_tmux.sh
   ```

   To enable reconstructed object proxies and visualization in `meshcat`, run instead:
   ```bash
   sh service_setup_tmux.sh --reconstruct --visualize
   ```

2. **Launch Perception Pipeline (restarted per handover):**
   Launches object tracking, hand detection, and dynamic state estimation. Restart this session for each handover attempt:
   ```bash
   cd ~/rgmc-2026/utils/visual_pipeline_setup
   sh vision_setup_tmux.sh
   ```

   To run with depth-based background clipping and logging, run:
   ```bash
   sh vision_setup_tmux.sh "LOG_DIR" --depth_clipping_right --depth_clipping_left
   ```


### Step B: Motion Planning Workstation

1. **Start Robot Drivers (once per session):**
   Connects to the UR7e controller, initialises MoveIt 2 hardware interfaces, and brings up sensor nodes. This session remains active and does **not** need to be restarted:
   ```bash
   cd ~/rgmc-2026/utils/ur
   sh ur_setup_tmux.sh
   ```

2. **Launch Motion Servoing & State Machine (restarted per handover):**
   Executes the handover state machine, visual servoing, tactile grasping, and object placement:
   ```bash
   cd ~/rgmc-2026/utils/ur
   sh moveit_setup_tmux.sh real
   ```

---

## 7. Running in Simulation

### Option 1: MoveIt 2 Mock Simulation
To test trajectory generation and MoveIt 2 state machine logic without physical hardware:
```bash
cd ~/rgmc-2026/utils/ur
sh moveit_setup_tmux.sh
```
This initializes MoveIt 2 with mock hardware interfaces.

### Option 2: Isaac Sim 4.5 Digital Twin (`handover_sim`)
For high-fidelity digital twin simulation including human hand motion trajectories, dynamic object physics, and camera perception streaming into ROS 2 topics:
* Refer to [`handover_sim/README.md`](./handover_sim/README.md) for full setup instructions using **NVIDIA Isaac Sim 4.5.0** and **Isaac Lab 1.x**.
* Sourcing and launching the handover stage:
  ```bash
  source ~/IsaacLab/env_isaaclab.sh
  cd ~/rgmc-2026/handover_sim/test_scripts
  ~/IsaacLab/isaaclab.sh -p test_stage.py
  ```
* For topic-based ROS 2 and MoveIt 2 closed-loop containerized control in Isaac Sim, see [`handover_sim/moveit_integration/ur10e_2f140_topic_based_ros2_control`](./handover_sim/moveit_integration/ur10e_2f140_topic_based_ros2_control/README.md).

---

## 8. Troubleshooting & Known Issues

1. **Cyclone DDS Multicast:** If workstations fail to discover ROS 2 topics, verify that multicast is enabled on your network interfaces and check `ROS_DOMAIN_ID`.
2. **Tactile Sensor Bluetooth Connection:** The custom tactile sensors stream over BLE. Ensure Bluetooth is enabled and pairs before running tactile nodes.
3. **RealSense Laser Power:** Camera exposure and laser power settings are configured in `service_setup_tmux.sh`. Adjust if lighting conditions in your environment cause depth noise.
4. **Isaac Sim IMU Driver:** In simulation, the RealSense IMU driver is disabled due to a known conflict with the PyTorch tensor backend.
