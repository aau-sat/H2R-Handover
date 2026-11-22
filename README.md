# H2R Handover System - BrokenArm (RGMC @ ICRA 2026)

Official open-source repository for the Human-to-Robot (H2R) Handover System developed by team **BrokenArm** from the [Smart System Technologies, Sensors, Actuators & Modular Robotics (SAT) group](https://www.aau.at/en/smart-systems-technologies/sensors-and-actuators/) at the [University of Klagenfurt](https://www.aau.at/en/), presented at the **RGMC 2026** (Robotic Grasping and Manipulation Competition) held in scope of **ICRA 2026** in Vienna, Austria.

Our system achieved **top score in the Qualification benchmark among participating teams** (71.41 points) and achieved a **75% success rate in the live Finals** under zero-shot physical handover of unknown containers across diverse lighting and environment conditions.

The core contribution is a robust, distributed, and highly-tunable architecture that seamlessly integrates:
* **Dual-Camera 3D Perception & Tracking:** Real-time object and hand tracking using open-vocabulary detectors (YOLOE), MediaPipe hand tracking, and spatial sensor fusion.
* **Human-Aware Diffusion Grasp Synthesis:** Rapid grasp generation with geometric, roll-isomorphism, and kinematic singularity filtering.
* **Closed-Loop Multi-Modal Interaction:** Real-time visual servoing, high-sensitivity custom tactile feedback for delicate grasping, and wrist-mounted force-torque touchdown detection.
* **Rapid Deployment:** Modular architecture that can be deployed on site and calibrated in a minimal amount of time.

To set up a local copy of the repository, see [Getting the Code](#getting-the-code).


If you have any questions, please reach out to the project maintainers:
* [Nadezhda Varzonova](mailto:varzonova.nadezhda@gmail.com)
  
---

## Table of Contents
- [H2R Handover System - BrokenArm (RGMC @ ICRA 2026)](#h2r-handover-system---brokenarm-rgmc--icra-2026)
  - [Table of Contents](#table-of-contents)
  - [About the Competition](#about-the-competition)
    - [Competition Flow:](#competition-flow)
    - [Resources \& Media:](#resources--media)
  - [Team](#team)
    - [Support](#support)
    - [Supervisors](#supervisors)
  - [Repository Structure](#repository-structure)
  - [Robot Hardware \& Sensor Suite](#robot-hardware--sensor-suite)
  - [System Environment \& Computing Infrastructure](#system-environment--computing-infrastructure)
    - [System Environment](#system-environment)
    - [Computing Infrastructure](#computing-infrastructure)
  - [Getting the Code](#getting-the-code)
  - [Quick Start / Running the System on a Real Robot](#quick-start--running-the-system-on-a-real-robot)
  - [Running the System in Simulation](#running-the-system-in-simulation)
    - [1. Fast Mock / ROS 2 Simulation](#1-fast-mock--ros-2-simulation)
    - [2. Full Dynamic Digital Twin Simulation (`handover_sim`)](#2-full-dynamic-digital-twin-simulation-handover_sim)
  - [External Dependencies: GraspGen Setup](#external-dependencies-graspgen-setup)
  - [External Dependencies: MoveIt 2 Setup](#external-dependencies-moveit-2-setup)
  - [Building Packages](#building-packages)
    - [Camera Calibration](#camera-calibration)
    - [Hardware \& Acceleration Requirements](#hardware--acceleration-requirements)
  - [SAM3 segmentation model](#sam3-segmentation-model)
  - [Roadmap \& Refactoring](#roadmap--refactoring)
  - [Acknowledgements \& Third-Party Credits](#acknowledgements--third-party-credits)
  - [Sponsorship](#sponsorship)

---

## About the Competition

The [RGMC 2026](https://sites.google.com/view/rgmcomp) focuses on advancing dynamic and generalizable robotic manipulation in human-centered environments.

### Competition Flow:
1. **Qualification Phase (CORSMAL):** Online benchmark phase requiring perception and physical reasoning under sensory ambiguity. Protocol can be found [here](https://corsmal.github.io/benchmark.html). Qualification results can be found [here](https://corsmal.github.io/events/rgmc/icra2026/leaderboards.html#corsmal-overall).
2. **On-Site Competition (ICRA 2026, Vienna):** Physical deployment on target robotic hardware, executing real-time human-to-robot handovers under varying constraints and dynamic conditions. Results and final rankings can be found [here](https://corsmal.github.io/events/rgmc/icra2026/leaderboards.html).

### Resources & Media:
* Project Report (under `BrokenArm`): [Read our Technical Report (PDF)](https://corsmal.github.io/events/rgmc/icra2026/teams.html)
* Video Demonstration: [Watch on YouTube](https://www.youtube.com/watch?v=0MkqkYvmN5c)

---

## Team

Team **BrokenArm**, [SAT group](https://www.aau.at/en/smart-systems-technologies/sensors-and-actuators/), [University of Klagenfurt](https://www.aau.at/en/):

* Nadezhda Varzonova (Team/Tech Lead)
* Daniel Bugelnig
* Valerian Pichler
* Markus Kräuter
* Christoph Rauter

The team consisted of 2 Bachelor students (Nadezhda, Markus) and 3 Master students (Daniel, Valerian, Christoph) at the time of the competition.

### Support
* Amon Peter Ferreira-Weratschnig (Technical support)
* Tobias Mitterer (Firmware support, on-site testing assistance)
* Barnaba Ubezio (Qualification assistance)

### Supervisors
* Hubert Zangl (lab head, main supervisor)
* Serkan Ergun (secondary supervisor)

---

## Repository Structure

```shell
├── handover_sim/                        # Simulation environment for testing handovers (Isaac Sim 4.5 / Isaac Lab)
├── utils/                               # Core modules, workspaces, and drivers
│   ├── config.py                        # Global configuration variables
│   ├── easy_handeye2_ws/                # Hand-eye calibration workspace
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
│   └── yolo-world/                      # Real-time object segmentation and detection
└── README.md
```

---

## Robot Hardware & Sensor Suite

The physical setup is built on standard industrial hardware augmented with heterogeneous sensing modalities:

* **Robot Manipulator:** Universal Robots **UR7e** 6-DOF industrial collaborative manipulator, offering reach and payload suitable for dynamic handovers. It connects directly to the motion workstation via a dedicated 1 Gbps hardware Ethernet link.
* **End-Effector (Gripper):** Robotiq **2F-140** parallel-jaw gripper, selected for its large 140 mm stroke, enabling variable-sized object acquisition and clean integration with grasp generation models.
* **Visual Perception (Dual Cameras):** Two static **Intel RealSense D435i** cameras monitoring the visual workspace, streaming RGB-D frames at 640×480 resolution at 30 FPS. Camera-to-robot extrinsics are calibrated using `easy_handeye2` via an end-effector calibration board (0.5–2 cm accuracy).
* **Force-Torque Sensor:** Wrist-mounted Robotiq **FT 300-S** Force-Torque Sensor, utilized for touchdown detection during object placement and real-time contact force monitoring.
* **Custom Tactile Finger Arrays:** Custom-built, barometric-pressure-sensor-based tactile sensors developed in the SAT lab by the Team Lead and the technical team. Mounted on the gripper fingers, each sensor features a 7-taxel grid streaming normal force measurements over Bluetooth Low Energy (BLE) at up to 60 FPS per finger with high sensitivity for delicate grasping.
  > **Note:** The tactile workspace (`utils/tactile_ws`) is designed modularly—any standard tactile sensor can be interfaced. For technical specifications, schematics, or inquiries regarding the lab's custom tactile sensor, please contact the SAT lab.

---

## System Environment & Computing Infrastructure

### System Environment
* **Operating System:** Ubuntu Linux 22.04 LTS
* **ROS 2 Distribution:** ROS 2 Humble Hawksbill
* **DDS Backbone:** **Eclipse Cyclone DDS** (`rmw_cyclonedds_cpp`) is used as the ROS 2 communication backbone across the network link to guarantee low-latency real-time performance.

### Computing Infrastructure
To guarantee strict real-time execution, the computational workload is bifurcated across two workstations, connected via a dedicated 2.5 Gbps Ethernet link:
* **Vision Pipeline Workstation:** Equipped with an AMD Ryzen 9 9950X 16-Core Processor and an NVIDIA RTX PRO 6000 Blackwell GPU (96 GB VRAM), executing object detection (YOLOE), SAM 3 segmentation, MediaPipe hand tracking, 3D object tracking, physical property estimation, and GraspGen synthesis.
* **Motion Planning Workstation:** Powered by an AMD Ryzen 9 5950X 16-Core Processor, exclusively reserved for real-time trajectory generation, MoveIt 2 Servo control, and direct hardware communication with the UR7e controller (1 Gbps link).

Both components are unified in this monorepo on the `dev` branch, allowing the complete architecture to be maintained together while running distributed across workstations or locally in simulation.

> **Note:** Experimental features developed in a Bachelor's thesis based on this system are on the `nad_thesis` branch. To learn about them, refer to the [thesis manuscript](https://drive.google.com/file/d/17Dt7FN3H-CJDMB9CM2PRGcGUX0tk9Wqs/view?usp=sharing).

---

## Getting the Code

Large files (meshes, simulation assets, model weights) are stored with [Git LFS](https://git-lfs.com), and third-party packages are included as Git submodules. Install Git LFS before cloning, then clone with submodules:
```bash
git lfs install
git clone --recurse-submodules https://github.com/aau-sat/RGMC-2026.git ~/rgmc-2026
```
The setup scripts expect the repository at `~/rgmc-2026`. If you cloned without `--recurse-submodules`, run `git submodule update --init --recursive` inside the repository.

---

## Quick Start / Running the System on a Real Robot

1. **Calibrate the Cameras:** Calibrate the cameras and refine calibration using the provided calibration tools in `utils/easy_handeye2_ws/`.
2. **Launch Perception (Vision Workstation):**
   In one terminal, run the following commands to start the persistent services which don't need to be restarted for every handover:
   ```bash
   cd utils/visual_pipeline_setup
   sh service_setup_tmux.sh
   ```
   In another terminal, launch the non-persistent perception scripts which need to be restarted for every handover:
   ```bash
   cd utils/visual_pipeline_setup
   sh vision_setup_tmux.sh
   ```
   Note that `tmux` sessions are used for the setup, so familiarize yourself with keyboard shortcuts for switching between windows etc. These setup scripts orchestrate the active nodes for the handover perception pipeline (segmentation, grasp generation service, VLM server, and camera streams).
   > **Note:** These setup scripts **show exactly which scripts are relevant for the handover pipeline**. This way you will not be lost in the multitude of scripts and `python` environments in the repository.
3. **Launch Motion Execution (Motion Workstation):**
    In one terminal, run the following commands to start the persistent robot setup scripts which launch drivers and start connection with the real robot. This `tmux` session doesn't need to be restarted for every handover:
   ```bash
   cd utils/ur
   sh ur_setup_tmux.sh
   ```
   In another terminal, launch the non-persistent motion execution scripts which need to be restarted for every handover:
   ```bash
   cd utils/ur
   sh moveit_setup_tmux.sh real
   ```
## Running the System in Simulation

The repository provides two simulation pathways:

### 1. Fast Mock / ROS 2 Simulation
To launch the robot motion pipeline using MoveIt 2 mock components (without needing physical hardware):
```bash
cd utils/ur
sh moveit_setup_tmux.sh
```
This starts the MoveIt 2 planning scene and mock controller stack.

### 2. Full Dynamic Digital Twin Simulation (`handover_sim`)
For high-fidelity simulation including human hand motion trajectories, dynamic object physics, and camera perception streaming into ROS 2 topics:
* See [`handover_sim/README.md`](./handover_sim/README.md) for full setup instructions using **NVIDIA Isaac Sim 4.5.0** and **Isaac Lab 1.x**.
* To launch the complete handover stage:
  ```bash
  source ~/IsaacLab/env_isaaclab.sh
  cd handover_sim/test_scripts
  ~/IsaacLab/isaaclab.sh -p test_stage.py
  ```
* For topic-based ROS 2 and MoveIt 2 closed-loop containerized control in Isaac Sim, refer to [`handover_sim/moveit_integration/ur10e_2f140_topic_based_ros2_control`](./handover_sim/moveit_integration/ur10e_2f140_topic_based_ros2_control/README.md).

---

## External Dependencies: GraspGen Setup

The grasp generation module utilizes [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) for deep-learning-based grasp pose synthesis. Due to its dedicated PyTorch environment and dependencies, GraspGen is expected as an external workspace at `~/packages/GraspGen/GraspGen`.

To set up GraspGen for the vision pipeline:
1. **Clone the repository:**
   ```bash
   mkdir -p ~/packages/GraspGen
   git clone https://github.com/NVlabs/GraspGen.git ~/packages/GraspGen/GraspGen
   ```
2. **Set up the Conda environment:**
   Follow the installation guide in [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) to create and activate the `graspgen` conda environment, and download the model checkpoints (including the Robotiq 2F-140 configuration).
3. **Deploy the custom three-stage grasp service script:**
   Copy the three-stage service script from this repository into the GraspGen `scripts/` directory:
   ```bash
   cp ~/rgmc-2026/utils/grasp_generation/scripts/grasp_service_three_stage.py ~/packages/GraspGen/GraspGen/scripts/
   ```
   Ensure the checkpoint file `models/checkpoints/graspgen_robotiq_2f_140.yml` is present in your GraspGen workspace.

Once configured, the grasp service and Meshcat visualization server are launched automatically by `utils/visual_pipeline_setup/service_setup_tmux.sh`.

---

## External Dependencies: MoveIt 2 Setup

The motion control module relies on **MoveIt 2** built and sourced from an external workspace:
1. Build MoveIt 2 from source in a separate ROS 2 workspace with the `humble` branch selected.
2. Source the MoveIt 2 workspace prior to executing robot motion scripts:
   ```bash
   source <path_to_moveit2_ws>/install/setup.bash
   ```
3. **Thesis Branch Compatibility (`nad_thesis`):** For the thesis extensions on the `nad_thesis` branch, the MoveIt 2 version corresponds exactly to the validated release tag:
   * [`thesis-ur7e-validated`](https://github.com/Incomprehensible/moveit2/releases/tag/thesis-ur7e-validated)

---

## Building Packages

Note that some `/utils` ROS2 packages need to be built with `colcon`.
The system-active packages that require building are contained in:
* `utils/ur`
* `utils/pose_tracking/depth_segmentation_tracking`
* `utils/tactile_ws`
* `utils/yolo-world`
* `utils/easy_handeye2_ws`
* `utils/ros2_ws`
* `utils/simple_ur_code/moveit_stuff`


### Camera Calibration

For eye-on-base calibration, `easy_handeye2` package is used. Refer to the [repository](https://github.com/marcoesposito1988/easy_handeye2)'s build instructions.

### Hardware & Acceleration Requirements
* **GPU Compute:** Perception modules (SAM 3, YOLO-World, InstantMesh, GraspGen) require an NVIDIA GPU with CUDA support. Ensure your NVIDIA driver and CUDA version match your PyTorch installation.
* **Conda Environments:** Various vision services utilize dedicated Conda / Python virtual environments (e.g., `sam3`, `graspgen`, `llm`) to prevent dependency conflicts, as launched in `service_setup_tmux.sh`.

---

## SAM3 segmentation model

Note that `sam3` requires obtaining the credentials from the HuggingFace website. If repository is cloned, follow the steps in official SAM3 repository.

---

## Roadmap & Refactoring

- [ ] **Dependency Packaging:** Integrate external submodules/libraries directly via Git submodules or native package management to eliminate manual copying.
- [ ] **Automated CI/CD:** Add simulation-in-the-loop tests for the full handover pipeline.

---

## Acknowledgements & Third-Party Credits

The H2R Handover System incorporates, adapts, or builds upon open-source codebases and research datasets:
* **[GENH2R Benchmark](https://github.com/GenH2R/GenH2R):** Dynamic handover motion datasets used in simulation.
* **[MANO Hand Model](https://mano.is.tue.mpg.de/):** 3D articulated human hand kinematic models (Max Planck Institute for Intelligent Systems).
* **[YCB Benchmarks](https://www.ycbbenchmarks.com/):** 3D object models and physical property benchmarks.
* **[UR_Robotiq](https://git.cps.unileoben.ac.at/Niko/UR_Robotiq) & [topic_based_ros2_control](https://gitlab.com/quentin.deyna/ur10e_2f140_topic_based_ros2_control):** ROS 2 control and MoveIt 2 integration by Quentin Deyna, Nikolas Schwarz (Uni Leoben), and PickNik Robotics.
* **[SAM 3](https://github.com/facebookresearch/sam3) & [Segment Anything 3D](https://github.com/facebookresearch/sam-3d-objects):** Zero-shot segmentation models by Meta AI Research.
* **[YOLO-World](https://github.com/AILab-CVC/YOLO-World):** Real-time open-vocabulary object detection.
* **[InstantMesh](https://github.com/TencentARC/InstantMesh):** Rapid 3D mesh generation from single images (Tencent ARC).
* **[GraspGen](https://github.com/NVlabs/GraspGen):** Deep learning grasp synthesis models.


## Sponsorship
The project has been founded and sponsored by the University of Klagenfurt, Department of Smart System Technologies, Sensors, Actuators & Modular Robotics group. This project is owned by SAT group.