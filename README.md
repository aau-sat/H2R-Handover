# H2R Handover System - BrokenArm (RGMC @ ICRA 2026)

Main codebase for the Human-to-Robot (H2R) Handover System developed by team **BrokenArm** from the [SAT group](https://www.aau.at/en/smart-systems-technologies/sensors-and-actuators/) at the [University of Klagenfurt](https://www.aau.at/en/), presented at the **RGMC 2026** at **ICRA 2026** in Vienna, Austria.

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
  - [Repository Structure](#repository-structure)
  - [Two-PC Setup](#two-pc-setup)
  - [Getting the Code](#getting-the-code)
  - [Quick Start / Running the System on a Real Robot](#quick-start--running-the-system-on-a-real-robot)
  - [Running the System in Simulation](#running-the-system-in-simulation)
  - [External Workspaces](#external-workspaces)
  - [Building packages](#building-packages)
  - [SAM3 segmentation model](#sam3-segmentation-model)
  - [Roadmap \& Refactoring](#roadmap--refactoring)

---

## About the Competition

The [RGMC 2026](https://sites.google.com/view/rgmcomp) focuses on advancing dynamic and generalizable robotic manipulation in human-centered environments.

### Competition Flow:
1. **Qualification Phase (CORSMAL):** Online benchmark phase requiring perception and physical reasoning under sensory ambiguity. Protocol can be found [here](https://corsmal.github.io/benchmark.html). Qualification results can be found [here](https://corsmal.github.io/events/rgmc/icra2026/qualification.html).
2. **On-Site Competition (ICRA 2026, Vienna):** Physical deployment on target robotic hardware, executing real-time human-to-robot handovers under varying constraints and dynamic conditions. Results and final rankings can be found [here](https://corsmal.github.io/events/rgmc/icra2026/competition.html).

### Resources & Media:
* Project Report: [Read our Technical Report (PDF)](https://corsmal.github.io/events/rgmc/icra2026/index.html)
* Video Demonstration: [Watch on YouTube](https://www.youtube.com/watch?v=0MkqkYvmN5c)

---

## Team

Team **BrokenArm**, [SAT group](https://www.aau.at/en/smart-systems-technologies/sensors-and-actuators/), [University of Klagenfurt](https://www.aau.at/en/):

* Nadezhda Varzonova
* Valerian Pichler
* Daniel Bugelnig
* Christoph Rauter
* Markus Kräuter

---

## Repository Structure

```shell
├── handover_sim/                        # legacy: Simulation environment for testing handovers
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

## Two-PC Setup

Due to deployment across two physical on-site machines with distinct code environments, the system is split into two parts:

* **Vision PC (Blackwell):** Vision pipeline.
* **Motion PC:** Motion control, physical connection to the robot.

During development, each part lived on its own branch (`dev` and `motion_planning`). This public repository is a single snapshot that merges the final state of both, so the code for both PCs is on one branch.

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
2. **Launch Perception (Vision PC):**
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
   Note that `tmux` sessions are used for the setup, so familiarize yourself with keyboard shortcuts for switching between windows etc.
   Also note that these setup scripts **show exactly which scripts are relevant for the handover pipeline**. This way you will not be lost in the multitude of scripts and `python` environments in the repository. For instructions of how to launch individual parts and how code works contact the members of [team BrokenArm](#team).
3. **Launch Motion Execution (Motion PC):**
    In one terminal, run the following commands to start the persistent, robot setup scripts which launch drivers and start connection with the real robot. This `tmux` session doesn't need to be restarted for every handover:
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
Perform the same steps as for the real robot until excluding step 3. 
Instead, do this:
```
cd utils/ur
sh moveit_setup_tmux.sh
```
This will not launch the real robot setup scripts and instead launch the simulation environment for testing handovers.

---

## External Workspaces

To run a visual pipeline, grasp generation scripts need to be launched. The grasp generation module depends on `GraspGen` model with its code environment. Due to legacy reasons, the `GraspGen` official repository was cloned and built outside of the main repository. The path to the external workspace can be found in visual pipeline start scripts. The path to the workspace (`~/packages/GraspGen/GraspGen`) is located on the Blackwell PC.

This repository only keeps and launches the client side of grasp generation module and keeps the current state of the grasp generation service script. The same service script is actually launched from the external workspace.

---

## Building packages

Note that some `/utils` ros2 packages need to be built with `colcon`.
The system-active packages that require building are:
* `utils/ur`
* `utils/pose_tracking/depth_segmentation_tracking`
* `utils/tactile_ws`
* `utils/yolo-world`
* `utils/easy_handeye2_ws`
* `utils/ros2_ws`
* `utils/simple_ur_code/moveit_stuff`

Moreover, visual pipeline is built for the **Blackwell PC** architecture. Many modules belonging to the pipeline utilize Nvidia GPU. The same build instructions are rarely applicable to the Nvidia GPUs from a different family.
The instructions used for building the visual pipeline packages can be found in `Anytype` - online note making software. Contact the team to use the team's account to access the notes.

---

## SAM3 segmentation model

Note that `sam3` requires obtaining the credentials from the HuggingFace website. If repository is cloned, follow the steps in official SAM3 repository.

---

## Roadmap & Refactoring

- [ ] **Monorepo Unification:** Merge `perception` and `planning` into a single modular architecture and upload it to dev->main branches.
- [ ] **Dependency Packaging:** Integrate external submodules/libraries directly via Git submodules or native package management to eliminate manual copying.
- [ ] **Automated CI/CD:** Add simulation-in-the-loop tests for the full handover pipeline.

