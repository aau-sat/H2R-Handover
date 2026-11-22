# Handover Simulation Environment (`handover_sim`)

This directory contains the simulation framework and digital twin assets developed for the **Human-to-Robot (H2R) Handover System** presented at **RGMC 2026 (ICRA 2026)**. It provides Isaac Sim/Isaac Lab scenes for simulating dynamic human handover motions, camera perception streams, and robot manipulator execution.

---

## Table of Contents
- [Handover Simulation Environment (`handover_sim`)](#handover-simulation-environment-handover_sim)
  - [Table of Contents](#table-of-contents)
  - [Platform \& Version Requirements](#platform--version-requirements)
  - [Directory Structure](#directory-structure)
  - [Getting Started](#getting-started)
    - [1. Sourcing Isaac Lab](#1-sourcing-isaac-lab)
    - [2. Testing Hand Motions (Isolated Simulation)](#2-testing-hand-motions-isolated-simulation)
    - [3. Testing the Full Handover Stage](#3-testing-the-full-handover-stage)
  - [MoveIt 2 \& Topic-Based ROS 2 Control (DevContainer)](#moveit-2--topic-based-ros-2-control-devcontainer)
  - [Configuration Reference](#configuration-reference)
  - [Known Limitations \& Technical Notes](#known-limitations--technical-notes)
  - [Third-Party Credits \& Academic Citations](#third-party-credits--academic-citations)

---

## Platform & Version Requirements

> [!IMPORTANT]
> **Compatibility Notice:** The simulation scripts and USD stage assets in this directory were developed and validated specifically against:
> - **NVIDIA Isaac Sim: `4.5.0`**
> - **NVIDIA Isaac Lab: `1.x`** (using the `isaacsim_4_5` application profiles)
> - **ROS 2: Humble Hawksbill**
> - **Python: 3.10** (within Isaac Lab Python environment)

If you are using newer releases (e.g., Isaac Lab >= 2.0 / Isaac Sim 5.x), package structures, experience app templates, and physics schema bindings may have changed. Please refer to NVIDIA's migration guides if upgrading.

---

## Directory Structure

```shell
handover_sim/
├── assets/
│   ├── motions/
│   │   └── genh2r/              # Motion trajectories (.npz) and scene metadata (scenes_data.csv)
│   ├── urdf/                    # URDF descriptions and meshes for hands, stages, and objects
│   └── usd/                     # USD stages, robots, MANO hand models, and YCB object models
│       ├── hand/                # MANO realistic hand models (mano.usd, mano_coarse.usd, mano_fine.usd)
│       ├── objects/             # Ported YCB object meshes with colliders and physics
│       ├── robot/               # UR10e + Robotiq gripper stages (stage2_mock.usd, stage_room.usd)
│       └── stage/               # Room and environmental meshes
├── moveit_integration/
│   └── ur10e_2f140_topic_based_ros2_control/   # Dual DevContainer ROS 2 + Isaac Sim MoveIt bridge
└── test_scripts/
    ├── config_motion/
    │   └── config_hand_motion.py # Central configuration (stages, hand models, stiffness, topics)
    ├── export_scene_data.py      # Scene dataset extraction script
    ├── test_stage.py             # Spawns full stage (robot + hand + object + wrist camera)
    ├── test_new_hand.py          # Isolated hand & object trajectory playback
    └── README.md                 # Test scripts execution notes
```

---

## Getting Started

### 1. Sourcing Isaac Lab

Before executing any simulation script, activate your Isaac Lab environment:

```bash
source ~/IsaacLab/env_isaaclab.sh
```

### 2. Testing Hand Motions (Isolated Simulation)

To verify hand trajectories and object grasp interaction without spawning the robot:

```bash
cd test_scripts
~/IsaacLab/isaaclab.sh -p test_new_hand.py
```
*(Or invoke via the environment's python: `python test_new_hand.py`)*

### 3. Testing the Full Handover Stage

To spawn the complete handover environment (UR10 arm, Robotiq 2F-85/140 gripper, animated human hand, object, and wrist-mounted RGB-D camera publishing to ROS 2 topics):

```bash
cd test_scripts
~/IsaacLab/isaaclab.sh -p test_stage.py
```

While running, `test_stage.py` streams camera feeds to standard ROS 2 topics and listens for robot joint commands.

---

## MoveIt 2 & Topic-Based ROS 2 Control (DevContainer)

For closed-loop trajectory planning and MoveIt 2 control connected directly to Isaac Sim:
- Navigate to `moveit_integration/ur10e_2f140_topic_based_ros2_control/`.
- This submodule provides a **Dual DevContainer** environment:
  1. **Isaac Sim DevContainer** (`.devcontainer/isaac/`): Runs GPU-accelerated simulation with the ROS 2 bridge.
  2. **ROS 2 DevContainer** (`.devcontainer/ros2/`): Compiles MoveIt 2 and robot drivers for hardware or mock simulation.
- Refer to [`ur10e_2f140_topic_based_ros2_control/README.md`](./moveit_integration/ur10e_2f140_topic_based_ros2_control/README.md) for full container build and startup instructions.

---

## Configuration Reference

Key simulation parameters are defined in [`test_scripts/config_motion/config_hand_motion.py`](./test_scripts/config_motion/config_hand_motion.py):

* **Isaac Experience**:
  ```python
  ISAAC_EXPERIENCE = "isaaclab.python.rendering.kit"  # Optimal rendering speed for multi-camera streams
  # Alternatives: "isaacsim.exp.full.kit", "isaacsim.exp.base.kit"
  ```
* **Stage Fidelity**: `StageConfig.BASIC` (`stage2_mock.usd`), `StageConfig.COLORED` (`stage_colored.usd`), or `StageConfig.DETAILED` (`stage_room.usd`).
* **Hand Models**: `HandConfig.MANO_BASIC`, `HandConfig.MANO_COARSE` (finger collisions disabled for thin objects), or `HandConfig.MANO_FINE` (full mesh colliders).
* **Joint Control Topic**: `JointsROS2CommandTopic.MOCK_HARDWARE` (`joint_states`) or `MANUAL_JOINTS_CONTROL` (`joint_command`).

---

## Known Limitations & Technical Notes

1. **Left Hand Collision Density:** Due to density parameters in the left hand mesh, a low-mass compensation (`LEFT_MASS = 0.00001`) is applied in `config_hand_motion.py` to prevent table fall-through.
2. **Camera IMU Sensor:** The onboard IMU sensor in the RealSense simulation model was disabled because its driver conflicts with the Isaac Lab PyTorch tensor backend.
3. **Reactive Stopping:** Recorded motion sequences continue through completion. For reactive robot handovers, the trajectory reader should be gated by proximity or tactile feedback triggers.
4. **Collision Tuning for Fine Objects:** For slender objects (such as mugs or thin handles), use `mano_coarse.usd` to prevent collision artifacts.

---

## Third-Party Credits & Academic Citations

This simulation module incorporates assets, trajectories, and frameworks from the following research projects:

1. **GENH2R Motion Dataset:**
   - Trajectory data in `assets/motions/genh2r/` is derived from the **GENH2R** benchmark.
   - *Citation:*
     ```bibtex
     @inproceedings{wang2024genh2r,
       title={GENH2R: Learning Generalizable Human-to-Robot Handover with Point Cloud Video},
       author={Wang, Zicheng and Chen, Bo and Wang, He and others},
       booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
       year={2024}
     }
     ```

2. **MANO Hand Model:**
   - Hand kinematic topologies and USD models in `assets/usd/hand/` and `assets/urdf/hand/` are based on the MANO model by the Max Planck Institute for Intelligent Systems.
   - *Citation:*
     ```bibtex
     @article{romero2017embodied,
       title={Embodied Hands: Modeling and Capturing Hands and Bodies Together},
       author={Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
       journal={ACM Transactions on Graphics (TOG)},
       volume={36},
       number={6},
       year={2017}
     }
     ```

3. **YCB Object Set:**
   - 3D models in `assets/usd/objects/ycb_ported/` originate from the YCB Object and Model Set.
   - *Citation:*
     ```bibtex
     @article{calli2015ycb,
       title={The YCB object and Model Set: Towards Common Benchmarks for Manipulation Research},
       author={Calli, Berk and Walsman, Aaron and Singh, Arjun and Srinivasa, Siddhartha and Abbeel, Pieter and Dollar, Aaron M.},
       journal={IEEE Transactions on Robotics},
       volume={31},
       number={5},
       pages={1206--1218},
       year={2015}
     }
     ```

4. **UR10e 2F-140 Topic-Based ROS 2 Control:**
   - The integration in `moveit_integration/ur10e_2f140_topic_based_ros2_control` is adapted from [Quentin Deyna's GitLab repository](https://gitlab.com/quentin.deyna/ur10e_2f140_topic_based_ros2_control), building upon [Nikolas Schwarz's `UR_Robotiq` (Uni Leoben)](https://git.cps.unileoben.ac.at/Niko/UR_Robotiq) and [PickNik Robotics' `topic_based_ros2_control`](https://github.com/PickNikRobotics/topic_based_ros2_control).

5. **NVIDIA Isaac Sim & Isaac Lab:**
   - Simulation engine powered by [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).
