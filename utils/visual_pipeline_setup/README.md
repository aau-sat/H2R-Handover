# RGMC Visual Pipeline Tmux Setup


### Window 0: "Vision Workspace_0" (Main Pipeline - 2x2 Grid)

This window contains the core visual perception components:

| Top-Left | Top-Right |
|----------|-----------|
| **FoundationPose** | **Object Detection (YOLO)** |
| Pose tracking from FoundationPose conda environment | YOLO-based handover detection |
| Activates: `foundationpose_ros` conda env | Activates: `env_isaaclab` conda env |
| Runs: `pose_tracker_node_scaling.py` | Runs: `ros2 run handover_detection test_yolo_rgb` |
| | |
| **Sam3 Segmentation** | **Mesh Generation (InstantMesh)** |
| SAM-based object segmentation service | Real-time 3D mesh generation |
| Activates: `sam3` conda env | Activates: `instantmesh` conda env |
| Runs: `segmentation_service.py` | Runs: `mesh_generator_node.py` |

### Window 1: "Vision_Workspace_1" (Extended Processing - 2x2 Grid)

Secondary processing and decision-making components:

| Top-Left | Top-Right |
|----------|-----------|
| **Stop Movement Detection** | **Grasp Coordinator** |
| Detects when object movement stops | Coordinates grasping strategy |
| Activates: `env_isaaclab` conda env | Activates: `env_isaaclab` conda env |
| Runs: `stop_movement_detection_node.py` | Runs: `grasp_coordinator_node_mirror.py` |
| | |
| **SAM Stream Bridge** | **SAM Stream Server** |
| Bridges SAM segmentation streams | Serves SAM segmentation streams |
| Activates: `env_isaaclab` conda env | Activates: `sam3` conda env |
| Runs: `sam_stream_bridge.py` | Runs: `sam_stream_server.py` |

### Window 2: "Topic Listener" (ROS2 Debugging - 2x2 Grid)

Monitors ROS2 topics for debugging and verification:

| Top-Left | Top-Right |
|----------|-----------|
| **Echo /detected_objects** | **Echo /goal_pose** |
| Displays detected object messages | Displays goal pose updates |
| | |
| **Echo /object_ready_to_grasp** | **Topic List** |
| Monitors object readiness | Lists all active ROS2 topics |

### Window 3: "utils" (Utilities - 2x2 Grid)

System utilities and development tools:

| Top-Left | Top-Right |
|----------|-----------|
| **RealSense Camera Launch** | **Code Editor** |
| Starts RealSense D435 camera with aligned streams | Opens VSCode for development |
| Command: `rs_launch.py` with RGBD + sync | Runs: `code .` |
| | |
| **GPU Monitor** | *(Empty)* |
| Real-time GPU usage monitoring | Available for manual use |
| Command: `watch -n 1 nvidia-smi` | |

## Starting the Session

```bash
bash setup_tmux_vp.sh
```

## Tmux Controls

### Basic Navigation

| Command | Shortcut | Action |
|---------|----------|--------|
| Switch windows | `Ctrl+B` then `n` / `p` | Next / Previous window |
| Switch windows by number | `Ctrl+B` then `0-3` | Jump to window 0, 1, 2, or 3 |
| Switch panes | `Ctrl+B` then arrow keys | Navigate between panes |
| Cycle panes | `Ctrl+B` then `o` | Move to next pane |

### Pane Management

| Command | Shortcut | Action |
|---------|----------|--------|
| Zoom into pane | `Ctrl+B` then `z` | Fullscreen current pane (press again to exit) |
| Kill pane | `Ctrl+B` then `x` | Close current pane |
| Break pane into window | `Ctrl+B` then `!` | Convert pane to new window |

### Window Management

| Command | Action |
|---------|--------|
| `Ctrl+B` then `c` | Create new window |
| `Ctrl+B` then `w` | List all windows |
| `Ctrl+B` then `,` | Rename current window |
| `Ctrl+B` then `&` | Kill current window |

### Session Management

| Command | Action |
|---------|--------|
| `Ctrl+B` then `d` | Detach from session |
| `tmux list-sessions` | List all tmux sessions |
| `tmux attach-session -t RGMC` | Reattach to RGMC session |
| `tmux kill-session -t RGMC` | Terminate RGMC session |

### Copy/Paste Mode

| Command | Action |
|---------|--------|
| `Ctrl+B` then `[` | Enter copy mode |
| `Space` | Start selection |
| `Enter` | Copy selection |
| `Ctrl+B` then `]` | Paste buffer |

### Terminate

`Ctrl+B` then `:kill-session` and enter