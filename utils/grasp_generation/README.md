# Grasp Generator Node

ROS2 node that listens for handover detection and generates grasp poses using a Unix socket-based grasp generation service.

## Overview

This node integrates handover detection with grasp generation by:
1. Monitoring `/handover_detected` topic for handover events
2. Capturing RGB, depth, and mask images from RealSense camera
3. Sending grasp generation requests via Unix socket to the grasp server
4. Publishing the resulting gripper pose to `/gripper_goal_pose`

## Dependencies

### ROS2 Packages
- `rclpy`
- `std_msgs`
- `sensor_msgs`
- `geometry_msgs`
- `cv_bridge`

### Python Packages
```bash
pip install -r requirements.txt
```

## Topics

### Subscribed Topics
- `/handover_detected` (std_msgs/Bool): Trigger for grasp generation
- `/camera/aligned_depth_to_color/image_raw` (sensor_msgs/Image): Depth image from RealSense
- `/camera/color/image_raw` (sensor_msgs/Image): RGB image from RealSense (stored for future use)

### Published Topics
- `/gripper_goal_pose` (geometry_msgs/PoseStamped): Computed grasp pose

## Usage

### Prerequisites
1. Ensure the grasp generation server is running and listening on `/tmp/graspgen` Unix socket
2. RealSense camera node is publishing depth images
3. Segmentation/mask node is publishing object masks

### Running the Node
```bash
# Make the script executable
chmod +x grasp_generator_node.py

# Run the node
ros2 run <package_name> grasp_generator_node.py

# Or run directly
python3 grasp_generator_node.py
```

## Communication Protocol

The node communicates with the grasp generation server using the following protocol:

### Request (Client → Server)
1. **Header** (12 bytes): [height, width, depth_dtype_size] as int32
2. **Depth Data**: H×W float32 array (in meters)
3. **Mask Data**: H×W uint8 array

### Response (Server → Client)
- **Grasp Matrix** (64 bytes): 4×4 float32 transformation matrix

The transformation matrix format:
```
[[ R11  R12  R13  tx ]
 [ R21  R22  R23  ty ]
 [ R31  R32  R33  tz ]
 [ 0    0    0    1  ]]
```

Where:
- R = 3×3 rotation matrix (converted to quaternion for ROS message)
- t = 3×1 translation vector (x, y, z position in meters)

## Configuration

### Frame ID
The default frame for the published pose is `camera_color_optical_frame`. Modify line 227 in the code if your setup uses a different frame:

```python
pose_msg.header.frame_id = 'your_frame_id'
```

### Socket Path
The Unix socket path is `/tmp/graspgen`. Change `SOCKET_PATH` constant if your server uses a different path.

### Topic Names
Adjust topic names in the `__init__` method if your setup uses different naming conventions.

## Error Handling

The node handles various error conditions:
- Missing depth or mask images
- Dimension mismatches between depth and mask
- Grasp server not running
- Socket communication errors
- Invalid grasp responses (zero matrix)

All errors are logged using the ROS2 logging system.

## Example Output

```
[INFO] [grasp_generator_node]: Grasp Generator Node initialized
[INFO] [grasp_generator_node]: Waiting for handover detection...
[INFO] [grasp_generator_node]: Handover detected! Generating grasp...
[INFO] [grasp_generator_node]: Received grasp matrix from server
[INFO] [grasp_generator_node]: Grasp pose published - Position: [-0.3499, -0.2603, 0.9984]
```

## Integration Notes

- The node prevents overlapping grasp requests using a `processing` flag
- Depth values are automatically converted from millimeters to meters
- The mask must be a grayscale (mono8) image with the object region marked
- RGB images are stored but not currently used (available for future extensions)
