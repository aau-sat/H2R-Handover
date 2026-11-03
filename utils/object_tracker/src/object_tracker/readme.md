# Depth-Clipper Node
Taking an rgb and depth stream from a realsense camera as well as a length parameter, it replaces further away pixels with gray after some post-processing.

## Topics
**Parametrers**:
- `camera_topic_rgb`: defaulting to `/camera/left/color/image_raw`
- `camera_topic_d`: defaulting to `/camera/left/aligned_depth_to_color/image_raw`
- `depth_clipped_topic`: defaulting to `/depth_clipped`
- `depth_threshold_mm`: defaulting to `2500` [mm]

The node takes the provided `camera_topic_rgb` and `camera_topic_d` topics, and publishes the processed stream to `depth_clipped_topic`.

## Setup
```sh
# left
ros2 run object_tracker depth_clipper_node --ros-args -p depth_clipped_topic:='/depth/clipped/left' -p camera_topic_rgb:='/camera/left/color/image_raw' -p camera_topic_d:='/camera/left/aligned_depth_to_color/image_raw'

# right
ros2 run object_tracker depth_clipper_node --ros-args -p depth_clipped_topic:='/depth/clipped/right' -p camera_topic_rgb:='/camera/right/color/image_raw' -p camera_topic_d:='/camera/right/aligned_depth_to_color/image_raw'
```

# Object-Tracker Node 
Taking RGB streams from one or both cameras and tracking objects using [YOLO](https://docs.ultralytics.com/models/yolo11/) or [YOLO-World](https://docs.ultralytics.com/models/yolo-world/), using logic to map the detections. Also optionally provides information about the hand position and cropped images of the detections.

`object_tracker_node` uses `yolo11x` when parameter `use_yolo_world` is set to `False` and `yolov8x-world` with said parameter set to `True` (default).

## Parameters
This node makes use of many parameters:
- **`use_yolo_world`** (default: `True`): 
    
    Defines which model to load. 
    When `True`, the node uses [YOLO-World](https://docs.ultralytics.com/models/yolo-world/), else [YOLO](https://docs.ultralytics.com/models/yolo11/).

- **`target_classes`**:
    Defines which classes the loaded model targets, which consist of IDs for YOLO and strings for YOLO-World. 
    
    The default depends on the model loaded with `use_yolo_world`:
    - **YOLO** (`use_yolo_world:=False`): 

        `[39, 40, 41, 45, 46, 47, 48, 49, 53, 54, 73, 75]`

    - **YOLO-World** (`use_yolo_world:=True`)

        `["glas", "cup", "mug", "bottle", "plate", "bowl", "food container"]`

- **`max_coast_frames`** (default: `10`): Number of frames to *hold* detections, e.g. to ignore frames without detections.

- **`publish_crops`** (default: `False`): Should cropped frames be published?

    These frames are published to `/object_tracker/crop/left` and `/object_tracker/crop/respectively`. If only one camera is supplying a stream, only the respective crop topic will have a stream.

- **`crop_padding`** (default: `20`): Padding to put around the bounding box for pushing cropped images (only if `publish_crops:=True`). Works like the single-value *padding* declaration in HTML: 20px are added on either side around the bounding box, effectively making the image bigger by 40px.

- **`hand_detection`** (default: `True`): Should hand detection be performed?

    The hand detection renders a dot on the debug view (only if `debug_topics:=True`) and publishes properties (`active_hand_center` and `active_palm_center`) to the respective `/object_tracker/detected_objects` JSON output. If this parameter is `False`, the properties will still exist in JSON but will be `null`.

- **`debug_topics`** (default: `True`): Pushes the debug topics consisting of the annotated frames for each camera (`object_tracker/det_left` and `object_tracker/det_right`) respectively. If only one stream is provided, only the corresponding detection topic will have data. If `hand_detection:=True`, the hand center will be drawn onto the debug frames as well. Furthermore, the model evaluation is set to *verbose*, printing inference information into the console.

- **`camera_topic_left` and `camera_topic_right`** (default: `' '`): See below.

## Topics
**Parametrized Input Topics**: 
- `camera_topic_left`: defaulting to `' '` (interpreted as undefined)
- `camera_topic_right`: defaulting to `' '` (interpreted as undefined)

**By default, no camera stream topics are provided!!!** 

This prevents errors when trying to pass an empty/unset parameter should only one stream be desired but both topics exist. Please declare the input topics using these parameters.
**Either one or both can be defined.**

**Output Topics**: 
- `/object_tracker/detected_objects`
- `/object_tracker/det_left` and `/object_tracker/det_right`: Camera frames with bounding boxes, classes, confidences and hand positions highlighted for easy debugging.
- `/object_tracker/crop/left` and `/object_tracker/crop/right`: Streams containing the detected objects cropped by their bounding boxes with a margin of `crop_padding` (parameter, defaulting to 20px).

## Setup
The order matters:
1. Setup conda environment and install `requirements.txt`
    ```sh
    # assuming you're in utils/object_tracker where this file is located.
    pip install -r requirements.txt
    ```
2. Setup gpu support for roboflow models:
    ```sh
    pip install inference-gpu
    ```
3. Manually install old `numpy` and ignore the error...
    ```sh
    pip install "numpy<2.0"
    ```

## Run
0. Compile
    ```sh
    colcon build
    ```
1. Source
    ```sh
    source install/setup.bash
    ```
2. Run
    ```sh
    ### YOLO World ###
    # Start node using YOLO World in debug mode with both cameras
    ros2 run object_tracker object_tracker_node --ros-args -p camera_topic_right:='/camera/right/color/image_raw' -p camera_topic_left:='/camera/left/color/image_raw' 

    # Start node using YOLO World without debug mode with both cameras
    ros2 run object_tracker object_tracker_node --ros-args -p camera_topic_right:='/camera/right/color/image_raw' -p camera_topic_left:='/camera/left/color/image_raw' -p debug_topics:=False

    ### YOLO ###
    # Start node using normal YOLO in debug mode with both cameras
    ros2 run object_tracker object_tracker_node --ros-args -p use_yolo_world:=False -p camera_topic_right:='/camera/right/color/image_raw' -p camera_topic_left:='/camera/left/color/image_raw' 

    # Start node using normal YOLO without debug mode with both cameras
    ros2 run object_tracker object_tracker_node --ros-args -p use_yolo_world:=False -p camera_topic_right:='/camera/right/color/image_raw' -p camera_topic_left:='/camera/left/color/image_raw' -p debug_topics:=False
    ```

Use `rviz2` for easy visualization!

Alternatively, but no longer recommended due to insane perfomance issues:
For `rqt_image_view`, you need to install the correct packages.

# Run all
Run depth-clipper nodes for each stream and call the object tracker node.
Maybe there will be a launch script, but for now do it yourself.
The assumptions are that the cameras are available at `/camera/left/...` and `/camera/right/...`.

```sh
# Left depth-clipper node
ros2 run object_tracker depth_clipper_node --ros-args -p depth_clipped_topic:='/depth/clipped/left' -p camera_topic_rgb:='/camera/left/color/image_raw' -p camera_topic_d:='/camera/left/aligned_depth_to_color/image_raw'

# Right depth-clipper node
ros2 run object_tracker depth_clipper_node --ros-args -p depth_clipped_topic:='/depth/clipped/right' -p camera_topic_rgb:='/camera/right/color/image_raw' -p camera_topic_d:='/camera/right/aligned_depth_to_color/image_raw'

# Run object tracker with depth-clipped topics
ros2 run object_tracker object_tracker_node --ros-args -p camera_topic_left:='/depth/clipped/left' -p camera_topic_right:='/depth/clipped/right'
```