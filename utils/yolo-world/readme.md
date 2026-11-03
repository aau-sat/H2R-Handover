# YOLO-based node
Taking stream from realme camera (or rosbag or anything streaming on the right topic).

**Topic listening**: `camera/camera/color/image_raw`

**Topic outputting**: `/yolo/detections`

## Setup
The order matters:
1. Setup conda environment and install `requirements.txt`
    ```sh
    # assuming you're in utils/yolo-world where this file is located.
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
    # YOLO direct
    ros2 run yolo_world yolo_world_node
    # With Roboflow and Inference
    ros2 run yolo_world yolo_world_roboflow_node
    ```


PS. If you wanna show the stream with `rqt_image_view`, you need to install the correct packages. See the wiki for more info.