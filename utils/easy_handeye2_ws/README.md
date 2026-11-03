# Calibration Visualization and Refinement

This package provides tools for visualizing and refining camera calibration results.

## How to Use
Ensure that you are on the `dev` branch (**Vision PC**) and have source the `python` environment for the `easy_handeye2_ws` workspace:
```bash
source ~/env_isaaclab/bin/activate
```

Make sure to locate the resulting calibration files in the `~/.ros2/easy_handeye2` root directory. There will be `left` and `right` camera calibration files. 
Original calibration files (e.g. `calibration_left_camera.calib`) have their own format, while refined calibration files (e.g. `left_refined_cam_to_base.yaml`) have a different format. Therefore, usually a special flag indicating refined calibration is needed when using the refined calibration files.

### Visualizing Calibration Results
Launch `realsense` cameras ros2 driver enabling the pointcloud topic output for each camera:
```bash
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=false enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 serial_no:=_346522071495 camera_name:=left pointcloud.enable:=true temporal_filter.enable:=true json_file_path:=/home/student-team/Downloads/HighDensityPreset.json
```

In second terminal:
```bash
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=false enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 serial_no:=_913522071225 camera_name:=right pointcloud.enable:=false temporal_filter.enable:=true json_file_path:=/home/student-team/Downloads/HighDensityPreset.json
```

In third terminal, launch `rviz2` to visualize the calibration results:
```bash
rviz2
```

Add the pointcloud topics to rviz2 visualization.
Alternatively, you can launch the simulated robot environment on the **Motion PC** and visualize the calibration results with pointclouds relative to the robot, which is more useful for debugging and measurements.

Launch the visualization script:
```bash
python visualize_cams_transforms.py
```

If you already refined calibration and want to visualize the refined results, you can launch the visualization script with the `--refined_first_to_base` or `--refined_second_to_base` flag depending on which camera calibration was refined. By convention first is `right` and second is `left`. For example, if you refined the left camera calibration, launch:
```bash
python visualize_cams_transforms.py --refined_second_to_base --refined_second_to_base_calibration left_refined_cam_to_base.yaml
```
This script will publish the transforms from the depth camera frames to base robot frame.

### Refining Calibration Results
Now, ensure that no camera transform publishing scripts are running (like the previous example). Launch the `rviz2` visualization to see the pointclouds.

To refine the calibration results, launch the refinement script. It has been changed multiple times so the parameter convention is a bit chaotic. The parameters you need to provide are `--first_to_base_calibration` and `--second_to_base_calibration` which can be either original calibration files for the first and second cameras or previously refined calibration files (in `.yaml` format). This way, you can continue refining iteratively and use both refined calibrations. 

Since this script has a weird convention, you choose which camera to refined by setting the `first` to either `left` or `right` and the other camera will be the `second`. I assume that the `first` camera transform will be refined because I don't remember now, but you can just confirm which camera transform is changing by visualizing the pointcloud outputs in `rviz2`. 

The `--file` parameter specifies the output file name.

To refine the single raw calibration, run the following command:
```bash
python refine_cam_transform_new.py --first_to_base_calibration /home/student-team/.ros2/easy_handeye2/calibrations/calibration_right_camera.calib  --second_to_base_calibration /home/student-team/.ros2/easy_handeye2/calibrations/calibration_left_camera.calib --file test_right_refined_cam_to_base.yaml
```

To pick up from the last calibration refinement, run the following command:
```bash
python refine_cam_transform_new.py --first_to_base_calibration right_refined_cam_to_base.yaml --second_to_base_calibration /home/student-team/.ros2/easy_handeye2/calibrations/calibration_left_camera.calib --file test_right_refined_cam_to_base.yaml
```
