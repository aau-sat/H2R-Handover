## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Export RGB, IR1 and IR2              ##
#####################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import os

import yaml

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # Optional: enable infrared stream if needed
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 0.5  # 0.5 meters
clipping_distance = clipping_distance_in_meters / depth_scale

# Get the absolute path to the subfolder
script_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_rgb = os.path.join(script_dir, "out1/rgb")
subfolder_ir1 = os.path.join(script_dir, "out1/ir1")
subfolder_ir2 = os.path.join(script_dir, "out1/ir2")
subfolder_masks = os.path.join(script_dir, "out1/masks")

# Check if the subfolder exists, and create it if it does not
if not os.path.exists(subfolder_rgb):
    os.makedirs(subfolder_rgb)
if not os.path.exists(subfolder_ir1):
    os.makedirs(subfolder_ir1)
if not os.path.exists(subfolder_ir2):
    os.makedirs(subfolder_ir2)
if not os.path.exists(subfolder_masks):
    os.makedirs(subfolder_masks)

# Create all 

RecordStream = False

# Streaming loop
index = 0
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        color_frame = frames.get_color_frame()
        ir1_frame = frames.get_infrared_frame(1)
        ir2_frame = frames.get_infrared_frame(2)

        # Validate that both frames are valid
        if not color_frame or not ir1_frame or not ir2_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        ir1_image = np.asanyarray(ir1_frame.get_data())
        ir2_image = np.asanyarray(ir2_frame.get_data())

        key = cv2.waitKey(1)

        # Start saving the frames if space is pressed once until it is pressed again
        # if key:
        if not RecordStream:
            time.sleep(0.05)  # wait for 1 second before starting recording
            RecordStream = True
            print("Recording started")
        else:
            RecordStream = False
            print("Recording stopped")

        if RecordStream:
            framename = index
            index += 1

            # Define the path to the image file within the subfolder
            image_path_rgb = os.path.join(subfolder_rgb, f"rgb{framename}.png")
            image_path_ir1 = os.path.join(subfolder_ir1, f"ir1{framename}.png")
            image_path_ir2 = os.path.join(subfolder_ir2, f"ir2{framename}.png")

            cv2.imwrite(image_path_ir1, ir1_image)
            cv2.imwrite(image_path_ir2, ir2_image)
            cv2.imwrite(image_path_rgb, color_image)

finally:
    pipeline.stop()