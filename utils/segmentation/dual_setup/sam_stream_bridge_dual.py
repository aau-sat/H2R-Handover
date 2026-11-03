#!/usr/bin/env python3

"""Launch two SamBridgeAsync instances in one process."""

import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor

from sam_stream_bridge import SamBridgeAsync


def main():
    rclpy.init()

    #left camera 
    bridge_left = SamBridgeAsync(
        camera="left",
        node_name="sam_bridge_left",
        image_topic="/camera/left/color/image_raw",
        #detection_topic="/object_tracker/detected_objects",  #new tracker
        detection_topic="/detected_objects/left",  # old tracker
        mask_topic="/hand/segmentation/left",
        push_addr="tcp://localhost:5555",
        sub_addr="tcp://localhost:5556",
        target_fps=30.0,
        track_hand=True,
        show_visualization=False, #set only one visualization to True 
    )

    #right camera
    bridge_right = SamBridgeAsync(
        camera="right",
        node_name="sam_bridge_right",
        image_topic="/camera/right/color/image_raw",
        #detection_topic="/object_tracker/detected_objects", #new tracker
        detection_topic="/detected_objects/right", # old tracker,
        mask_topic="/hand/segmentation/right",
        push_addr="tcp://localhost:5565",
        sub_addr="tcp://localhost:5566",
        target_fps=30.0,
        track_hand=True,
        show_visualization=False,
    )

    executor = MultiThreadedExecutor(num_threads=4)
    try:
        executor.add_node(bridge_left)
        executor.add_node(bridge_right)
        executor.spin()
    finally:
        bridge_left.destroy_node()
        bridge_right.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()