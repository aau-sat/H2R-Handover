#!/usr/bin/env python3
"""
ROS 2 node movement detection for pose tracking.
"""

import collections
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Bool
from enum import Enum

class State(Enum):
    IDLE = 0        
    MOVING = 1      
    STABILIZING = 2 
    STOPPED = 3     


class StopDetectionStateMachine:
    def __init__(self, start_threshold: float = 0.05, 
                stability_threshold: float = 0.01, 
                window_length: int = 20,
                stable_percentage_threshold: float = 0.8, 
                init_height: float = None):
        """
        Initialize the state machine parameters.

        Args:
            start_threshold (float): Meters moved to transition IDLE -> MOVING.
            stability_threshold (float): Max meters/frame to consider STABILIZING.
            stable_frame_window (int): Frames required to transition to STOPPED.
            stable_percentage_threshold (float): Percentage of stable frames required to transition to STOPPED.
            init_height (float): Initial height of the object.
        """
        self.state = State.IDLE
        self.start_threshold = start_threshold
        self.stability_threshold = stability_threshold
        self.frame_window = window_length
        
        self.initial_pose = None
        self.previous_pose = None
        self.sliding_window = collections.deque(maxlen=window_length)
        self.stable_percentage_threshold = stable_percentage_threshold
        self.init_height = init_height

    def process_pose(self, current_position: Point) -> bool:
        """
        Updates the state machine with a new position.

        Args:
            current_position (Point): The new object position.

        Returns:
            bool: True if the state stopp transistioned, false otherwise.
        """
        if self.initial_pose is None:
            self.initial_pose = current_position
            self.previous_pose = current_position
            self.init_height = current_position.z
            return False


        self.delta_dist = self._calculate_euclidean_distance(current_position, self.previous_pose)
        self.total_disp = self._calculate_euclidean_distance(current_position, self.initial_pose)

        stopped = False

        # State logic
        if self.state == State.IDLE:
            #print("IDLE ")
            if self.total_disp > self.start_threshold:
                self.state = State.MOVING

        elif self.state == State.MOVING:
            if self.delta_dist < self.stability_threshold:
                self.state = State.STABILIZING
                self.sliding_window.clear()  # Reset sliding window when we start stabilizing


        elif self.state == State.STABILIZING:
            if self.delta_dist < self.stability_threshold:
                self.sliding_window.append(True)
            else:
                self.sliding_window.append(False)
            if len(self.sliding_window) == self.frame_window:
                #print(f"sliding window size: {len(self.sliding_window)}")
                stable_percentage = sum(self.sliding_window) / self.frame_window        
                #print(f"Stable Percentage: {stable_percentage:.2f}")
                if stable_percentage >= self.stable_percentage_threshold:
                    if self._is_on_table(current_position):
                        print("Object is on the table for whatever reason.")
                        self.state = State.IDLE  # Reset to IDLE after stopping
                        self.sliding_window.clear()  # Clear the sliding window for the next round
                        self.initial_pose = current_position  # Update initial pose for next detection
                    else:
                        print("Stopped!")
                        self.state = State.STOPPED
                        stopped = True
                else:
                    self.state = State.MOVING  # Not stable enough, go back to MOVING


        elif self.state == State.STOPPED:

            if self._is_on_table(current_position):
                print("Object is on the table finally.")
                self.state = State.IDLE
                self.initial_pose = current_position
                stopped = False
            

        self.previous_pose = current_position
        
        return stopped

    def _is_on_table(self, position: Point) -> bool:
        """
        Checks if the object is close to the table height (Z-axis).
        Adjust the comparison (< or >) based on your specific camera frame!
        """
        return self.total_disp < self.start_threshold 
    
    def get_current_state(self):
        return self.state

    def _calculate_euclidean_distance(self, pos1: Point, pos2: Point):
        return math.sqrt(
            (pos1.x - pos2.x)**2 +
            (pos1.y - pos2.y)**2 +
            (pos1.z - pos2.z)**2
        )


class MovementDetectionNode(Node):
    """
    ROS 2 Node wrapper for the StopDetectionStateMachine.
    """

    def __init__(self):
        super().__init__('stop_detection_node')

        self.sm = StopDetectionStateMachine(
            start_threshold=0.05,
            stability_threshold=0.015,
            window_length=30, 
            stable_percentage_threshold=0.9,
            init_height=-0.15
        )
        self.current_pose = None
        self.stop_pose = None
        self.first_callback = False


        #Subscriber to pose topic
        self.pose_sub = self.create_subscription(
            PoseStamped,
            #'/estimated_pose',
            '/goal_pose',
            self.pose_callback,
            10
        )
        
        self.trigger_pub = self.create_publisher(Bool, '/object_ready_for_grasp', 10)

        self.get_logger().info("Stop Detection Node started.")
        

    def pose_callback(self, msg: PoseStamped):
        """
        Receives pose, updates SM, and publishes if triggered.
        """
        if not self.first_callback:
            self.get_logger().info("Received first pose message, starting detection.")
            self.first_callback = True
        self.current_pose = msg.pose.position
        
        # Update the decoupled state machine
        stopp_detected = self.sm.process_pose(self.current_pose)
        
        # Log state changes for debugging
        current_state = self.sm.get_current_state()
        #self.get_logger().info(f"Current State: {current_state.name}")

        if stopp_detected:
            self.publish_stop_signal()
            self.stop_pose = msg.pose  # Store the pose at which we stopped
            stopp_detected = False
        else:
            msg = Bool()
            msg.data = False
            self.trigger_pub.publish(msg)

    def publish_stop_signal(self):
        msg = Bool()
        msg.data = True
        self.trigger_pub.publish(msg)
        self.get_logger().info("TRIGGER: Object is stable. Signal published.")
        self.get_logger().info(f"Stopped object pose: {self.stop_pose}")



def main(args=None):
    rclpy.init(args=args)
    node = MovementDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()