#!/usr/bin/env python3

import math
import statistics
import threading
from collections import deque

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import WrenchStamped


class TouchdownDetector(Node):
    def __init__(self):
        super().__init__('touchdown_detector')

        self.declare_parameter('force_threshold', 10.0)
        self.declare_parameter('calibration_samples', 100)
        self.declare_parameter('window_size', 5)

        self.threshold = self.get_parameter('force_threshold').get_parameter_value().double_value
        self.cal_limit = self.get_parameter('calibration_samples').get_parameter_value().integer_value
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value

        # State Machine Variables
        self.state_lock = threading.Lock()
        self.state = 'IDLE'  # States: IDLE, CALIBRATING, DETECTING
        
        self.bias_fx = 0.0
        self.bias_fy = 0.0
        self.bias_fz = 0.0
        
        # Efficient queues for calibration and detection
        self.calibration_buffer = []
        self.force_window = deque(maxlen=self.window_size)

        self.callback_group = ReentrantCallbackGroup()

        # Internal FT Sensor Subscriber
        self.ft_sub = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_broadcaster/wrench',
            self.ft_callback,
            10,
            callback_group=self.callback_group
        )

        self.touchdown_status_pub = self.create_publisher(Bool, '/touchdown_status', 10)
        self.service = self.create_service(
            Trigger,
            'detect_touchdown',
            self.start_detection_callback,
            callback_group=self.callback_group,
        )
        self.get_logger().info(f"Service ready. Median filter size: {self.window_size}")

    def ft_callback(self, msg):
        """Driven entirely by incoming sensor data (Event-Driven)."""
        fx, fy, fz = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z

        with self.state_lock:
            if self.state == 'CALIBRATING':
                self.calibration_buffer.append((fx, fy, fz))
                
                if len(self.calibration_buffer) >= self.cal_limit:
                    # Calculate average bias
                    self.bias_fx = sum(f[0] for f in self.calibration_buffer) / self.cal_limit
                    self.bias_fy = sum(f[1] for f in self.calibration_buffer) / self.cal_limit
                    self.bias_fz = sum(f[2] for f in self.calibration_buffer) / self.cal_limit
                    
                    self.state = 'DETECTING'
                    self.get_logger().info(f'Bias calculated. Monitoring for > {self.threshold}N.')

            elif self.state == 'DETECTING':
                # Apply bias and calculate magnitude
                adj_fx = fx - self.bias_fx
                adj_fy = fy - self.bias_fy
                adj_fz = fz - self.bias_fz
                
                current_force = math.sqrt(adj_fx**2 + adj_fy**2 + adj_fz**2)
                
                # Deque automatically handles the sliding window size
                self.force_window.append(current_force)

                if len(self.force_window) == self.window_size:
                    median_force = statistics.median(self.force_window)
                    
                    if median_force > self.threshold:
                        self.get_logger().warn(f'TOUCHDOWN DETECTED: Median {median_force:.2f}N')
                        self.publish_touchdown()
                        self.state = 'IDLE' # Reset state after detection

    def start_detection_callback(self, request, response):
        del request
        with self.state_lock:
            if self.state != 'IDLE':
                response.success = False
                response.message = 'Touchdown detection is already running.'
                return response

            # Reset buffers and transition state
            self.calibration_buffer.clear()
            self.force_window.clear()
            self.state = 'CALIBRATING'

        response.success = True
        response.message = 'Touchdown calibration started.'
        self.get_logger().info('Touchdown requested. Calculating current bias...')
        return response

    def publish_touchdown(self):
        status_msg = Bool()
        status_msg.data = True
        # Simplified publish burst without relying on thread sleeping if possible, 
        # but kept here for backwards compatibility with your requirement.
        for _ in range(5):
            self.touchdown_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TouchdownDetector()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()