#!/usr/bin/env python3

# DOCUMENTATION:
# With this command the service can be called to start the touchdown detection process:
# ros2 service call /detect_touchdown std_srvs/srv/Trigger {}
# The detections are published on the topic /touchdown_status:
# ros2 topic echo /touchdown_status

import math
import statistics
import threading
import time

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

        # Removed 'port' parameter as we now use the internal ROS topic
        self.declare_parameter('force_threshold', 10.0)
        self.declare_parameter('calibration_samples', 100)
        self.declare_parameter('window_size', 5)

        self.threshold = self.get_parameter('force_threshold').get_parameter_value().double_value
        self.cal_limit = self.get_parameter('calibration_samples').get_parameter_value().integer_value
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value

        # Threading locks
        self.state_lock = threading.Lock()
        self.sensor_lock = threading.Lock()  # Protects access to the latest sensor data

        self.stop_event = threading.Event()
        self.worker_thread = None
        self.calibrating = False
        self.detection_active = False
        self.touchdown_detected = False
        
        self.latest_ft = None  # Stores the latest (Fx, Fy, Fz) tuple
        
        self.latest_median_force = 0.0
        self.latest_current_force = 0.0
        self.bias_fx = 0.0
        self.bias_fy = 0.0
        self.bias_fz = 0.0
        self.force_buffer = []

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
        self.get_logger().info(f"Service ready on /detect_touchdown using INTERNAL sensor. Median filter size: {self.window_size}")

    def ft_callback(self, msg):
        """Continually updates the latest force data from the internal sensor."""
        with self.sensor_lock:
            self.latest_ft = (
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z
            )

    def _get_latest_ft(self):
        """Thread-safe getter for the latest force values."""
        with self.sensor_lock:
            return self.latest_ft

    def start_detection_callback(self, request, response):
        del request

        # Ensure we are actually receiving sensor data before starting
        if self._get_latest_ft() is None:
            response.success = False
            response.message = 'Internal FT sensor data not yet received.'
            self.get_logger().warning(response.message)
            return response

        with self.state_lock:
            busy = self.worker_thread is not None and self.worker_thread.is_alive()
            if busy or self.calibrating or self.detection_active:
                response.success = False
                response.message = 'Touchdown detection is already running.'
                return response

            self.stop_event.clear()
            self.worker_thread = threading.Thread(target=self.run_detection_sequence, daemon=True)
            self.worker_thread.start()

        response.success = True
        response.message = 'Touchdown calibration started.'
        self.get_logger().info('Touchdown detection service request accepted.')
        return response

    def publish_touchdown(self):
        status_msg = Bool()
        status_msg.data = True

        for _ in range(5):
            self.touchdown_status_pub.publish(status_msg)
            time.sleep(0.05)

    def run_detection_sequence(self):
        try:
            with self.state_lock:
                self.calibrating = True
                self.detection_active = False
                self.touchdown_detected = False
                self.latest_median_force = 0.0
                self.latest_current_force = 0.0
                self.force_buffer = []

            self.get_logger().info('Touchdown requested. Calculating current bias...')

            bias_fx = 0.0
            bias_fy = 0.0
            bias_fz = 0.0
            
            # Calibration loop
            for _ in range(self.cal_limit):
                if self.stop_event.is_set():
                    return

                ft_values = self._get_latest_ft()
                if ft_values is not None:
                    bias_fx += ft_values[0]
                    bias_fy += ft_values[1]
                    bias_fz += ft_values[2]
                
                # Sleep briefly to sample over a span of time
                time.sleep(0.005) 

            bias_fx /= self.cal_limit
            bias_fy /= self.cal_limit
            bias_fz /= self.cal_limit

            with self.state_lock:
                self.bias_fx = bias_fx
                self.bias_fy = bias_fy
                self.bias_fz = bias_fz
                self.calibrating = False
                self.detection_active = True

            self.get_logger().info(
                f'Bias calculated. Monitoring for > {self.threshold}N (Median of {self.window_size}).'
            )

            # Detection loop
            while rclpy.ok() and not self.stop_event.is_set():
                ft_values = self._get_latest_ft()
                
                if ft_values is not None:
                    fx = ft_values[0] - self.bias_fx
                    fy = ft_values[1] - self.bias_fy
                    fz = ft_values[2] - self.bias_fz

                    self.latest_current_force = math.sqrt(fx**2 + fy**2 + fz**2)
                    self.force_buffer.append(self.latest_current_force)

                    if len(self.force_buffer) > self.window_size:
                        self.force_buffer.pop(0)

                    if len(self.force_buffer) == self.window_size:
                        self.latest_median_force = statistics.median(self.force_buffer)
                        
                        if self.latest_median_force > self.threshold and not self.touchdown_detected:
                            self.touchdown_detected = True
                            self.get_logger().warn(
                                f'TOUCHDOWN DETECTED: Median {self.latest_median_force:.2f}N'
                            )
                            self.publish_touchdown()
                            break
                    else:
                        self.latest_median_force = 0.0
                
                time.sleep(0.005)

        except Exception as error:
            self.get_logger().error(f'Error during touchdown detection: {error}')
        finally:
            with self.state_lock:
                self.calibrating = False
                self.detection_active = False


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
        node.stop_event.set()
        if node.worker_thread is not None and node.worker_thread.is_alive():
            node.worker_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()