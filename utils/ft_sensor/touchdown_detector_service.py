#!/usr/bin/env python3


# DOCUMENTATION:
# With this command the serice can be called to start the touchdown detection process:
# ros2 service call /detect_touchdown std_srvs/srv/Trigger {}
# The detections are published on the topic /touchdown_status:
# ros2 topic echo /touchdown_status

import math
import statistics
import threading
import time

import rclpy
from ft300python.ft300_stream import FT300Stream
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


class TouchdownDetector(Node):
	def __init__(self):
		super().__init__('touchdown_detector')

		self.declare_parameter('port', '/dev/ttyUSB0')
		self.declare_parameter('force_threshold', 10.0)
		self.declare_parameter('calibration_samples', 100)
		self.declare_parameter('window_size', 5)

		self.port = self.get_parameter('port').get_parameter_value().string_value
		self.threshold = self.get_parameter('force_threshold').get_parameter_value().double_value
		self.cal_limit = self.get_parameter('calibration_samples').get_parameter_value().integer_value
		self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value

		self.state_lock = threading.Lock()
		self.stop_event = threading.Event()
		self.worker_thread = None
		self.calibrating = False
		self.detection_active = False
		self.touchdown_detected = False
		self.latest_median_force = 0.0
		self.latest_current_force = 0.0
		self.bias_fx = 0.0
		self.bias_fy = 0.0
		self.bias_fz = 0.0
		self.force_buffer = []

		self.ft_stream = None
		try:
			self.ft_stream = FT300Stream(self.port, timeout=1, zero_reset=True)
			self.get_logger().info(
				f"FT300 connected on {self.port}. Median filter size: {self.window_size}"
			)
		except Exception as error:
			self.get_logger().error(f"Failed to connect FT300 on {self.port}: {error}")

		self.callback_group = ReentrantCallbackGroup()
		self.touchdown_status_pub = self.create_publisher(Bool, '/touchdown_status', 10)
		self.service = self.create_service(
			Trigger,
			'detect_touchdown',
			self.start_detection_callback,
			callback_group=self.callback_group,
		)
		self.get_logger().info("Service ready on /detect_touchdown")

	def start_detection_callback(self, request, response):
		del request

		if self.ft_stream is None:
			response.success = False
			response.message = 'FT300 is not connected.'
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
		if self.ft_stream is None:
			return

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
			for _ in range(self.cal_limit):
				if self.stop_event.is_set():
					return

				ft_values = self.ft_stream.get_force_torque()
				bias_fx += ft_values[0]
				bias_fy += ft_values[1]
				bias_fz += ft_values[2]
				time.sleep(0.002)

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

			while rclpy.ok() and not self.stop_event.is_set():
				ft_values = self.ft_stream.get_force_torque()
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