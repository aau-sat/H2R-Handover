#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import time
import statistics
from ft300python.ft300_stream import FT300Stream
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from touchdown_detection.action import DetectTouchdown


class TouchdownDetector(Node):
	def __init__(self):
		super().__init__('touchdown_detector')

		self.declare_parameter('port', '/dev/ttyUSB0')
		self.declare_parameter('force_threshold', 10.0) # 9.0
		self.declare_parameter('calibration_samples', 100)
		self.declare_parameter('window_size', 5)

		self.port = self.get_parameter('port').get_parameter_value().string_value
		self.threshold = self.get_parameter('force_threshold').get_parameter_value().double_value
		self.cal_limit = self.get_parameter('calibration_samples').get_parameter_value().integer_value
		self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value

		self.ft_stream = None
		try:
			self.ft_stream = FT300Stream(self.port, timeout=1, zero_reset=True)
			self.get_logger().info(
				f"FT300 connected on {self.port}. Median filter size: {self.window_size}"
			)
		except Exception as error:
			self.get_logger().error(f"Failed to connect FT300 on {self.port}: {error}")

		self.callback_group = ReentrantCallbackGroup()
		self.action_server = ActionServer(
			self,
			DetectTouchdown,
			'detect_touchdown',
			execute_callback=self.execute_callback,
			goal_callback=self.goal_callback,
			cancel_callback=self.cancel_callback,
			callback_group=self.callback_group,
		)
		self.get_logger().info("Action server ready on /detect_touchdown")

	def goal_callback(self, goal_request):
		del goal_request
		if self.ft_stream is None:
			self.get_logger().error("Rejecting goal: FT300 is not connected.")
			return GoalResponse.REJECT

		self.get_logger().info("Received touchdown detection goal.")
		return GoalResponse.ACCEPT

	def cancel_callback(self, goal_handle):
		del goal_handle
		self.get_logger().info("Received request to cancel touchdown detection goal.")
		return CancelResponse.ACCEPT

	def execute_callback(self, goal_handle):
		result = DetectTouchdown.Result()

		if self.ft_stream is None:
			self.get_logger().error("Cannot execute goal: FT300 is not connected.")
			goal_handle.abort()
			result.status = False
			return result

		self.get_logger().info("Touchdown requested. Calculating current bias...")

		try:
			bias_fx, bias_fy, bias_fz = 0.0, 0.0, 0.0
			for _ in range(self.cal_limit):
				if goal_handle.is_cancel_requested:
					goal_handle.canceled()
					result.status = False
					self.get_logger().info("Goal canceled during calibration phase.")
					return result

				ft_values = self.ft_stream.get_force_torque()
				bias_fx += ft_values[0]
				bias_fy += ft_values[1]
				bias_fz += ft_values[2]
				time.sleep(0.002)

			bias_fx /= self.cal_limit
			bias_fy /= self.cal_limit
			bias_fz /= self.cal_limit

			self.get_logger().info(
				f"Bias calculated. Monitoring for > {self.threshold}N (Median of {self.window_size})."
			)

			force_buffer = []
			median_force = 0.0

			while rclpy.ok():
				if goal_handle.is_cancel_requested:
					goal_handle.canceled()
					result.status = False
					self.get_logger().info("Goal canceled during detection phase.")
					return result

				ft_values = self.ft_stream.get_force_torque()
				fx = ft_values[0] - bias_fx
				fy = ft_values[1] - bias_fy
				fz = ft_values[2] - bias_fz

				current_magnitude = math.sqrt(fx**2 + fy**2 + fz**2)
				force_buffer.append(current_magnitude)

				if len(force_buffer) > self.window_size:
					force_buffer.pop(0)

				if len(force_buffer) == self.window_size:
					median_force = statistics.median(force_buffer)
					if median_force > self.threshold:
						self.get_logger().warn(
							f"TOUCHDOWN DETECTED: Median {median_force:.2f}N"
						)
						goal_handle.succeed()
						result.status = True
						return result

				time.sleep(0.005)

			goal_handle.abort()
			result.status = False
			return result

		except Exception as error:
			self.get_logger().error(f"Error during touchdown detection: {error}")
			goal_handle.abort()
			result.status = False
			return result


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