#!/usr/bin/env python3

"""
Monitor left/right goal pose topics and report their frame rates.
"""

from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class GoalPoseRateMonitor(Node):
	def __init__(self):
		super().__init__('goal_pose_rate_monitor')

		self.declare_parameter('goal_pose_left_topic', '/goal_pose_left')
		self.declare_parameter('goal_pose_right_topic', '/goal_pose_right')
		self.declare_parameter('report_period_sec', 1.0)
		self.declare_parameter('window_size', 120)

		self.left_topic = str(
			self.get_parameter('goal_pose_left_topic').value or '/goal_pose_left'
		)
		self.right_topic = str(
			self.get_parameter('goal_pose_right_topic').value or '/goal_pose_right'
		)

		report_period_value = self.get_parameter('report_period_sec').value
		window_size_value = self.get_parameter('window_size').value
		self.report_period_sec = float(
			report_period_value if report_period_value is not None else 1.0
		)
		self.window_size = int(window_size_value if window_size_value is not None else 120)

		self.left_times = deque(maxlen=self.window_size)
		self.right_times = deque(maxlen=self.window_size)

		self.left_count_interval = 0
		self.right_count_interval = 0

		self.left_total = 0
		self.right_total = 0

		self.create_subscription(PoseStamped, self.left_topic, self.left_cb, 50)
		self.create_subscription(PoseStamped, self.right_topic, self.right_cb, 50)
		self.create_timer(self.report_period_sec, self.report_rates)

		self.get_logger().info(
			'Goal pose FPS monitor started with '
			f'left={self.left_topic}, right={self.right_topic}, '
			f'report_period={self.report_period_sec:.2f}s, window_size={self.window_size}'
		)

	def _now_sec(self):
		return self.get_clock().now().nanoseconds * 1e-9

	def left_cb(self, _msg: PoseStamped):
		now = self._now_sec()
		self.left_times.append(now)
		self.left_count_interval += 1
		self.left_total += 1

	def right_cb(self, _msg: PoseStamped):
		now = self._now_sec()
		self.right_times.append(now)
		self.right_count_interval += 1
		self.right_total += 1

	def _window_fps(self, times):
		if len(times) < 2:
			return 0.0
		dt = times[-1] - times[0]
		if dt <= 1e-9:
			return 0.0
		return (len(times) - 1) / dt

	def report_rates(self):
		left_window_fps = self._window_fps(self.left_times)
		right_window_fps = self._window_fps(self.right_times)

		left_interval_fps = self.left_count_interval / self.report_period_sec
		right_interval_fps = self.right_count_interval / self.report_period_sec

		left_last_seen = f'{self.left_times[-1]:.3f}' if self.left_times else 'never'
		right_last_seen = f'{self.right_times[-1]:.3f}' if self.right_times else 'never'

		self.get_logger().info(
			'[FPS] '
			f'left({self.left_topic}): interval={left_interval_fps:.2f}, window={left_window_fps:.2f}, total={self.left_total}, last={left_last_seen} | '
			f'right({self.right_topic}): interval={right_interval_fps:.2f}, window={right_window_fps:.2f}, total={self.right_total}, last={right_last_seen}'
		)

		self.left_count_interval = 0
		self.right_count_interval = 0


def main(args=None):
	rclpy.init(args=args)
	node = GoalPoseRateMonitor()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
