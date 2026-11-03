#!/usr/bin/env python3

"""
Fused object pose tracker with hand-based fallback.

Subscribes to:
- `/tracked_hand_pose`: Hand 3D position (PoseStamped)
- `/goal_pose`: Object pose from segmentation tracking (PoseStamped)

Publishes:
- `/estimated_object_pose_fused`: Final pose (either segmentation-based or hand+offset)

Behavior:
- Primary: Track object via segmentation masks
- Fallback: If segmentation fails (deviation or dropout), use hand pose + cached offset
- Offset management: Compute and update offsets whenever both hand and object poses available
- No auto-recovery: Once fallback triggered, stays in fallback mode
"""

import json
import math
from collections import deque

import cv2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String



def pose_to_array(pose_msg):
    return np.array(
        [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z],
        dtype=np.float32,
    )


def array_to_pose_stamped(xyz, frame_id, stamp):
    pose_msg = PoseStamped()
    pose_msg.header.stamp = stamp
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.position.x = float(xyz[0])
    pose_msg.pose.position.y = float(xyz[1])
    pose_msg.pose.position.z = float(xyz[2])
    pose_msg.pose.orientation.w = 1.0
    return pose_msg


class HandPoseFallbackEstimator(Node):
    def __init__(self):
        super().__init__('hand_pose_fallback_estimator')

        self.declare_parameter('hand_pose_topic', '/tracked_hand_pose')
        self.declare_parameter('object_pose_topic', '/goal_pose')
        self.declare_parameter('hand_tracker_json_topic', '/hand_tracker')
        self.declare_parameter('handover_detected_topic', '/handover_detected')
        self.declare_parameter('mask_left_topic', '/segmentation/left')
        self.declare_parameter('mask_right_topic', '/segmentation/right')
        self.declare_parameter('output_pose_topic', '/estimated_object_pose_fused')
        self.declare_parameter('target_frame', 'base')
        self.declare_parameter('deviation_threshold_m', 0.05)
        self.declare_parameter('max_dropout_frames', 5)
        self.declare_parameter('sync_queue_size', 20)
        self.declare_parameter('sync_slop_s', 0.05)
        self.declare_parameter('debug_status_period_s', 10.0)

        self.hand_pose_topic = str(self.get_parameter('hand_pose_topic').value or '/tracked_hand_pose')
        self.object_pose_topic = str(self.get_parameter('object_pose_topic').value or '/goal_pose')
        self.hand_tracker_json_topic = str(self.get_parameter('hand_tracker_json_topic').value or '/hand_tracker')
        self.handover_detected_topic = str(self.get_parameter('handover_detected_topic').value or '/handover_detected')
        self.mask_left_topic = str(self.get_parameter('mask_left_topic').value or '/segmentation/left')
        self.mask_right_topic = str(self.get_parameter('mask_right_topic').value or '/segmentation/right')
        self.output_pose_topic = str(self.get_parameter('output_pose_topic').value or '/estimated_object_pose_fused')
        self.target_frame = str(self.get_parameter('target_frame').value or 'base')
        self.deviation_threshold_m = float(self.get_parameter('deviation_threshold_m').value or 0.05)
        self.max_dropout_frames = int(self.get_parameter('max_dropout_frames').value or 5)
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value or 20)
        self.sync_slop_s = float(self.get_parameter('sync_slop_s').value or 0.05)
        self.debug_status_period_s = float(self.get_parameter('debug_status_period_s').value or 10.0)

        self.pose_pub = self.create_publisher(PoseStamped, self.output_pose_topic, 10)

        sensor_qos = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Synchronize hand/object pose streams to ensure fusion uses time-aligned inputs.
        self.hand_pose_sub = Subscriber(self, PoseStamped, self.hand_pose_topic, qos_profile=sensor_qos)
        self.object_pose_sub = Subscriber(self, PoseStamped, self.object_pose_topic, qos_profile=sensor_qos)
        self.pose_sync = ApproximateTimeSynchronizer(
            [self.hand_pose_sub, self.object_pose_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_s,
        )
        self.pose_sync.registerCallback(self.synced_pose_callback)
        self.hand_tracker_json_sub = self.create_subscription(String, self.hand_tracker_json_topic, self.hand_tracker_json_callback, sensor_qos)
        self.handover_detected_sub = self.create_subscription(Bool, self.handover_detected_topic, self.handover_detected_callback, 10)
        self.mask_left_sub = self.create_subscription(Image, self.mask_left_topic, self.mask_left_callback, sensor_qos)
        self.mask_right_sub = self.create_subscription(Image, self.mask_right_topic, self.mask_right_callback, sensor_qos)
        self.status_timer = self.create_timer(self.debug_status_period_s, self.debug_status_callback)

        # State tracking
        self.last_hand_pose = None
        self.last_object_pose = None
        self.last_hand_tracker_payload = None
        self.handover_detected = False
        self.in_fallback_mode = False
        self.consecutive_dropout_count = 0
        self.dropout_history = deque(maxlen=10)
        self.left_mask_has_pixels = False
        self.right_mask_has_pixels = False
        self.left_mask_received = False
        self.right_mask_received = False

        # Per-view offset caches: map view_name -> offset_xyz (numpy array)
        self.view_offsets = {
            'left': None,
            'right': None,
        }
        self.offset_confidence = {
            'left': 0,
            'right': 0,
        }

        # Statistics
        self.pub_count = 0
        self.fallback_triggers_deviation = 0
        self.fallback_triggers_dropout = 0
        self.offset_updates = 0

        self.get_logger().info(
            'Hand-pose fallback estimator started: '
            f'deviation_threshold={self.deviation_threshold_m:.3f}m, '
            f'max_dropout_frames={self.max_dropout_frames}, '
            f'sync_queue_size={self.sync_queue_size}, '
            f'sync_slop_s={self.sync_slop_s:.3f}s, '
            f'handover_topic={self.handover_detected_topic}, '
            f'mask_left_topic={self.mask_left_topic}, '
            f'mask_right_topic={self.mask_right_topic}, '
            f'output_topic={self.output_pose_topic}'
        )

    def hand_tracker_json_callback(self, msg):
        """Cache the latest hand tracker payload for offset computation."""
        try:
            payload = json.loads(msg.data) if msg.data else {}
            self.last_hand_tracker_payload = payload
        except json.JSONDecodeError:
            pass

    def handover_detected_callback(self, msg):
        handover_now = bool(msg.data)
        if handover_now and not self.handover_detected:
            self.get_logger().info('handover_detected=true received. Enabling mask-dropout fallback trigger.')
        elif (not handover_now) and self.handover_detected:
            self.get_logger().info('handover_detected=false received. Resetting mask-dropout counter.')
            self.consecutive_dropout_count = 0
        self.handover_detected = handover_now

    def _mask_has_foreground(self, mask_msg):
        if not mask_msg.data:
            return False
        mask_np = np.frombuffer(mask_msg.data, dtype=np.uint8)
        return bool(np.any(mask_np > 0))

    def mask_left_callback(self, msg):
        self.left_mask_received = True
        self.left_mask_has_pixels = self._mask_has_foreground(msg)

    def mask_right_callback(self, msg):
        self.right_mask_received = True
        self.right_mask_has_pixels = self._mask_has_foreground(msg)

    def _update_dropout_from_masks(self):
        """Count consecutive frames where both masks are empty after handover."""
        if not self.handover_detected:
            self.consecutive_dropout_count = 0
            return

        if not self.left_mask_received or not self.right_mask_received:
            return

        if self.left_mask_has_pixels or self.right_mask_has_pixels:
            self.consecutive_dropout_count = 0
            return

        self.handle_dropout_frame()

    def synced_pose_callback(self, hand_msg, object_msg):
        """
        Main callback: process synchronized hand/object poses and decide between normal and fallback mode.
        """
        self.last_hand_pose = hand_msg
        self._update_dropout_from_masks()

        current_object_xyz = pose_to_array(object_msg)

        # Check for deviation from last known object pose
        deviation_detected = False
        if self.last_object_pose is not None:
            last_object_xyz = pose_to_array(self.last_object_pose)
            deviation = float(np.linalg.norm(current_object_xyz - last_object_xyz))
            if deviation > self.deviation_threshold_m:
                deviation_detected = True
                self.get_logger().warn(
                    f'Object pose deviation detected: {deviation:.4f}m (threshold: {self.deviation_threshold_m:.4f}m)'
                )

        if self.left_mask_has_pixels or self.right_mask_has_pixels:
            self.consecutive_dropout_count = 0

        # If normal mode: try to update offsets and check for trigger conditions
        if not self.in_fallback_mode:
            self._try_update_offsets(current_object_xyz, hand_msg)

            if deviation_detected:
                self.get_logger().warn('Entering fallback mode due to deviation.')
                self.in_fallback_mode = True
                self.fallback_triggers_deviation += 1
                self.consecutive_dropout_count = 0
                # Publish hand + offset
                fallback_pose = self._compute_fallback_pose(hand_msg)
                self.publish_pose(fallback_pose)
            elif self.handover_detected and self.consecutive_dropout_count >= self.max_dropout_frames:
                self.get_logger().warn(
                    'Entering fallback mode due to mask dropout after handover: '
                    f'{self.consecutive_dropout_count} >= {self.max_dropout_frames}'
                )
                self.in_fallback_mode = True
                self.fallback_triggers_dropout += 1
                self.consecutive_dropout_count = 0
                fallback_pose = self._compute_fallback_pose(hand_msg)
                self.publish_pose(fallback_pose)
            else:
                # Publish segmentation pose normally
                self.publish_pose(object_msg)
        else:
            # In fallback mode: update offset opportunistically, continue using hand + offset
            self._try_update_offsets(current_object_xyz, hand_msg)
            fallback_pose = self._compute_fallback_pose(hand_msg)
            self.publish_pose(fallback_pose)

        self.last_object_pose = object_msg

    def _try_update_offsets(self, object_xyz, hand_pose):
        """Computes and updates the spatial offset between the hand and object.

        This method calculates the offset and updates the internal state. 
        It logs the new offset only if it is the first update or if the L1 
        distance (Manhattan distance) between the new and previous offset 
        exceeds 5 cm (0.05 meters).

        Args:
            object_xyz (numpy.ndarray): The 3D coordinates of the object.
            hand_pose (Pose): The current pose of the hand.

        Returns:
            None
        """
        if self.last_hand_tracker_payload is None:
            return

        hand_xyz = pose_to_array(hand_pose)
        offset = object_xyz - hand_xyz

        active_view = self.last_hand_tracker_payload.get('active_view')
        if active_view in ('left', 'right'):
            prev_offset = self.view_offsets.get(active_view)

            # Update the offset for the active view
            self.view_offsets[active_view] = offset
            self.offset_confidence[active_view] += 1
            self.offset_updates += 1

            # Log if it's the first update OR if the L1 change is > 0.05m (5cm)
            if prev_offset is None or np.sum(np.abs(offset - prev_offset)) > 0.05:
                self.get_logger().info(
                    f'Updated offset for {active_view} view: '
                    f'offset=[{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]'
                )

    def _compute_fallback_pose(self, hand_pose):
        """
        Compute fallback pose: hand_pose + offset.
        Uses the most confident offset available.
        """
        hand_xyz = pose_to_array(hand_pose)

        best_view = None
        best_confidence = -1
        for view_name in ('left', 'right'):
            if self.view_offsets[view_name] is not None and self.offset_confidence[view_name] > best_confidence:
                best_confidence = self.offset_confidence[view_name]
                best_view = view_name

        if best_view is None:
            # No offset available; just return hand pose
            self.get_logger().warn('No offset available; returning hand pose directly.')
            return hand_pose

        offset = self.view_offsets[best_view]
        fallback_xyz = hand_xyz + offset

        fallback_pose = array_to_pose_stamped(fallback_xyz, hand_pose.header.frame_id, hand_pose.header.stamp)
        return fallback_pose

    def handle_dropout_frame(self):
        """
        Called when we detect a frame with no segmentation from either view.
        """
        self.consecutive_dropout_count += 1
        self.dropout_history.append(self.consecutive_dropout_count)

        if not self.in_fallback_mode and self.consecutive_dropout_count >= self.max_dropout_frames:
            self.get_logger().warn(
                f'Entering fallback mode due to dropout: {self.consecutive_dropout_count} >= {self.max_dropout_frames}'
            )
            self.in_fallback_mode = True
            self.fallback_triggers_dropout += 1

    def publish_pose(self, pose_msg):
        """Publish the pose to output topic."""
        self.pose_pub.publish(pose_msg)
        self.pub_count += 1

    def debug_status_callback(self):
        """Periodic status logging."""
        self.get_logger().info(
            'Status: '
            f'in_fallback={self.in_fallback_mode}, '
            f'handover={self.handover_detected}, '
            f'dropout_count={self.consecutive_dropout_count}, '
            f'mask_left_fg={self.left_mask_has_pixels}, '
            f'mask_right_fg={self.right_mask_has_pixels}, '
            f'published={self.pub_count}, '
            f'fallback_dev={self.fallback_triggers_deviation}, '
            f'fallback_drop={self.fallback_triggers_dropout}, '
            f'offset_updates={self.offset_updates}, '
            f'offsets_left={self.view_offsets["left"] is not None}, '
            f'offsets_right={self.view_offsets["right"] is not None}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = HandPoseFallbackEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
