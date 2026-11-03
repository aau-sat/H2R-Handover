#!/usr/bin/env python3
"""
Merged Tactile Sensor Node: BLE Streaming + Contact Detection
Combines BLE pressure streaming with on-node contact detection to minimize latency.
"""

import asyncio
import signal
import struct
import math
import time
from typing import Any
from statistics import mean, pstdev

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tactile_detection.msg import TactileDetection

from bleak import BleakClient

# BLE Configuration
BLE_NOTIFY_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

# Publisher Parameters
DEFAULT_BLE_ADDRESS = "D9:E9:6D:70:68:D0"
DEFAULT_NUM_CHANNELS = 7
DEFAULT_HEADER_BYTES = 3
DEFAULT_PAYLOAD_BYTES = DEFAULT_NUM_CHANNELS * 4  # bytes (4 bytes per float)
DEFAULT_EXPECTED_MIN_LEN = DEFAULT_HEADER_BYTES + DEFAULT_PAYLOAD_BYTES
DEFAULT_DEVICE_UID = "DEVB9975E99211783F"
DEFAULT_ADAPTER = "hci0"

# Contact Detection Parameters
DEFAULT_ALPHA = 0.8
DEFAULT_CONTACT_SENSOR_COUNT = 2
DEFAULT_CONTACT_DETECTION_STRATEGY = "threshold"  # "threshold" or "derivative"
DEFAULT_DETECTION_MODE = "continuous"  # "continuous" or "lock-in"

# Derivative-based Contact Detection Parameters
DEFAULT_DERIVATIVE_SPIKE_THRESHOLD = 20.0

# Calibration Parameters
DEFAULT_CALIBRATION_SAMPLES = 100
DEFAULT_STD_DEV_HIGH_MULTIPLIERS = [20.0] * DEFAULT_NUM_CHANNELS  # Multiplier for std deviation to set high threshold
DEFAULT_STD_DEV_LOW_MULTIPLIERS = [1.0] * DEFAULT_NUM_CHANNELS  # Multiplier for std deviation to set low threshold
DEFAULT_MIN_STD_EPS = 1e-6

# BLE Connection Watchdog Parameters
DEFAULT_BLE_DATA_TIMEOUT_MS = 300  # Timeout in milliseconds if no data received


class TactileSensorNode(Node):
    def __init__(self):
        super().__init__("tactile_sensor_node")

        def get_param_value(name: str, default: Any) -> Any:
            value = self.get_parameter(name).value
            return default if value is None else value

        def normalize_multipliers(value: Any, default: list[float]) -> list[float]:
            if value is None:
                return default.copy()
            if isinstance(value, (int, float)):
                return [float(value)] * self.num_channels
            if isinstance(value, (list, tuple)):
                values = [float(v) for v in value]
                if len(values) == self.num_channels:
                    return values
            return default.copy()

        # BLE Parameters
        self.declare_parameter("ble_address", DEFAULT_BLE_ADDRESS)
        self.declare_parameter("num_channels", DEFAULT_NUM_CHANNELS)
        self.declare_parameter("header_bytes", DEFAULT_HEADER_BYTES)
        self.declare_parameter("payload_bytes", DEFAULT_PAYLOAD_BYTES)
        self.declare_parameter("expected_min_len", DEFAULT_EXPECTED_MIN_LEN)
        self.declare_parameter("device_uid", DEFAULT_DEVICE_UID)
        self.declare_parameter("adapter", DEFAULT_ADAPTER)
        self.declare_parameter("publish_pressure", True)

        self.ble_address = str(get_param_value("ble_address", DEFAULT_BLE_ADDRESS))
        self.num_channels = int(get_param_value("num_channels", DEFAULT_NUM_CHANNELS))
        self.header_bytes = int(get_param_value("header_bytes", DEFAULT_HEADER_BYTES))
        self.payload_bytes = int(get_param_value("payload_bytes", DEFAULT_PAYLOAD_BYTES))
        self.expected_min_len = int(get_param_value("expected_min_len", DEFAULT_EXPECTED_MIN_LEN))
        self.device_uid = str(get_param_value("device_uid", DEFAULT_DEVICE_UID))
        self.adapter = str(get_param_value("adapter", DEFAULT_ADAPTER))
        self.publish_pressure = bool(get_param_value("publish_pressure", False))

        # Contact Detection Parameters
        self.declare_parameter("alpha", DEFAULT_ALPHA)
        self.declare_parameter("contact_sensor_count", DEFAULT_CONTACT_SENSOR_COUNT)
        self.declare_parameter("contact_detection_strategy", DEFAULT_CONTACT_DETECTION_STRATEGY)
        self.declare_parameter("calibration_samples", DEFAULT_CALIBRATION_SAMPLES)
        self.declare_parameter("std_dev_high_multipliers", DEFAULT_STD_DEV_HIGH_MULTIPLIERS)
        self.declare_parameter("std_dev_low_multipliers", DEFAULT_STD_DEV_LOW_MULTIPLIERS)
        self.declare_parameter("min_std_eps", DEFAULT_MIN_STD_EPS)
        self.declare_parameter("derivative_spike_threshold", DEFAULT_DERIVATIVE_SPIKE_THRESHOLD)
        self.declare_parameter("detection_mode", DEFAULT_DETECTION_MODE)

        # BLE Connection Watchdog Parameters
        self.declare_parameter("ble_data_timeout_ms", DEFAULT_BLE_DATA_TIMEOUT_MS)

        self.alpha = float(get_param_value("alpha", DEFAULT_ALPHA))
        self.contact_sensor_count = int(get_param_value("contact_sensor_count", DEFAULT_CONTACT_SENSOR_COUNT))
        self.contact_detection_strategy = str(get_param_value("contact_detection_strategy", DEFAULT_CONTACT_DETECTION_STRATEGY))
        self.calibration_samples = int(get_param_value("calibration_samples", DEFAULT_CALIBRATION_SAMPLES))
        self.std_dev_high_multipliers = normalize_multipliers(
            get_param_value("std_dev_high_multipliers", DEFAULT_STD_DEV_HIGH_MULTIPLIERS),
            DEFAULT_STD_DEV_HIGH_MULTIPLIERS,
        )
        self.std_dev_low_multipliers = normalize_multipliers(
            get_param_value("std_dev_low_multipliers", DEFAULT_STD_DEV_LOW_MULTIPLIERS),
            DEFAULT_STD_DEV_LOW_MULTIPLIERS,
        )
        self.min_std_eps = float(get_param_value("min_std_eps", DEFAULT_MIN_STD_EPS))
        self.derivative_spike_threshold = float(get_param_value("derivative_spike_threshold", DEFAULT_DERIVATIVE_SPIKE_THRESHOLD))
        self.detection_mode = str(get_param_value("detection_mode", DEFAULT_DETECTION_MODE))
        # BLE Connection Watchdog Parameters
        self.ble_data_timeout_ms = int(get_param_value("ble_data_timeout_ms", DEFAULT_BLE_DATA_TIMEOUT_MS))
        self.last_data_time = asyncio.get_event_loop().time()

        # Publishers
        if self.publish_pressure:
            self.pressure_pub = self.create_publisher(
                Float32MultiArray,
                f"/{self.device_uid}/Pressure",
                10,
            )
        
        self.contact_pub = self.create_publisher(
            TactileDetection,
            f"/{self.device_uid}/Contact",
            10,
        )

        # Service for calibration
        self.calibration_service = self.create_service(
            Trigger,
            f"/{self.device_uid}/calibrate_contact",
            self.calibrate_callback,
        )

        # State variables - Pressure streaming
        self.current_pressure = [0.0] * self.num_channels

        # State variables - Contact detection
        self.smoothed_p = [0.0] * self.num_channels
        self.prev_smoothed_p = [0.0] * self.num_channels  # For derivative calculation
        self.has_prev_smoothed_p = False
        self.sensor_active = [False] * self.num_channels
        self.is_contact = False

        # Calibration state
        self.is_calibrating = True
        self.calibration_buffer = []
        self.baseline_mean = [0.0] * self.num_channels
        self.baseline_std = [1.0] * self.num_channels
        self.high_thresholds = [math.inf] * self.num_channels
        self.low_thresholds = [math.inf] * self.num_channels
        self.is_calibrated = False

        # BLE Connection Watchdog state
        self.last_data_time = None
        self.connection_lost_logged = False

        self.get_logger().info("Tactile Sensor Node initialized")
        self.get_logger().info(
            f"BLE Address: {self.ble_address}"
        )
        self.get_logger().info(
            f"Device UID: {self.device_uid}"
        )
        self.get_logger().info(
            f"Call '/{self.device_uid}/calibrate_contact' service to calibrate sensors"
        )
        self.get_logger().info(
            "Parameters: "
            f"ble_address={self.ble_address}, num_channels={self.num_channels}, "
            f"alpha={self.alpha}, contact_sensor_count={self.contact_sensor_count}, "
            f"contact_detection_strategy={self.contact_detection_strategy}, "
            f"calibration_samples={self.calibration_samples}, "
            f"std_dev_high_multipliers={self.std_dev_high_multipliers}, "
            f"std_dev_low_multipliers={self.std_dev_low_multipliers}, "
            f"derivative_spike_threshold={self.derivative_spike_threshold}, "
            f"ble_data_timeout_ms={self.ble_data_timeout_ms}"
        )

    def publish_tactile_event(self, detected: bool, disconnected: bool):
        """Publish the current tactile contact and connection state."""
        event_msg = TactileDetection()
        event_msg.detected = detected
        event_msg.disconnected = disconnected
        self.contact_pub.publish(event_msg)

    def calibrate_callback(self, request, response):
        """ROS2 service to start calibration."""
        if self.contact_detection_strategy == "derivative":
            response.success = False
            response.message = "Calibration is not required for derivative contact detection"
            return response

        if self.is_calibrating:
            response.success = False
            response.message = "Calibration already in progress"
            return response

        self.get_logger().info(
            f"Starting calibration: collecting {self.calibration_samples} samples..."
        )
        self.is_calibrating = True
        self.calibration_buffer = []
        response.success = True
        response.message = f"Calibration started, collecting {self.calibration_samples} samples"
        return response

    def compute_calibration(self):
        """Compute baseline statistics from calibration buffer."""
        if len(self.calibration_buffer) == 0:
            self.get_logger().warn("Calibration buffer is empty!")
            return

        # Convert to per-channel lists
        channels = list(zip(*self.calibration_buffer))

        # Compute mean and std per channel
        self.baseline_mean = [mean(ch) for ch in channels]
        self.baseline_std = [pstdev(ch) if len(ch) > 1 else 0.0 for ch in channels]

        # Compute per-sensor thresholds based on calibrated mean and std deviation
        std = [max(v, self.min_std_eps) for v in self.baseline_std]
        self.high_thresholds = [
            m + (high_multiplier * s)
            for m, s, high_multiplier in zip(self.baseline_mean, std, self.std_dev_high_multipliers)
        ]
        self.low_thresholds = [
            m + (low_multiplier * s)
            for m, s, low_multiplier in zip(self.baseline_mean, std, self.std_dev_low_multipliers)
        ]

        # Start from the calibrated baseline and clear the per-sensor contact state
        self.smoothed_p = self.baseline_mean.copy()
        self.prev_smoothed_p = self.baseline_mean.copy()
        self.has_prev_smoothed_p = True
        self.sensor_active = [False] * self.num_channels

        self.is_calibrated = True

        self.get_logger().info("=== Calibration Complete ===")
        self.get_logger().info(f"Baseline Mean: {self.baseline_mean}")
        self.get_logger().info(f"Baseline Std: {self.baseline_std}")
        self.get_logger().info(f"High Thresholds: {self.high_thresholds}")
        self.get_logger().info(f"Low Thresholds: {self.low_thresholds}")

    def mark_ble_activity(self):
        """Record that a BLE notification was received."""
        self.last_data_time = time.monotonic()
        self.connection_lost_logged = False

    def process_pressure_data(self, raw_p):
        """
        Process incoming pressure data from BLE.
        Handles calibration and contact detection in-process for minimal latency.
        Supports two detection strategies: threshold-based and derivative-based.
        """
        if len(raw_p) != self.num_channels:
            self.get_logger().warn(
                f"Expected {self.num_channels} channels, got {len(raw_p)}"
            )
            return

        # Store raw pressure for pressure topic
        self.current_pressure = [float(v) for v in raw_p]

        if self.publish_pressure:
            # Publish pressure data immediately (high frequency)
            pressure_msg = Float32MultiArray()
            pressure_msg.data = self.current_pressure
            self.pressure_pub.publish(pressure_msg)

        # Handle calibration mode
        if self.contact_detection_strategy != "derivative" and self.is_calibrating:
            self.calibration_buffer.append(list(raw_p))

            if len(self.calibration_buffer) >= self.calibration_samples:
                self.is_calibrating = False
                self.compute_calibration()
            self.publish_tactile_event(detected=False, disconnected=False)
            return

        # Smooth the raw pressure values using exponential moving average
        self.smoothed_p = [
            (self.alpha * raw) + ((1 - self.alpha) * smooth)
            for raw, smooth in zip(raw_p, self.smoothed_p)
        ]

        # Derivative strategy is calibration-free and does not depend on the resting pressure level.
        # Initialize the previous sample on the first valid frame to avoid a startup spike.
        if self.contact_detection_strategy == "derivative" and not self.has_prev_smoothed_p:
            self.prev_smoothed_p = self.smoothed_p.copy()
            self.has_prev_smoothed_p = True
            self.is_contact = False
            self.publish_tactile_event(detected=False, disconnected=False)
            self.get_logger().debug("Derivative strategy initialized; waiting for next sample")
            return

        # Threshold strategy still requires calibration.
        if self.contact_detection_strategy != "derivative" and not self.is_calibrated:
            return

        if self.detection_mode == "lock-in" and self.is_contact:
            self.publish_tactile_event(detected=True, disconnected=False)
            return
        
        # Select detection strategy
        if self.contact_detection_strategy == "derivative":
            self._detect_contact_derivative()
        else:  # Default to "threshold"
            self._detect_contact_threshold()
            # self._detect_contact_threshold_hysteresis()

        if self.is_contact:
            self.publish_tactile_event(detected=True, disconnected=False)

        # Update previous smoothed values for next iteration
        self.prev_smoothed_p = self.smoothed_p.copy()
        self.has_prev_smoothed_p = True
        
        self.get_logger().debug(
            f"Strategy: {self.contact_detection_strategy}, Contact: {self.is_contact}"
        )

    def _detect_contact_threshold_hysteresis(self):
        """
        Threshold-based contact detection using hysteresis.
        Detects contact when pressure exceeds high thresholds.
        """
        # Detect contact per sensor with hysteresis
        active_sensor_count = 0
        for i in range(self.num_channels):
            if self.sensor_active[i]:
                if self.smoothed_p[i] < self.low_thresholds[i]:
                    self.sensor_active[i] = False
            else:
                if self.smoothed_p[i] > self.high_thresholds[i]:
                    self.sensor_active[i] = True
                    # self.get_logger().info('Sensor {} activated (value: {:.3f} > high threshold: {:.3f})'.format(
                    #     i, self.smoothed_p[i], self.high_thresholds[i]
                    # ))

            if self.sensor_active[i]:
                active_sensor_count += 1

        # Contact is active when at least N sensors are above their own thresholds
        self.is_contact = active_sensor_count >= self.contact_sensor_count

        # self.get_logger().debug(
        #     f"Active sensors: {active_sensor_count}, Contact: {self.is_contact}"
        # )
    
    def _detect_contact_threshold(self):
        """
        Threshold-based contact detection without hysteresis.
        Prefer _detect_contact_threshold_hysteresis for noisy sensors.
        """
        # Detect contact per sensor using a simple upper threshold.
        active_sensor_count = 0
        for i in range(self.num_channels):
            if self.smoothed_p[i] > self.high_thresholds[i]:
                self.sensor_active[i] = True
            else:
                self.sensor_active[i] = False

            if self.sensor_active[i]:
                active_sensor_count += 1

        # Contact is active when at least N sensors are above their own thresholds
        self.is_contact = active_sensor_count >= self.contact_sensor_count

        self.get_logger().debug(
            f"Active sensors: {active_sensor_count}, Contact: {self.is_contact}"
        )

    def _detect_contact_derivative(self):
        """
        Derivative-based contact detection using rate of change.
        Detects rapid positive pressure spikes characteristic of contact.
        This strategy is calibration-free and independent of the resting pressure level.
        """
        # Detect contact per sensor based on positive derivative only
        active_sensor_count = 0
        for i in range(self.num_channels):
            # Compute rate of change: derivative of smoothed pressure
            derivative = self.smoothed_p[i] - self.prev_smoothed_p[i]

            # Contact produces a positive spike, so only positive excursions count.
            self.sensor_active[i] = derivative > self.derivative_spike_threshold
            # self.get_logger().info(
            #     f"Sensor {i}: Smoothed={self.smoothed_p[i]:.3f}, Derivative={derivative:.3f}"
            # )

            if self.sensor_active[i]:
                active_sensor_count += 1

        # Contact is active when at least N sensors show sufficient rate of change
        self.is_contact = active_sensor_count >= self.contact_sensor_count

        self.get_logger().debug(
            f"Active sensors (derivative): {active_sensor_count}, Contact: {self.is_contact}"
        )

    @staticmethod
    def decode_taxels(data: bytearray, num_channels: int, header_bytes: int, payload_bytes: int, expected_min_len: int):
        """Decode BLE pressure packet."""
        if len(data) < expected_min_len:
            return None

        payload = data[header_bytes:header_bytes + payload_bytes]

        # Big-endian 32-bit floats: unpack all channel values at once
        try:
            values = struct.unpack(
                ">" + "f" * num_channels,
                bytes(payload),
            )
        except struct.error:
            return None

        return values


async def run_ble(node: TactileSensorNode, stop_event: asyncio.Event):
    """Main BLE connection and data streaming loop."""
    loop = asyncio.get_running_loop()
    latest_pressure_sample = None
    sample_event = asyncio.Event()

    def publish_latest_sample(values):
        """Keep only the most recent decoded sample for processing."""
        nonlocal latest_pressure_sample
        latest_pressure_sample = values
        sample_event.set()

    async def watchdog_task(timeout_ms: int):
        """
        Watchdog task that monitors if data is being received.
        Raises exception if no data received for timeout_ms milliseconds.
        """
        while not stop_event.is_set():
            await asyncio.sleep(0.1)  # Check every 100ms
            
            if node.last_data_time is None:
                continue
                
            elapsed_ms = (time.monotonic() - node.last_data_time) * 1000
            
            if elapsed_ms > timeout_ms:
                if not node.connection_lost_logged:
                    node.get_logger().warn(
                        f"No data received for {elapsed_ms:.0f}ms (timeout={timeout_ms}ms). "
                        "Triggering reconnection..."
                    )
                    node.connection_lost_logged = True
                node.publish_tactile_event(detected=False, disconnected=True)
                # Raise error only once to trigger reconnection
                raise TimeoutError(
                    f"BLE data timeout: no data for {elapsed_ms:.0f}ms"
                )

    def notification_handler(_, data: bytearray):
        """Handle incoming BLE notifications with pressure data."""
        node.mark_ble_activity()

        values = TactileSensorNode.decode_taxels(
            data,
            node.num_channels,
            node.header_bytes,
            node.payload_bytes,
            node.expected_min_len,
        )
        if values is None:
            node.get_logger().warn(
                f"Received packet too short: {len(data)} bytes"
            )
            return

        loop.call_soon_threadsafe(publish_latest_sample, values)

    node.get_logger().info(f"Connecting to BLE device {node.ble_address} ...")

    try:
        async with BleakClient(node.ble_address, bluez={"adapter": node.adapter}) as client:
            if not client.is_connected:
                raise RuntimeError(f"Failed to connect to {node.ble_address}")

            node.publish_tactile_event(detected=False, disconnected=False)
            node.get_logger().info("BLE connected")
            node.get_logger().info(
                f"Starting notifications on {BLE_NOTIFY_CHAR_UUID}"
            )

            node.last_data_time = time.monotonic()
            node.connection_lost_logged = False

            await client.start_notify(BLE_NOTIFY_CHAR_UUID, notification_handler)

            # Run main loop, watchdog, and sample processing concurrently.
            async def main_loop():
                """Main event processing loop."""
                while not stop_event.is_set():
                    rclpy.spin_once(node, timeout_sec=0.0)
                    await asyncio.sleep(0.01)

            async def processing_loop():
                """Process only the newest decoded BLE sample."""
                nonlocal latest_pressure_sample
                while not stop_event.is_set():
                    await sample_event.wait()
                    sample_event.clear()

                    if stop_event.is_set():
                        break

                    sample = latest_pressure_sample
                    if sample is None:
                        continue

                    node.process_pressure_data(sample)

            try:
                watchdog = asyncio.create_task(watchdog_task(node.ble_data_timeout_ms))
                main_loop_task = asyncio.create_task(main_loop())
                processing_task = asyncio.create_task(processing_loop())
                
                # Wait for either task to complete (exception in watchdog will propagate here)
                done, pending = await asyncio.wait(
                    [watchdog, main_loop_task, processing_task],
                    return_when=asyncio.FIRST_EXCEPTION
                )
                
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Check if watchdog raised an exception
                for task in done:
                    exc = task.exception()
                    if exc:
                        raise exc
                        
            except TimeoutError as e:
                node.get_logger().warn(f"Data timeout detected: {e}")
                # Fall through to cleanup
            finally:
                sample_event.set()
                node.get_logger().info("Stopping notifications...")
                await client.stop_notify(BLE_NOTIFY_CHAR_UUID)

    except TimeoutError as e:
        node.get_logger().error(f"BLE watchdog timeout: {e}")
        node.publish_tactile_event(detected=False, disconnected=True)
        if not stop_event.is_set():
            node.get_logger().info("Waiting 0.1 seconds before reconnection attempt...")
            await asyncio.sleep(0.1)
            if not stop_event.is_set():
                node.get_logger().info("Reconnecting to BLE device...")
                await run_ble(node, stop_event)
    except RuntimeError as e:
        node.get_logger().error(f"BLE connection failed: {e}")
        node.publish_tactile_event(detected=False, disconnected=True)
        if not stop_event.is_set():
            node.get_logger().info("Waiting 0.1 seconds before reconnection attempt...")
            await asyncio.sleep(0.1)
            if not stop_event.is_set():
                node.get_logger().info("Reconnecting to BLE device...")
                await run_ble(node, stop_event)
    except Exception as e:
        node.get_logger().error(f"BLE error: {type(e).__name__}: {e}")
        node.publish_tactile_event(detected=False, disconnected=True)
        if not stop_event.is_set():
            node.get_logger().info("Waiting 0.1 seconds before reconnection attempt...")
            await asyncio.sleep(0.1)
            if not stop_event.is_set():
                node.get_logger().info("Reconnecting to BLE device...")
                await run_ble(node, stop_event)
    finally:
        node.get_logger().info("Unknown BLE error occurred, still attempting reconnection...")
        if not stop_event.is_set():
            node.get_logger().info("Waiting 0.1 seconds before reconnection attempt...")
            await asyncio.sleep(0.1)
            if not stop_event.is_set():
                node.get_logger().info("Reconnecting to BLE device...")
                await run_ble(node, stop_event)


async def main_async():
    """Async entry point for ROS2 node with BLE integration."""
    rclpy.init()
    node = TactileSensorNode()
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    def request_stop():
        node.get_logger().info("Ctrl+C received, shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            pass

    try:
        await run_ble(node, stop_event)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
