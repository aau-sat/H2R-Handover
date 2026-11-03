#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from std_srvs.srv import Trigger
from rclpy.qos import QoSProfile, ReliabilityPolicy
from statistics import mean, pstdev
import math

# Contact detection parameters
DEFAULT_ALPHA = 1.0 #0.8
DEFAULT_NUM_CHANNELS = 7
DEFAULT_CONTACT_SENSOR_COUNT = 2

# Calibration parameters
DEFAULT_CALIBRATION_SAMPLES = 100  # Number of samples to collect for baseline
DEFAULT_STD_DEV_HIGH_MULTIPLIER = 3.0  # HIGH_THRESHOLD = mean + (std_dev * this)
DEFAULT_STD_DEV_LOW_MULTIPLIER = 1.0   # LOW_THRESHOLD = mean + (std_dev * this)
DEFAULT_MIN_STD_EPS = 1e-6

DEFAULT_DEVICE_UID = "DEVB9975E99211783F"

class TaxelContactDetector(Node):
    def __init__(self):
        super().__init__("taxel_contact_detector")

        def get_param_value(name, default):
            value = self.get_parameter(name).value
            return default if value is None else value

        # Parameters
        self.declare_parameter("alpha", DEFAULT_ALPHA)
        self.declare_parameter("num_channels", DEFAULT_NUM_CHANNELS)
        self.declare_parameter("contact_sensor_count", DEFAULT_CONTACT_SENSOR_COUNT)
        self.declare_parameter("calibration_samples", DEFAULT_CALIBRATION_SAMPLES)
        self.declare_parameter("std_dev_high_multiplier", DEFAULT_STD_DEV_HIGH_MULTIPLIER)
        self.declare_parameter("std_dev_low_multiplier", DEFAULT_STD_DEV_LOW_MULTIPLIER)
        self.declare_parameter("min_std_eps", DEFAULT_MIN_STD_EPS)
        self.declare_parameter("device_uid", DEFAULT_DEVICE_UID)

        self.alpha = float(get_param_value("alpha", DEFAULT_ALPHA))
        self.num_channels = int(get_param_value("num_channels", DEFAULT_NUM_CHANNELS))
        self.contact_sensor_count = int(get_param_value("contact_sensor_count", DEFAULT_CONTACT_SENSOR_COUNT))
        self.calibration_samples = int(get_param_value("calibration_samples", DEFAULT_CALIBRATION_SAMPLES))
        self.std_dev_high_multiplier = float(get_param_value("std_dev_high_multiplier", DEFAULT_STD_DEV_HIGH_MULTIPLIER))
        self.std_dev_low_multiplier = float(get_param_value("std_dev_low_multiplier", DEFAULT_STD_DEV_LOW_MULTIPLIER))
        self.min_std_eps = float(get_param_value("min_std_eps", DEFAULT_MIN_STD_EPS))
        self.device_uid = str(get_param_value("device_uid", DEFAULT_DEVICE_UID))
        
        # Subscribe to pressure topic with best-effort QoS
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            f"/{self.device_uid}/Pressure",
            self.pressure_callback,
            qos_profile,
        )
        
        # Publisher for contact status
        self.contact_pub = self.create_publisher(
            Bool,
            f"/{self.device_uid}/Contact",
            10,
        )
        
        # Service for calibration
        self.calibration_service = self.create_service(
            Trigger,
            f"/{self.device_uid}/calibrate_contact",
            self.calibrate_callback,
        )
        
        # State variables
        self.smoothed_p = [0.0] * self.num_channels
        self.sensor_active = [False] * self.num_channels
        self.is_contact = False
        
        # Calibration state
        self.is_calibrating = False
        self.calibration_buffer = []
        self.baseline_mean = [0.0] * self.num_channels
        self.baseline_std = [1.0] * self.num_channels
        self.high_thresholds = [math.inf] * self.num_channels
        self.low_thresholds = [math.inf] * self.num_channels
        self.is_calibrated = False
        
        self.get_logger().info("Taxel Contact Detector initialized")
        self.get_logger().info(
            f"Call '/{self.device_uid}/calibrate_contact' service to calibrate sensors"
        )
        self.get_logger().info(
            "Parameters: "
            f"alpha={self.alpha}, num_channels={self.num_channels}, "
            f"contact_sensor_count={self.contact_sensor_count}, "
            f"calibration_samples={self.calibration_samples}, "
            f"std_dev_high_multiplier={self.std_dev_high_multiplier}, "
            f"std_dev_low_multiplier={self.std_dev_low_multiplier}, "
            f"min_std_eps={self.min_std_eps}, device_uid={self.device_uid}"
        )
    
    def calibrate_callback(self, request, response):
        """ROS2 service to start calibration."""
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
        
        # Convert to numpy array (shape: [samples, channels])
        channels = list(zip(*self.calibration_buffer))

        # Compute mean and std per channel
        self.baseline_mean = [mean(ch) for ch in channels]
        self.baseline_std = [pstdev(ch) if len(ch) > 1 else 0.0 for ch in channels]
        
        # Compute per-sensor thresholds based on the calibrated mean and std deviation
        std = [max(v, self.min_std_eps) for v in self.baseline_std]
        self.high_thresholds = [
            m + (self.std_dev_high_multiplier * s)
            for m, s in zip(self.baseline_mean, std)
        ]
        self.low_thresholds = [
            m + (self.std_dev_low_multiplier * s)
            for m, s in zip(self.baseline_mean, std)
        ]

        # Start from the calibrated baseline and clear the per-sensor contact state
        self.smoothed_p = self.baseline_mean.copy()
        self.sensor_active = [False] * self.num_channels
        
        self.is_calibrated = True
        
        self.get_logger().info("=== Calibration Complete ===")
        self.get_logger().info(f"Baseline Mean: {self.baseline_mean}")
        self.get_logger().info(f"Baseline Std: {self.baseline_std}")
        self.get_logger().info(f"High Thresholds: {self.high_thresholds}")
        self.get_logger().info(f"Low Thresholds: {self.low_thresholds}")
    
    def pressure_callback(self, msg: Float32MultiArray):
        """Process incoming pressure data and detect contact."""
        raw_p = list(msg.data)
        
        if len(raw_p) != self.num_channels:
            self.get_logger().warn(
                f"Expected {self.num_channels} channels, got {len(raw_p)}"
            )
            return
        
        # Handle calibration mode
        if self.is_calibrating:
            self.calibration_buffer.append(raw_p.copy())
            
            if len(self.calibration_buffer) >= self.calibration_samples:
                self.is_calibrating = False
                self.compute_calibration()
            return
        
        # Skip contact detection if not calibrated
        if not self.is_calibrated:
            # self.get_logger().warn(
            #     "Sensor not calibrated yet. Call calibration service first."
            # )
            return
        
        # Smooth the raw pressure values using exponential moving average
        self.smoothed_p = [
            (self.alpha * raw) + ((1 - self.alpha) * smooth)
            for raw, smooth in zip(raw_p, self.smoothed_p)
        ]

        active_sensor_count = 0
        for i in range(self.num_channels):
            # if self.sensor_active[i]:
            #     if self.smoothed_p[i] < self.low_thresholds[i]:
            #         self.sensor_active[i] = False
            # else:
            if self.smoothed_p[i] > self.high_thresholds[i]:
                self.sensor_active[i] = True
            else:
                self.sensor_active[i] = False

            if self.sensor_active[i]:
                active_sensor_count += 1

        # Contact is active when at least two sensors are above their own thresholds
        self.is_contact = active_sensor_count >= self.contact_sensor_count
        
        if self.is_contact:
            # Publish contact status
            status_msg = Bool()
            status_msg.data = self.is_contact
            self.contact_pub.publish(status_msg)
        
        self.get_logger().debug(
            f"Active sensors: {active_sensor_count}"
        )


def main():
    rclpy.init()
    node = TaxelContactDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
