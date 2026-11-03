import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from std_srvs.srv import Trigger
import math
import time
import statistics  # Added for median calculation
from ft300python.ft300_stream import FT300Stream


class TouchdownDetector(Node):
    def __init__(self):
        super().__init__('touchdown_detector')
      
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('force_threshold', 2.0)
        self.declare_parameter('calibration_samples', 100)
        self.declare_parameter('window_size', 5) # Added parameter for flexibility
        
        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.threshold = self.get_parameter('force_threshold').get_parameter_value().double_value
        self.cal_limit = self.get_parameter('calibration_samples').get_parameter_value().integer_value
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        self.calibrated = False
        self.touchdown_achieved = False
        self.bias_fx = 0.0
        self.bias_fy = 0.0
        self.bias_fz = 0.0
        self.force_buffer = [] # Buffer for rolling samples

        # --- Sensor Setup ---
        try:
            self.ft_stream = FT300Stream(self.port, timeout=1, zero_reset=True)
            self.get_logger().info(f"FT300 connected on {self.port}. Median filter size: {self.window_size}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect: {e}")
            return

        # timer to publish real force data for testing
        self.publish_timer = self.create_timer(0.005, self.publish_force_data)
        self.force_publisher = self.create_publisher(Float32, '/median_force', 10)
        self.current_force_publisher = self.create_publisher(Float32, '/current_force', 10) # For debugging
        self.touchdown_publisher = self.create_publisher(Bool, '/touchdown_status', 10)
       
    def publish_force_data(self):
        if not self.calibrated:
            self.publish_timer.cancel()  # Stop the timer until calibrated
            self.calibrate()
            return  # Don't publish until calibrated

        median_force = 0.0
      
        try:
            ft_values = self.ft_stream.get_force_torque()
              
            # Subtract the bias
            fx = ft_values[0] - self.bias_fx
            fy = ft_values[1] - self.bias_fy
            fz = ft_values[2] - self.bias_fz
              
            # Resultant force (Magnitude)
            current_magnitude = math.sqrt(fx**2 + fy**2 + fz**2)

            # Maintain the sliding window
            self.force_buffer.append(current_magnitude)
            if len(self.force_buffer) > self.window_size:
                self.force_buffer.pop(0)

            # Only check once the buffer is full to avoid premature triggers
            if len(self.force_buffer) == self.window_size:
                median_force = statistics.median(self.force_buffer)
                self.force_publisher.publish(Float32(data=median_force))

            if median_force > self.threshold:
                self.get_logger().warn(f"TOUCHDOWN DETECTED: Median {median_force:.4f}N")
                self.touchdown_achieved = True
            
            self.touchdown_publisher.publish(Bool(data=self.touchdown_achieved))
            self.current_force_publisher.publish(Float32(data=current_magnitude))

            print(f"Current Force: {current_magnitude:.4f}N, Median: {median_force:.4f}N, Touchdown: {self.touchdown_achieved}")
        except Exception as e:
           self.get_logger().error(f"Error during detection: {e}")

    def calibrate(self):
        self.get_logger().info("Calibrating sensor...")
      
        # 1. Dynamic Calibration (Taring)
        bias_fx, bias_fy, bias_fz = 0.0, 0.0, 0.0
        for _ in range(self.cal_limit):
            ft = self.ft_stream.get_force_torque()
            bias_fx += ft[0]
            bias_fy += ft[1]
            bias_fz += ft[2]
            time.sleep(0.002)
      
        self.bias_fx /= self.cal_limit
        self.bias_fy /= self.cal_limit
        self.bias_fz /= self.cal_limit

        self.get_logger().info(f"Bias calculated. Monitoring for > {self.threshold}N (Median of {self.window_size}).")
        self.calibrated = True
        self.publish_timer.reset()  # Restart the timer to begin publishing and detection

def main(args=None):
   rclpy.init(args=args)
   node = TouchdownDetector()
  
   executor = rclpy.executors.MultiThreadedExecutor()
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

