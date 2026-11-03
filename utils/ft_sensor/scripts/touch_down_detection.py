import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup # <--- Import this
from std_srvs.srv import Trigger
import math
import statistics
import threading
from ft_sensor_msgs.msg import SensorData

class TouchdownDetector(Node):
    def __init__(self):
        super().__init__('touchdown_detector')
        
        # --- Parameters ---
        self.declare_parameter('force_threshold', 12.0)
        self.declare_parameter('calibration_samples', 100)
        self.declare_parameter('window_size', 5)
        
        self.threshold = self.get_parameter('force_threshold').get_parameter_value().double_value
        self.cal_limit = self.get_parameter('calibration_samples').get_parameter_value().integer_value
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value

        # --- Internal State ---
        self.is_monitoring = False
        self.touchdown_detected = False
        self.final_median = 0.0
        
        self.cal_buffer = []
        self.force_buffer = []
        self.bias = [0.0, 0.0, 0.0]
        
        self.detection_event = threading.Event()

        # --- CRITICAL CHANGE: Callback Group ---
        # Using a Reentrant group allows the executor to run 
        # the subscriber even while the service is waiting.
        self.cb_group = ReentrantCallbackGroup()

        self.subscription = self.create_subscription(
            SensorData, 
            '/sensor_data', 
            self.sensor_data_cb, 
            10,
            callback_group=self.cb_group # <--- Assign group
        )
        
        self.srv = self.create_service(
            Trigger, 
            'detect_touchdown', 
            self.handle_touchdown_request,
            callback_group=self.cb_group # <--- Assign group
        )

        self.get_logger().info("Touchdown Detector Node initialized.")

    def handle_touchdown_request(self, request, response):
        self.get_logger().info("Touchdown requested. Resetting buffers...")
        
        self.cal_buffer = []
        self.force_buffer = []
        self.touchdown_detected = False
        self.detection_event.clear()
        self.is_monitoring = True
        
        # This will now work because sensor_data_cb can run in parallel
        finished = self.detection_event.wait(timeout=None) 
        
        self.is_monitoring = False
        
        if finished and self.touchdown_detected:
            response.success = True
            response.message = f"Touchdown detected at {self.final_median:.2f}N"
        else:
            response.success = False
            response.message = "Detection timed out or failed."
            
        return response

    def sensor_data_cb(self, msg: SensorData):
        if not self.is_monitoring:
            return

        if len(self.cal_buffer) < self.cal_limit:
            self.cal_buffer.append([msg.fx, msg.fy, msg.fz])
            if len(self.cal_buffer) == self.cal_limit:
                self.bias[0] = sum(p[0] for p in self.cal_buffer) / self.cal_limit
                self.bias[1] = sum(p[1] for p in self.cal_buffer) / self.cal_limit
                self.bias[2] = sum(p[2] for p in self.cal_buffer) / self.cal_limit
                self.get_logger().info("Calibration complete.")
            return

        fx, fy, fz = msg.fx - self.bias[0], msg.fy - self.bias[1], msg.fz - self.bias[2]
        current_magnitude = math.sqrt(fx**2 + fy**2 + fz**2)
        
        self.force_buffer.append(current_magnitude)
        if len(self.force_buffer) > self.window_size:
            self.force_buffer.pop(0)

        if len(self.force_buffer) == self.window_size:
            median_val = statistics.median(self.force_buffer)
            if median_val > self.threshold:
                self.final_median = median_val
                self.touchdown_detected = True
                self.detection_event.set() 

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