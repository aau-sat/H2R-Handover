#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from force_estimation.msg import SensorData
from ft300python.ft300_stream import FT300Stream
import threading
import time

PORT = "/dev/ttyUSB0"

class SensorDataPublisher(Node):
    def __init__(self, port):
        super().__init__('sensor_data_publisher')
        
        # Initialize FT300 Hardware
        self.ft_stream = FT300Stream(port, timeout=1, zero_reset=False)
        
        # ROS2 Publisher
        self.publisher = self.create_publisher(SensorData, 'sensor_data', 10)
        
        self.running = True
        self.thread = threading.Thread(target=self._sensor_loop, daemon=True)
        self.thread.start()
        
        self.get_logger().info("FT Sensor Publisher Started (Data Only)")

    def _sensor_loop(self):
        while self.running and rclpy.ok():
            try:
                ft_values = self.ft_stream.get_force_torque() # [fx, fy, fz, tx, ty, tz]
                
                if ft_values:
                    msg = SensorData()
                    msg.header = Header()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    
                    # Map FT values
                    msg.fx, msg.fy, msg.fz = ft_values[0], ft_values[1], ft_values[2]
                    msg.tx, msg.ty, msg.tz = ft_values[3], ft_values[4], ft_values[5]


                    self.publisher.publish(msg)
                
                time.sleep(0.001) 

            except Exception as e:
                if self.running:
                    self.get_logger().warn(f"Sensor Read Error: {e}")
                time.sleep(0.01)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = SensorDataPublisher(PORT)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()