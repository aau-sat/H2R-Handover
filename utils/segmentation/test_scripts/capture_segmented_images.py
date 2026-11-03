#!/usr/bin/env python3
"""
Script to capture synchronized RGB, depth, and segmentation mask images.
Auto-captures continuously by default. Use --manual for manual mode.
"""

import os
import json
import socket
import argparse
import threading
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import message_filters


SOCKET_PATH = "/tmp/sam3_socket"


class ImageCaptureNode(Node):
    """
    ROS 2 node that captures synchronized RGB, depth, and segmentation images.
    """
    def __init__(self, rgb_topic, depth_topic, output_dir, auto_capture=True, capture_delay=0.2):
        super().__init__("image_capture_node")
        
        self.bridge = CvBridge()
        self.output_dir = output_dir
        self.capture_index = 1  # Start from 1 for better UX
        self.auto_capture = auto_capture
        self.capture_delay = capture_delay  # Seconds between auto-captures
        
        # Latest detection
        self.latest_detection = None
        self.last_valid_detection = None
        
        # Capture control
        self.capture_requested = False
        self.capture_lock = threading.Lock()
        self.last_capture_time = 0
        self.is_capturing = False  # Flag to prevent overlapping captures
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subscribe to detected objects
        self.create_subscription(String, "/detected_objects", self.detection_callback, 10)
        
        # Synchronized RGB and depth subscriptions
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.images_callback)
        
        self.get_logger().info(f"Image capture node initialized")
        self.get_logger().info(f"RGB topic: {rgb_topic}")
        self.get_logger().info(f"Depth topic: {depth_topic}")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info("Waiting for data...")
    
    def detection_callback(self, msg):
        """
        Callback for detected objects topic.
        
        Args:
            msg (std_msgs.msg.String): JSON string with detection data
        """
        try:
            detections = json.loads(msg.data)
            if detections:
                # Get best detection
                best = max(detections, key=lambda x: x.get("conf", 0.0))
                
                bbox = best.get("bbox")
                label = best.get("class")
                
                if bbox and label and len(bbox) == 4:
                    self.latest_detection = {
                        "bbox": bbox,
                        "label": label,
                        "conf": best.get("conf", 0.0)
                    }
                    self.last_valid_detection = self.latest_detection
                else:
                    self.latest_detection = None
            else:
                self.latest_detection = None
        except Exception as e:
            self.get_logger().error(f"Detection parse error: {e}")
    
    def images_callback(self, rgb_msg, depth_msg):
        """
        Callback for synchronized RGB and depth images.
        
        Args:
            rgb_msg (sensor_msgs.msg.Image): RGB image message
            depth_msg (sensor_msgs.msg.Image): Depth image message
        """
        # Convert images for visualization (always show live feed)
        rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        
        # Visualize the streams
        self.visualize_streams(rgb_frame, depth_frame)
        
        with self.capture_lock:
            # Check if we should capture
            current_time = time.time()
            
            if self.auto_capture:
                # Auto-capture mode: check if enough time has passed
                if self.is_capturing:
                    return  # Skip if already processing a capture
                    
                if current_time - self.last_capture_time < self.capture_delay:
                    return  # Not enough time has passed
            else:
                # Manual mode: only capture if requested
                if not self.capture_requested:
                    return
            
            # Use latest or last valid detection
            detection = self.latest_detection or self.last_valid_detection
            
            if not detection:
                if self.auto_capture:
                    return  # Silently skip in auto mode
                else:
                    self.get_logger().warn("No detection available, skipping capture")
                    self.capture_requested = False
                return
            
            try:
                # Mark as capturing to prevent overlaps
                self.is_capturing = True
                
                # Convert RGB to format expected by segmentation service
                rgb_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                
                h, w = rgb_rgb.shape[:2]
                bbox = detection["bbox"]
                label = detection["label"]
                x1, y1, x2, y2 = map(int, bbox)
                
                self.get_logger().info(f"Capturing with detection: {label} [{x1}, {y1}, {x2}, {y2}]")
                
                # Get segmentation mask
                mask = self.call_segmentation_service(rgb_rgb, h, w, x1, y1, x2, y2, label)
                
                if mask is not None and mask.size > 0:
                    # Save all three images
                    self.save_image_set(rgb_frame, depth_frame, mask, detection)
                    self.get_logger().info("=" * 60)
                    self.get_logger().info(f"✓ Successfully saved image set #{self.capture_index}")
                    self.get_logger().info(f"  Output directory: {self.output_dir}")
                    self.get_logger().info("=" * 60)
                    self.capture_index += 1
                    # Update last capture time
                    self.last_capture_time = current_time
                else:
                    self.get_logger().warn("Failed to get segmentation mask")
                
                self.capture_requested = False
                self.is_capturing = False
                
            except Exception as e:
                self.get_logger().error(f"Image capture error: {e}")
                self.capture_requested = False
                self.is_capturing = False
    
    def call_segmentation_service(self, frame, h, w, x1, y1, x2, y2, label):
        """
        Call the Unix socket segmentation service.
        
        Args:
            frame (np.ndarray): RGB image (H, W, 3)
            h, w (int): Image dimensions
            x1, y1, x2, y2 (int): Bounding box
            label (str): Object label
            
        Returns:
            np.ndarray: Segmentation mask or None on failure
        """
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            # Set a 60 second timeout for the entire operation
            client.settimeout(60.0)
            client.connect(SOCKET_PATH)
            
            # Encode label
            label_bytes = label.encode()
            label_len = len(label_bytes)
            
            # Send header
            header = np.array([h, w, x1, y1, x2, y2, label_len], dtype=np.int32)
            client.sendall(header.tobytes())
            
            # Send label
            client.sendall(label_bytes)
            
            # Send image
            img_bytes = frame.astype(np.uint8).tobytes()
            client.sendall(img_bytes)
            
            self.get_logger().info(f"Data sent to segmentation service. Waiting for mask (this may take 10-30 seconds)...")
            
            # Receive mask
            mask_size = h * w
            mask_data = b""
            
            # Receive with timeout
            start_time = time.time()
            while len(mask_data) < mask_size:
                try:
                    remaining = mask_size - len(mask_data)
                    chunk = client.recv(min(4096, remaining))
                    if not chunk:
                        # Connection closed by server
                        elapsed = time.time() - start_time
                        self.get_logger().error(
                            f"Connection closed by server after {elapsed:.1f}s. "
                            f"Received {len(mask_data)}/{mask_size} bytes"
                        )
                        break
                    mask_data += chunk
                    
                    # Log progress for large transfers
                    if len(mask_data) % 50000 == 0 or len(mask_data) == mask_size:
                        progress = (len(mask_data) / mask_size) * 100
                        self.get_logger().info(f"Receiving mask: {progress:.1f}% ({len(mask_data)}/{mask_size} bytes)")
                        
                except socket.timeout:
                    elapsed = time.time() - start_time
                    self.get_logger().error(f"Socket timeout after {elapsed:.1f}s while receiving mask")
                    break
            
            client.close()
            
            if len(mask_data) == mask_size:
                mask = np.frombuffer(mask_data, dtype=np.uint8).reshape((h, w))
                self.get_logger().info("Mask received successfully")
                return mask
            else:
                self.get_logger().error(
                    f"Incomplete mask: {len(mask_data)}/{mask_size} bytes. "
                    "Segmentation service may have crashed or returned invalid data."
                )
                return None
                
        except socket.timeout:
            self.get_logger().error("Segmentation service timeout (60s). Service may be overloaded.")
            return None
        except ConnectionRefusedError:
            self.get_logger().error("Connection refused. Is the segmentation service running?")
            return None
        except Exception as e:
            self.get_logger().error(f"Segmentation service error: {e}")
            return None
    
    def visualize_streams(self, rgb_frame, depth_frame):
        """
        Visualize RGB and depth streams in OpenCV windows.
        
        Args:
            rgb_frame (np.ndarray): RGB image (BGR format)
            depth_frame (np.ndarray): Depth image
        """
        # Create a copy for visualization with detection bbox
        rgb_vis = rgb_frame.copy()
        
        # Draw detection bbox if available
        if self.latest_detection:
            bbox = self.latest_detection["bbox"]
            label = self.latest_detection["label"]
            conf = self.latest_detection["conf"]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bbox
            cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            label_text = f"{label} {conf:.2f}"
            cv2.putText(rgb_vis, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add capture count overlay
        cv2.putText(rgb_vis, f"Captured: {self.capture_index - 1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Normalize and colorize depth for visualization
        depth_vis = np.zeros_like(depth_frame, dtype=np.uint8)
        valid_depth = depth_frame > 0
        if np.any(valid_depth):
            depth_normalized = cv2.normalize(
                depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        else:
            depth_colormap = np.zeros((depth_frame.shape[0], depth_frame.shape[1], 3), dtype=np.uint8)
        
        # Add depth stats overlay
        if np.any(valid_depth):
            min_d = depth_frame[valid_depth].min()
            max_d = depth_frame[valid_depth].max()
            mean_d = depth_frame[valid_depth].mean()
            cv2.putText(depth_colormap, f"Depth: {min_d:.0f}-{max_d:.0f}mm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(depth_colormap, f"Mean: {mean_d:.0f}mm", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show the images
        cv2.imshow("RGB Stream", rgb_vis)
        cv2.imshow("Depth Stream", depth_colormap)
        cv2.waitKey(1)
    
    def save_image_set(self, rgb, depth, mask, detection):
        """
        Save RGB, depth, and mask images with the same index.
        
        Args:
            rgb (np.ndarray): RGB image (BGR format)
            depth (np.ndarray): Depth image
            mask (np.ndarray): Segmentation mask
            detection (dict): Detection information
        """
        idx = self.capture_index
        
        # Save RGB (as BGR for OpenCV)
        rgb_path = os.path.join(self.output_dir, f"{idx:04d}_rgb.png")
        cv2.imwrite(rgb_path, rgb)
        
        # Save depth (as 16-bit PNG to preserve depth values)
        depth_path = os.path.join(self.output_dir, f"{idx:04d}_depth.png")
        if depth.dtype != np.uint16:
            # Convert to uint16 if needed
            depth_uint16 = depth.astype(np.uint16)
        else:
            depth_uint16 = depth
        cv2.imwrite(depth_path, depth_uint16)
        
        # Save mask
        mask_path = os.path.join(self.output_dir, f"{idx:04d}_mask.png")
        cv2.imwrite(mask_path, mask)
        
        # Save metadata
        meta_path = os.path.join(self.output_dir, f"{idx:04d}_meta.json")
        metadata = {
            "index": idx,
            "label": detection["label"],
            "bbox": detection["bbox"],
            "confidence": detection["conf"],
            "rgb_file": f"{idx:04d}_rgb.png",
            "depth_file": f"{idx:04d}_depth.png",
            "mask_file": f"{idx:04d}_mask.png"
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def request_capture(self):
        """Request a capture on the next synchronized frame."""
        with self.capture_lock:
            self.capture_requested = True
            self.get_logger().info("Capture requested, waiting for next frame...")


def input_thread(node):
    """Thread to handle user input - only for stopping in auto mode."""
    if node.auto_capture:
        print("\n" + "="*60)
        print("AUTO-CAPTURE MODE")
        print("="*60)
        print(f"Capturing images every {node.capture_delay} seconds")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Just wait for Ctrl+C
        try:
            while rclpy.ok():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            rclpy.shutdown()
    else:
        print("\n" + "="*60)
        print("MANUAL CAPTURE MODE")
        print("="*60)
        print("Press ENTER to capture an image set")
        print("Type 'q' or 'quit' and press ENTER to exit")
        print("="*60 + "\n")
        
        import sys
        while rclpy.ok():
            try:
                sys.stdout.flush()
                user_input = input(">>> Press ENTER to capture (or 'q' to quit): ").strip().lower()
                
                if user_input in ['q', 'quit', 'exit']:
                    print("Exiting...")
                    rclpy.shutdown()
                    break
                
                # Empty input means capture
                node.request_capture()
                
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                rclpy.shutdown()
                break


def main():
    parser = argparse.ArgumentParser(
        description="Capture synchronized RGB, depth, and segmentation mask images"
    )
    parser.add_argument(
        "--rgb-topic",
        type=str,
        default="/camera/cam_right/color/image_raw",
        help="RGB image topic (default: /camera/cam_right/color/image_raw)"
    )
    parser.add_argument(
        "--depth-topic",
        type=str,
        default="/camera/cam_right/aligned_depth_to_color/image_raw",
        help="Aligned depth image topic (default: /camera/cam_right/aligned_depth_to_color/image_raw)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="captured_images",
        help="Output directory for saved images (default: captured_images)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="Auto-capture mode (default: True)"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Manual capture mode (press Enter to capture)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between auto-captures in seconds (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Manual mode overrides auto mode
    auto_capture = not args.manual
    
    rclpy.init()
    node = ImageCaptureNode(
        args.rgb_topic, 
        args.depth_topic, 
        args.output_dir,
        auto_capture=auto_capture,
        capture_delay=args.delay
    )
    
    # Start input thread
    input_handler = threading.Thread(target=input_thread, args=(node,), daemon=True)
    input_handler.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
