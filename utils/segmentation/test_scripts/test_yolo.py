"""
YOLO Object Detection Tester
============================

This script runs a YOLOv8 model on local images to detect specific object classes
(e.g., bottles, cups, keyboards, etc.) related to a handover scenario. It is a 
stripped-down version of a ROS2 node, intended for standalone testing without ROS.

Functionality:
--------------
1.  **Input:** Reads images from a specified input folder (default: 'input_images').
    - Supports .jpg, .jpeg, .png, .bmp.
2.  **Inference:** Loads 'yolov8n.pt' and detects objects belonging to a 
    pre-defined list of classes (FILTERED_IDS).
3.  **Filtering:** Filters detections based on class ID and a confidence threshold.
4.  **Output:** - Appends detection results (bounding boxes, labels, confidence) as JSON 
      lines to 'output/results.txt'.
    - Visually displays the image with drawn bounding boxes (unless --no-show is used).

Usage:
------
    python test_yolo.py                     # Process all images in default folders
    python test_yolo.py --single            # Process only the first image
    python test_yolo.py --no-show           # Batch process (no GUI)
    python test_yolo.py --input <dir>       # Specify input directory
    python test_yolo.py --output <dir>      # Specify output directory
"""


import cv2
import os
import glob
import json
import argparse
from ultralytics import YOLO

# ================= CONFIGURATION =================
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
MODEL_PATH = "/home/nadezhda/rgmc-2026/utils/handover_detection/src/handover_detection/yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.8

# Classes to detect (Same as your ROS node)
FILTERED_IDS = [39, 40, 41, 45, 46, 47, 48, 49, 53, 54, 73, 75]
# =================================================

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Test Script on Local Images")
    
    parser.add_argument('--input', type=str, default="test_images", 
                        help='Path to the input folder containing images.')
    
    parser.add_argument('--output', type=str, default="output", 
                        help='Path to the output folder for results.')
    
    parser.add_argument('--single', action='store_true', 
                        help='If set, only process the first image found.')
    
    parser.add_argument('--no-show', action='store_true', 
                        help='If set, do not display the images (useful for batch processing).')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created output directory: {args.output}")

    # Output file path
    output_txt_path = os.path.join(args.output, "results.txt")
    
    # Clear previous results file
    with open(output_txt_path, 'w') as f:
        f.write("") 

    # 1. Load the YOLO model
    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # 2. Get list of images
    image_paths = []
    for ext in EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(args.input, ext)))
    
    image_paths.sort()

    if not image_paths:
        print(f"No images found in folder: {args.input}")
        return

    # Handle the --single argument
    if args.single:
        image_paths = [image_paths[0]]
        print("Mode: Processing ONLY the first image.")
    else:
        print(f"Mode: Processing ALL {len(image_paths)} images.")

    print(f"Results will be saved to: {output_txt_path}")
    if not args.no_show:
        print("Press 'space' or 'enter' for next image, 'q' to quit early.")

    # 3. Process images
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Run YOLO inference
        results = model.track(frame, persist=False, verbose=False)
        
        # Prepare dictionary for this image
        image_result = {
            "image": filename,
            "detections": []
        }

        if results:
            result = results[0]
            if result.boxes and result.boxes.id is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    
                    if cls_id not in FILTERED_IDS:
                        continue

                    confidence = box.conf.item()
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = model.names[cls_id]

                    # Add to result dictionary
                    detection_data = {
                        "class": class_name,
                        "id": cls_id,
                        "confidence": round(confidence, 4),
                        "bbox": [x1, y1, x2, y2]
                    }
                    image_result["detections"].append(detection_data)

                    # Draw graphics (only if we are showing images)
                    if not args.no_show:
                        label = f"{class_name} {confidence:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Write dictionary to text file (append mode)
        with open(output_txt_path, 'a') as f:
            f.write(json.dumps(image_result) + "\n")
        
        print(f"Processed {filename}: Found {len(image_result['detections'])} objects.")

        # Display Logic
        if not args.no_show:
            cv2.imshow("YOLO Test", frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("Quitting early...")
                break

    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()