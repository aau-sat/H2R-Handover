import socket
import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox
import cv2
import json
from ultralytics import YOLO
# ... (Keep your existing imports from sam3.model.box_ops etc.)

# ================= CONFIGURATION =================
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
MODEL_PATH = "/home/nadezhda/rgmc-2026/utils/handover_detection/src/handover_detection/yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.8

# Classes to detect (Same as your ROS node)
FILTERED_IDS = [39, 40, 41, 45, 46, 47, 48, 49, 53, 54, 73, 75]
# =================================================

SOCKET_PATH = "/tmp/sam3_socket"

def run_server():
    # Model Initialization (Keep model in GPU memory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    print(f"Loading YOLO model from {MODEL_PATH}...")
    yolo_model = YOLO(MODEL_PATH)

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    print(f"SAM3 Server listening on {SOCKET_PATH}...")

    while True:
        conn, _ = server.accept()
        try:
            # 1. Read Metadata (Header)
            # Format: H, W, x1, y1, x2, y2, label_len (Total 28 bytes if using int32)
            # header = conn.recv(28)
            # h, w, x1, y1, x2, y2, l_len = np.frombuffer(header, dtype=np.int32)
            # label = conn.recv(l_len).decode()
            header = conn.recv(8)
            h, w = np.frombuffer(header, dtype=np.int32)

            # 2. Read Raw Image Bytes
            img_size = h * w * 3
            img_data = b""
            while len(img_data) < img_size:
                img_data += conn.recv(img_size - len(img_data))
            
            frame = np.frombuffer(img_data, dtype=np.uint8).reshape((h, w, 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLO inference
            results = yolo_model.track(frame, persist=False, verbose=False)
        
            # Prepare dictionary for this image
            detections = []
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
                        class_name = yolo_model.names[cls_id]

                        # Add to result dictionary
                        detection_data = {
                            "class": class_name,
                            "id": cls_id,
                            "confidence": round(confidence, 4),
                            "bbox": [x1, y1, x2, y2]
                        }
                        detections.append(detection_data)
            detection = max(detections, key=lambda x: x["confidence"])

            # frame is in openCV format (H, W, C) with RGB channels
            image = Image.fromarray(frame)
            width, height = image.size
            label = detection["class"]
            conf = detection["confidence"]
            x1, y1, x2, y2 = detection["bbox"]
            
            # Convert and normalize bbox for SAM3
            box_xywh = torch.tensor([float(x1), float(y1), float(x2-x1), float(y2-y1)]).view(-1, 4)
            box_cxcywh = box_xywh_to_cxcywh(box_xywh)
            norm_box = normalize_bbox(box_cxcywh, width, height).flatten().tolist()
            sam_box = [[norm_box]]

            inputs = processor(
                images=image,
                text=label,
                input_boxes=sam_box,
                input_boxes_labels=[[1]],
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
            
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            mask_final = np.array([])
            if len(results["masks"]) > 0:
                idx = torch.argmax(results["scores"])
                best_mask = results["masks"][idx]
                mask_final = (best_mask.cpu().numpy().astype(np.uint8) * 255)
            
            # 4. Send back the mask
            conn.sendall(mask_final.tobytes())
            # break
        finally:
            conn.close()
            torch.cuda.empty_cache()
            exit(0)

if __name__ == "__main__":
    run_server()