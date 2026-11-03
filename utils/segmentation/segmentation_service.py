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

SOCKET_PATH = "/tmp/sam3_socket"

def run_server():
    # Model Initialization (Keep model in GPU memory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

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
            header = conn.recv(28)
            h, w, x1, y1, x2, y2, l_len = np.frombuffer(header, dtype=np.int32)
            label = conn.recv(l_len).decode()
            print(f"Received image parameters: size ({h}, {w}) with box ({x1}, {y1}, {x2}, {y2}) and label '{label}'")

            # 2. Read Raw Image Bytes
            img_size = h * w * 3
            img_data = b""
            while len(img_data) < img_size:
                img_data += conn.recv(img_size - len(img_data))
            
            print(f"Received image data of length: {len(img_data)} bytes")
            frame = np.frombuffer(img_data, dtype=np.uint8).reshape((h, w, 3))
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # frame is in openCV format (H, W, C) with RGB channels
            image = Image.fromarray(frame)

            # debug: dump image
            image.save("debug_input_image.png")
            
            # if bbox is missing or invalid, only use label
            if x1 == x2 or y1 == y2:
                inputs = processor(
                    images=image,
                    text=label,
                    return_tensors="pt"
                ).to(device)
            else:
                # Convert and normalize bbox for SAM3
                box_xywh = torch.tensor([float(x1), float(y1), float(x2-x1), float(y2-y1)]).view(-1, 4)
                box_cxcywh = box_xywh_to_cxcywh(box_xywh)
                norm_box = normalize_bbox(box_cxcywh, w, h).flatten().tolist()
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
                threshold=0.3,
                mask_threshold=0.3,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            print(f"Segmentation results: {len(results['masks'])} masks found with scores {results['scores'].cpu().numpy()}")

            mask_final = np.array([])
            if len(results["masks"]) > 0:
                idx = torch.argmax(results["scores"])
                best_mask = results["masks"][idx]
            else:
                print("No masks detected, sending empty mask.")
                best_mask = torch.zeros((1, h, w), dtype=torch.uint8)
            mask_final = (best_mask.cpu().numpy().astype(np.uint8) * 255)
            
            # 4. Send back the mask
            print(f"Sending back mask of shape: {mask_final.shape} and dtype: {mask_final.dtype}")
            conn.sendall(mask_final.tobytes())
        finally:
            conn.close()
            torch.cuda.empty_cache() # enable if generation of the mesh is used and vram memory is not enough

if __name__ == "__main__":
    run_server()
