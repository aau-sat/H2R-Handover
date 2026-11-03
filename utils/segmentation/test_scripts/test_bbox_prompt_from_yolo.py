"""
SAM3 Assisted Segmentation from YOLO Results
===========================================

This script automates SAM3 instance segmentation by reading object detections 
from a YOLO 'results.txt' file. It uses the YOLO bounding boxes and class 
names as spatial and semantic prompts to generate high-fidelity masks.

Functionality:
--------------
1.  **Input:** Parses a JSON-line formatted YOLO results file via --yolo_results.
2.  **Mapping:** Automatically locates corresponding images in the --input_img 
    directory using filenames provided in the results file.
3.  **Selection Mode:** Allows user to choose between processing only the 
    highest confidence detection or all detections per image via --mode.
4.  **Coordination:** Converts YOLO [x1, y1, x2, y2] boxes into SAM3-compatible 
    normalized [cx, cy, w, h] format.
5.  **Inference:** Runs SAM3 on the selected detections using the YOLO class 
    label as the text prompt and the box as the spatial constraint.
6.  **Output:** Saves binary masks to the user-defined --output directory.

Usage:
------
    # Process only the best detection per image
    python test_bbox_prompt_from_yolo.py --input_img ./images --yolo_results ./results.txt --mode top

    # Process all detections per image
    python test_bbox_prompt_from_yolo.py --input_img ./images --yolo_results ./results.txt --mode all
"""

import cv2
import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox

def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Segmentation guided by YOLO results.")
    
    parser.add_argument("--input_img", type=str, required=True, 
                        help="Path to the directory containing the source images.")
    
    parser.add_argument("--yolo_results", type=str, required=True, 
                        help="Path to the YOLO results.txt JSON-lines file.")
    
    parser.add_argument("--output", type=str, default="sam_masks", 
                        help="Directory where the output masks will be saved.")
    
    parser.add_argument("--mode", type=str, choices=["top", "all"], default="top",
                        help="Choose 'top' for only the highest confidence detection, or 'all' for all detections.")
    
    parser.add_argument("--visualize", action='store_true', 
                        help="If set, displays the mask overlay for each processed detection.")
    
    return parser.parse_args()

def overlap_mask(img, mask):
    """Overlay a single mask on image for visualization."""
    mask_np = mask.cpu().numpy().astype(np.uint8)
    mask_255 = (mask_np * 255).astype(np.uint8)
    img_rgba = img.convert("RGBA")
    
    overlay = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = 255  # R
    overlay[..., 1] = 255  # G
    overlay[..., 3] = (mask_255 * 0.5).astype(np.uint8) # Alpha
    
    overlay_img = Image.fromarray(overlay)
    return Image.alpha_composite(img_rgba, overlay_img)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(f"Loading SAM3 Model to {device}...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    if not os.path.exists(args.yolo_results):
        print(f"Error: Results file not found at {args.yolo_results}")
        return

    with open(args.yolo_results, 'r') as f:
        lines = f.readlines()

    print(f"Processing mode: {args.mode.upper()}")

    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        img_name = data["image"]
        detections = data["detections"]

        if not detections:
            continue

        # Selection logic based on mode
        if args.mode == "top":
            target_detections = [max(detections, key=lambda x: x["confidence"])]
        else:
            target_detections = detections

        img_path = os.path.join(args.input_img, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} not found. Skipping.")
            continue

        image = Image.open(img_path)
        width, height = image.size

        for i, det in enumerate(target_detections):
            label = det["class"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            
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

            if len(results["masks"]) > 0:
                idx = torch.argmax(results["scores"])
                best_mask = results["masks"][idx]
                
                # Dynamic naming based on mode
                suffix = "top" if args.mode == "top" else str(i)
                mask_name = f"{os.path.splitext(img_name)[0]}_mask_{suffix}.png"
                mask_path = os.path.join(args.output, mask_name)
                
                mask_final = (best_mask.cpu().numpy().astype(np.uint8) * 255)
                cv2.imwrite(mask_path, mask_final)

                if args.visualize:
                    viz = overlap_mask(image, best_mask)
                    plt.imshow(viz)
                    plt.title(f"{img_name} | {label} ({conf:.2f}) | Mode: {args.mode}")
                    plt.axis("off")
                    plt.show()

    print(f"Success. Masks saved to: {args.output}")

if __name__ == "__main__":
    main()