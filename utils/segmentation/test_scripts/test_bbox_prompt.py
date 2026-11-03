"""
test_bbox_prompt.py
===================

This script demonstrates the use of Segment Anything Model 3 (SAM3) to perform 
segmentation using a combination of text prompts and explicit bounding box 
inputs. It focuses on isolating objects within a defined spatial region to 
improve segmentation accuracy.

Functionality:
--------------
1.  **Input:** Loads a specific test image (default: bowl.jpeg) and defines a 
    manual bounding box in [x, y, w, h] format.
2.  **Preprocessing:** Converts and normalizes bounding box coordinates into 
    [cx, cy, w, h] format relative to the image size for SAM3 consumption.
3.  **Inference:** Executes the SAM3 model using both the text prompt and the 
    spatial box constraint to generate precise instance masks.
4.  **Visualization:** Overlays the resulting segmentation masks onto the 
    original image with transparency for visual inspection.
5.  **Output:** Identifies the highest-confidence mask and exports it as a 
    binary grayscale image (anchor_mask.png) to the source image directory.

Usage:
------
    python test_bbox_prompt.py                      # Run with default 'cup' prompt
    python test_bbox_prompt.py --prompts bowl       # Specify a different text prompt
    python test_bbox_prompt.py --prompts plate tray # Test with multiple labels
"""

from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import argparse
import cv2
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox

parser = argparse.ArgumentParser(description="Process some images and prompts.")
# parser.add_argument("--input", type=str, required=True, help="Path to the input image")
parser.add_argument("--prompts", nargs='+', default=['cup'], help="List of text prompts")
# parser.add_argument("--output", type=str, default="anchor_mask", help="Output image name")
args = parser.parse_args()

def overlap_mask(img, masks, alpha=0.5, mask_index=0, all=False):
    """Overlay mask on image with given alpha transparency."""
    result = []
    if all:
        for i in range(masks.shape[0]):
            mask = masks[i].cpu().numpy().astype(np.uint8)  # -> HxW
            mask_255 = (mask * 255).astype(np.uint8)
            img = image.convert("RGBA")
            alpha = (mask_255 * 128).astype(np.uint8)

            print(mask_255.shape)
            
            overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            overlay[...,0] = 255     # R
            overlay[...,1] = 255       # G
            overlay[...,2] = 0       # B
            overlay[...,3] = alpha    # A


            overlay_img = Image.fromarray(overlay)

            # Überlagern
            result.append(Image.alpha_composite(img, overlay_img))
    else:
        mask = masks[mask_index,0].cpu().numpy().astype(np.uint8)  # -> HxW
        mask_255 = (mask * 255).astype(np.uint8)
        img = image.convert("RGBA")
        alpha = (mask_255 * 128).astype(np.uint8)
        
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        overlay[...,0] = 255     # R
        overlay[...,1] = 255       # G
        overlay[...,2] = 0       # B
        overlay[...,3] = alpha    # A


        overlay_img = Image.fromarray(overlay)

        # Überlagern
        result.append(Image.alpha_composite(img, overlay_img))
    return result

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Load image
image_path = f"/home/nadezhda/RGMC/sam3/assets/images/bowl.jpeg"
image = Image.open(image_path)
box_input_xywh = torch.tensor([300.0, 850.0, 650.0, 500.0]).view(-1, 4)
box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
width, height = image.size

norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
box = [[norm_box_cxcywh]]

# Pre-process image and compute vision embeddings once
inputs = processor(
    images=image,
    text=args.prompts[0],  # Use the first prompt to get initial boxes
    input_boxes=box,
    input_boxes_labels=[[1]],  # 0 = negative (exclude this region)
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

# Run multiple text prompts efficiently
text_prompts = args.prompts
all_results = []
all_results.append({"prompt": args.prompts[0], "results": results})
for item in all_results:
    print(f"Prompt '{item['prompt']}': {len(item['results']['masks'])} objects found")
    print(f'Mask shapes: {[mask.shape for mask in item["results"]["masks"]]}')

# Visualize results for all prompts
for prompt_item in all_results:
    if len(prompt_item["results"]["masks"]) == 0:
        continue
    print(f"Visualizing results for prompt: {prompt_item['prompt']}")
    mask_index = 0  # Index of the mask to visualize
    # print confidence scores
    print(f"Scores: {prompt_item['results']['scores']}")
    overlapped_images = overlap_mask(image, prompt_item["results"]["masks"], mask_index=mask_index, all=True)
    plt.imshow(overlapped_images[0])
    plt.axis("off")
    plt.show()

# export most confident mask out of all prompts
most_confident_mask = None
highest_score = -1
for item in all_results:
    scores = item["results"]["scores"]
    masks = item["results"]["masks"]
    if len(item['results']['masks']) == 0:
        continue
    for i, score in enumerate(scores):
        if score > highest_score:
            highest_score = score
            most_confident_mask = masks[i]

output_mask = "anchor_mask.png"
output_mask = os.path.join(pathlib.Path(image.filename).parent, output_mask)
cv2.imwrite(output_mask, most_confident_mask.cpu().numpy().astype(np.uint8)*255)