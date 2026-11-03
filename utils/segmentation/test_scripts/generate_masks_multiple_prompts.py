"""
generate_masks_multiple_prompts.py
=================================

This script utilizes the Segment Anything Model 3 (SAM3) to perform text-prompted 
instance segmentation on local images. It processes a directory of images, 
identifies objects based on natural language descriptions, and exports the 
highest-confidence segmentation mask.

Functionality:
--------------
1.  **Input:** Reads images from a user-specified directory via the --input flag.
    - Supports .png, .jpg, and .jpeg formats.
2.  **Inference:** Leverages 'facebook/sam3' to compute vision embeddings and 
    processes multiple text prompts per image efficiently.
3.  **Refinement:** Compares results across all provided prompts to select 
    the single most confident mask for final export.
4.  **Visualization:** Optionally displays the image with a semi-transparent 
    yellow mask overlay using matplotlib for verification.
5.  **Output:** Saves the binary mask (0 or 255) of the top detection as an 
    image file to the specified output directory.

Usage:
------
    python generate_masks_multiple_prompts.py --input <dir> --output <dir>
    python generate_masks_multiple_prompts.py --input <dir> --prompts cup bottle
    python generate_masks_multiple_prompts.py --input <dir> --visualize
    python generate_masks_multiple_prompts.py --input <dir> --output <dir> --visualize
"""

from transformers import Sam3Processor, Sam3Model
import torch
# ... rest of your code


from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import argparse
import cv2

parser = argparse.ArgumentParser(description="Process some images and prompts.")
parser.add_argument("--input", type=str, required=True, help="Path to the input directory")
parser.add_argument("--prompts", nargs='+', default=['mandarine'], help="List of text prompts")
parser.add_argument("--output", type=str, help="Output directory name")
parser.add_argument("--visualize", action='store_true', help="Whether to visualize the results")
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

# Produce mask for a every image in a directory with multiple prompts
img_dir = args.input
image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(img_dir, image_file)
    image = Image.open(image_path)

    # Pre-process image and compute vision embeddings once
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

    # Run multiple text prompts efficiently
    text_prompts = args.prompts
    all_results = []

    for prompt in text_prompts:
        text_inputs = processor(text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(vision_embeds=vision_embeds, **text_inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=img_inputs.get("original_sizes").tolist()
        )[0]
        all_results.append({"prompt": prompt, "results": results})

    for item in all_results:
        print(f"Prompt '{item['prompt']}': {len(item['results']['masks'])} objects found")
        print(f'Mask shapes: {[mask.shape for mask in item["results"]["masks"]]}')

    if args.visualize:
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

    output_mask = os.path.join(args.output, image_file)
    cv2.imwrite(output_mask, most_confident_mask.cpu().numpy().astype(np.uint8)*255)