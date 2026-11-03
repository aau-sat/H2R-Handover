import os
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import cv2
import argparse

parser = argparse.ArgumentParser(description="Mask of a single object.")
parser.add_argument("--prompt", type=str, default='hand', help="Objecttext prompt")
args = parser.parse_args()

mask_index = 0  # Index of the mask to visualize

side = ['left', 'right']  # or 'left'
prompt = ['cup']  # or ['hand']
def overlap_mask(img, masks, alpha=0.5, mask_index=mask_index, all=False):
    """Overlay mask on image with given alpha transparency."""
    result = []
    if all:
        for i in range(masks.shape[0]):
            mask = masks[i,0].cpu().numpy().astype(np.uint8)  # -> HxW
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

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
id = 0
for side in ['left', 'right']:
    for prompt in ['box',]:
        try:
            for i in range(0,21):
                img_path = f"/home/student-team/rgmc-2026/utils/realsense/captured_data/rgb_{side}_{i}.png"
                image = Image.open(img_path)
                # change img dimensions to (640, 480)
                print(f"Filename: {image.filename}")
                inference_state = processor.set_image(image)
                # Prompt the model with text
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)

                # Get the masks, bounding boxes, and scores
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                
                
                for id in range(masks.shape[0]):
                    output_mask= f"mask_{side}_{prompt}_{i}_{id}.png"
                    output_mask = os.path.join(pathlib.Path(image.filename).parent, output_mask)
                    cv2.imwrite(output_mask, masks[id,0].cpu().numpy().astype(np.uint8)*255)
                # show the results

                print(f"Masks shape: {masks.shape}, mask type: {type(masks)}")
                print(f"Image size: {image.size}, image type: {type(image)}")
                # # Visualize 
                # plt.figure(figsize=(10, 10))
                # plt.imshow(image)
                # for overlay_img in overlap_mask(image, masks, alpha=0.5, mask_index=mask_index, all=True):  # Overlay all masks
                #     plt.imshow(overlay_img)
                # plt.axis('off')
                # plt.show()
        except Exception as e:
            pass



