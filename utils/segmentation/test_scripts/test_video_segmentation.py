import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor


def overlap_mask(img, mask, alpha=0.5):
    """Overlay mask on image with given alpha transparency."""
    mask = mask[0,0].cpu().numpy().astype(np.uint8)  # -> HxW
    mask_255 = (mask * 255).astype(np.uint8)
    img = img.convert("RGBA")
    alpha = (mask_255 * 128).astype(np.uint8)

    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[...,0] = 255     # R
    overlay[...,1] = 255       # G
    overlay[...,2] = 0       # B
    overlay[...,3] = alpha    # A

    overlay_img = Image.fromarray(overlay)

    # Überlagern
    result = Image.alpha_composite(img, overlay_img)
    return result


#################################### For Video ####################################


video_predictor = build_sam3_video_predictor()
video_path = "/home/nadezhda/RGMC/sam3/assets/videos/hand_taking_glass.mp4" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="glass",
    )
)
output = response["outputs"]
print(f"Output: {output}, shape: {type(output)}")
