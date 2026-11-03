import os
import argparse
import sys
from PIL import Image

def merge_mask_and_image(image_path, mask_path, output_path):
    """
    Combines an RGB image and a grayscale mask into a single RGBA image.
    
    Args:
        image_path (str): Path to the source RGB image.
        mask_path (str): Path to the binary or grayscale mask.
        output_path (str): Path where the resulting RGBA image will be saved.
    """
    
    # 1. Check if input files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found at {mask_path}")
        sys.exit(1)

    try:
        # 2. Load images
        # Convert image to RGB to ensure 3 channels
        img = Image.open(image_path).convert("RGB")
        # Convert mask to Grayscale (L) to ensure 1 channel (0-255)
        mask = Image.open(mask_path).convert("L")

        # 3. Validation & Pre-processing
        # Best Practice: Ensure mask matches image dimensions. 
        # If not, resize mask to fit image (nearest neighbor for hard masks, bilinear for soft).
        if img.size != mask.size:
            print(f"Warning: Size mismatch! Image: {img.size}, Mask: {mask.size}. Resizing mask to fit...")
            mask = mask.resize(img.size, Image.NEAREST)

        # 4. Apply Alpha Channel
        # putalpha adds or replaces the alpha layer of the image
        img.putalpha(mask)

        # 5. Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 6. Save result
        # Ensure we save as PNG because JPEG does not support transparency
        if not output_path.lower().endswith(".png"):
            output_path += ".png"
            
        img.save(output_path, "PNG")
        print(f"Success: RGBA image saved to '{output_path}'")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # CLI Argument Setup
    parser = argparse.ArgumentParser(description="Merge an RGB image with a Mask to create a transparent PNG.")
    
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to the source image (jpg, png, etc.)")
    parser.add_argument("-m", "--mask", type=str, required=True, help="Path to the mask image (white = visible, black = transparent)")
    parser.add_argument("-o", "--output", type=str, default="output.png", help="Path for the output file (default: output.png)")
    
    args = parser.parse_args()
    
    merge_mask_and_image(args.image, args.mask, args.output)