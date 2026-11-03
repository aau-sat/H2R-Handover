#!/usr/bin/env python3
"""
Download and cache SAM3 model locally for offline use.
Run this once with internet access to cache the model.
"""
import os
import sys
from pathlib import Path

# Default cache location (relative to this script's directory)
SCRIPT_DIR = Path(__file__).parent
DEFAULT_CACHE_DIR = SCRIPT_DIR / "sam3_model_cache"


def download_sam3_model(cache_dir=None):
    """Download SAM3 model and processor to a local cache directory."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading SAM3 model to {cache_dir}...")
    
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as e:
        print(f"ERROR: Could not import transformers. {e}")
        print("Make sure the environment is activated: conda activate sam3")
        return False
    
    try:
        print("Downloading Sam3Model...")
        model = Sam3Model.from_pretrained("facebook/sam3", cache_dir=str(cache_dir))
        print(f"✓ Model saved to {cache_dir}")
        
        print("Downloading Sam3Processor...")
        processor = Sam3Processor.from_pretrained("facebook/sam3", cache_dir=str(cache_dir))
        print(f"✓ Processor saved to {cache_dir}")
        
        print(f"\n✓ SAM3 model and processor cached successfully!")
        print(f"  Cache directory: {cache_dir}")
        print(f"\n  To run the service offline, use:")
        print(f"  export SAM3_MODEL_PATH={cache_dir}")
        print(f"  export SAM3_OFFLINE=1")
        print(f"  python segmentation_service_multiprompt.py")
        return True
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        return False


if __name__ == "__main__":
    custom_cache = sys.argv[1] if len(sys.argv) > 1 else None
    success = download_sam3_model(custom_cache)
    sys.exit(0 if success else 1)
