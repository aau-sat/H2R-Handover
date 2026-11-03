"""
Advanced client examples for Qwen3-VL Inference API Server.

Examples:
- Batch inference on multiple images
- Async inference
- Custom configurations
"""

import requests
import asyncio
from pathlib import Path
from typing import List, Optional
import json

SERVER_URL = "http://localhost:8000"


class Qwen3VLClient:
    """Synchronous client for Qwen3-VL Inference API"""
    
    def __init__(self, server_url: str = SERVER_URL, timeout: int = 120):
        self.server_url = server_url
        self.timeout = timeout
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to server"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            print(f"✓ Connected to {self.server_url}")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to server: {e}")
    
    def infer(self, image_path: str, prompt: str = "Describe this image.") -> Optional[dict]:
        """Run single inference"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    f"{self.server_url}/infer",
                    files={"file": (image_path.name, f)},
                    data={"prompt": prompt},
                    timeout=self.timeout
                )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during inference: {e}")
            return None
    
    def batch_infer(self, image_paths: List[str], prompt: str = "Describe this image.", verbose: bool = True) -> List[dict]:
        """Run inference on multiple images"""
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            if verbose:
                print(f"[{i}/{len(image_paths)}] Processing {Path(image_path).name}...")
            
            result = self.infer(image_path, prompt)
            if result:
                results.append({
                    "image": str(image_path),
                    **result
                })
        
        return results


class AsyncQwen3VLClient:
    """Asynchronous client for Qwen3-VL Inference API"""
    
    def __init__(self, server_url: str = SERVER_URL, timeout: int = 120):
        self.server_url = server_url
        self.timeout = timeout
    
    async def infer(self, image_path: str, prompt: str = "Describe this image.") -> Optional[dict]:
        """Run single inference asynchronously"""
        import aiohttp
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            async with aiohttp.ClientSession() as session:
                with open(image_path, "rb") as f:
                    form = aiohttp.FormData()
                    form.add_field("file", f, filename=image_path.name)
                    form.add_field("prompt", prompt)
                    
                    async with session.post(
                        f"{self.server_url}/infer",
                        data=form,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        else:
                            error = await resp.text()
                            print(f"Error: {resp.status} - {error}")
                            return None
        except Exception as e:
            print(f"Error during async inference: {e}")
            return None
    
    async def batch_infer(self, image_paths: List[str], prompt: str = "Describe this image.") -> List[dict]:
        """Run inference on multiple images concurrently"""
        tasks = [
            self.infer(image_path, prompt)
            for image_path in image_paths
        ]
        
        results = []
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            print(f"[{i}/{len(image_paths)}] Processing...")
            result = await coro
            if result:
                results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    # Example 1: Single image inference
    print("=" * 50)
    print("Example 1: Single Image Inference")
    print("=" * 50)
    
    client = Qwen3VLClient()
    result = client.infer(
        "/home/student-team/rgmc-2026/utils/vlm/captured_data/left_00002_20260505_120120_536787.png",
        "Describe this image in detail."
    )
    
    if result:
        print(f"Response: {result['response']}\n")
    
    # Example 2: Batch inference
    print("=" * 50)
    print("Example 2: Batch Inference")
    print("=" * 50)
    
    image_dir = Path("/home/student-team/rgmc-2026/utils/vlm/captured_data")
    images = list(image_dir.glob("*.png"))[:3]  # First 3 images
    
    if images:
        batch_results = client.batch_infer(
            [str(img) for img in images],
            "What's the main object in this image?"
        )
        
        print(f"\nBatch Results ({len(batch_results)} images):")
        for result in batch_results:
            print(f"  - {Path(result['image']).name}: {result['response'][:100]}...")
    
    # Example 3: Async batch inference
    print("\n" + "=" * 50)
    print("Example 3: Async Batch Inference")
    print("=" * 50)
    
    async def async_example():
        async_client = AsyncQwen3VLClient()
        
        if images:
            results = await async_client.batch_infer(
                [str(img) for img in images],
                "Identify any objects in this image."
            )
            
            print(f"\nAsync Batch Results ({len(results)} images):")
            for result in results:
                print(f"  Response: {result.get('response', 'N/A')[:100]}...")
    
    try:
        asyncio.run(async_example())
    except ImportError:
        print("aiohttp not installed. Install with: pip install aiohttp")
