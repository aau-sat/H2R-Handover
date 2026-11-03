"""
Client for the Qwen3-VL Inference API Server.
Make requests to the running server for image inference.

Usage:
    python client.py --image /path/to/image.jpg --prompt "Describe this image."
    python client.py --image image.jpg
"""

import requests
import argparse
import json
from pathlib import Path
import time
import base64

SERVER_URL = "http://localhost:8000"


def check_server_health():
    """Check if the server is running and healthy"""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        return True, health
    except requests.exceptions.ConnectionError:
        return False, {"error": "Cannot connect to server"}
    except Exception as e:
        return False, {"error": str(e)}


def run_inference(image_path, prompt="Describe this image."):
    """
    Send image to server for inference.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt for the model
    
    Returns:
        Response from the server
    """
    
    # Validate image file
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Error: Image file not found: {image_path}")
        return None
    
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    if image_file.suffix.lower() not in allowed_extensions:
        print(f"Error: Unsupported file type. Allowed: {', '.join(allowed_extensions)}")
        return None
    
    print(f"Sending request to {SERVER_URL}/infer")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    
    response = None
    try:
        start_time = time.time()
        with open(image_path, "rb") as f:
            files = {"file": (image_file.name, f)}
            data = {"prompt": prompt}
            
            response = requests.post(
                f"{SERVER_URL}/infer",
                files=files,
                data=data,
                timeout=120  # 2 minute timeout for inference
            )
        
        response.raise_for_status()
        elapsed = time.time() - start_time
        result = response.json()
        # attach client-side measured round-trip time
        try:
            result["api_call_time_sec"] = round(elapsed, 6)
        except Exception:
            # if result isn't a dict, ignore
            pass

        print("\nInference successful!")
        print(f"\nResponse: {result.get('response', '')}")
        print(f"API call time (client RTT): {elapsed:.6f} sec")
        
        return result
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to server at {SERVER_URL}")
        print("Make sure the server is running: python server.py")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Inference may still be processing.")
        return None
    except requests.exceptions.HTTPError as e:
        err_response = e.response if e.response is not None else response
        status_code = err_response.status_code if err_response is not None else "unknown"
        print(f"Error: HTTP {status_code}")
        if err_response is not None:
            try:
                print(f"Details: {err_response.json()}")
            except:
                print(f"Details: {err_response.text}")
        else:
            print(f"Details: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def run_inference_base64(image_base64, prompt="Describe this image."):
    """
    Send base64 encoded image to server for inference.
    
    Args:
        image_base64: Base64 encoded image string (with or without data: URI prefix)
        prompt: Text prompt for the model
    
    Returns:
        Response from the server
    """
    
    if not image_base64:
        print("Error: Empty base64 image string")
        return None
    
    print(f"Sending base64 inference request to {SERVER_URL}/infer_base64")
    print(f"Prompt: {prompt}")
    
    response = None
    try:
        start_time = time.time()
        
        # Prepare payload with base64 image
        payload = {
            "image_base64": image_base64,
            "prompt": prompt
        }
        
        response = requests.post(
            f"{SERVER_URL}/infer_base64",
            json=payload,
            timeout=120  # 2 minute timeout for inference
        )
        
        response.raise_for_status()
        elapsed = time.time() - start_time
        result = response.json()
        # attach client-side measured round-trip time
        try:
            result["api_call_time_sec"] = round(elapsed, 6)
        except Exception:
            # if result isn't a dict, ignore
            pass
        
        # print("\n✓ Inference successful!")
        # print(f"\nResponse: {result.get('response', '')}")
        print(f"API call time (client RTT): {elapsed:.6f} sec")
        if result.get('inference_time_sec'):
            print(f"Server inference time: {result['inference_time_sec']:.6f} sec")
        
        return result
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to server at {SERVER_URL}")
        print("Make sure the server is running: python server.py")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Inference may still be processing.")
        return None
    except requests.exceptions.HTTPError as e:
        err_response = e.response if e.response is not None else response
        status_code = err_response.status_code if err_response is not None else "unknown"
        print(f"Error: HTTP {status_code}")
        if err_response is not None:
            try:
                print(f"Details: {err_response.json()}")
            except:
                print(f"Details: {err_response.text}")
        else:
            print(f"Details: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def infer_answer_from_base64(image_base64, prompt="Describe this image."):
    """
    Run inference from a base64 image + prompt and return only the answer text.

    Args:
        image_base64: Base64 encoded image string (with or without data: URI prefix)
        prompt: Text prompt for the model

    Returns:
        Model answer as a string, or an empty string on failure
    """
    result = run_inference_base64(image_base64=image_base64, prompt=prompt)
    if not result:
        return ""
    return str(result.get("response", ""))


def run_inference_from_file_base64(image_path, prompt="Describe this image."):
    """
    Load image from file, encode to base64, and run inference.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt for the model
    
    Returns:
        Response from the server
    """
    
    # Validate image file
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Error: Image file not found: {image_path}")
        return None
    
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    if image_file.suffix.lower() not in allowed_extensions:
        print(f"Error: Unsupported file type. Allowed: {', '.join(allowed_extensions)}")
        return None
    
    try:
        # Read file and encode to base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        b64_str = base64.b64encode(image_bytes).decode('ascii')
        
        # Determine MIME type based on file extension
        ext = image_file.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(ext, "image/jpeg")
        
        # Create data URI
        data_uri = f"data:{mime_type};base64,{b64_str}"
        
        print(f"Encoded {image_path} to base64")
        
        # Call base64 inference function
        return run_inference_base64(data_uri, prompt)
    
    except Exception as e:
        print(f"Error encoding image to base64: {str(e)}")
        return None


def test_base64_inference():
    """
    Test function to load an image, convert to base64, and run inference.
    Uses a specific test image from the captured_data directory.
    """
    test_image_path = "/home/student-team/rgmc-2026/utils/vlm/captured_data/left_00006_20260505_120212_946744.png"
    test_prompt = "Describe this image."
    
    print("=" * 60)
    print("BASE64 INFERENCE TEST")
    print("=" * 60)
    print(f"Test image: {test_image_path}")
    print(f"Prompt: {test_prompt}")
    print()
    
    # Open and read the image file
    try:
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()
        print(f"Image opened and read into bytes: {len(image_bytes)} bytes")
    except Exception as e:
        print(f"Failed to read image: {e}")
        return False
    
    # Convert to base64
    try:
        image_base64 = base64.b64encode(image_bytes).decode('ascii')
        print(f"Image converted to base64: {len(image_base64)} characters")
    except Exception as e:
        print(f"Failed to encode to base64: {e}")
        return False
    
    # Create data URI
    try:
        data_uri = f"data:image/png;base64,{image_base64}"
        print(f"Data URI created: {len(data_uri)} characters")
        print(f"  URI prefix: {data_uri[:50]}...")
    except Exception as e:
        print(f"Failed to create data URI: {e}")
        return False
    
    print()
    
    # Run inference with base64 image
    result = run_inference_base64(data_uri, test_prompt)
    
    if result:
        print("\n" + "=" * 60)
        print("TEST PASSED")
        print("=" * 60)
        print(f"Response: {result.get('response', 'N/A')}")
        print(f"Server inference time: {result.get('inference_time_sec', 'N/A')} sec")
        print(f"Client RTT: {result.get('api_call_time_sec', 'N/A')} sec")
        return True
    else:
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("=" * 60)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Client for Qwen3-VL Inference API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py --image photo.jpg
  python client.py --image photo.jpg --prompt "What's in this image?"
  python client.py --image photo.jpg --prompt "Count the objects"
  python client.py --test  # Run base64 inference test with predefined image
        """
    )
    
    parser.add_argument(
        "--image",
        required=False,
        help="Path to the image file to analyze"
    )
    parser.add_argument(
        "--prompt",
        default="You see a white table. In the center of the table is a object. Respond only with the object name. Be specific as you can. Box alone is not enough.",
        help="Text prompt for the model (default: 'You see a white table. In the center of the table is a object. Respond only with the object name. Be specific as you can (rice bag or spaghetti box). Box alone is not enough.')"
    )
    parser.add_argument(
        "--check-health",
        action="store_true",
        help="Check server health and exit"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run base64 inference test with predefined image"
    )
    
    args = parser.parse_args()
    
    # Check server health
    is_healthy, health_info = check_server_health()
    
    if not is_healthy:
        print(f"Server health check failed: {health_info}")
        print(f"\nStart the server with:")
        print(f"  python server.py")
        print(f"Or if you have uvicorn installed:")
        print(f"  uvicorn server:app --host 0.0.0.0 --port 8000")
        return
    
    print(f"Server is healthy")
    print(f"  Model loaded: {health_info.get('model_loaded')}")
    print(f"  Model: {health_info.get('model_name')}")
    
    if args.check_health:
        return
    
    # Run test if requested
    if args.test:
        test_base64_inference()
        return
    
    # Require image if not running test
    if not args.image:
        print("Error: --image is required (or use --test for predefined test)")
        parser.print_help()
        return
    
    # Run inference
    print()
    result = run_inference(args.image, args.prompt)
    
    if result:
        print("\nFull response:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
