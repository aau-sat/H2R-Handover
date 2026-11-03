"""
FastAPI server for Qwen3-VL inference.
Loads the model once and serves inference requests via HTTP API.

Run with: uvicorn server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import time
import base64

# Configuration
local_model_path = "/home/student-team/Qwen3-VL-8B-Instruct"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# Initialize FastAPI app
app = FastAPI(title="Qwen3-VL Inference Server", version="1.0")

# Global variables to hold the loaded model and processor
model = None
processor = None


@app.on_event("startup")
async def startup_event():
    """Load model and processor when server starts"""
    global model, processor
    
    print(f"Loading model: {MODEL_NAME}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     MODEL_NAME,
    #     dtype="auto",
    #     device_map="auto"
    # )
    print("Model loaded")
    
    print("Loading processor")
    #processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(local_model_path)
    print("Processor loaded")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when server shuts down"""
    global model, processor
    
    if model is not None:
        del model
    if processor is not None:
        del processor
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Resources cleaned up")


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    prompt: str = Form(default="Describe this image.")
):
    """
    Run inference on an uploaded image.
    
    Args:
        file: Image file to analyze
        prompt: Text prompt for the model
    
    Returns:
        JSON with the model's response
    """
    
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Server may still be starting up.")
    
    # Validate file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Start timing the inference work (after saving file)
        start_time = time.time()

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_file_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Preparation for inference
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        inference_time = time.time() - start_time

        return JSONResponse({
            "success": True,
            "prompt": prompt,
            "response": output_text[0] if output_text else "",
            "model": MODEL_NAME,
            "inference_time_sec": round(inference_time, 6)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/infer_base64")
async def infer_base64(payload: dict):
    """
    Run inference on a base64-encoded image sent as JSON.
    JSON body: {"image_base64": "data:image/png;base64,..." or "base64_string", "prompt": "Describe this image."}
    
    Args:
        payload: JSON dict with image_base64 and optional prompt
    
    Returns:
        JSON with the model's response
    """
    
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Server may still be starting up.")
    
    image_b64 = payload.get("image_base64")
    prompt = payload.get("prompt", "Describe this image.")
    
    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing 'image_base64' in JSON payload")
    
    try:
        # Decode base64 (handle data: URI format)
        if "," in image_b64:
            # data:image/png;base64,... format
            _, b64_data = image_b64.split(",", 1)
        else:
            # raw base64
            b64_data = image_b64
        
        img_bytes = base64.b64decode(b64_data)
        
        # Write to temporary file
        temp_file_path = f"/tmp/infer_base64_{int(time.time() * 1000)}.png"
        with open(temp_file_path, "wb") as f:
            f.write(img_bytes)
        
        # Start timing the inference work
        start_time = time.time()

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_file_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Preparation for inference
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        inference_time = time.time() - start_time

        return JSONResponse({
            "success": True,
            "prompt": prompt,
            "response": output_text[0] if output_text else "",
            "model": MODEL_NAME,
            "inference_time_sec": round(inference_time, 6)
        })
    
    except base64.binascii.Error as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/health")
async def health_check():
    """Check if server is running and model is loaded"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and processor is not None,
        "model_name": MODEL_NAME
    }


@app.get("/")
async def root():
    """API documentation"""
    return {
        "name": "Qwen3-VL Inference Server",
        "endpoints": {
            "POST /infer": "Run inference on an image (multipart/form-data: file + prompt)",
            "POST /infer_base64": "Run inference on a base64-encoded image (JSON: image_base64 + prompt)",
            "GET /health": "Check server health and model status",
            "GET /docs": "Interactive API documentation (Swagger UI)"
        },
        "example_curl": 'curl -X POST -F "file=@image.jpg" -F "prompt=Describe this image." http://localhost:8000/infer'
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# cd /home/student-team/rgmc-2026/utils/vlm uvicorn server:app --host 0.0.0.0 --port 8000