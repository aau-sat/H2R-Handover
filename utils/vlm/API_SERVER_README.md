# Qwen3-VL API Server & Client Setup

## Installation

Install required packages:

```bash
pip install fastapi uvicorn requests python-multipart
# or
pip install -r requirements.txt
```

## Quick Start

### 1. Start the Server

```bash
# Option A: Using uvicorn (recommended)
uvicorn server:app --host 0.0.0.0 --port 8000

# Option B: Running directly with Python
python server.py
```

The server will load the model on startup. First run may take a few minutes.

Once loaded, you'll see:
```
✓ Model loaded
✓ Processor loaded
```

### 2. Run Inference (in another terminal)

```bash
# Basic usage
python client.py --image /path/to/image.jpg

# With custom prompt
python client.py --image /path/to/image.jpg --prompt "What objects are in this image?"

# Check server health
python client.py --check-health
```

### 3. Using the API Directly

With `curl`:
```bash
curl -X POST \
  -F "file=@image.jpg" \
  -F "prompt=Describe this image." \
  http://localhost:8000/infer
```

With Python `requests`:
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/infer",
        files={"file": f},
        data={"prompt": "Describe this image."}
    )
    
print(response.json())
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/health` | GET | Server health check and model status |
| `/infer` | POST | Run inference on an image |

### POST `/infer`

**Request:**
- `file` (multipart/form-data): Image file to analyze
- `prompt` (form-data): Text prompt (optional, default: "Describe this image.")

**Response:**
```json
{
  "success": true,
  "prompt": "Describe this image.",
  "response": "Model's description...",
  "model": "Qwen/Qwen3-VL-8B-Instruct"
}
```

## Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Architecture

```
┌─────────────────────────────────────────────┐
│   Client (client.py / HTTP requests)        │
└──────────────────┬──────────────────────────┘
                   │ HTTP
                   ▼
┌─────────────────────────────────────────────┐
│   FastAPI Server (server.py)                │
│   - Loads model on startup                  │
│   - Handles /infer requests                 │
│   - Manages GPU memory                      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Qwen3-VL Model      │
        │  (stays in memory)   │
        └──────────────────────┘
```

## Benefits

✓ **Single model load** - Model stays in memory, no reload per request
✓ **Multiple clients** - Serve multiple inference requests
✓ **HTTP API** - Easy to integrate with other services
✓ **Scalable** - Can be deployed with Gunicorn/Uvicorn workers
✓ **Interactive docs** - Auto-generated Swagger UI for testing

## Troubleshooting

**Server won't start:**
- Check if port 8000 is already in use: `lsof -i :8000`
- Use different port: `uvicorn server:app --port 8001`

**Model download is slow:**
- First run downloads the 8B model (~16GB)
- Subsequent runs use cached model from `~/.cache/huggingface`

**Out of memory errors:**
- Reduce batch size or max_new_tokens in server.py
- Use a smaller model if available

**Connection refused:**
- Make sure server is running
- Check firewall settings
- Verify correct hostname/port in client
