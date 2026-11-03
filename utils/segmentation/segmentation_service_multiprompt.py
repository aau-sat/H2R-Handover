import socket
import os
import struct
import time
import json
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox

SOCKET_PATH = "/tmp/sam3_socket"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(SCRIPT_DIR, "sam3_model_cache")

MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
MODEL_PATH = os.environ.get("SAM3_MODEL_PATH", "").strip() or DEFAULT_CACHE_DIR
OFFLINE_MODE = True  # Always run offline


def find_cached_model_path():
    """Find the actual cached model directory in the HF cache structure."""
    hf_cache_path = os.path.join(MODEL_PATH, "models--facebook--sam3", "snapshots")
    if not os.path.exists(hf_cache_path):
        return None
    
    # Find the first (and usually only) snapshot directory
    snapshots = os.listdir(hf_cache_path)
    if snapshots:
        snapshot_dir = os.path.join(hf_cache_path, snapshots[0])
        if os.path.exists(os.path.join(snapshot_dir, "config.json")):
            return snapshot_dir
    
    return None


def ensure_model_downloaded():
    """Download SAM3 model to cache if not already present."""
    # Check if model is already cached (either in direct format or HF cache format)
    cached_path = find_cached_model_path()
    if cached_path and os.path.exists(os.path.join(cached_path, "config.json")):
        print(f"Using cached SAM3 model at {cached_path}")
        return True
    
    # Model not cached, download it now
    print(f"SAM3 model not found in cache")
    print("Downloading SAM3 model (first run only, requires internet)...")
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    try:
        from transformers import Sam3Model, Sam3Processor
        
        print(f"Downloading Sam3Model...")
        Sam3Model.from_pretrained(MODEL_ID, cache_dir=MODEL_PATH)
        print("Model downloaded")
        
        print(f"Downloading Sam3Processor...")
        Sam3Processor.from_pretrained(MODEL_ID, cache_dir=MODEL_PATH)
        print("Processor downloaded")
        
        print("SAM3 model cached successfully. Subsequent runs will be fully offline.\n")
        return True
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download SAM3 model from Hugging Face: {exc}\n"
            f"Make sure you have internet access and the transformers library installed."
        ) from exc


def load_sam3_component(component_cls):
    # Find the actual cached model in HF cache structure
    source = find_cached_model_path()
    if not source:
        source = MODEL_PATH
    
    load_kwargs = {"local_files_only": True}

    try:
        return component_cls.from_pretrained(source, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load SAM3 component from '{source}'. "
            f"Model may not be cached. Try running download_sam3_model.py first or "
            f"set SAM3_MODEL_PATH to a valid model directory."
        ) from exc


def mask_centroid(mask_tensor):
    mask_np = mask_tensor.squeeze().cpu().numpy()
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return None
    return np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float32)


def recv_exact(conn, size):
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        data += chunk
    return data


def send_masks(conn, masks):
    # Protocol:
    #   uint32 object_ok (0/1)
    #   uint32 hand_ok   (0/1)
    #   repeat 2 times:
    #     uint32 payload_size
    #     payload bytes (H*W uint8 mask)
    if len(masks) == 0:
        raise ValueError("send_masks requires at least one mask entry")

    # Backward-compatible behavior:
    # - single-label requests return one ok flag + one mask
    # - dual-label requests return object_ok/hand_ok + two masks
    if len(masks) == 1:
        object_ok = 1 if masks[0]["ok"] else 0
        conn.sendall(struct.pack("<I", object_ok))
    else:
        masks = masks[:2]
        object_ok = 1 if masks[0]["ok"] else 0
        hand_ok = 1 if masks[1]["ok"] else 0
        conn.sendall(struct.pack("<II", object_ok, hand_ok))

    for mask in masks:
        payload = mask["data"].tobytes()
        conn.sendall(struct.pack("<I", len(payload)))
        conn.sendall(payload)


def parse_labels(raw_label):
    """
    Supported label encodings in request payload:
      - legacy string: "bottle" -> ["bottle", "hand"]
      - object-only prefix: "single:bottle" -> ["bottle"]
      - json list: '["bottle"]' or '["bottle","hand"]'
      - json object: '{"labels": ["bottle"]}'
    """
    s = str(raw_label).strip()
    if not s:
        return ["object", "hand"]

    if s.lower().startswith("single:"):
        one = s.split(":", 1)[1].strip()
        return [one] if one else ["object"]

    if s.startswith("[") or s.startswith("{"):
        try:
            payload = json.loads(s)
            if isinstance(payload, list):
                labels = [str(x).strip() for x in payload if str(x).strip()]
                if len(labels) in (1, 2):
                    return labels
            if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
                labels = [str(x).strip() for x in payload["labels"] if str(x).strip()]
                if len(labels) in (1, 2):
                    return labels
        except Exception:
            pass

    return [s, "hand"]

def run_server():
    # Ensure model is downloaded and cached (first run may require internet)
    ensure_model_downloaded()
    
    # Model Initialization (Keep model in GPU memory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sam3_component(Sam3Model).to(device)
    processor = load_sam3_component(Sam3Processor)

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    print(f"SAM3 Server listening on {SOCKET_PATH}...")

    while True:
        conn, _ = server.accept()
        try:
            request_start = time.perf_counter()

            # 1. Read Metadata (Header)
            # Format: H, W, x1, y1, x2, y2, label_len (Total 28 bytes if using int32)
            header = recv_exact(conn, 28)
            h, w, x1, y1, x2, y2, l_len = np.frombuffer(header, dtype=np.int32)
            label_raw = recv_exact(conn, int(l_len)).decode().strip()

            labels = parse_labels(label_raw)
            print(
                f"Received image of size ({h}, {w}) with box ({x1}, {y1}, {x2}, {y2}) "
                f"and labels {labels} (raw='{label_raw}')"
            )
            
            # 2. Read Raw Image Bytes
            img_size = h * w * 3
            img_data = recv_exact(conn, int(img_size))
            
            frame = np.frombuffer(img_data, dtype=np.uint8).reshape((h, w, 3))

            image = Image.fromarray(frame)

            response_masks = []
            first_mask_centroid = None
            for label_idx, label in enumerate(labels):
                fallback_mask = np.zeros((h, w), dtype=np.uint8)
                success = False
                best_mask = None
            
                try:
                    # if bbox is missing or invalid, only use label for first one
                    if x1 == x2 or y1 == y2 or label_idx != 0:
                        inputs = processor(
                            images=image,
                            text=label,
                            return_tensors="pt"
                        ).to(device)
                    else:
                        box_xywh = torch.tensor([float(x1), float(y1), float(x2 - x1), float(y2 - y1)]).view(-1, 4)
                        box_cxcywh = box_xywh_to_cxcywh(box_xywh)
                        norm_box = normalize_bbox(box_cxcywh, w, h).flatten().tolist()
                        sam_box = [[norm_box]]
                        inputs = processor(
                            images=image,
                            text=label,
                            input_boxes=sam_box,
                            input_boxes_labels=[[1]],
                            return_tensors="pt"
                        ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)

                    results = processor.post_process_instance_segmentation(
                        outputs,
                        threshold=0.45, # 0.5
                        mask_threshold=0.45, # 0.5
                        target_sizes=inputs.get("original_sizes").tolist()
                    )[0]

                    if len(results["masks"]) > 0:
                        use_closest_hand = (
                            len(labels) == 2
                            and label_idx == 1
                            and label.lower() == "hand"
                            and first_mask_centroid is not None
                        )

                        if use_closest_hand:
                            ref_centroid = first_mask_centroid
                            if ref_centroid is None:
                                ref_centroid = np.array([0.0, 0.0], dtype=np.float32)
                            min_dist = float("inf")
                            idx = 0
                            for cand_idx, cand_mask in enumerate(results["masks"]):
                                c = mask_centroid(cand_mask)
                                if c is None:
                                    continue
                                dist = float(np.linalg.norm(c - ref_centroid))
                                if dist < min_dist:
                                    min_dist = dist
                                    idx = cand_idx
                        else:
                            idx = torch.argmax(results["scores"])

                        best_mask = results["masks"][idx]
                        success = True
                    else:
                        print(f"No mask detected for label '{label}', sending empty mask.")
                except Exception as e:
                    print(f"Mask generation failed for label '{label}': {e}")

                if best_mask is not None:
                    mask_final = best_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
                else:
                    mask_final = fallback_mask

                response_masks.append({"data": mask_final, "ok": success})

                if label_idx == 0 and best_mask is not None:
                    first_mask_centroid = mask_centroid(best_mask)

            # 4. Send back all masks in one framed response
            send_masks(conn, response_masks)
            elapsed_ms = (time.perf_counter() - request_start) * 1000.0
            print(
                f"Request done: labels={len(labels)}, masks={min(2, len(response_masks))}, "
                f"elapsed_ms={elapsed_ms:.2f}"
            )
        finally:
            conn.close()
            #torch.cuda.empty_cache() # enable if generation of the mesh is used and vram memory is not enough

if __name__ == "__main__":
    run_server()
