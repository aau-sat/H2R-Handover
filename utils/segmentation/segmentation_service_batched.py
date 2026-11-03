import socket
import os
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import struct

SOCKET_PATH = "/tmp/sam3_batched_socket"


def recv_exact(conn, size):
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        data += chunk
    return data

def run_server():
    # Model Initialization (Keep model in GPU memory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    print(f"SAM3 Server listening on {SOCKET_PATH}...")

    while True:
        conn, _ = server.accept()
        try:
            # 1. Read Metadata (Header)
            # Format: num_images, H, W, label_len (all int32)
            header = recv_exact(conn, 16)
            num_images, h, w, l_len = np.frombuffer(header, dtype=np.int32)
            num_images = int(num_images)
            h = int(h)
            w = int(w)
            l_len = int(l_len)
            if num_images <= 0 or h <= 0 or w <= 0 or l_len <= 0:
                raise ValueError(
                    f"Invalid request metadata: num_images={num_images}, h={h}, w={w}, label_len={l_len}"
                )

            label = recv_exact(conn, l_len).decode().strip()
            print(
                f"Received batch metadata: num_images={num_images}, size=({h}, {w}), label='{label}'"
            )

            # 2. Read Raw Image Bytes for all images in one go
            img_size = h * w * 3
            total_img_size = num_images * img_size
            img_data = recv_exact(conn, total_img_size)

            print(f"Received batched image data: {len(img_data)} bytes")
            images = []
            for i in range(num_images):
                start = i * img_size
                end = start + img_size
                frame = np.frombuffer(img_data[start:end], dtype=np.uint8).reshape((h, w, 3))
                images.append(Image.fromarray(frame))

            # 3. Batched text-only inference (no bboxes in this protocol)
            text_prompts = [label] * num_images
            inputs = processor(
                images=images,
                text=text_prompts,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
            
            original_sizes = inputs.get("original_sizes")
            if original_sizes is None:
                raise RuntimeError("SAM3 processor did not return original_sizes for batched request")

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.3,
                mask_threshold=0.3,
                target_sizes=original_sizes.tolist()
            )

            # 4. Build one best-mask per image
            out_masks = []
            for i, result in enumerate(results):
                if len(result["masks"]) > 0:
                    idx = int(torch.argmax(result["scores"]))
                    best_mask = result["masks"][idx]
                    mask_final = (best_mask.cpu().numpy().astype(np.uint8) * 255)
                    if mask_final.ndim == 3:
                        mask_final = mask_final[0]
                    print(
                        f"Image {i}: masks={len(result['masks'])}, best_score={float(result['scores'][idx]):.4f}"
                    )
                else:
                    print(f"Image {i}: no masks detected, sending empty mask")
                    mask_final = np.zeros((h, w), dtype=np.uint8)

                out_masks.append(mask_final)

            # 5. Send all masks back at once
            # Response format: num_images (int32) + concatenated masks (num_images * H * W uint8)
            payload = b"".join(mask.tobytes() for mask in out_masks)
            conn.sendall(struct.pack("<i", num_images))
            conn.sendall(payload)
        finally:
            conn.close()
            torch.cuda.empty_cache() # enable if generation of the mesh is used and vram memory is not enough
            exit(0)

if __name__ == "__main__":
    run_server()
