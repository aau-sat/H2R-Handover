import time
import cv2
import torch
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor
import numpy as np

# Settings
SOURCE = "/dev/video4"          # RTSP / file / device
SCALE_FACTOR = 1
USE_COMPILE = False             # True: might be faster, but needs long warmup
COMPILE_MODE = "reduce-overhead"  # "default" , "reduce-overhead" ,"max-autotune"
SHOW_EVERY_N = 1                # UI update frequency
RESET_EVERY = 50  


# model 
acc = Accelerator()
device = acc.device

model = Sam3VideoModel.from_pretrained(
    "facebook/sam3",
    local_files_only=True
).to(device, dtype=torch.bfloat16).eval()

if USE_COMPILE:
    print("Compiling model (can take minutes)...")
    model = torch.compile(model, mode=COMPILE_MODE)

processor = Sam3VideoProcessor.from_pretrained("facebook/sam3", local_files_only=True)

cap = cv2.VideoCapture(SOURCE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {SOURCE}")

# Session helpers
def make_session(text_prompt: str):
    # Keep session storage on CPU when possible to save VRAM.
    # If Sam3 needs GPU storage, switch video_storage_device back to device.
    sess = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",        # preprocess on CPU
        video_storage_device="cpu",     # keep history off GPU if allowed
        dtype=torch.bfloat16,
    )
    sess = processor.add_text_prompt(inference_session=sess, text=text_prompt)
    return sess

def reset_session(text_prompt: str):
    print(f"\n[INFO] Switching target to: '{text_prompt}'")
    # Drop old session and clear caches
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return make_session(text_prompt)

current_prompt = "mug"
streaming_inference_session = reset_session(current_prompt)

print("\nControls: q=quit, p=person, m=mug, c=chair, d=dog")
print("Starting stream...")

# Stats
processed_frames = 0
t0 = time.time()
last_fps_t = t0
last_fps_frames = 0
fps = 0.0

# Main

with torch.inference_mode():
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        processed_frames += 1

        if processed_frames % RESET_EVERY == 0:
            streaming_inference_session = reset_session(current_prompt)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess on CPU to keep tensors off GPU as long as possible
        inputs = processor(images=frame_rgb, device="cpu", return_tensors="pt")

        # Move only pixel_values to GPU
        pixel_values = inputs.pixel_values[0].to(device, non_blocking=True)

        # Inference with autocast bf16 to save VRAM
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            model_outputs = model(
                inference_session=streaming_inference_session,
                frame=pixel_values,
                reverse=False,
            )

        processed_outputs = processor.postprocess_outputs(
            streaming_inference_session,
            model_outputs,
            original_sizes=inputs.original_sizes,  # still CPU
        )

        # Viz
        if processed_outputs and "masks" in processed_outputs:
            masks_tensor = processed_outputs["masks"]  # likely on GPU/CPU depending on impl
            if masks_tensor is not None and masks_tensor.shape[0] > 0:
                combined_mask = masks_tensor.any(dim=0)
                combined_mask = combined_mask.to("cpu", non_blocking=True).numpy()
                mask_uint8 = (combined_mask * 255).astype(np.uint8)

                green_overlay = np.zeros_like(frame_bgr)
                green_overlay[:, :, 1] = mask_uint8
                frame_bgr = cv2.addWeighted(frame_bgr, 1.0, green_overlay, 0.5, 0)

        # FPS
        if processed_frames - last_fps_frames >= 10:
            now = time.time()
            dt = now - last_fps_t
            fps = (processed_frames - last_fps_frames) / max(dt, 1e-6)
            last_fps_t = now
            last_fps_frames = processed_frames

        # ui
        if processed_frames % SHOW_EVERY_N == 0:
            cv2.putText(frame_bgr, f"Tracking: {current_prompt}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Frames: {processed_frames}  FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("SAM 3 Fast Stream", frame_bgr)

        del pixel_values, model_outputs, processed_outputs, inputs
        if torch.cuda.is_available() and (processed_frames % 5 == 0):
            torch.cuda.empty_cache()

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            current_prompt = "person"
            streaming_inference_session = reset_session(current_prompt)
        elif key == ord("m"):
            current_prompt = "mug"
            streaming_inference_session = reset_session(current_prompt)
        elif key == ord("c"):
            current_prompt = "chair"
            streaming_inference_session = reset_session(current_prompt)
        elif key == ord("d"):
            current_prompt = "dog"
            streaming_inference_session = reset_session(current_prompt)

cap.release()
cv2.destroyAllWindows()

elapsed = time.time() - t0
print(f"✓ Done. Processed frames: {processed_frames} in {elapsed:.1f}s ({processed_frames/max(elapsed,1e-6):.2f} FPS avg)")