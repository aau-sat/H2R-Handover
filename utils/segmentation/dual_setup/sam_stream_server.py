#!/usr/bin/env python3

"""SAM3 async stream server with configurable runtime parameters.

Default behavior matches the previous script:
- Binds PULL on tcp://*:5555
- Binds PUB on tcp://*:5556
- Tracks sessions per label with periodic reset
"""

import time
import pickle
import traceback
import zmq
import numpy as np
import torch
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor


class SamStreamServer:
    """Asynchronous SAM3 server using ZeroMQ PULL/PUB sockets."""

    def __init__(
        self,
        pull_addr="tcp://*:5555",
        pub_addr="tcp://*:5556",
        use_compile=False,
        compile_mode="reduce-overhead",
        reset_every=1000,
        print_every=100,
        rcv_hwm=50,
        snd_hwm=50,
        rcv_timeout_ms=100,
        local_files_only=True,
    ):
        self.pull_addr = pull_addr
        self.pub_addr = pub_addr
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.reset_every = reset_every
        self.print_every = print_every
        self.rcv_hwm = rcv_hwm
        self.snd_hwm = snd_hwm
        self.rcv_timeout_ms = rcv_timeout_ms
        self.local_files_only = local_files_only

        print("[DEBUG] SamStreamServer config:")
        print(f"[DEBUG]   pull_addr={self.pull_addr}")
        print(f"[DEBUG]   pub_addr={self.pub_addr}")
        print(f"[DEBUG]   use_compile={self.use_compile} compile_mode={self.compile_mode}")
        print(f"[DEBUG]   reset_every={self.reset_every} print_every={self.print_every}")
        print(f"[DEBUG]   rcv_hwm={self.rcv_hwm} snd_hwm={self.snd_hwm} rcv_timeout_ms={self.rcv_timeout_ms}")

        self.acc = Accelerator()
        self.device = self.acc.device
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print("[INFO] Loading SAM3...")
        self.model = Sam3VideoModel.from_pretrained(
            "facebook/sam3",
            local_files_only=self.local_files_only,
        ).to(self.device, dtype=self.dtype).eval()

        if self.use_compile and torch.cuda.is_available():
            print("[INFO] Compiling model (warmup cost)...")
            self.model = torch.compile(self.model, mode=self.compile_mode)

        self.processor = Sam3VideoProcessor.from_pretrained(
            "facebook/sam3",
            local_files_only=self.local_files_only,
        )

        self.ctx = zmq.Context()

        self.pull = self.ctx.socket(zmq.PULL)
        self.pull.bind(self.pull_addr)
        self.pull.setsockopt(zmq.RCVHWM, self.rcv_hwm)
        self.pull.setsockopt(zmq.LINGER, 0)
        self.pull.setsockopt(zmq.RCVTIMEO, self.rcv_timeout_ms)

        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(self.pub_addr)
        self.pub.setsockopt(zmq.SNDHWM, self.snd_hwm)
        self.pub.setsockopt(zmq.LINGER, 0)

        print(f"[INFO] SAM async server running: PULL {self.pull_addr}, PUB {self.pub_addr}")

        self.sessions = {}
        self.processed = 0
        self.skipped_by_drain = 0
        self.t0 = time.time()

    def make_session(self, text_prompt: str):
        """Create and prime a new SAM3 inference session."""
        sess = self.processor.init_video_session(
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype if torch.cuda.is_available() else torch.float32,
        )
        sess = self.processor.add_text_prompt(inference_session=sess, text=text_prompt)
        sess = self.processor.add_text_prompt(
            inference_session=sess,
            text="person, hands, fingers, arms",
        )
        return sess

    def get_session(self, label: str) -> dict:
        if label not in self.sessions:
            print(f"[DEBUG] Creating new session for label '{label}'")
            self.sessions[label] = {"sess": self.make_session(label), "count": 0}
        return self.sessions[label]

    def maybe_reset(self, label: str):
        """Reset a label session every N frames to avoid long-term drift."""
        entry = self.get_session(label)
        entry["count"] += 1
        if self.reset_every > 0 and entry["count"] % self.reset_every == 0:
            print(f"[DEBUG] Resetting session for label '{label}' at frame count {entry['count']}")
            self.sessions[label] = {"sess": self.make_session(label), "count": 0}

    def _decode_request(self, data: bytes):
        req = pickle.loads(data)
        timestamp_key = req["timestamp_key"]
        h = int(req["h"])
        w = int(req["w"])
        label = req.get("label", None)
        image = req["image"]
        return timestamp_key, h, w, label, image

    def _recv_latest(self):
        """Receive one message and drain backlog, keeping only the newest."""
        try:
            data = self.pull.recv()
        except zmq.error.Again:
            if self.processed > 0 and self.processed % self.print_every == 0:
                dt = time.time() - self.t0
                print(
                    f"[INFO] processed={self.processed} avg_fps={self.processed/max(dt,1e-6):.2f} drained={self.skipped_by_drain}"
                )
            return None

        latest_data = data
        drained_count = 0
        while True:
            try:
                latest_data = self.pull.recv(flags=zmq.NOBLOCK)
                self.skipped_by_drain += 1
                drained_count += 1
            except zmq.error.Again:
                break

        if drained_count > 0:
            print(f"[DEBUG] Drained {drained_count} old messages, using latest")
        return latest_data

    def run(self):
        """Run the server loop until interrupted."""
        with torch.inference_mode():
            while True:
                try:
                    data = self._recv_latest()
                    if data is None:
                        continue
                except KeyboardInterrupt:
                    print("[DEBUG] KeyboardInterrupt received in run loop")
                    break

                try:
                    timestamp_key, h, w, label, image = self._decode_request(data)
                    # label = 'hand palm' # DEBUG
                    print(
                        f"[DEBUG] Decoded request: timestamp={timestamp_key}, size=({h},{w}), label='{label}'"
                    )
                except Exception as e:
                    print(f"[WARN] Bad request payload: {e}")
                    continue

                if not label:
                    print("[WARN] No label provided, skipping frame")
                    continue

                if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
                    print(
                        f"[WARN] Image has unexpected format: type={type(image)}, "
                        f"shape={getattr(image, 'shape', 'N/A')}; skipping."
                    )
                    continue

                h_img, w_img = image.shape[:2]
                h, w = h_img, w_img

                entry = self.get_session(label)
                sess = entry["sess"]
                self.maybe_reset(label)
                print(f"[DEBUG] Using session for label '{label}' (frame count: {entry['count']})")

                inputs = self.processor(images=image, device="cpu", return_tensors="pt")
                pixel_values = inputs.pixel_values[0].to(self.device, non_blocking=True)
                model_outputs = None

                try:
                    if torch.cuda.is_available():
                        with torch.autocast(device_type="cuda", dtype=self.dtype):
                            model_outputs = self.model(
                                inference_session=sess,
                                frame=pixel_values,
                                reverse=False,
                            )
                    else:
                        model_outputs = self.model(
                            inference_session=sess,
                            frame=pixel_values,
                            reverse=False,
                        )

                    processed_outputs = self.processor.postprocess_outputs(
                        sess,
                        model_outputs,
                        original_sizes=inputs.original_sizes,
                    )
                except Exception as e:
                    print(f"[WARN] SAM inference failed: {e}")
                    traceback.print_exc()
                    continue
                finally:
                    del pixel_values
                    if model_outputs is not None:
                        del model_outputs

                mask_uint8 = None
                try:
                    masks_tensor = processed_outputs.get("masks", None) if processed_outputs else None
                    if masks_tensor is not None and hasattr(masks_tensor, "shape") and masks_tensor.shape[0] > 0:
                        single_mask = masks_tensor[0]
                        single_mask = single_mask.to("cpu", non_blocking=True).numpy()

                        binary_mask = single_mask > 0.0
                        mask_uint8 = binary_mask.astype(np.uint8) * 255

                        border = 3
                        mask_uint8[:border, :] = 0
                        mask_uint8[-border:, :] = 0
                        mask_uint8[:, :border] = 0
                        mask_uint8[:, -border:] = 0
                except Exception as e:
                    print(f"[WARN] Postprocess mask failed: {e}")

                if mask_uint8 is None:
                    print("[WARN] No mask generated, sending empty mask")
                    mask_uint8 = np.zeros((h, w), dtype=np.uint8)
                else:
                    print(
                        f"[DEBUG] Mask generated: shape={mask_uint8.shape}, non-zero pixels={np.count_nonzero(mask_uint8)}"
                    )

                if mask_uint8.shape != (h, w):
                    print(f"[WARN] Mask shape mismatch: {mask_uint8.shape} vs expected ({h},{w}), cropping...")
                    mask_uint8 = mask_uint8[:h, :w]
                    if mask_uint8.shape != (h, w):
                        mask_uint8 = np.zeros((h, w), dtype=np.uint8)

                resp = {
                    "timestamp_key": timestamp_key,
                    "h": h,
                    "w": w,
                    "mask": mask_uint8.tobytes(),
                }

                try:
                    self.pub.send(pickle.dumps(resp), flags=0)
                    print(f"[DEBUG] Published mask for timestamp={timestamp_key}")
                except Exception as e:
                    print(f"[WARN] Failed to publish mask: {e}")
                    traceback.print_exc()

                self.processed += 1

                del inputs, processed_outputs
                if torch.cuda.is_available() and (self.processed % 20 == 0):
                    torch.cuda.empty_cache()

        print("[INFO] Server exiting.")

    def close(self):
        """Close sockets and terminate ZMQ context."""
        print("[DEBUG] Closing SamStreamServer sockets/context")
        try:
            self.pull.close(0)
        except Exception:
            pass
        try:
            self.pub.close(0)
        except Exception:
            pass
        try:
            self.ctx.term()
        except Exception:
            pass


def main():
    # Preserves previous default behavior.
    server = SamStreamServer(
        pull_addr="tcp://*:5555",
        pub_addr="tcp://*:5556",
        use_compile=False,
        compile_mode="reduce-overhead",
        reset_every=10000,
        print_every=10000,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
