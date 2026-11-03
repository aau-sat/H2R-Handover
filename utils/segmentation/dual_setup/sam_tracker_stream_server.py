#!/usr/bin/env python3
"""
SAM3 Tracker Video Async Server (Modular)

Provides asynchronous ZeroMQ-based inference using the Sam3TrackerVideoModel.
Refactored with class structure for multiple instantiation support.
"""

import time
import pickle
import traceback
from collections import deque
import zmq
import numpy as np
import torch
import cv2
from accelerate import Accelerator
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from transformers import Sam3Processor, Sam3Model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox


class SamTrackerStreamServer:
    """Asynchronous SAM3 Tracker server using ZeroMQ PULL/PUB sockets."""

    def __init__(
        self,
        pull_addr="tcp://*:5555",
        pub_addr="tcp://*:5556",
        use_compile=False,
        compile_mode="reduce-overhead",
        reset_every=150,
        print_every=100,
        rcv_hwm=50,
        snd_hwm=50,
        rcv_timeout_ms=100,
        local_files_only=True,
        init_with_bbox_only=False,
    ):
        """
        Initialize the SAM3 Tracker Stream Server.

        Args:
            pull_addr: ZMQ PULL socket address for receiving inference requests
            pub_addr: ZMQ PUB socket address for publishing mask responses
            use_compile: Whether to compile model with torch.compile
            compile_mode: Torch compile mode (e.g., "reduce-overhead")
            reset_every: Reinitialize session every N frames
            print_every: Print stats every N frames
            rcv_hwm: ZMQ receive high water mark
            snd_hwm: ZMQ send high water mark
            rcv_timeout_ms: ZMQ receive timeout in ms
            local_files_only: Load models from local cache only
            init_with_bbox_only: If True, always initialize/reinitialize using input bbox,
                never with SAM-generated start mask.
        """
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
        self.init_with_bbox_only = init_with_bbox_only

        print("[DEBUG] SamTrackerStreamServer config:")
        print(f"[DEBUG]   pull_addr={self.pull_addr}")
        print(f"[DEBUG]   pub_addr={self.pub_addr}")
        print(f"[DEBUG]   use_compile={self.use_compile} compile_mode={self.compile_mode}")
        print(f"[DEBUG]   reset_every={self.reset_every} print_every={self.print_every}")
        print(f"[DEBUG]   init_with_bbox_only={self.init_with_bbox_only}")

        # Initialize device and dtype
        self.acc = Accelerator()
        self.device = self.acc.device
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Load models
        print("[INFO] Loading SAM3 Tracker models...")
        self.model = Sam3TrackerVideoModel.from_pretrained(
            "facebook/sam3",
            local_files_only=self.local_files_only,
        ).to(self.device, dtype=self.dtype).eval()

        if self.use_compile and torch.cuda.is_available():
            print("[INFO] Compiling model (warmup cost)...")
            self.model = torch.compile(self.model, mode=self.compile_mode)

        self.processor = Sam3TrackerVideoProcessor.from_pretrained(
            "facebook/sam3",
            local_files_only=self.local_files_only,
        )

        # Load SAM3 static model (for initial mask generation)
        self.sam_model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.sam_processor = Sam3Processor.from_pretrained("facebook/sam3")

        # ZMQ Sockets
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

        print(
            f"[INFO] SAM3 Tracker server running: PULL {self.pull_addr}, PUB {self.pub_addr}"
        )

        # Session management
        self.sessions = {}
        self.processed = 0
        self.skipped_by_drain = 0
        self.t0 = time.time()
        self.reinit = False

    def make_session(self) -> dict:
        """Initializes a new, empty inference session for the SAM3 Tracker."""
        return self.processor.init_video_session(
            inference_device=self.device,
            dtype=self.dtype,
        )

    def get_session(self, obj_id: str) -> dict:
        """Retrieves or creates a tracking session for a specific object ID."""
        if obj_id not in self.sessions:
            self.sessions[obj_id] = {
                "sess": self.make_session(),
                "frame_idx": 0,
                "target_initialized": False,
                "hand_currently_tracked": False,
                "area_history": deque(maxlen=25),
                "shape_pos_history": deque(maxlen=25),
            }
        return self.sessions[obj_id]

    @staticmethod
    def should_reinitialize_from_mask(
        mask_uint8: np.ndarray,
        area_history: deque,
        shape_pos_history: deque,
        only_basic_checks: bool = True,
    ) -> tuple[bool, str]:
        """Return (reinit_required, reason) based on mask quality checks."""
        binary = (mask_uint8 > 0).astype(np.uint8)
        area = int(binary.sum())
        if area == 0:
            return True, "empty-mask"

        n_labels, _ = cv2.connectedComponents(binary, connectivity=8)

        if n_labels >= 3:
            comp_count = n_labels - 1
            return True, f"split-mask components={comp_count}"
        if only_basic_checks:
            return False, "ok"

        ys, xs = np.where(binary > 0)
        if xs.size == 0:
            return True, "empty-mask-after-components"

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        box_w = max(1, x_max - x_min + 1)
        box_h = max(1, y_max - y_min + 1)

        cx = float(xs.mean())
        cy = float(ys.mean())
        aspect_ratio = float(box_w / box_h)
        fill_ratio = float(area / float(box_w * box_h))

        if len(shape_pos_history) >= 6:
            hist = np.array(shape_pos_history, dtype=np.float32)
            med_cx = float(np.median(hist[:, 0]))
            med_cy = float(np.median(hist[:, 1]))
            med_aspect = max(float(np.median(hist[:, 2])), 1e-6)
            med_fill = max(float(np.median(hist[:, 3])), 1e-6)
            med_area = max(float(np.median(hist[:, 4])), 1.0)

            frame_h, frame_w = binary.shape
            frame_diag = max(float(np.hypot(frame_h, frame_w)), 1.0)
            center_shift = float(np.hypot(cx - med_cx, cy - med_cy) / frame_diag)

            area_ratio = float(area / med_area)
            aspect_change = float(
                max(
                    aspect_ratio / med_aspect,
                    med_aspect / max(aspect_ratio, 1e-6),
                )
            )
            fill_change = float(max(fill_ratio / med_fill, med_fill / max(fill_ratio, 1e-6)))

            if center_shift > 0.35 and area_ratio > 1.4:
                return (
                    True,
                    f"position-jump shift={center_shift:.2f} area_ratio={area_ratio:.2f}",
                )
            if aspect_change > 2.8 and area_ratio > 1.2:
                return (
                    True,
                    f"shape-jump aspect_change={aspect_change:.2f} area_ratio={area_ratio:.2f}",
                )
            if fill_change > 2.5 and area_ratio > 1.2:
                return (
                    True,
                    f"shape-fill-jump fill_change={fill_change:.2f} area_ratio={area_ratio:.2f}",
                )

        area_history.append(area)
        shape_pos_history.append((cx, cy, aspect_ratio, fill_ratio, float(area)))
        return False, "ok"

    def obtain_start_mask(
        self, label: str, image: np.ndarray, bbox: list[int]
    ) -> np.ndarray | None:
        """Obtain a starting mask for the tracker based on initial bbox and label."""
        x1, y1, x2, y2 = bbox
        h, w, _ = image.shape

        box_xywh = torch.tensor(
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        ).view(-1, 4)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)
        norm_box = normalize_bbox(box_cxcywh, w, h).flatten().tolist()
        sam_box = [[norm_box]]

        inputs = self.sam_processor(
            images=image,
            text=label,
            input_boxes=sam_box,
            input_boxes_labels=[[1]],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        results = self.sam_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        if len(results["masks"]) > 0:
            idx = torch.argmax(results["scores"])
            best_mask = results["masks"][idx]
            mask_final = best_mask.cpu().numpy().astype(np.uint8) * 255
        else:
            print("[WARN] No masks detected in obtain_start_mask.")
            return None

        torch.cuda.empty_cache()
        return mask_final

    def _decode_request(self, data: bytes):
        """Decode incoming ZMQ request."""
        req = pickle.loads(data)
        timestamp_key = req["timestamp_key"]
        h = int(req["h"])
        w = int(req["w"])
        obj_id = req.get("obj_id", "default_target")
        bbox = req.get("bbox", None)
        hand_point = req.get("hand_point", None)
        label = req.get("label", "object")
        image = req["image"]
        return timestamp_key, h, w, obj_id, bbox, hand_point, label, image

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
        # Keep current behavior by default: try SAM start mask once, then bbox.
        # With init_with_bbox_only=True always use bbox path.
        use_start_mask = not self.init_with_bbox_only

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
                    (
                        timestamp_key,
                        h,
                        w,
                        obj_id,
                        bbox,
                        hand_point,
                        label,
                        image,
                    ) = self._decode_request(data)
                    print(
                        f"[DEBUG] Decoded request: timestamp={timestamp_key}, obj_id='{obj_id}', label='{label}'"
                    )
                except Exception as e:
                    print(f"[WARN] Bad request payload: {e}")
                    continue

                if not label:
                    print("[WARN] No label provided, skipping frame")
                    continue

                if (
                    not isinstance(image, np.ndarray)
                    or image.ndim != 3
                    or image.shape[2] != 3
                ):
                    print(
                        f"[WARN] Image has unexpected format: type={type(image)}, "
                        f"shape={getattr(image, 'shape', 'N/A')}; skipping."
                    )
                    continue

                # Generate start mask only when enabled. BBox-only mode skips this path entirely.
                if use_start_mask and bbox is not None:
                    start_mask = self.obtain_start_mask(label=label, image=image, bbox=bbox)
                    use_start_mask = False
                else:
                    start_mask = None

                entry = self.get_session(obj_id)
                current_frame = entry["frame_idx"]

                inputs = self.processor(images=image, device="cpu", return_tensors="pt")
                pixel_values = inputs.pixel_values[0].to(self.device, non_blocking=True)

                # Initialize object
                if not entry["target_initialized"]:
                    if bbox is not None:
                        print(f"[DEBUG] Initializing target '{obj_id}' with bbox {bbox}")
                        self.processor.add_inputs_to_inference_session(
                            inference_session=entry["sess"],
                            frame_idx=current_frame,
                            obj_ids=[1],
                            input_boxes=[[bbox]] if start_mask is None else None,
                            input_masks=start_mask,
                            original_size=inputs.original_sizes[0],
                        )
                        entry["target_initialized"] = True
                        self.reinit = True
                        
                    else:
                        continue
                else:
                    self.reinit = False

                # Dynamic hand re-initialization
                if hand_point is not None:
                    if not entry["hand_currently_tracked"]:
                        print(
                            f"[DEBUG] Hand appeared at {hand_point}, masking as object ID 2"
                        )
                        self.processor.add_inputs_to_inference_session(
                            inference_session=entry["sess"],
                            frame_idx=current_frame,
                            obj_ids=[2],
                            input_points=[[hand_point]],
                            input_labels=[[[1]]],
                            original_size=inputs.original_sizes[0],
                        )
                        entry["hand_currently_tracked"] = True
                else:
                    if entry["hand_currently_tracked"]:
                        print(
                            f"[DEBUG] Hand disappeared. Resuming normal tracking for object '{obj_id}'"
                        )
                        entry["hand_currently_tracked"] = False

                try:
                    if torch.cuda.is_available():
                        with torch.autocast(device_type="cuda", dtype=self.dtype):
                            outputs = self.model(
                                inference_session=entry["sess"], frame=pixel_values
                            )
                    else:
                        outputs = self.model(
                            inference_session=entry["sess"], frame=pixel_values
                        )

                    video_res_masks = self.processor.post_process_masks(
                        [outputs.pred_masks],
                        original_sizes=inputs.original_sizes,
                        binarize=False,
                    )[0]

                    entry["frame_idx"] += 1

                except Exception as e:
                    print(f"[WARN] Inference failed: {e}")
                    traceback.print_exc()
                    continue

                # Extract mask of object
                mask_uint8 = np.zeros((h, w), dtype=np.uint8)
                if video_res_masks is not None and video_res_masks.numel() > 0:
                    active_obj_ids = entry["sess"].obj_ids
                    if 1 in active_obj_ids:
                        target_index = active_obj_ids.index(1)
                        single_mask = (
                            video_res_masks[target_index]
                            .squeeze()
                            .to(dtype=torch.float32)
                            .to("cpu", non_blocking=True)
                            .numpy()
                        )
                        mask_uint8 = (single_mask > 0.0).astype(np.uint8) * 255

                # Sanity check
                must_reinit, reason = self.should_reinitialize_from_mask(
                    mask_uint8,
                    entry["area_history"],
                    entry["shape_pos_history"],
                    only_basic_checks=False,
                )
                if must_reinit:
                    print(f"[WARN] Reinitializing '{obj_id}' due to {reason}. Skipping mask.")
                    if obj_id in self.sessions:
                        del self.sessions[obj_id]
                    continue

                resp = {
                    "timestamp_key": timestamp_key,
                    "h": h,
                    "w": w,
                    "mask": mask_uint8.tobytes(),
                    "reinit": self.reinit
                    
                }

                try:
                    self.pub.send(pickle.dumps(resp), flags=0)
                    print(
                        f"[DEBUG] Published mask for timestamp={timestamp_key}"
                    )
                except Exception as e:
                    print(f"[WARN] Failed to publish mask: {e}")
                    traceback.print_exc()

                self.processed += 1

                # Cleanup and reinitialization
                if self.reset_every > 0 and entry["frame_idx"] >= self.reset_every:
                    print(
                        f"[INFO] Reached {self.reset_every} frames. Reinitializing '{obj_id}'."
                    )
                    if obj_id in self.sessions:
                        del self.sessions[obj_id]

                del inputs, pixel_values
                if torch.cuda.is_available() and (self.processed % 20 == 0):
                    torch.cuda.empty_cache()

        print("[INFO] Server exiting.")

    def close(self):
        """Close sockets and terminate ZMQ context."""
        print("[DEBUG] Closing SamTrackerStreamServer sockets/context")
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
    """Main entry point: start a single SAM3 Tracker server instance."""
    init_with_bbox_only = False  # Set True to always initialize with bbox only.

    server = SamTrackerStreamServer(
        pull_addr="tcp://*:5555",
        pub_addr="tcp://*:5556",
        use_compile=False,
        compile_mode="reduce-overhead",
        reset_every=150,
        print_every=100,
        init_with_bbox_only=init_with_bbox_only,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()