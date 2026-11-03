#!/usr/bin/env python3

import argparse
import socket
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_SOCKET_PATH = "/tmp/sam3_socket"


def recv_exact(conn, size):
	data = b""
	while len(data) < size:
		chunk = conn.recv(size - len(data))
		if not chunk:
			raise ConnectionError("Connection closed while receiving data")
		data += chunk
	return data


def parse_args():
	parser = argparse.ArgumentParser(
		description="Send one image + labels to segmentation_service_multiprompt and save returned masks."
	)
	parser.add_argument("--image", type=str, help="Path to input image")
	parser.add_argument(
		"--labels",
		nargs="+",
		help="Primary label to query. Example: --labels cup"
	)
	parser.add_argument("--socket", type=str, default=DEFAULT_SOCKET_PATH, help="Unix socket path")
	parser.add_argument("--output-dir", type=str, default=".", help="Directory for output masks")
	parser.add_argument("--prefix", type=str, default="mask", help="Output filename prefix")
	parser.add_argument(
		"--bbox",
		type=int,
		nargs=4,
		metavar=("X1", "Y1", "X2", "Y2"),
		default=[0, 0, 0, 0],
		help="Optional bbox prompt. Use 0 0 0 0 for text-only mode"
	)
	return parser.parse_args()


def main():
	args = parse_args()

	if not args.image:
		raise ValueError("Missing --image")
	if not args.labels:
		raise ValueError("Missing --labels")

	image_path = Path(args.image)
	if not image_path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")

	image = Image.open(image_path).convert("RGB")
	frame = np.array(image, dtype=np.uint8)
	h, w = frame.shape[:2]

	x1, y1, x2, y2 = args.bbox
	# Server now expects one primary label and auto-adds hand internally.
	label_str = args.labels[0]
	label_bytes = label_str.encode("utf-8")
	label_len = len(label_bytes)

	# Request protocol:
	#   int32 h, w, x1, y1, x2, y2, label_len
	#   label bytes (utf-8, length=label_len)
	#   image bytes (H*W*3 uint8 RGB)
	header = np.array([h, w, x1, y1, x2, y2, label_len], dtype=np.int32)
    
	with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
		client.connect(args.socket)
		client.sendall(header.tobytes())
		client.sendall(label_bytes)
		client.sendall(frame.tobytes())

		# Response protocol:
		#   uint32 object_ok (0/1)
		#   uint32 hand_ok   (0/1)
		#   repeat 2 times:
		#     uint32 payload_size
		#     payload bytes (H*W uint8 mask)
		object_ok, hand_ok = struct.unpack("<II", recv_exact(client, 8))
		print(
			f"Received mask(s), "
			f"object_ok={bool(object_ok)}, hand_ok={bool(hand_ok)}"
		)

		out_dir = Path(args.output_dir)
		out_dir.mkdir(parents=True, exist_ok=True)
		label_order = [label_str, "hand"]

		for i in range(2):
			payload_size = struct.unpack("<I", recv_exact(client, 4))[0]
			payload = recv_exact(client, payload_size)
			mask = np.frombuffer(payload, dtype=np.uint8).reshape((h, w))

			label_name = label_order[i] if i < len(label_order) else f"label{i}"
			safe_label = label_name.replace(" ", "_")
			out_path = out_dir / f"{args.prefix}_{safe_label}_{i}.png"
			cv2.imwrite(str(out_path), mask)
			print(f"Saved: {out_path}")


if __name__ == "__main__":
	main()


# python3 /home/student-team/rgmc-2026/utils/segmentation/test_scripts/multiprompt_single_image.py \
#   --image /home/student-team/rgmc-2026/utils/realsense/left_top/rgb_left_13.png \
#   --labels cup \
#   --output-dir /home/student-team/rgmc-2026/utils/segmentation/test_scripts \
#   --prefix sam