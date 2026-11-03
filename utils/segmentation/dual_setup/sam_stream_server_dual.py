#!/usr/bin/env python3

"""Run two SAM servers using multiprocessing.

Set server type with either:
- Env var: SAM_SERVER_TYPE=sam|tracker
- CLI arg: python sam_stream_server_dual.py sam|tracker
"""

import multiprocessing
import os
import sys
from sam_stream_server import SamStreamServer
from sam_tracker_stream_server import SamTrackerStreamServer


def _resolve_server_type(default_type: str = "tracker") -> str:
    server_type = default_type
    if len(sys.argv) > 1:
        server_type = sys.argv[1].strip().lower()
    else:
        server_type = os.getenv("SAM_SERVER_TYPE", default_type).strip().lower()

    if server_type not in ("sam", "tracker"):
        raise ValueError(
            f"Invalid server type '{server_type}'. Use 'sam' or 'tracker'."
        )
    return server_type


def _run_server(pull_addr: str, pub_addr: str, server_type: str) -> None:
    """Initializes and runs a single SAM3 stream server instance.

    Args:
        pull_addr (str): The ZeroMQ PULL address for receiving images.
        pub_addr (str): The ZeroMQ PUB address for sending masks.
        server_type (str): Server implementation, either "sam" or "tracker".
    """
    print(
        f"[DUAL] Starting {server_type} server: PULL={pull_addr} PUB={pub_addr}"
    )

    # Initialize INSIDE process to avoid CUDA/ZMQ multiprocessing issues.
    if server_type == "tracker":
        server = SamTrackerStreamServer(
            pull_addr=pull_addr,
            pub_addr=pub_addr,
            use_compile=False,
            compile_mode="reduce-overhead",
            reset_every=150,
            print_every=100,
            init_with_bbox_only=True
        )
    else:
        server = SamStreamServer(
            pull_addr=pull_addr,
            pub_addr=pub_addr,
            use_compile=False,
            compile_mode="reduce-overhead",
            reset_every=1000,
            print_every=100,
        )
    
    try:
        server.run()
    finally:
        print(f"[DUAL] Stopping server: PULL={pull_addr} PUB={pub_addr}")
        server.close()


def main() -> None:
    """Main entry point to launch dual SAM stream servers using multiprocessing."""
    server_type = _resolve_server_type(default_type="tracker")
    print(
        f"[DUAL] Initializing dual '{server_type}' servers via multiprocessing..."
    )

    # Best practice on Ubuntu/Linux for PyTorch multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Define processes
    p1 = multiprocessing.Process(
        target=_run_server,
        args=("tcp://*:5555", "tcp://*:5556", server_type),
        daemon=True
    )
    p2 = multiprocessing.Process(
        target=_run_server,
        args=("tcp://*:5565", "tcp://*:5566", server_type),
        daemon=True
    )

    print("[DUAL] Launching processes for left and right servers")
    p1.start()
    p2.start()
    
    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("[DUAL] KeyboardInterrupt detected, terminating processes...")
        p1.terminate()
        p2.terminate()

    print("[DUAL] Both server processes exited")


if __name__ == "__main__":
    main()