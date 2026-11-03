#!/usr/bin/env python3

"""Single SAM server runner.

This script initializes and runs a single SAM or Tracker stream server.
It is designed to be launched via a Bash script for process isolation.
"""

import argparse
import sys
from utils.segmentation.dual_setup.sam_stream_server import SamStreamServer
from utils.segmentation.dual_setup.sam_tracker_stream_server import SamTrackerStreamServer


def run_server(pull_addr: str, pub_addr: str, server_type: str) -> None:
    """Initializes and runs a single SAM3 stream server instance.

    Args:
        pull_addr (str): The ZeroMQ PULL address for receiving images.
        pub_addr (str): The ZeroMQ PUB address for sending masks.
        server_type (str): Server implementation, either 'sam' or 'tracker'.

    Raises:
        ValueError: If an unsupported server_type is provided.
    """
    print(f"[SERVER] Starting {server_type} server: PULL={pull_addr} PUB={pub_addr}")

    if server_type == "tracker":
        server = SamTrackerStreamServer(
            pull_addr=pull_addr,
            pub_addr=pub_addr,
            use_compile=False,
            compile_mode="reduce-overhead",
            reset_every=150,
            print_every=100,
        )
    elif server_type == "sam":
        server = SamStreamServer(
            pull_addr=pull_addr,
            pub_addr=pub_addr,
            use_compile=False,
            compile_mode="reduce-overhead",
            reset_every=1000,
            print_every=100,
        )
    else:
        raise ValueError(f"Invalid server type '{server_type}'. Use 'sam' or 'tracker'.")
    
    try:
        server.run()
    finally:
        print(f"[SERVER] Stopping server: PULL={pull_addr} PUB={pub_addr}")
        server.close()


def main() -> None:
    """Parses command-line arguments and starts the server."""
    parser = argparse.ArgumentParser(description="Run a single SAM stream server.")
    parser.add_argument("--pull", type=str, required=True, help="ZMQ PULL address")
    parser.add_argument("--pub", type=str, required=True, help="ZMQ PUB address")
    parser.add_argument("--type", type=str, choices=["sam", "tracker"], default="sam", help="Server type")
    
    args = parser.parse_args()

    try:
        run_server(pull_addr=args.pull, pub_addr=args.pub, server_type=args.type)
    except KeyboardInterrupt:
        print("\n[SERVER] KeyboardInterrupt detected, terminating process...")
        sys.exit(0)


if __name__ == "__main__":
    main()