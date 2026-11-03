#!/usr/bin/env python3

import asyncio
import signal
import struct
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from bleak import BleakClient

BLE_ADDRESS = "D9:E9:6D:70:68:D0"
# BLE_ADDRESS = "F8:9B:D5:57:96:0C"
# BLE_ADDRESS = "FD:7A:24:4C:0B:36"
# BLE_ADDRESS = "C9:3F:23:86:7B:03"

# TODO: replace with the characteristic UUID that sends the notifications
BLE_NOTIFY_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

DEFAULT_BLE_ADDRESS = "D9:E9:6D:70:68:D0"
DEFAULT_NUM_CHANNELS = 7
DEFAULT_HEADER_BYTES = 3
DEFAULT_PAYLOAD_BYTES = DEFAULT_NUM_CHANNELS * 4  # bytes (4 bytes per float)
DEFAULT_EXPECTED_MIN_LEN = DEFAULT_HEADER_BYTES + DEFAULT_PAYLOAD_BYTES
DEFAULT_DEVICE_UID = "DEVB9975E99211783F"
# DEVICE_UID = "DEV925C67C4A7659FB3"
# DEVICE_UID = "DEV17FE769C0253BC55"


class TaxelPublisher(Node):
    def __init__(self):
        super().__init__("ble_electrode_publisher")

        def get_param_value(name: str, default: Any) -> Any:
            value = self.get_parameter(name).value
            return default if value is None else value

        self.declare_parameter("ble_address", DEFAULT_BLE_ADDRESS)
        self.declare_parameter("num_channels", DEFAULT_NUM_CHANNELS)
        self.declare_parameter("header_bytes", DEFAULT_HEADER_BYTES)
        self.declare_parameter("payload_bytes", DEFAULT_PAYLOAD_BYTES)
        self.declare_parameter("expected_min_len", DEFAULT_EXPECTED_MIN_LEN)
        self.declare_parameter("device_uid", DEFAULT_DEVICE_UID)

        self.ble_address = str(get_param_value("ble_address", DEFAULT_BLE_ADDRESS))
        self.num_channels = int(get_param_value("num_channels", DEFAULT_NUM_CHANNELS))
        self.header_bytes = int(get_param_value("header_bytes", DEFAULT_HEADER_BYTES))
        self.payload_bytes = int(get_param_value("payload_bytes", DEFAULT_PAYLOAD_BYTES))
        self.expected_min_len = int(get_param_value("expected_min_len", DEFAULT_EXPECTED_MIN_LEN))
        self.device_uid = str(get_param_value("device_uid", DEFAULT_DEVICE_UID))

        self.pub = self.create_publisher(
            Float32MultiArray,
            f"/{self.device_uid}/Pressure",
            10,
        )

        self.get_logger().info(
            "Parameters: "
            f"ble_address={self.ble_address}, num_channels={self.num_channels}, "
            f"header_bytes={self.header_bytes}, payload_bytes={self.payload_bytes}, "
            f"expected_min_len={self.expected_min_len}, device_uid={self.device_uid}"
        )

    def publish_taxels(self, values):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in values]
        self.pub.publish(msg)


def decode_taxels(data: bytearray, num_channels: int, header_bytes: int, payload_bytes: int, expected_min_len: int):
    if len(data) < expected_min_len:
        return None

    payload = data[header_bytes:header_bytes + payload_bytes]

    # Big-endian 32-bit floats: unpack all channel values at once
    try:
        # ensure we have a bytes object for struct.unpack
        values = struct.unpack(
            ">" + "f" * num_channels,
            bytes(payload),
        )
    except struct.error:
        return None

    return values


async def run_ble(node: TaxelPublisher, stop_event: asyncio.Event):
    def notification_handler(_, data: bytearray):
        values = decode_taxels(
            data,
            node.num_channels,
            node.header_bytes,
            node.payload_bytes,
            node.expected_min_len,
        )
        if values is None:
            node.get_logger().warn(
                f"Received packet too short: {len(data)} bytes"
            )
            return

        node.publish_taxels(values)

    node.get_logger().info(f"Connecting to BLE device {node.ble_address} ...")

    async with BleakClient(node.ble_address) as client:
        if not client.is_connected:
            raise RuntimeError(f"Failed to connect to {node.ble_address}")

        node.get_logger().info("BLE connected")
        node.get_logger().info(
            f"Starting notifications on {BLE_NOTIFY_CHAR_UUID}"
        )

        await client.start_notify(BLE_NOTIFY_CHAR_UUID, notification_handler)

        try:
            while not stop_event.is_set():
                rclpy.spin_once(node, timeout_sec=0.0)
                await asyncio.sleep(0.01)
        finally:
            node.get_logger().info("Stopping notifications...")
            await client.stop_notify(BLE_NOTIFY_CHAR_UUID)

    node.get_logger().info("BLE disconnected")


async def main_async():
    rclpy.init()
    node = TaxelPublisher()
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    def request_stop():
        node.get_logger().info("Ctrl+C received, shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            pass

    try:
        await run_ble(node, stop_event)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
