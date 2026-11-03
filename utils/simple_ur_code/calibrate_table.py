import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
import numpy as np
import json
import os

JOINTS = [
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "shoulder_pan_joint",
]

Z_VARIATION_THRESHOLD = 0.05
CALIBRATION_FILE = "obstacles/table.json"

class PoseRecorder(Node):
    def __init__(self):
        super().__init__('pose_recorder')
        self.joint_positions = {}
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for FK service...")

    def joint_state_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def wait_for_joint_states(self):
        while True:
            missing = [name for name in JOINTS if name not in self.joint_positions]
            if not missing:
                break
            self.get_logger().info(f"Waiting for joint states: {missing}")
            rclpy.spin_once(self, timeout_sec=0.1)

    def get_current_pose(self) -> Pose:
        self.wait_for_joint_states()
        req = GetPositionFK.Request()
        req.header.frame_id = 'world'
        req.fk_link_names = ['tool0']
        req.robot_state = RobotState()
        req.robot_state.joint_state.name = JOINTS
        req.robot_state.joint_state.position = [self.joint_positions[n] for n in JOINTS]

        future = self.fk_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp.error_code.val != 1:
            raise RuntimeError("FK service failed")
        return resp.pose_stamped[0].pose

    def record_poses(self, num_poses=4):
        poses = []
        for i in range(num_poses):
            input(f"\nMove the robot to position {i+1} and press Enter...")
            pose = None
            while pose is None:
                pose = self.get_current_pose()
            poses.append(pose)
        return poses

    def get_avg_z(self, poses):
        z_values = [p.position.z for p in poses]
        z_min = min(z_values)
        z_max = max(z_values)
        self.get_logger().info(f"Z min: {z_min:.3f}, Z max: {z_max:.3f}, variation: {z_max - z_min:.3f}")
        if (z_max - z_min) > Z_VARIATION_THRESHOLD:
            raise RuntimeError(f"Z variation too large! ({z_max - z_min:.3f} > {Z_VARIATION_THRESHOLD})")
        return float(np.mean(z_values))

def main():
    rclpy.init()
    node = PoseRecorder()
    os.makedirs("obstacles", exist_ok=True)
    try:
        z_offset = float(input("Z-Offset (length of the gripper):" ))
        poses = node.record_poses(num_poses=2)
        avg_z = node.get_avg_z(poses)

        table_dimensions = [1.75, 1.0, 1.0]

        table_json = {
            "id": "table",
            "frame_id": "world",
            "primitives": [
                {
                    "type": "BOX",
                    "dimensions": table_dimensions
                }
            ],
            "pose": {
                "position": {
                    "x": 0.0,
                    "y": table_dimensions[1]/2 + 0.18,
                    "z": avg_z - table_dimensions[2]/2 - z_offset
                },
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "w": 0.0
                }
            },
            "operation": "ADD",
            "attached": False
        }

        with open(CALIBRATION_FILE, "w") as f:
            json.dump(table_json, f, indent=4)
        node.get_logger().info(f"Calibration saved to {CALIBRATION_FILE}")

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
