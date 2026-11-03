import rclpy
from rclpy.node import Node
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject, ObjectColor
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import json
import os
from std_msgs.msg import ColorRGBA

class AddObstacleNode(Node):
    def __init__(self):
        super().__init__('add_obstacle_node')
        
        # Declare parameter so you can change the obstacle via launch
        self.declare_parameter('obstacle_json_path', '')
        json_path = self.get_parameter('obstacle_json_path').get_parameter_value().string_value

        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.scene_client.wait_for_service()

        if os.path.exists(json_path):
            self.apply_obstacle(json_path)
        else:
            self.get_logger().error(f"Obstacle file not found: {json_path}")

    def apply_obstacle(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        scene = PlanningScene()
        scene.is_diff = True
        
        co = CollisionObject()
        co.id = data["id"]
        co.header.frame_id = data["frame_id"]
        
        primitive = SolidPrimitive()
        primitive.type = getattr(SolidPrimitive, data["primitives"][0]["type"])
        
        primitive.dimensions = data["primitives"][0]["dimensions"]

        # Mapping pose logic
        pose = Pose()
        pose.position.x = data["pose"]["position"]["x"]
        pose.position.y = data["pose"]["position"]["y"]
        pose.position.z = data["pose"]["position"]["z"]
        pose.orientation.x = data["pose"]["orientation"]["x"]
        pose.orientation.y = data["pose"]["orientation"]["y"]
        pose.orientation.z = data["pose"]["orientation"]["z"]
        pose.orientation.w = data["pose"]["orientation"]["w"]

        co.primitives.append(primitive)
        co.primitive_poses.append(pose) # Ensure pose is defined here
        co.operation = CollisionObject.ADD
        scene.world.collision_objects.append(co)

        # Color
        oc = ObjectColor()
        oc.id = data["id"]
        if "color" in data:
            oc.color = ColorRGBA(
                r=float(data["color"]["r"]),
                g=float(data["color"]["g"]),
                b=float(data["color"]["b"]),
                a=float(data["color"]["a"])
            )
        else: #fallback
            oc.color = ColorRGBA(
                r=0.3,
                g=1.0,
                b=0.2,
                a=1.0
            )
        scene.object_colors.append(oc)

        req = ApplyPlanningScene.Request()
        req.scene = scene
        self.scene_client.call_async(req)
        self.get_logger().info(f"Loaded scenario obstacle: {data['id']}")


def main():
    rclpy.init()
    node = AddObstacleNode()
    # Let it run briefly to ensure the async call finishes, then shut down
    rclpy.spin_once(node, timeout_sec=10.0)
    node.destroy_node()
    rclpy.shutdown()