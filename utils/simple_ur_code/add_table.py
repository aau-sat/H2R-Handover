import rclpy
from rclpy.node import Node
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import json
import os

JSON_FILES = {
    #"gripper": "obstacles/virtual_gripper.json",
    "base": "obstacles/base.json",
    "table": "obstacles/table.json"
}

class AddObstaclesNode(Node):
    def __init__(self):
        super().__init__('add_obstacles_node')

        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for ApplyPlanningScene service...')

        scene = PlanningScene()
        scene.is_diff = True

        # gripper_file = JSON_FILES["gripper"]
        # if os.path.exists(gripper_file):
        #     with open(gripper_file, "r") as f:
        #         gripper_data = json.load(f)

        #     primitive = SolidPrimitive()
        #     primitive.type = getattr(SolidPrimitive, gripper_data["primitives"][0]["type"])
        #     primitive.dimensions = gripper_data["primitives"][0]["dimensions"]

        #     pose = Pose()
        #     pose.position.x = gripper_data["pose"]["position"]["x"]
        #     pose.position.y = gripper_data["pose"]["position"]["y"]
        #     pose.position.z = gripper_data["pose"]["position"]["z"]
        #     pose.orientation.x = gripper_data["pose"]["orientation"]["x"]
        #     pose.orientation.y = gripper_data["pose"]["orientation"]["y"]
        #     pose.orientation.z = gripper_data["pose"]["orientation"]["z"]
        #     pose.orientation.w = gripper_data["pose"]["orientation"]["w"]

        #     aco = AttachedCollisionObject()
        #     aco.link_name = gripper_data.get("link_name", "tool0")
        #     aco.object.header.frame_id = gripper_data["frame_id"]
        #     aco.object.id = gripper_data["id"]
        #     aco.object.primitives.append(primitive)
        #     aco.object.primitive_poses.append(pose)
        #     aco.object.operation = aco.object.ADD
        #     scene.robot_state._attached_collision_objects.append(aco)
        #     self.get_logger().info("Added attached object: gripper")


        base_file = JSON_FILES["base"]
        if os.path.exists(base_file):
            with open(base_file, "r") as f:
                base_data = json.load(f)

            primitive = SolidPrimitive()
            primitive.type = getattr(SolidPrimitive, base_data["primitives"][0]["type"])
            primitive.dimensions = base_data["primitives"][0]["dimensions"]

            pose = Pose()
            pose.position.x = base_data["pose"]["position"]["x"]
            pose.position.y = base_data["pose"]["position"]["y"]
            pose.position.z = base_data["pose"]["position"]["z"]
            pose.orientation.x = base_data["pose"]["orientation"]["x"]
            pose.orientation.y = base_data["pose"]["orientation"]["y"]
            pose.orientation.z = base_data["pose"]["orientation"]["z"]
            pose.orientation.w = base_data["pose"]["orientation"]["w"]

            co = CollisionObject()
            co.id = base_data["id"]
            co.header.frame_id = base_data["frame_id"]
            co.primitives.append(primitive)
            co.primitive_poses.append(pose)
            co.operation = CollisionObject.ADD
            scene.world.collision_objects.append(co)
            self.get_logger().info("Added world object: base")

        table_file = JSON_FILES["table"]
        if os.path.exists(table_file):
            with open(table_file, "r") as f:
                table_data = json.load(f)

            primitive = SolidPrimitive()
            primitive.type = getattr(SolidPrimitive, table_data["primitives"][0]["type"])
            primitive.dimensions = table_data["primitives"][0]["dimensions"]

            pose = Pose()
            pose.position.x = table_data["pose"]["position"]["x"]
            pose.position.y = table_data["pose"]["position"]["y"]
            pose.position.z = table_data["pose"]["position"]["z"]
            pose.orientation.x = table_data["pose"]["orientation"]["x"]
            pose.orientation.y = table_data["pose"]["orientation"]["y"]
            pose.orientation.z = table_data["pose"]["orientation"]["z"]
            pose.orientation.w = table_data["pose"]["orientation"]["w"]

            co = CollisionObject()
            co.id = table_data["id"]
            co.header.frame_id = table_data["frame_id"]
            co.primitives.append(primitive)
            co.primitive_poses.append(pose)
            co.operation = CollisionObject.ADD
            scene.world.collision_objects.append(co)
            self.get_logger().info("Added world object: table")

        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self.scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info("All obstacles added from JSON!")

def main():
    rclpy.init()
    node = AddObstaclesNode()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
