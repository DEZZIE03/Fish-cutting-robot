import time

import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive


class SimpleMoveNode(Node):
    def __init__(self):
        super().__init__("simple_move_node")

        # Give ROS time to fully start before using MoveIt
        self.get_logger().info("Starting MoveIt...")
        time.sleep(5)

        # Initialize MoveIt and get the robot arm (panda_arm group)
        self.moveit = MoveItPy(node_name="moveit_py")
        self.arm = self.moveit.get_planning_component("panda_arm")

        self.get_logger().info("MoveIt Ready!")

        # Add objects (table + fish) into the planning scene
        self.add_environment()

        # Then move the robot
        self.move_robot()

    def add_environment(self):
        # Access the planning scene (where objects live)
        planning_scene_monitor = self.moveit.get_planning_scene_monitor()

        with planning_scene_monitor.read_write() as scene:

            # ---------- TABLE ----------
            # Create a box to represent the table
            table = CollisionObject()
            table.header.frame_id = "panda_link0"  # base frame of robot
            table.id = "table"

            table_primitive = SolidPrimitive()
            table_primitive.type = SolidPrimitive.BOX
            table_primitive.dimensions = [0.8, 1.2, 0.4]  # size (x, y, z)

            table_pose = Pose()
            table_pose.position.x = 0.5   # in front of robot
            table_pose.position.y = 0.0
            table_pose.position.z = -0.2  # half below origin so top is near z=0

            table.primitives.append(table_primitive)
            table.primitive_poses.append(table_pose)
            table.operation = CollisionObject.ADD

            # Add table to scene
            scene.apply_collision_object(table)

            # ---------- FISH ----------
            # Create a small box to represent the fish
            fish = CollisionObject()
            fish.header.frame_id = "panda_link0"
            fish.id = "fish"

            fish_primitive = SolidPrimitive()
            fish_primitive.type = SolidPrimitive.BOX
            fish_primitive.dimensions = [0.2, 0.05, 0.05]  # long + thin

            fish_pose = Pose()
            fish_pose.position.x = 0.5
            fish_pose.position.y = 0.0
            fish_pose.position.z = 0.05  # sits on top of table

            fish.primitives.append(fish_primitive)
            fish.primitive_poses.append(fish_pose)
            fish.operation = CollisionObject.ADD

            # Add fish to scene
            scene.apply_collision_object(fish)

        self.get_logger().info("Table and fish added to scene.")

    def move_robot(self):
        # Always start from current robot position
        self.get_logger().info("Setting start state...")
        self.arm.set_start_state_to_current_state()

        # ---------- TARGET POSE ----------
        # This is where you want the robot hand to go
        target_pose = Pose()
        target_pose.position.x = 0.45
        target_pose.position.y = 0.0
        target_pose.position.z = 0.20  # above the fish (safe position)

        # Orientation of the tool (keep fixed for now)
        target_pose.orientation.x = 0.0
        target_pose.orientation.y = 1.0
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.0

        # MoveIt requires PoseStamped (pose + reference frame)
        target_pose_stamped = PoseStamped()
        target_pose_stamped.header.frame_id = "panda_link0"
        target_pose_stamped.pose = target_pose

        # Tell MoveIt where we want the end-effector to go
        self.get_logger().info("Setting pose goal...")
        self.arm.set_goal_state(
            pose_stamped_msg=target_pose_stamped,
            pose_link="panda_link8",  # end-effector link
        )

        # ---------- PLAN ----------
        # MoveIt computes a path from current → goal
        self.get_logger().info("Planning...")
        plan = self.arm.plan()

        # ---------- EXECUTE ----------
        if plan:
            self.get_logger().info("Executing...")
            self.moveit.execute(plan.trajectory, controllers=[])
            self.get_logger().info("Done.")
        else:
            self.get_logger().error("Planning failed")


def main(args=None):
    rclpy.init(args=args)
    node = SimpleMoveNode()

    # Run node briefly (we don't need a full loop yet)
    rclpy.spin_once(node, timeout_sec=5.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

# Hi!