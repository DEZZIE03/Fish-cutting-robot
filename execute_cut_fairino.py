#What I will be coding in on friday when i go into to see the arm 
import time

import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy
from geometry_msgs.msg import Pose, PoseStamped



# Ask the lab or check launch files
PLANNING_GROUP = "PUT_FAIRINO_PLANNING_GROUP_HERE"
BASE_FRAME = "PUT_FAIRINO_BASE_FRAME_HERE"
EE_LINK = "PUT_FAIRINO_END_EFFECTOR_LINK_HERE"


class MoveAboveTable(Node):
    def __init__(self):
        super().__init__("move_above_table")

        self.get_logger().info("Starting MoveIt...")
        time.sleep(5)

        # Initialize MoveIt
        self.moveit = MoveItPy(node_name="moveit_py")

        # Get robot arm
        self.arm = self.moveit.get_planning_component(PLANNING_GROUP)

        self.get_logger().info("MoveIt Ready!")

        # Run single test motion
        self.move_above_table()

    # Move above the table 
    def move_above_table(self):

        # Always start from current robot state
        self.arm.set_start_state_to_current_state()

        # Move robot manually using the UI
        # Record X, Y, Z (mm convert to meters)
        # Convert mm → meters (divide by 1000)

        x = -0.312097 
        y = 0.762458   
        z = 0.35   

        target_pose = Pose()
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z

        # Use ONE fixed orientation(fix in the lab if needed)
        target_pose.orientation.x = 0.0
        target_pose.orientation.y = 1.0
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.0

        # Wrap in PoseStamped
        target_pose_stamped = PoseStamped()
        target_pose_stamped.header.frame_id = BASE_FRAME
        target_pose_stamped.pose = target_pose

        # Set goal
        self.arm.set_goal_state(
            pose_stamped_msg=target_pose_stamped,
            pose_link=EE_LINK,
        )

        self.get_logger().info(
            f"Planning to ({x:.3f}, {y:.3f}, {z:.3f})"
        )

        # Plan motion
        plan = self.arm.plan()

        # Execute motion
        if plan:
            self.get_logger().info("Executing...")
            self.moveit.execute(plan.trajectory, controllers=[])
            self.get_logger().info("Done.")
        else:
            self.get_logger().error("Planning failed.")


def main(args=None):
    rclpy.init(args=args)

    node = MoveAboveTable()

    # Spin briefly
    rclpy.spin_once(node, timeout_sec=5.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
