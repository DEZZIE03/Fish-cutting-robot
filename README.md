# Fish-cutting-robot
# Fish-cutting-robot-
# ============================================
# FULL COMMAND WORKFLOW (ROS2 + MoveIt + Code)
# ============================================

# 🟢 1. Open VS Code (optional but recommended)
# code ~/ros2_ws/src

# 🟢 2. Build your workspace
# cd ~/ros2_ws
# source /opt/ros/jazzy/setup.bash
# colcon build
# source ~/ros2_ws/install/setup.bash

# 🟢 3. Start the robot simulation
# source /opt/ros/jazzy/setup.bash
# ros2 launch moveit_resources_panda_moveit_config demo.launch.py

# 👉 Wait ~10 seconds until RViz fully loads

# 🟢 4. Run your robot code
# (Open a NEW terminal)

# source /opt/ros/jazzy/setup.bash
# source ~/ros2_ws/install/setup.bash
# ros2 launch fish_cutting_demo execute_cut.launch.py

# ============================================
# 🔁 WHENEVER YOU CHANGE CODE
# ============================================

# After editing in VS Code:

# cd ~/ros2_ws
# colcon build
# source ~/ros2_ws/install/setup.bash

# Then re-run:

# ros2 launch fish_cutting_demo execute_cut.launch.py
