import sys
import threading
import time
import os
import json

sys.path.insert(0, './linux')
from fairino import Robot

ROBOT_IP = os.getenv("FAIRINO_ROBOT_IP", "192.168.58.2")
PATH_FILE = './paths/Linear_Bezier_Test_Path_1.json'   # change if needed

robot = Robot.RPC(ROBOT_IP)

if not robot.is_connect:
    print(f"couldn't connect to {ROBOT_IP}")
    sys.exit(1)

# Start state thread
t = threading.Thread(target=robot.robot_state_routine_thread)
t.daemon = True
t.start()
time.sleep(0.5)

# Check ESTOP
err, estop = robot.GetRobotEmergencyStopState()
print("estop:", bool(estop))
if estop:
    print("Release E-STOP")
    sys.exit(1)

# Enable robot
enable_ret = robot.RobotEnable(1)
print("RobotEnable:", enable_ret)
time.sleep(1)

# Set AUTO mode
mode_ret = robot.Mode(0)
print("Mode:", mode_ret)
time.sleep(1)

# Load path
try:
    with open(PATH_FILE, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Error: path file not found: {PATH_FILE}")
    robot.RobotEnable(0)
    sys.exit(1)

waypoints = data.get("waypoints", [])
if len(waypoints) < 2:
    print("Error: not enough waypoints in path file.")
    robot.RobotEnable(0)
    sys.exit(1)

print(f'Loaded path: {data.get("path_name", "Unknown")}')
print(f"Waypoint count: {len(waypoints)}")

def waypoint_to_desc_pos(wp):
    return [
        wp["x"],
        wp["y"],
        wp["z"],
        wp["rx"],
        wp["ry"],
        wp["rz"]
    ]

# -----------------------------
# 1) MoveJ to first waypoint
# -----------------------------
start_pos = waypoint_to_desc_pos(waypoints[0])
print("Start waypoint:", start_pos)

ret = robot.robot.GetInverseKin(0, start_pos, -1)
print("GetInverseKin return for start:", ret)

if ret[0] != 0:
    print("IK failed for start waypoint with error code:", ret[0])
    robot.RobotEnable(0)
    sys.exit(1)

start_joint_pos = [ret[1], ret[2], ret[3], ret[4], ret[5], ret[6]]
print("Start joint_pos:", start_joint_pos)

print("Moving to first waypoint with MoveJ...")
err = robot.MoveJ(
    joint_pos=start_joint_pos,
    tool=0,
    user=0,
    desc_pos=start_pos,
    vel=20,
    acc=0,
    ovl=100,
    exaxis_pos=[0.0, 0.0, 0.0, 0.0],
    blendT=-1.0,
    offset_flag=0,
    offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
print("MoveJ result:", err)

if err != 0:
    print("MoveJ to start failed.")
    robot.RobotEnable(0)
    sys.exit(1)

time.sleep(1)

# -----------------------------
# 2) MoveL through the rest
# -----------------------------
print("Following path with MoveL...")

for i in range(1, len(waypoints)):
    cur_pos = waypoint_to_desc_pos(waypoints[i])
    print(f"Waypoint {i}: {cur_pos}")

    # Solve IK for current Cartesian target
    ret = robot.robot.GetInverseKin(0, cur_pos, -1)
    print("GetInverseKin return:", ret)

    if ret[0] != 0:
        print(f"IK failed at waypoint {i} with error code:", ret[0])
        robot.RobotEnable(0)
        sys.exit(1)

    joint_pos = [ret[1], ret[2], ret[3], ret[4], ret[5], ret[6]]
    print("Target joint_pos:", joint_pos)

    # For path following, use MoveL
    err = robot.MoveL(
        joint_pos=joint_pos,
        tool=0,
        user=0,
        desc_pos=cur_pos,
        vel=20,
        acc=0,
        ovl=100,
        blendR=5.0,   # try differnt numbers depending on behavior
        exaxis_pos=[0.0, 0.0, 0.0, 0.0],
        search=0,
        offset_flag=0,
        offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    print("MoveL result:", err)

    if err != 0:
        print(f"MoveL failed at waypoint {i}")
        robot.RobotEnable(0)
        sys.exit(1)

time.sleep(2)

disable_ret = robot.RobotEnable(0)
print("RobotEnable off:", disable_ret)
print("done")
