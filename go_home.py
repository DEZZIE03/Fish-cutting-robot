"""
Move the robot to its original safe home position at low speed.

This is the position the robot was at when we first ran test_move.py:
  J1=-92.71  J2=-58.64  J3=100.49  J4=-78.58  J5=-6.31  J6=1.63

Run this whenever the arm is in an awkward or unknown position and you
want to return it to a known-safe starting point.
"""

import sys
import threading
import time

sys.path.insert(0, "./linux")
from fairino import Robot  # type: ignore

# The original home captured from the very first test_move.py run.
# Only change this if you re-teach a new home.
HOME = [-92.7091632503094, -58.63849299969059, 100.4868309096534,
        -78.5774820157797, -6.312678875309406, 1.632503094059406]

ROBOT_IP = "192.168.58.2"
VEL = 40  # slow on purpose — this is a recovery move

robot = Robot.RPC(ROBOT_IP)
if not robot.is_conect:
    print(f"can't connect to {ROBOT_IP}")
    sys.exit(1)

t = threading.Thread(target=robot.robot_state_routine_thread, daemon=True)
t.start()
time.sleep(0.5)

err, estop = robot.GetRobotEmergencyStopState()
err, codes = robot.GetRobotErrorCode()
print(f"estop: {bool(estop)}  error codes: {codes}")

if estop:
    print("E-stop is active. Release it first.")
    sys.exit(1)

err, current = robot.GetActualJointPosDegree()
print(f"current joints: {[round(v, 2) for v in current]}")
print(f"target (home):  {[round(v, 2) for v in HOME]}")
print(f"speed: {VEL}%  (slow, intentional)")

input("\nPress ENTER to move to home (Ctrl-C to abort)...")

robot.RobotEnable(1)
time.sleep(1)
robot.Mode(0)
time.sleep(1)

err = robot.MoveJ(HOME, tool=0, user=0, vel=VEL, blendT=0)
print("MoveJ sent, err:", err)

# Poll until every joint is within 0.5° of home, or 90 second timeout.
# blendT=0 is non-blocking so we must wait manually — a fixed sleep isn't
# reliable because the distance to home varies each time.
print("Waiting for robot to reach home", end="", flush=True)
timeout = 90
start   = time.time()
while time.time() - start < timeout:
    time.sleep(0.5)
    _, joints = robot.GetActualJointPosDegree()
    diffs = [abs(joints[i] - HOME[i]) for i in range(6)]
    print(".", end="", flush=True)
    if max(diffs) < 0.5:
        break
print()  # newline after the dots

err, final = robot.GetActualJointPosDegree()
diffs = [abs(final[i] - HOME[i]) for i in range(6)]
if max(diffs) < 1.0:
    print(f"arrived at: {[round(v, 2) for v in final]}")
else:
    print(f"stopped at: {[round(v, 2) for v in final]}  (max error {max(diffs):.1f}° — may need another run)")

robot.RobotEnable(0)
print("done")
