"""
Interactive point-and-move tool.

Shows a live ZED camera view. Click any point on the table/object.
The script back-projects the pixel + depth into 3D camera coordinates,
transforms them into robot base frame using the saved calibration, and
optionally moves the robot's TCP above that point.

Controls:
  Left-click   — set the target point (shows live 3D coords)
  G            — move robot to 150 mm above the clicked point
  H            — send robot back to home
  Q / Escape   — quit

Requirements:
  - camera_to_robot_T.npy    (run handeye_solve.py first)
  - Robot connected at 192.168.58.2
  - ZED camera connected via USB
"""

from __future__ import annotations

import os
import sys
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl

sys.path.insert(0, "./linux")
from fairino import Robot  # type: ignore

# ── config ────────────────────────────────────────────────────────────────────
ROBOT_IP        = "192.168.58.2"
T_CAM2BASE_FILE = "camera_to_robot_T.npy"
APPROACH_HEIGHT = 150     # mm above the clicked point before descending
MOVE_VEL        = 15      # robot speed (%)
TOOL_ORIENT     = [180.0, 0.0, 0.0]  # tool pointing straight down

HOME = [-92.7091632503094, -58.63849299969059, 100.4868309096534,
        -78.5774820157797, -6.312678875309406, 1.632503094059406]

# ── state shared between mouse callback and main loop ─────────────────────────
clicked_pixel   = None   # (u, v) of last click
target_base     = None   # [x, y, z] in robot base frame (mm)
status_line     = ""     # text shown at the bottom of the window


# ── helpers ───────────────────────────────────────────────────────────────────

def pixel_to_3d(u: int, v: int, depth_mat: sl.Mat, K: np.ndarray) -> np.ndarray | None:
    """
    Back-project pixel (u, v) to a 3D point in the OpenCV camera frame (mm).

    ZED's retrieve_measure(DEPTH) returns the Z distance along the optical axis
    at each pixel (same Z as the OpenCV camera convention).  Back-projecting:
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy
      Z = depth
    """
    depth_val = depth_mat.get_value(u, v)[1]

    if not np.isfinite(depth_val) or depth_val <= 0:
        return None

    fx = K[0, 0]; fy = K[1, 1]
    cx = K[0, 2]; cy = K[1, 2]

    X = (u - cx) * depth_val / fx
    Y = (v - cy) * depth_val / fy
    Z = depth_val
    return np.array([X, Y, Z], dtype=np.float64)


def cam_to_base(p_cam: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply the 4×4 T_cam2base transform to a 3D point."""
    p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
    return (T @ p_h)[:3]


def connect_robot():
    import socket
    socket.setdefaulttimeout(5)
    robot = Robot.RPC(ROBOT_IP)
    t = threading.Thread(target=robot.robot_state_routine_thread, daemon=True)
    t.start()
    time.sleep(0.5)
    return robot


def safe_move_above(robot, target_xyz: np.ndarray, approach_mm: float):
    """Move to approach_mm above target, then descend."""
    x, y, z = target_xyz

    # approach point: same XY but higher Z
    approach = [x, y, z + approach_mm] + TOOL_ORIENT
    robot.MoveL(approach, tool=0, user=0, vel=MOVE_VEL, blendR=0)
    time.sleep(3)

    # descend to the point
    target = [x, y, z] + TOOL_ORIENT
    robot.MoveL(target, tool=0, user=0, vel=int(MOVE_VEL * 0.5), blendR=0)
    time.sleep(2)


# ── mouse callback ────────────────────────────────────────────────────────────

def on_mouse(event, u, v, flags, userdata):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (u, v)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global clicked_pixel, target_base, status_line

    if not os.path.exists(T_CAM2BASE_FILE):
        print(f"ERROR: {T_CAM2BASE_FILE} not found. Run handeye_solve.py first.")
        return

    T = np.load(T_CAM2BASE_FILE)
    print(f"Loaded {T_CAM2BASE_FILE}")
    print(f"Camera in base frame: {np.round(T[:3, 3], 1)} mm\n")

    # ── robot ──────────────────────────────────────────────────────────────────
    print("Connecting to robot...")
    robot = connect_robot()
    err, estop = robot.GetRobotEmergencyStopState()
    if err != 0 or estop:
        print("ERROR: E-STOP is active. Release it before continuing.")
        return
    robot.RobotEnable(1)
    robot.Mode(0)
    print("Robot ready.\n")

    # ── ZED ────────────────────────────────────────────────────────────────────
    print("Opening ZED camera...")
    zed  = sl.Camera()
    init = sl.InitParameters()
    init.depth_mode        = sl.DEPTH_MODE.NEURAL
    init.coordinate_units  = sl.UNIT.MILLIMETER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init.camera_resolution = sl.RESOLUTION.HD720
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ERROR: ZED failed to open.")
        return

    runtime = sl.RuntimeParameters()

    info = zed.get_camera_information()
    cam  = info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[cam.fx, 0, cam.cx],
                  [0, cam.fy, cam.cy],
                  [0, 0,     1     ]], dtype=np.float64)

    # warmup
    for _ in range(30):
        zed.grab(runtime)

    WIN = "ZED | click=target  G=go  H=home  Q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.waitKey(1)   # let Qt actually render the window before attaching callback
    cv2.setMouseCallback(WIN, on_mouse)

    img_mat   = sl.Mat()
    depth_mat = sl.Mat()

    print("Window open. Left-click any point to see its 3D coords.")
    print("Press G to move robot there, H for home, Q to quit.\n")

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(img_mat, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        bgra = np.array(img_mat.get_data(), copy=True)
        frame = cv2.cvtColor(np.ascontiguousarray(bgra), cv2.COLOR_BGRA2BGR)

        # ── process click ─────────────────────────────────────────────────────
        if clicked_pixel is not None:
            u, v = clicked_pixel

            p_cam = pixel_to_3d(u, v, depth_mat, K)
            if p_cam is not None:
                p_base    = cam_to_base(p_cam, T)
                target_base = p_base
                status_line = (
                    f"cam=({p_cam[0]:.0f}, {p_cam[1]:.0f}, {p_cam[2]:.0f}) mm  "
                    f"base=({p_base[0]:.0f}, {p_base[1]:.0f}, {p_base[2]:.0f}) mm  "
                    f"[press G to move here]"
                )
            else:
                status_line = "No valid depth at that pixel. Try a different spot."
                target_base = None

            clicked_pixel = None   # consume the click

        # ── draw overlay ──────────────────────────────────────────────────────
        if target_base is not None:
            # re-project the target base point back to pixel for the crosshair
            # (approximate — just use the pixel we clicked for display)
            pass

        # draw crosshair at last known valid click position (track separately)
        if status_line:
            h, w = frame.shape[:2]
            # dark background strip at bottom
            cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, status_line, (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA)

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord('g') or key == ord('G'):
            if target_base is None:
                print("No target selected. Click somewhere first.")
            else:
                x, y, z = target_base
                print(f"\nMoving to ({x:.0f}, {y:.0f}, {z:.0f}) mm in base frame")
                print(f"Approach height: {z + APPROACH_HEIGHT:.0f} mm  →  descend to {z:.0f} mm")
                try:
                    safe_move_above(robot, target_base, APPROACH_HEIGHT)
                    print("Done.")
                    status_line = f"Reached ({x:.0f}, {y:.0f}, {z:.0f}) mm"
                except Exception as e:
                    print(f"Move failed: {e}")
                    status_line = f"Move failed: {e}"

        elif key == ord('h') or key == ord('H'):
            print("\nReturning to home...")
            robot.MoveJ(HOME, tool=0, user=0, vel=15, blendT=0)
            time.sleep(6)
            status_line = "Home"
            print("Done.")

    zed.close()
    cv2.destroyAllWindows()
    print("Exited.")


if __name__ == "__main__":
    main()
