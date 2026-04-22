from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import cv2
import numpy as np
import pyzed.sl as sl
from scipy.spatial.transform import Rotation

sys.path.insert(0, "./linux")
from fairino import Robot  # type: ignore

ROBOT_IP = "192.168.58.2"
MARKER_SIZE_MM = 60.0
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250
TARGET_MARKER_ID = 0
T_CAM2BASE_FILE = "camera_to_robot_T.npy"
META_FILE = "camera_to_robot_meta.json"

HOME = [-92.7091632503094, -58.63849299969059, 100.4868309096534,
        -78.5774820157797, -6.312678875309406, 1.632503094059406]

VERIFY_OFFSETS = [
    [5, -5, 5, -5, 8, 8],
    [-15, 5, -5, 10, -10, 12],
    [8, 8, 8, -8, 0, -8],
]


def _normalize_pose_result(result: Any) -> tuple[int, list[float] | None]:
    if isinstance(result, tuple):
        if len(result) >= 2 and isinstance(result[0], (int, np.integer)):
            err = int(result[0])
            pose = result[1]
            if pose is None:
                return err, None
            return err, [float(v) for v in pose[:6]]
    if isinstance(result, (list, tuple)):
        return 0, [float(v) for v in result[:6]]
    return -1, None


def get_tcp_pose(robot) -> list[float]:
    for call in (lambda: robot.GetActualTCPPose(), lambda: robot.GetActualTCPPose(0)):
        try:
            err, pose = _normalize_pose_result(call())
            if err == 0 and pose is not None and len(pose) >= 6:
                return pose[:6]
        except TypeError:
            continue
        except Exception:
            continue
    raise RuntimeError("GetActualTCPPose failed in all supported call styles")


def tcp_to_R_t(tcp: list[float], euler_order: str) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, rx, ry, rz = tcp[:6]
    R = Rotation.from_euler(euler_order, [rx, ry, rz], degrees=True).as_matrix().astype(np.float64)
    t = np.array([x, y, z], dtype=np.float64)
    return R, t


def open_zed() -> sl.Camera:
    zed = sl.Camera()
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init.camera_resolution = sl.RESOLUTION.HD720
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")
    return zed


def get_intrinsics(zed: sl.Camera):
    info = zed.get_camera_information()
    cam = info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    try:
        dist = np.array(list(cam.disto), dtype=np.float64).reshape(-1, 1)
    except Exception:
        dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist


def grab_bgr(zed: sl.Camera, runtime: sl.RuntimeParameters) -> np.ndarray | None:
    img = sl.Mat()
    for _ in range(10):
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(img, sl.VIEW.LEFT)
            bgra = np.array(img.get_data(), copy=True)
            return cv2.cvtColor(np.ascontiguousarray(bgra), cv2.COLOR_BGRA2BGR)
        time.sleep(0.02)
    return None


def detect_marker(bgr, K, dist):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or TARGET_MARKER_ID not in ids.ravel():
        return None, None, None

    idx = list(ids.ravel()).index(TARGET_MARKER_ID)
    corner = corners[idx]
    half = MARKER_SIZE_MM / 2.0
    obj_pts = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float32,
    )
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        corner.reshape(4, 2),
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None, None, None
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    reproj = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - corner.reshape(4, 2), axis=1)))
    return rvec.ravel(), tvec.ravel(), reproj


def connect_robot():
    robot = Robot.RPC(ROBOT_IP)
    time.sleep(0.5)
    err, estop = robot.GetRobotEmergencyStopState()
    if err != 0 or estop:
        raise RuntimeError("E-STOP active or connection failed. Check robot.")
    err, codes = robot.GetRobotErrorCode()
    if err != 0 or any(c != 0 for c in codes):
        raise RuntimeError(f"Robot has active errors: {codes}")
    robot.RobotEnable(1)
    robot.Mode(0)
    return robot


def movej(robot, joints: list[float], vel: float = 20.0, settle: float = 6.0):
    robot.MoveJ(joints, tool=0, user=0, vel=vel, blendT=0)
    time.sleep(settle)


def main():
    if not os.path.exists(T_CAM2BASE_FILE) or not os.path.exists(META_FILE):
        print(f"ERROR: need both {T_CAM2BASE_FILE} and {META_FILE}. Run handeye_solve_fixed.py first.")
        return

    T_cam2base = np.load(T_CAM2BASE_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    euler_order = meta["best_order"]
    marker_offset_tcp = np.array(meta["marker_offset_tcp_mm"], dtype=np.float64)

    print(f"Loaded transform from {T_CAM2BASE_FILE}")
    print(f"Loaded metadata  from {META_FILE}")
    print(f"Best Euler order: {euler_order}")
    print(f"Marker offset in TCP frame: {np.round(marker_offset_tcp, 1)} mm\n")

    robot = connect_robot()
    print("Robot connected. Moving to home first...")
    movej(robot, HOME, vel=15, settle=6)
    print(f"Home TCP: {[round(v, 3) for v in get_tcp_pose(robot)]}\n")

    zed = open_zed()
    runtime = sl.RuntimeParameters()
    K, dist = get_intrinsics(zed)
    for _ in range(30):
        zed.grab(runtime)

    position_errors = []
    offset_consistency_errors = []

    for i, offset in enumerate(VERIFY_OFFSETS):
        target = [HOME[j] + offset[j] for j in range(6)]
        input(f"[Pose {i+1}/{len(VERIFY_OFFSETS)}] Press ENTER to move...")
        print(f"  Moving to J = {[round(j, 1) for j in target]}")
        movej(robot, target, vel=20, settle=6)

        tcp = get_tcp_pose(robot)
        R_g2b, t_g2b = tcp_to_R_t(tcp, euler_order)
        marker_pred_from_robot = (R_g2b @ marker_offset_tcp) + t_g2b

        bgr = grab_bgr(zed, runtime)
        if bgr is None:
            print("  Camera grab failed — skipping")
            continue

        rvec, tvec, reproj = detect_marker(bgr, K, dist)
        if tvec is None:
            print("  ArUco NOT detected — marker out of view? Skipping.")
            continue

        p_cam = np.array([tvec[0], tvec[1], tvec[2], 1.0], dtype=np.float64)
        marker_from_camera = (T_cam2base @ p_cam)[:3]

        err_vec = marker_from_camera - marker_pred_from_robot
        pos_err = float(np.linalg.norm(err_vec))
        offset_norm = float(np.linalg.norm(marker_from_camera - np.array(tcp[:3], dtype=np.float64)))
        expected_offset_norm = float(np.linalg.norm(marker_offset_tcp))
        offset_consistency = abs(offset_norm - expected_offset_norm)

        position_errors.append(pos_err)
        offset_consistency_errors.append(offset_consistency)

        print(f"\n  Reproj error                : {reproj:.2f} px")
        print(f"  Marker from robot model     : {np.round(marker_pred_from_robot, 1)} mm")
        print(f"  Marker from camera transform: {np.round(marker_from_camera, 1)} mm")
        print(f"  3D disagreement             : {pos_err:.1f} mm")
        print(f"  Offset norm from TCP        : {offset_norm:.1f} mm")
        print(f"  Expected offset norm        : {expected_offset_norm:.1f} mm")
        print(f"  Offset norm error           : {offset_consistency:.1f} mm")

    print("\nReturning to home...")
    movej(robot, HOME, vel=15, settle=6)
    zed.close()

    if position_errors:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Mean 3D disagreement   : {float(np.mean(position_errors)):.1f} mm")
        print(f"Std 3D disagreement    : {float(np.std(position_errors)):.1f} mm")
        print(f"Mean offset norm error : {float(np.mean(offset_consistency_errors)):.1f} mm")
        print(f"Std offset norm error  : {float(np.std(offset_consistency_errors)):.1f} mm")

        mean_err = float(np.mean(position_errors))
        if mean_err < 10:
            print("Result: GOOD")
        elif mean_err < 25:
            print("Result: ACCEPTABLE")
        else:
            print("Result: POOR")
    else:
        print("\nNo poses completed successfully.")


if __name__ == "__main__":
    main()
