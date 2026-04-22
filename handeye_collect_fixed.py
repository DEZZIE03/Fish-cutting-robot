from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyzed.sl as sl

sys.path.insert(0, "./linux")
from fairino import Robot  # type: ignore

MARKER_SIZE_MM = 60.0
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250
TARGET_MARKER_ID = 0

OFFSETS = np.array([
    [0, 0, 0, 0, 0, 0],
    [12, 0, 0, 0, 0, 0],
    [-12, 0, 0, 0, 0, 0],
    [0, 10, 0, 0, 0, 0],
    [0, -10, 0, 0, 0, 0],
    [0, 0, 10, 0, 0, 0],
    [0, 0, -10, 0, 0, 0],
    [0, 0, 0, 12, 0, 0],
    [0, 0, 0, -12, 0, 0],
    [0, 0, 0, 0, 12, 0],
    [0, 0, 0, 0, -12, 0],
    [0, 0, 0, 0, 0, 15],
    [0, 0, 0, 0, 0, -15],
    [10, 8, 0, 0, 0, 12],
    [-10, 8, 0, 0, 0, -12],
    [8, -8, 8, 0, 0, 0],
    [-8, -8, -8, 0, 0, 0],
    [0, 8, 0, 10, -10, 10],
    [0, 0, 0, 20, 15, 0],
    [0, 0, 0, -20, 15, 0],
    [0, 0, 0, 20, -15, 20],
    [0, 0, 0, -20, -15, -20],
    [0, 0, 0, 0, 20, 25],
], dtype=float)


@dataclass
class Sample:
    idx: int
    unix_time: float
    joint_target_deg: list[float]
    joint_actual_deg: list[float] | None
    tcp_pose_mm_deg: list[float] | None
    aruco_found: bool
    rvec: list[float] | None
    tvec_mm: list[float] | None
    reproj_err_px: float | None
    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    image_size: list[int]
    image_path: str | None


def open_zed() -> sl.Camera:
    zed = sl.Camera()
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")
    return zed


def get_intrinsics(zed: sl.Camera):
    info = zed.get_camera_information()
    cam = info.camera_configuration.calibration_parameters.left_cam
    w = info.camera_configuration.resolution.width
    h = info.camera_configuration.resolution.height
    K = np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    try:
        dist = np.array(list(cam.disto), dtype=np.float64).reshape(-1, 1)
    except Exception:
        dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist, w, h


def grab_bgr(zed: sl.Camera, runtime: sl.RuntimeParameters) -> np.ndarray | None:
    img = sl.Mat()
    for _ in range(10):
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(img, sl.VIEW.LEFT)
            bgra = np.array(img.get_data(), copy=True)
            return cv2.cvtColor(np.ascontiguousarray(bgra), cv2.COLOR_BGRA2BGR)
        time.sleep(0.02)
    return None


def detect_marker(bgr: np.ndarray, K: np.ndarray, dist: np.ndarray, marker_size_mm: float):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    vis = bgr.copy()

    if ids is None or TARGET_MARKER_ID not in ids.ravel():
        return None, None, None, vis

    target_idx = list(ids.ravel()).index(TARGET_MARKER_ID)
    corner = corners[target_idx]

    half = marker_size_mm / 2.0
    obj_pts = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, corner.reshape(4, 2), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None, None, vis

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    reproj_err = float(np.linalg.norm(proj.reshape(4, 2) - corner.reshape(4, 2), axis=1).mean())

    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    cv2.drawFrameAxes(vis, K, dist, rvec, tvec, marker_size_mm * 0.5)
    return rvec.ravel().tolist(), tvec.ravel().tolist(), reproj_err, vis


def connect_robot(ip: str):
    robot = Robot.RPC(ip)
    if not robot.is_connect:
        raise RuntimeError(f"Cannot connect to robot at {ip}")
    t = threading.Thread(target=robot.robot_state_routine_thread, daemon=True)
    t.start()
    time.sleep(0.5)
    return robot


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


def get_tcp_pose(robot) -> list[float] | None:
    for call in (lambda: robot.GetActualTCPPose(), lambda: robot.GetActualTCPPose(0)):
        try:
            err, pose = _normalize_pose_result(call())
            if err == 0 and pose is not None and len(pose) >= 6:
                return pose[:6]
        except TypeError:
            continue
        except Exception:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.58.2")
    ap.add_argument("--out", default="handeye_samples.jsonl")
    ap.add_argument("--images-dir", default="handeye_images")
    ap.add_argument("--vel", type=float, default=15.0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        robot = connect_robot(args.robot_ip)
    except RuntimeError as e:
        print(e)
        return 1

    err, estop = robot.GetRobotEmergencyStopState()
    if err != 0 or estop:
        print("E-stop is active or status read failed.")
        return 1

    err, codes = robot.GetRobotErrorCode()
    print(f"estop: False  error codes: {codes}")

    err, seed_joints = robot.GetActualJointPosDegree()
    if err != 0 or seed_joints is None:
        print("Could not read joint positions.")
        return 1

    seed = np.array(seed_joints, dtype=float)
    targets = (seed + OFFSETS).tolist()

    print(f"Home pose:        {np.round(seed, 2).tolist()}")
    print(f"Poses to collect: {len(targets)}")
    print(f"Speed: {args.vel}%   Settle: {args.settle}s")

    if args.dry_run:
        for i, t in enumerate(targets):
            print(f"  {i:02d}: {np.round(t, 2).tolist()}")
        return 0

    try:
        zed = open_zed()
    except RuntimeError as e:
        print(e)
        return 1

    K, dist, w, h = get_intrinsics(zed)
    print(f"ZED intrinsics: fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")

    runtime = sl.RuntimeParameters()
    print(f"Warming up ZED ({args.warmup} frames)...")
    n = 0
    while n < args.warmup:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            n += 1

    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    robot.RobotEnable(1)
    time.sleep(1)
    robot.Mode(0)
    time.sleep(1)

    found_count = 0
    print(f"\nImages → {images_dir}/   Data → {out_path}")
    input("Press ENTER to begin collection (Ctrl-C to abort)...\n")

    with out_path.open("w", encoding="utf-8") as f:
        for idx, target in enumerate(targets):
            print(f"[{idx+1}/{len(targets)}] moving to pose {idx:02d}...")
            err = robot.MoveJ(target, tool=0, user=0, vel=float(args.vel), blendT=0)
            if err != 0:
                print(f"  MoveJ error {err}, skipping")
                continue

            time.sleep(float(args.settle))

            err_j, joints_actual = robot.GetActualJointPosDegree()
            tcp = get_tcp_pose(robot)

            if err_j == 0 and joints_actual is not None:
                print(f"  joints : {[round(v, 2) for v in joints_actual]}")
            if tcp is not None:
                print(f"  tcp    : {[round(v, 1) for v in tcp]}")
            else:
                print("  tcp    : FAILED TO READ")

            bgr = grab_bgr(zed, runtime)
            if bgr is None:
                print("  ZED grab failed, skipping")
                continue

            rvec, tvec, reproj, vis = detect_marker(bgr, K, dist, MARKER_SIZE_MM)
            img_path = str(images_dir / f"pose_{idx:03d}.png")
            cv2.imwrite(img_path, vis)

            if rvec is not None:
                found_count += 1
                print(f"  ArUco  : found  reproj={reproj:.2f}px  tvec={[round(v, 1) for v in tvec]} mm")
            else:
                print("  ArUco  : NOT found (marker out of view?)")

            sample = Sample(
                idx=idx,
                unix_time=time.time(),
                joint_target_deg=[float(v) for v in target],
                joint_actual_deg=[float(v) for v in joints_actual] if (err_j == 0 and joints_actual is not None) else None,
                tcp_pose_mm_deg=tcp,
                aruco_found=rvec is not None,
                rvec=rvec,
                tvec_mm=tvec,
                reproj_err_px=reproj,
                camera_matrix=K.tolist(),
                dist_coeffs=[float(v) for v in dist.ravel()],
                image_size=[int(w), int(h)],
                image_path=img_path,
            )
            f.write(json.dumps(asdict(sample)) + "\n")
            f.flush()

    print("\nReturning to home pose...")
    robot.MoveJ(seed.tolist(), tool=0, user=0, vel=float(args.vel), blendT=0)
    time.sleep(float(args.settle) + 1)

    robot.RobotEnable(0)
    zed.close()

    print(f"\nDone. {found_count}/{len(targets)} poses had ArUco detected.")
    if found_count < 10:
        print("Warning: fewer than 10 good detections. Calibration may be poor.")
        print("  Check handeye_images/ — is the marker visible in the camera view?")
    print(f"Data saved to: {out_path}")
    print("Next step: python3 handeye_solve_fixed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
