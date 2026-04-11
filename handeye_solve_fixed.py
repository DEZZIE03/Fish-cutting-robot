from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def build_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def tcp_to_R_t(tcp: list[float], euler_order: str) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, rx, ry, rz = tcp[:6]
    R = Rotation.from_euler(euler_order, [rx, ry, rz], degrees=True).as_matrix().astype(np.float64)
    t = np.array([x, y, z], dtype=np.float64)
    return R, t


def solve_direct(R_g2b_arr: np.ndarray, t_g2b_arr: np.ndarray, tvec_arr: np.ndarray,
                 n_trials: int = 80, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    def objective(params: np.ndarray) -> float:
        rv = params[:3]
        t_c2b = params[3:6]
        c = params[6:9]
        R_c2b = Rotation.from_rotvec(rv).as_matrix()
        pred = (R_c2b @ tvec_arr.T).T + t_c2b
        gt = np.einsum('nij,j->ni', R_g2b_arr, c) + t_g2b_arr
        diff = pred - gt
        return 0.5 * float(np.dot(diff.ravel(), diff.ravel()))

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_trials):
        x0 = np.concatenate([
            rng.standard_normal(3) * 0.5,
            rng.standard_normal(3) * 300,
            rng.standard_normal(3) * 40,
        ])
        res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 10000, "ftol": 1e-14, "gtol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res

    params = best.x
    R_c2b = Rotation.from_rotvec(params[:3]).as_matrix()
    t_c2b = params[3:6]
    c = params[6:9]

    pred = (R_c2b @ tvec_arr.T).T + t_c2b
    gt = np.einsum('nij,j->ni', R_g2b_arr, c) + t_g2b_arr
    rmse = float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1))))
    return R_c2b, t_c2b, c, rmse


def load_samples(path: Path):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def run_for_order(samples: list[dict], euler_order: str, trials: int):
    good = [s for s in samples if s.get("aruco_found") and s.get("tcp_pose_mm_deg") is not None and s.get("tvec_mm") is not None]
    R_g2b_list, t_g2b_list, tvec_list = [], [], []
    for s in good:
        try:
            R, t = tcp_to_R_t(s["tcp_pose_mm_deg"], euler_order)
        except Exception:
            continue
        R_g2b_list.append(R)
        t_g2b_list.append(t)
        tvec_list.append(np.array(s["tvec_mm"], dtype=np.float64))

    if len(R_g2b_list) < 6:
        raise RuntimeError(f"Not enough usable samples for order {euler_order}")

    R_g2b_arr = np.array(R_g2b_list)
    t_g2b_arr = np.array(t_g2b_list)
    tvec_arr = np.array(tvec_list)
    R_c2b, t_c2b, marker_offset, rmse = solve_direct(R_g2b_arr, t_g2b_arr, tvec_arr, n_trials=trials)
    T = build_4x4(R_c2b, t_c2b)
    return {
        "order": euler_order,
        "rmse_mm": rmse,
        "T_cam2base": T,
        "marker_offset_tcp_mm": marker_offset,
        "num_samples": int(len(R_g2b_arr)),
        "det": float(np.linalg.det(R_c2b)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="handeye_samples.jsonl")
    ap.add_argument("--out", default="camera_to_robot_T.npy")
    ap.add_argument("--meta", default="camera_to_robot_meta.json")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--orders", nargs="*", default=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
    args = ap.parse_args()

    path = Path(args.samples)
    if not path.exists():
        print(f"Samples file not found: {path}")
        print("Run handeye_collect_fixed.py first.")
        return 1

    samples = load_samples(path)
    print(f"Total samples: {len(samples)}")

    results = []
    for order in args.orders:
        try:
            result = run_for_order(samples, order, args.trials)
            results.append(result)
            print(f"Order {order}: RMSE={result['rmse_mm']:.2f} mm, det={result['det']:.6f}, samples={result['num_samples']}")
        except Exception as e:
            print(f"Order {order}: failed ({e})")

    if not results:
        print("No valid solve results.")
        return 1

    results.sort(key=lambda r: r["rmse_mm"])
    best = results[0]

    np.save(args.out, best["T_cam2base"])

    serializable = {
        "best_order": best["order"],
        "best_rmse_mm": float(best["rmse_mm"]),
        "num_samples": int(best["num_samples"]),
        "marker_offset_tcp_mm": [float(v) for v in best["marker_offset_tcp_mm"]],
        "T_cam2base": best["T_cam2base"].tolist(),
        "all_results": [
            {
                "order": r["order"],
                "rmse_mm": float(r["rmse_mm"]),
                "det": float(r["det"]),
                "num_samples": int(r["num_samples"]),
                "marker_offset_tcp_mm": [float(v) for v in r["marker_offset_tcp_mm"]],
            }
            for r in results
        ],
    }

    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nSaved best transform to: {args.out}")
    print(f"Saved metadata to:      {args.meta}")
    print(f"Best Euler order:       {best['order']}")
    print(f"Best RMSE:              {best['rmse_mm']:.2f} mm")
    print(f"Marker offset from TCP: {np.round(best['marker_offset_tcp_mm'], 1)} mm")
    print("\nBest T_cam2base (camera coords → robot base, mm):")
    print(np.round(best["T_cam2base"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
