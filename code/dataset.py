"""
dataset.py — build a tidy clip-level table from the synthetic projection model.

Mirrors the real pipeline: each row = one captured "clip" with its true angle
and the four estimator outputs, plus observable 2D features for the ML baseline.
When real data arrives, replace generate_synthetic() with a loader for
truth.csv + the per-clip estimates from pose_extract/angles, and the rest of the
analysis (analyze.py) is unchanged.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import sim_proof as S

THETAS = list(range(20, 161, 10))   # 20..160 deg
PHIS = [0, 15, 30, 45, 60, 75, 90]


def _features(lm2):
    """Observable 2D features (no truth, no azimuth) for a learned baseline."""
    s = lm2["shoulder"]
    arm = np.linalg.norm(lm2["elbow"] - s)
    trunk = np.linalg.norm(lm2["hip"] - s)
    ratio = arm / trunk if trunk > 1e-9 else 0.0
    return arm, trunk, ratio


def generate_synthetic(mode="persp", noise2d=0.004, noise3d=0.01,
                       reps=20, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arm_len, trunk_len = S.calibrate_lengths(mode)
    rows = []
    for phi in PHIS:
        for theta in THETAS:
            scene = S.build_scene(theta)
            for r in range(reps):
                lm2, lm3 = S.render(scene, phi, mode, noise2d, noise3d, rng)
                est = S.estimate_all(lm2, lm3, arm_len, trunk_len)
                f_arm, f_trunk, f_ratio = _features(lm2)
                rows.append({
                    "true_angle": theta, "azimuth": phi, "rep": r,
                    "baseline_nose": est["baseline_nose"],
                    "2d_trunk": est["2d_trunk"],
                    "bonelength": est["bonelength"],
                    "3d_world": est["3d_world"],
                    "feat_armlen": f_arm, "feat_trunklen": f_trunk, "feat_ratio": f_ratio,
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_synthetic()
    print(df.head())
    print("rows:", len(df), "| azimuths:", sorted(df.azimuth.unique()),
          "| angles:", df.true_angle.min(), "-", df.true_angle.max())
    df.to_csv("synthetic_dataset.csv", index=False)
    print("saved synthetic_dataset.csv")
