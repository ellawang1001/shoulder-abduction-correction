"""
sim_proof.py — synthetic proof-of-concept for the paper's central premise:
does the training-free bone-length foreshortening correction recover the true
shoulder-abduction angle across camera viewpoints, where raw-2D fails?

No real data / no MediaPipe needed. We build a known 3D arm, project it through
a camera at varying azimuth (orthographic + perspective), optionally add
landmark noise, then run each estimator from angles.py and report error vs truth.

Body frame: X = lateral (+right), Y = vertical (+up), Z = anterior (+forward).
Right-arm abduction stays in the coronal (X-Y) plane: elbow = L*(sinT, -cosT, 0).
Camera azimuth phi orbits horizontally about Y; phi=0 is frontal.
"""

from __future__ import annotations
import numpy as np
import angles as A

# --- scene geometry (meters) ---
ARM_LEN = 0.30      # shoulder -> elbow (upper arm)
TRUNK_LEN = 0.50    # shoulder -> hip
NOSE_UP = 0.20      # nose sits this far above the shoulder (rough stand-in)


def build_scene(theta_deg: float):
    """Return dict of 3D landmarks (body frame) for true abduction theta."""
    t = np.radians(theta_deg)
    shoulder = np.array([0.0, 0.0, 0.0])
    elbow = shoulder + ARM_LEN * np.array([np.sin(t), -np.cos(t), 0.0])
    hip = shoulder + TRUNK_LEN * np.array([0.0, -1.0, 0.0])
    nose = shoulder + np.array([0.0, NOSE_UP, 0.0])
    return {"shoulder": shoulder, "elbow": elbow, "hip": hip, "nose": nose}


def _Ry(phi_deg: float) -> np.ndarray:
    p = np.radians(phi_deg)
    return np.array([[np.cos(p), 0, np.sin(p)],
                     [0, 1, 0],
                     [-np.sin(p), 0, np.cos(p)]])


def project(P3: np.ndarray, phi_deg: float, mode: str, d: float = 2.0, f: float = 1.0):
    """Rotate point by camera azimuth then project. Returns (2D image xy, 3D rotated)."""
    Pr = _Ry(phi_deg) @ P3
    if mode == "ortho":
        uv = Pr[:2].copy()
    elif mode == "persp":
        depth = d - Pr[2]                      # camera at +d looking toward -Z
        uv = f * Pr[:2] / depth
    else:
        raise ValueError(mode)
    return uv, Pr


def render(scene: dict, phi: float, mode: str, noise2d: float = 0.0,
           noise3d: float = 0.0, rng: np.random.Generator | None = None):
    """Project all landmarks; optionally add Gaussian landmark noise."""
    lm2, lm3 = {}, {}
    for k, P in scene.items():
        uv, Pr = project(P, phi, mode)
        lm2[k], lm3[k] = uv, Pr
    if noise2d and rng is not None:
        for k in lm2:
            lm2[k] = lm2[k] + rng.normal(0, noise2d, size=2)
    if noise3d and rng is not None:
        for k in lm3:
            lm3[k] = lm3[k] + rng.normal(0, noise3d, size=3)
    return lm2, lm3


def calibrate_lengths(mode: str):
    """One-time frontal (phi=0) calibration of true segment lengths in image units,
    exactly as a real study would calibrate from a frontal reference frame."""
    s = build_scene(90.0)
    lm2, _ = render(s, 0.0, mode)
    arm_len = np.linalg.norm(lm2["elbow"] - lm2["shoulder"])
    trunk_len = np.linalg.norm(lm2["hip"] - lm2["shoulder"])
    return arm_len, trunk_len


def estimate_all(lm2, lm3, arm_len, trunk_len):
    return {
        "baseline_nose": A.baseline_nose(lm2["nose"], lm2["shoulder"], lm2["elbow"]),
        "2d_trunk":      A.abduction_2d_trunk(lm2["shoulder"], lm2["hip"], lm2["elbow"]),
        "bonelength":    A.abduction_bonelength(lm2["shoulder"], lm2["hip"], lm2["elbow"],
                                                arm_len, trunk_len),
        "3d_world":      A.abduction_3d(lm3["shoulder"], lm3["hip"], lm3["elbow"]),
    }


METHODS = ["baseline_nose", "2d_trunk", "bonelength", "3d_world"]
THETAS = [30, 60, 90, 120, 150]
PHIS = [0, 30, 45, 60, 90]


def run_block(title, mode, noise2d=0.0, noise3d=0.0, reps=1, seed=42):
    rng = np.random.default_rng(seed)
    arm_len, trunk_len = calibrate_lengths(mode)
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"  (calibrated image lengths: arm={arm_len:.4f}, trunk={trunk_len:.4f})")
    # per-method accumulated absolute error
    abserr = {m: [] for m in METHODS}
    for phi in PHIS:
        print(f"\n  azimuth phi = {phi:>2}deg   | " +
              " | ".join(f"{m:>13}" for m in METHODS) + "   (signed err vs true)")
        for theta in THETAS:
            scene = build_scene(theta)
            # average over reps when noise is on
            ests = {m: [] for m in METHODS}
            for _ in range(reps):
                lm2, lm3 = render(scene, phi, mode, noise2d, noise3d, rng)
                e = estimate_all(lm2, lm3, arm_len, trunk_len)
                for m in METHODS:
                    ests[m].append(e[m])
            row = []
            for m in METHODS:
                est = float(np.mean(ests[m]))
                err = est - theta
                abserr[m].append(abs(err))
                row.append(f"{err:+13.1f}")
            print(f"    true={theta:>3}deg     | " + " | ".join(row))
    print("\n  --- mean absolute error (deg) across all theta x phi ---")
    for m in METHODS:
        print(f"    {m:>13}: {np.mean(abserr[m]):6.2f}")
    return {m: float(np.mean(abserr[m])) for m in METHODS}


if __name__ == "__main__":
    print("Synthetic proof-of-concept: shoulder-abduction angle vs camera azimuth")
    print("Methods: baseline_nose (app), 2d_trunk (anat. fix), bonelength (ours), 3d_world (reference)")

    mae_o = run_block("BLOCK 1 — ORTHOGRAPHIC, no noise (clean theory)", "ortho")
    mae_p = run_block("BLOCK 2 — PERSPECTIVE (d=2.0 m), no noise (realistic distortion)", "persp")
    mae_n = run_block("BLOCK 3 — PERSPECTIVE + landmark noise (2D sd=0.004, 3D sd=0.01 m), 200 reps",
                      "persp", noise2d=0.004, noise3d=0.01, reps=200)

    print(f"\n{'#'*78}\nSUMMARY — mean absolute error (deg), lower is better\n{'#'*78}")
    print(f"  {'method':>13} | {'ortho':>8} | {'persp':>8} | {'persp+noise':>12}")
    for m in METHODS:
        print(f"  {m:>13} | {mae_o[m]:8.2f} | {mae_p[m]:8.2f} | {mae_n[m]:12.2f}")
