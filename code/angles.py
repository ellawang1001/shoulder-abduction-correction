"""
angles.py — shoulder-abduction angle estimators under study.

Each estimator takes landmark coordinates (2D image-normalized, and/or 3D world)
and returns an abduction angle in degrees. The SAME functions are used on
(a) synthetic projected data for the proof-of-concept (sim_proof.py) and later
(b) real MediaPipe landmarks extracted from video (pose_extract.py).

Conventions
-----------
A shoulder-abduction angle is the angle at the SHOULDER between the trunk
(shoulder->hip) and the upper arm (shoulder->elbow). 0deg = arm down alongside
trunk; 90deg = arm horizontal; 180deg = arm overhead.

Landmarks are passed as length-2 (x,y) or length-3 (x,y,z) numpy arrays.
"""

from __future__ import annotations
import numpy as np


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle (deg) between two vectors, robust to fp drift."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    c = np.dot(u, v) / (nu * nv)
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


# --- Condition 1: BASELINE — raw 2D law of cosines, nose-referenced -----------
# Reproduces the app's calculateAngle() (apps/web/src/lib/ai-coach.utils.ts:18-29)
# applied to the (nose, shoulder, elbow) triple the app uses for arm raises.

def angle_2d_lawcos(a2: np.ndarray, b2: np.ndarray, c2: np.ndarray) -> float:
    """2D law-of-cosines angle at vertex b2 (the app's exact formula)."""
    a2, b2, c2 = np.asarray(a2, float)[:2], np.asarray(b2, float)[:2], np.asarray(c2, float)[:2]
    return _angle_between(a2 - b2, c2 - b2)


def baseline_nose(nose2, shoulder2, elbow2) -> float:
    """Condition 1: app baseline — angle at shoulder using NOSE as reference."""
    return angle_2d_lawcos(nose2, shoulder2, elbow2)


# --- Condition 2: ANATOMICAL-PLANE FIX — trunk-referenced 2D ------------------

def abduction_2d_trunk(shoulder2, hip2, elbow2) -> float:
    """Condition 2: 2D abduction = angle at shoulder between (hip) and (elbow).
    Same projected coords as baseline, but anatomically correct reference."""
    shoulder2 = np.asarray(shoulder2, float)[:2]
    return _angle_between(np.asarray(hip2, float)[:2] - shoulder2,
                          np.asarray(elbow2, float)[:2] - shoulder2)


# --- Condition 3: BONE-LENGTH FORESHORTENING CORRECTION (the hero, 2D only) ---

def _lift_segment(p2_distal: np.ndarray, p2_vertex: np.ndarray,
                  true_len: float, sign: float = +1.0) -> np.ndarray:
    """Recover a 3D segment (camera frame) from its 2D projection + known true
    length. Orthographic model: |3D|^2 = u^2 + v^2 + z^2 = true_len^2, so the
    out-of-plane component magnitude is z = sqrt(true_len^2 - p^2). Sign is the
    depth ambiguity (toward/away from camera); caller supplies a prior."""
    d2 = np.asarray(p2_distal, float)[:2] - np.asarray(p2_vertex, float)[:2]
    p = np.linalg.norm(d2)
    z2 = true_len * true_len - p * p
    z = sign * np.sqrt(z2) if z2 > 0 else 0.0  # Zero inferred depth when projected length reaches or exceeds calibration.
    return np.array([d2[0], d2[1], z])


def abduction_bonelength(shoulder2, hip2, elbow2,
                         arm_len: float, trunk_len: float,
                         arm_sign: float = +1.0, trunk_sign: float = +1.0) -> float:
    """Condition 3: training-free geometric correction. Re-inflate each
    foreshortened 2D segment to its known true length to recover the 3D
    direction (camera frame), then take the frame-invariant angle between them.
    `arm_len`/`trunk_len` are the true segment lengths in the SAME units as the
    2D coords. Real-video calibration uses the median segment length across
    six frontal clip-median landmark sets per recording session; synthetic
    calibration is specified separately in the simulation routines."""
    arm3 = _lift_segment(elbow2, shoulder2, arm_len, arm_sign)
    trunk3 = _lift_segment(hip2, shoulder2, trunk_len, trunk_sign)
    return _angle_between(trunk3, arm3)


# --- Condition 4: 3D WORLD LANDMARKS (the reference) ----------------------------

def abduction_3d(shoulder3, hip3, elbow3) -> float:
    """Condition 4: angle computed directly from 3D world landmarks."""
    shoulder3 = np.asarray(shoulder3, float)[:3]
    return _angle_between(np.asarray(hip3, float)[:3] - shoulder3,
                          np.asarray(elbow3, float)[:3] - shoulder3)


# Condition 5 (ML-style learned correction) lives in a separate module once real
# data exists; it is a fitted regressor, not a closed-form function.
