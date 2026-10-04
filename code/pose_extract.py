"""
pose_extract.py — run MediaPipe Pose on each clip in ./video and produce
real_estimates.csv (one row per clip, same schema as dataset.generate_synthetic),
Included for processing provenance; not needed for CSV-based reproduction.

The Tasks API full model uses detection/presence/tracking thresholds of 0.5.
All pose-detected frames contribute; no visibility-threshold filter is executed.
2D angles use PIXEL coords (x*W, y*H) to fix aspect-ratio
distortion; the 3D condition uses MediaPipe world landmarks.

Filename convention: subjNN_ang{deg}_az{deg}_r{rep}.mp4
The 38 cm tape measurement belongs to participant A (sessions 1 and 3).
The constant below is historical descriptive metadata, not a measured length
for every participant or an input to the correction. Calibration uses the
median projected length across frontal clip-median coordinates per session.
"""
from __future__ import annotations
import os, re, glob, urllib.request
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import angles as A

VIDEO_DIR = "video"
ARM_LEN_CM = 38.0
LM = {"nose": 0, "Lsh": 11, "Rsh": 12, "Lel": 13, "Rel": 14,
      "Lhip": 23, "Rhip": 24, "Lwr": 15, "Rwr": 16}
VIS_TH = 0.5  # Historical unused constant; no visibility exclusion was applied.

# Tasks API: full model ~= app's modelComplexity=1
MODEL_PATH = "models/pose_landmarker_full.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"downloading pose model -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def make_landmarker():
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False)
    return vision.PoseLandmarker.create_from_options(opts)


def parse_name(fn: str):
    m = re.match(r"subj(\d+)_ang(\d+)_az(\d+)_r(\d+)", os.path.basename(fn))
    if not m:
        return None
    return {"subject_id": int(m[1]), "true_angle": int(m[2]),
            "azimuth": int(m[3]), "rep": int(m[4])}


def extract_clip(path):
    """Return dict of per-clip MEDIAN landmark positions (pixels for 2D, meters
    for 3D world) for both arms + nose/hips, plus frame stats."""
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    acc2d = {k: [] for k in LM}      # pixel coords (x*W, y*H)
    acc3d = {k: [] for k in LM}      # world meters
    accvis = {k: [] for k in LM}
    n_total = 0
    landmarker = make_landmarker()
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_total += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(fi * 1000.0 / fps)
        fi += 1
        res = landmarker.detect_for_video(mp_img, ts)
        if not res.pose_landmarks or not res.pose_world_landmarks:
            continue
        lm = res.pose_landmarks[0]
        wlm = res.pose_world_landmarks[0]
        for k, i in LM.items():
            acc2d[k].append([lm[i].x * W, lm[i].y * H])
            acc3d[k].append([wlm[i].x, wlm[i].y, wlm[i].z])
            accvis[k].append(lm[i].visibility)
    landmarker.close()
    cap.release()
    n_det = len(acc2d["nose"])
    if n_det == 0:
        return {"n_total": n_total, "n_det": 0}
    out = {"n_total": n_total, "n_det": n_det}
    for k in LM:
        out[k + "_2d"] = np.median(np.array(acc2d[k]), axis=0)
        out[k + "_3d"] = np.median(np.array(acc3d[k]), axis=0)
        out[k + "_vis"] = float(np.mean(accvis[k]))
    return out


def studied_side(clips_raw, info):
    """Pick the consistently-raised arm: side with larger mean 3D abduction in
    the high-angle (>=120) clips. Returns 'L' or 'R'."""
    def abd(c, side):
        sh, hip, el = c[f"{side}sh_3d"], c[f"{side}hip_3d"], c[f"{side}el_3d"]
        return A.abduction_3d(sh, hip, el)
    L, R = [], []
    for fn, c in clips_raw.items():
        if c.get("n_det", 0) == 0 or info[fn]["true_angle"] < 120:
            continue
        L.append(abd(c, "L")); R.append(abd(c, "R"))
    mL, mR = np.nanmean(L) if L else 0, np.nanmean(R) if R else 0
    return "L" if mL >= mR else "R"


def clip_estimates(c, side, arm_len_px, trunk_len_px):
    sh2, el2, hip2 = c[f"{side}sh_2d"], c[f"{side}el_2d"], c["nose_2d"]  # nose for baseline
    hip2_real = c[f"{side}hip_2d"]
    sh3, el3, hip3 = c[f"{side}sh_3d"], c[f"{side}el_3d"], c[f"{side}hip_3d"]
    arm_px = float(np.linalg.norm(c[f"{side}el_2d"] - c[f"{side}sh_2d"]))
    trunk_px = float(np.linalg.norm(c[f"{side}hip_2d"] - c[f"{side}sh_2d"]))
    return {
        "baseline_nose": A.baseline_nose(c["nose_2d"], c[f"{side}sh_2d"], c[f"{side}el_2d"]),
        "2d_trunk": A.abduction_2d_trunk(c[f"{side}sh_2d"], c[f"{side}hip_2d"], c[f"{side}el_2d"]),
        "bonelength": A.abduction_bonelength(c[f"{side}sh_2d"], c[f"{side}hip_2d"], c[f"{side}el_2d"],
                                             arm_len_px, trunk_len_px),
        "3d_world": A.abduction_3d(sh3, hip3, el3),
        "feat_armlen": arm_px, "feat_trunklen": trunk_px,
        "feat_ratio": arm_px / trunk_px if trunk_px > 1e-6 else 0.0,
        "studied_side": side,
        "vis_shoulder": c[f"{side}sh_vis"], "vis_elbow": c[f"{side}el_vis"],
    }


def run_extract(files, info, out_csv="real_estimates.csv"):
    """files: list of paths; info: {path -> {subject_id,true_angle,azimuth,rep}}."""
    ensure_model()
    print(f"extracting {len(files)} clips -> {out_csv}")
    clips_raw = {}
    for i, fn in enumerate(files, 1):
        c = extract_clip(fn)
        clips_raw[fn] = c
        det = c.get("n_det", 0)
        m = info[fn]
        flag = " <-- NO POSE" if det == 0 else (" <-- few frames" if det < 5 else "")
        print(f"  [{i:2}/{len(files)}] {os.path.basename(fn):16} "
              f"ang{m['true_angle']}_az{m['azimuth']}_r{m['rep']:<2} "
              f"det/total={det}/{c.get('n_total',0)}{flag}")

    side = studied_side(clips_raw, info)
    print(f"\nstudied arm side (auto): {side}")

    arms, trunks = [], []
    for fn, c in clips_raw.items():
        if c.get("n_det", 0) and info[fn]["azimuth"] == 0:
            arms.append(np.linalg.norm(c[f"{side}el_2d"] - c[f"{side}sh_2d"]))
            trunks.append(np.linalg.norm(c[f"{side}hip_2d"] - c[f"{side}sh_2d"]))
    arm_len_px, trunk_len_px = float(np.median(arms)), float(np.median(trunks))
    print(f"calibrated (az=0): arm={arm_len_px:.1f}px  trunk={trunk_len_px:.1f}px  "
          f"(=> {arm_len_px / ARM_LEN_CM:.2f} px/cm; arm=38cm)")

    rows = []
    for fn, c in clips_raw.items():
        row = dict(info[fn]); row["clip"] = os.path.basename(fn)
        if c.get("n_det", 0) == 0:
            for k in ["baseline_nose", "2d_trunk", "bonelength", "3d_world",
                      "feat_armlen", "feat_trunklen", "feat_ratio"]:
                row[k] = np.nan
            row["studied_side"] = side; row["n_det"] = 0
        else:
            row.update(clip_estimates(c, side, arm_len_px, trunk_len_px))
            row["n_det"] = c["n_det"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["azimuth", "true_angle", "rep"])
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(df)} rows, {df['baseline_nose'].notna().sum()} usable)")
    return df


def main():
    files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    info = {}
    for fn in files:
        m = parse_name(fn)
        if m:
            info[fn] = m
    files = [f for f in files if f in info]
    run_extract(files, info, "real_estimates.csv")


if __name__ == "__main__":
    main()
