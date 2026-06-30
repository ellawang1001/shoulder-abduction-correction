# Reducing camera-angle error in single-camera shoulder abduction measurement with a geometric correction — data and code

This repository accompanies the manuscript and lets anyone reproduce every number
and figure. No raw video is included (videos were not retained beyond extraction,
to protect participants); the released data are per-clip angle measurements.

## Contents
- `dataset_per_clip.csv` — real data, 270 clips, 8 subjects (see DATA_DICTIONARY.md).
- `synthetic_dataset.csv` — synthetic projection-model data (2100 rows).
- `DATA_DICTIONARY.md` — column definitions.
- `code/` — Python analysis pipeline.

## Code
- `angles.py` — the four angle estimators (baseline nose-ref, raw 2D trunk-ref, bone-length correction, 3D world).
- `metrics.py` — error metrics, ICC(2,1), Bland–Altman, bootstrap CI (pure numpy).
- `sim_proof.py`, `dataset.py` — synthetic projection model and synthetic dataset generation.
- `pose_extract.py`, `extract_users.py` — run MediaPipe on videos to produce the per-clip CSVs (needs the raw videos, not included).
- `combine_8subj.py` — builds the 8-subject table and prints all reported statistics.
- `analyze.py` — produces the metrics tables and figures from a per-clip CSV.

## Reproduce
```
pip install numpy pandas matplotlib mediapipe opencv-contrib-python
# statistics + tables (from the released CSV):
python combine_8subj.py
# figures (real):
python analyze.py real dataset_per_clip.csv figures_out
# synthetic theory + data-efficiency figures:
python sim_proof.py
python analyze.py            # synthetic
```

## Key results (real, azimuth ≤ 60°, n = 216 clips, 8 subjects)
Raw 2D MAE 14.6° → bone-length correction 11.2° → 3D reference 10.7°; reduction
3.4° (23%), 95% CI 2.2–4.7°, paired t-test p < 0.001; improvement held for all
8 subjects (subject-level sign test p = 0.008).

## Ethics
Adult participants gave informed consent; data are de-identified; no medical
intervention was performed.
