# Foreshortening contributes to camera-angle error in static single-camera shoulder abduction measurement

Data and analysis code accompanying the revised JEI manuscript by Ella Wang and Zhirong Wang. This deposit reproduces the statistical analyses and four figures from the released clip-level data. Raw participant videos are not publicly distributed. Their absence from this deposit does not mean that they were deleted.

## Reproduce

From this directory:

```powershell
python -m pip install -r requirements.txt
python -B code/reproduce.py
```

The pinned package versions were tested with Python 3.14. Scripts resolve their input and output paths relative to this directory, so they also work when invoked from another working directory. CSV-based reproduction needs only NumPy, pandas, and matplotlib; it does not require MediaPipe, OpenCV, Microsoft Word, private videos, a downloaded pose model, or files elsewhere in the authors' workspace.

The command regenerates statistics, the calibration sensitivity experiment, Figures 1–4, and the original synthetic demonstration log, then checks data hashes and key results. It does not rewrite the two input CSVs. Existing files with the same names inside results/ and figures/ are regenerated.

## Contents

- `dataset_per_clip.csv`: unchanged real-video data, 270 clips from eight adults in nine sessions. Participant A has two sessions.
- `synthetic_dataset.csv`: unchanged original noisy simulation, 2,100 trials.
- `DATA_DICTIONARY.md`: column definitions, reference targets, session mapping, and processing details.
- `code/reproduce.py`: complete CSV-based reproduction command.
- `code/revision_analysis.py`: revised statistics, individual results, offset analysis, calibration sensitivity, and Figures 2–4.
- `code/make_fig1.py`: revised geometric schematic.
- `code/angles.py`, `metrics.py`, `sim_proof.py`, `dataset.py`: estimator, statistical, and simulation helpers.
- `code/analyze.py`, `combine_8subj.py`: compatibility entry points for the revised reproduction workflow. They no longer require nine unpublished session CSVs.
- `code/pose_extract.py`, `extract_users.py`: original extraction logic retained as provenance, not part of CSV-based reproduction. Extraction would require private videos, a model, and a separate compatible MediaPipe/OpenCV environment. Comments about visibility were corrected without changing extraction logic.
- `results/`: statistics, participant results, simulation outputs, runtime environment, and validation report.
- `figures/`: revised Figures 1–4 in PNG (300 dpi) and SVG format.
- `CHANGELOG.md`: revision changes and unchanged input hashes.
- `CITATION.cff`: citation metadata for the dataset and code.

## Repository and archive

Repository: https://github.com/ellawang1001/shoulder-abduction-correction

OSF project: https://osf.io/8a3rw/

The public analysis layout is the same for this repository and the revised OSF deposit. The two input CSVs retain their original names and contents. Git history and local environment folders are excluded from the generated file manifest. Signed forms, editorial correspondence, and manuscript-review documents are not part of this repository.

## Analysis conventions

The primary improvement test uses eight participant-mean error differences, a two-sided sign test, and a participant bootstrap. Pooled MAE and ICC confidence intervals use 10,000 participant-cluster bootstrap samples, seed 42; all clips and both sessions for participant A stay together. Original per-clip intervals for paired improvements are exploratory. Pooled estimates weight clips equally, whereas Figure 3 weights participants equally.

At camera azimuths ≤ 60° (216 clips), pooled MAE is 14.6° for raw 2D, 11.2° after correction, and 10.7° for the 3D reference. All eight participants improved; the primary sign-test p-value is 0.008 after rounding. See results/statistics.json for unrounded values and confidence intervals.

The initial noisy `sim_proof.py` demonstration averages angle estimates within each condition before calculating absolute error. Figure 2 and the new calibration sensitivity experiment instead use per-trial MAE on their specified grids. These quantities are not interchangeable. Undefined raw/baseline angles for a degenerate clean projection are recorded as null in the demonstration JSON and nan in its text log.

Sensitivity calibration uses median projected lengths across six noiseless postures at 0°, ±5°, and ±10° camera azimuth. Test landmarks and their noise remain identical across offsets. The summary includes 4,800 trials at test azimuths ≤ 60° per calibration; these are simulated observations, not additional participants.

All pose-detected frames contributed to the original clip-median coordinates; no visibility-threshold exclusion was applied. Calibration was session-specific, using six frontal clips. The nose baseline is an app-derived reference-choice comparator and is not presented as a clinically validated shoulder-abduction measure. Physical upper-arm length of 38 cm was measured for participant A and was not an input to the correction.

## Scope and ethics

Adult participants provided informed consent. Released measurements are de-identified; no medical intervention was performed. This public deposit excludes raw videos, signed forms, editorial correspondence, and private review documents. Re-extracting landmarks from videos is outside the self-contained CSV-based workflow.


## Additional sensitivity analysis (2026-10-14)

Run `python -B code/sensitivity_audit.py` to reproduce the fixed same-sign versus opposite-sign comparison from the original CSV and to recompute all Table 4 trials. Results are in `results/depth_sign_trials.csv`, `depth_sign_summary.csv`, and `sensitivity_audit.json`. The alternative signs are an exploratory check, not an independently validated estimator.
