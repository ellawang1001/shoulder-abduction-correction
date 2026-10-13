# Revised shoulder-abduction analysis

This directory contains the unchanged de-identified CSV inputs and the analysis used for the JEI revision. The original submission is preserved separately in `../02_review/original/`.

## Reproduce the revision

From this directory, with Python and the packages listed in `requirements.txt`:

```powershell
python code/revision_analysis.py
python -B code/sensitivity_audit.py
python code/make_fig1.py
python code/build_manuscript.py
python code/build_handoff.py
powershell -NoProfile -ExecutionPolicy Bypass -File code/export_pdf_isolated.ps1
python code/validate_package.py
python code/make_public_archive.py
python code/finalize_package.py
```

Scripts locate inputs relative to their own paths, so the working directory is not significant. PDF export requires Microsoft Word on Windows. The analysis and plots do not require Word, MediaPipe, raw videos, or a downloaded pose model. `results/build_environment.json` records the tested package versions. The final command refreshes the file manifest after PDF export and validation.

## Contents

- `data/real_estimates_8subj.csv`: original 270 clip-level observations, unchanged.
- `data/synthetic_dataset.csv`: original 2,100 noisy synthetic observations, unchanged.
- `code/angles.py`, `metrics.py`, `sim_proof.py`, `dataset.py`: original estimator, metric, and simulation helpers.
- `code/revision_analysis.py`: participant statistics, cluster confidence intervals, offset analysis, calibration sensitivity, and Figures 2–4.
- `code/make_fig1.py`: revised schematic.
- `source/manuscript_revised.md`: manuscript source; tables are included here.
- `source/reviewer_responses.json`: authored response text and exact pointers into the editor's original comments.
- `source/figures/`: vector SVG and high-resolution TIFF copies; submission PNGs are in `../01_submission/figures/`.
- `source/tables/`: separate CSV and Word versions of the manuscript tables.
- `results/`: machine-readable numerical outputs, including every new simulated observation.

## Analysis conventions

The primary comparison uses one mean error difference per participant. Participant A's two sessions remain one participant. Pooled MAE weights clips equally, so participant A contributes twice as many clips as each other participant. Figure 3 weights participants equally. Confidence intervals for pooled MAE and ICC resample eight participant clusters, retaining all their clips, using 10,000 percentile-bootstrap samples and seed 42. Original per-clip improvement intervals are descriptive/exploratory.

The initial `sim_proof.py` noisy demonstration computes the error of the mean estimate within each condition. Figure 2 and the added sensitivity experiment compute the mean absolute error of individual trials. Their grids differ and their aggregate errors are not interchangeable. The manuscript states these distinctions.

Sensitivity calibration uses median projected lengths from six noiseless postures at each specified calibration azimuth. Test landmarks, including their noise, are identical across offsets. Positive and negative offsets use `sim_proof._Ry`'s rotation convention. Real observations are not modified.

## Limits of this reproducibility package

`pose_extract.py` is included for provenance. It is not a self-contained raw-video reproduction workflow: running it requires the original videos, a MediaPipe model file, OpenCV, and the appropriate MediaPipe environment. Identifiable videos are not part of this public-analysis package. The code did not execute a visibility-threshold filter; all detected-pose frames contributed to coordinate medians. Existing CSVs are sufficient to reproduce the revised statistical results and figures.

The public GitHub and OSF links refer to the existing deposits. Earlier revision materials were uploaded according to the project record. The 2026-10-14 depth-sign analysis and calibration audit have only been staged locally and require public synchronization before resubmission; response P16 records this pending action. Do not publish the editor's letter, school approval signatures, consent documents, or the whole submission package. Document-building scripts refer to private editorial materials and should be omitted from a public analysis-only deposit unless adapted.

`code/sensitivity_audit.py` reconstructs the original corrected angles from saved projected lengths, raw 2D angles, and session-specific frontal calibration, then compares fixed same-sign and opposite-sign depth assignments. It also recomputes all 30,000 Table 4 estimates and compares them with the saved trials. Outputs are `results/depth_sign_trials.csv`, `depth_sign_summary.csv`, and `sensitivity_audit.json`. There is no target-based per-clip sign selection, no change to the original estimator or inputs, and no independent validation of the alternative signs.
