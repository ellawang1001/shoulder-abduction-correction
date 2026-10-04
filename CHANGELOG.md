# Revision deposit changes

Prepared locally on 2026-10-04 for the revised manuscript, "Foreshortening contributes to camera-angle error in static single-camera shoulder abduction measurement."

## Unchanged observations and estimator calculations

- The real dataset remains 270 clips, nine recording sessions, and eight participants.
- The original synthetic dataset remains 2,100 trials.
- No real observations were added, excluded, or re-extracted. Estimator calculations are unchanged.
- The real CSV retains the original public filename, `dataset_per_clip.csv`.

SHA-256 input hashes:

```text
dataset_per_clip.csv
1db2ddb663a3eefcceff31ae3b4ba84761e0530580ea9241f16f7a5747dfdea2

synthetic_dataset.csv
6358cc97c5bfcce30138cb5864a5d767039132de069347f06c1d3f47b42f24a9
```

## Added or updated

- Participant-cluster confidence intervals for pooled MAE and ICC; participant-level inference retained as primary.
- Per-participant MAE, individual Figure 3 points, zero-target estimates, and exploratory offset analysis.
- Calibration sensitivity at 0°, ±5°, and ±10°, with complete simulated trial output.
- Updated Figures 1–4, including the simulation geometry panel and participant dots.
- Corrected visibility, reference-angle, session-specific calibration, and 38 cm measurement descriptions.
- Corrected the README's claim that videos were no longer retained: raw videos are not publicly distributed.
- Added a single CSV-based reproduction command and self-contained validation.
- Replaced old `analyze.py` and `combine_8subj.py` entry points with explicit wrappers for the revised workflow. The latter no longer depends on unpublished per-session CSVs. The old exploratory kNN/data-efficiency workflow is not part of this revised analysis.
- Recorded dependencies, runtime versions, input hashes, and generated output hashes.

The original synthetic demonstration and the per-trial MAE used for Figure 2 have different aggregation conventions; both are explained in the README and revised manuscript. Undefined clean-projection angles are not silently converted to valid measurements.

This local preparation does not update OSF or GitHub. The depositor is responsible for selecting the intended revision when uploading and for maintaining a clear version history. No private editorial or signed participant documents are included.
