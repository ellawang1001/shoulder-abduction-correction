# Data dictionary

## dataset_per_clip.csv

One row represents one video clip. There are 270 clips, nine recording sessions, and eight adults. No rows were added, removed, or recalculated for the revision.

| Field | Meaning |
|---|---|
| subject_id | Recording-session ID, not unique participant ID. IDs 1 and 3 both belong to participant A. |
| person | Participant A–H. Mapping: 1→A, 2→B, 3→A, 4→C, 5→D, 6→E, 7→F, 8→G, 9→H. |
| round | Within-participant session: A has rounds 1 and 2; others have round 1. |
| true_angle | Goniometer-set target angle checked before filming (0, 30, 60, 90, 120, 150 degrees). These are discrete targets, not continuously recorded goniometer readings. |
| azimuth | Nominal horizontal camera azimuth relative to the participant: 0, 30, 45, 60, 90 degrees. The phone stayed fixed while the participant turned. |
| rep | Original within-condition repetition label. |
| clip | Original video filename, retained to trace the observation. Video files are not distributed in this package. |
| baseline_nose | 2D nose–shoulder–elbow angle in degrees. Authors' app-derived reference-choice comparator. |
| 2d_trunk | 2D hip–shoulder–elbow angle in degrees, computed from pixel coordinates. |
| bonelength | Angle in degrees after calibrated-length depth reconstruction, with positive depth signs. |
| 3d_world | Hip–shoulder–elbow angle in degrees from MediaPipe world landmarks; a same-system comparison, not external ground truth. |
| feat_armlen | Projected shoulder-to-elbow length in pixels from clip-median landmark coordinates. |
| feat_trunklen | Projected shoulder-to-hip length in pixels from clip-median landmark coordinates. |
| feat_ratio | feat_armlen / feat_trunklen; this is not the arm's p/L calibration ratio. |
| studied_side | Automatically selected arm, L or R. |
| vis_shoulder, vis_elbow | Mean visibility scores across pose-detected frames; descriptive, not exclusion criteria. |
| n_det | Number of frames with a detected pose contributing to clip-level coordinate medians. No visibility-threshold filter was applied. |

Calibration lengths were calculated separately within each session from the six frontal clips. Participant A's measured physical upper-arm length was 38 cm; it is not a calibration constant used in the angle estimator.

## synthetic_dataset.csv

2,100 simulated observations: 15 target angles (20–160 degrees), seven azimuths (0–90 degrees), and 20 repeats. Angle columns have the same estimator definitions; `3d_world` here refers to noisy simulated 3D points rather than MediaPipe output. Projected feature lengths use the simulation's image units, not real-video pixels. No participant, session, or visibility fields are present.

## New sensitivity output

`results/calibration_sensitivity_trials.csv` has 30,000 rows: five calibration azimuths × five test azimuths × six target postures × 200 repeats. All five calibrations share the same 6,000 test observations. `arm_calibration` and `trunk_calibration` are lengths in simulated image units. `calibration_sensitivity_summary.csv` uses only test azimuths ≤ 60 degrees (4,800 observations per calibration). The number of rows is not a count of independent biological samples.
