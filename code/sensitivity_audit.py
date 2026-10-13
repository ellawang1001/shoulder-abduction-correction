"""Recompute sign sensitivity and independently audit Table 4; never alter inputs.

Run with python -B code/sensitivity_audit.py. No videos or MediaPipe required.
The CSV retains projected segment lengths and their included 2D angle, sufficient
to reconstruct the dot product. Session calibration is recomputed from six clips.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from angles import abduction_bonelength
from sim_proof import build_scene, render, estimate_all, METHODS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'


def sign_audit():
    df = pd.read_csv(ROOT / 'dataset_per_clip.csv')
    frontal = df[df.azimuth == 0]
    assert frontal.groupby('subject_id').size().eq(6).all()
    lengths = frontal.groupby('subject_id')[['feat_armlen', 'feat_trunklen']].median()
    rows = []
    for _, r in df.iterrows():
        arm, trunk = lengths.loc[r.subject_id]
        p, q = r.feat_armlen, r.feat_trunklen
        theta = np.radians(r['2d_trunk'])
        # Rigid in-plane rotation preserves all angles and projected lengths.
        sh = np.zeros(2)
        hip = np.array([q, 0.])
        elbow = np.array([p*np.cos(theta), p*np.sin(theta)])
        same = abduction_bonelength(sh, hip, elbow, arm, trunk, +1, +1)
        opposite = abduction_bonelength(sh, hip, elbow, arm, trunk, +1, -1)
        assert np.isclose(same, abduction_bonelength(sh, hip, elbow, arm, trunk, -1, -1), atol=1e-10)
        assert np.isclose(opposite, abduction_bonelength(sh, hip, elbow, arm, trunk, -1, +1), atol=1e-10)
        # Separate scalar dot-product calculation, including p > L clamping.
        za = np.sqrt(max(0., arm*arm-p*p))
        zt = np.sqrt(max(0., trunk*trunk-q*q))
        scalar = np.degrees(np.arccos(np.clip((p*q*np.cos(theta)-za*zt)/
                              (np.hypot(p, za)*np.hypot(q, zt)), -1, 1)))
        assert np.isclose(opposite, scalar, atol=1e-10)
        rows.append(dict(person=r.person, subject_id=r.subject_id, clip=r['clip'],
                         true_angle=r.true_angle, azimuth=r.azimuth,
                         same_sign=same, opposite_sign=opposite,
                         absolute_angle_change=abs(opposite-same)))
    out = pd.DataFrame(rows)
    residual = float(abs(out.same_sign-df.bonelength).max())
    assert residual < 1e-9, residual
    out.to_csv(OUT/'depth_sign_trials.csv', index=False)
    summaries = []
    for label, frame in [('azimuth_le_60', out[out.azimuth <= 60]),
                         ('azimuth_90', out[out.azimuth == 90]), ('all', out)]:
        for person, group in [('pooled', frame), *list(frame.groupby('person'))]:
            summaries.append(dict(range=label, person=person, clips=len(group),
                same_sign_mae=float(abs(group.same_sign-group.true_angle).mean()),
                opposite_sign_mae=float(abs(group.opposite_sign-group.true_angle).mean()),
                mean_angle_change=float(group.absolute_angle_change.mean()),
                max_angle_change=float(group.absolute_angle_change.max())))
    pd.DataFrame(summaries).to_csv(OUT/'depth_sign_summary.csv', index=False)
    return dict(max_original_angle_reconstruction_error=residual,
                pooled=[r for r in summaries if r['person']=='pooled'],
                interpretation='Fixed global sign alternatives; no target-based per-clip selection. '
                'Sensitivity analysis, not validation of physical depth signs or an optimized estimator.')


def calibration_audit():
    rng = np.random.default_rng(42)
    tests = []
    for phi in [0, 30, 45, 60, 90]:
        for theta in [0, 30, 60, 90, 120, 150]:
            for rep in range(200):
                lm2, lm3 = render(build_scene(theta), phi, 'persp', .004, .01, rng)
                tests.append((phi, theta, rep, lm2, lm3))
    saved = pd.read_csv(OUT/'calibration_sensitivity_trials.csv')
    saved_summary = pd.read_csv(OUT/'calibration_sensitivity_summary.csv').set_index('calibration_azimuth')
    details = []
    for delta in [-10, -5, 0, 5, 10]:
        lengths = []
        for theta in [0, 30, 60, 90, 120, 150]:
            lm, _ = render(build_scene(theta), delta, 'persp')
            lengths.append([np.linalg.norm(lm['elbow']-lm['shoulder']),
                            np.linalg.norm(lm['hip']-lm['shoulder'])])
        arm, trunk = np.median(lengths, axis=0)
        rows = []
        for phi, theta, rep, lm2, lm3 in tests:
            rows.append(dict(test_azimuth=phi, true_angle=theta, rep=rep,
                             **estimate_all(lm2, lm3, arm, trunk)))
        frame = pd.DataFrame(rows)
        original = saved[saved.calibration_azimuth == delta].reset_index(drop=True)
        assert np.array_equal(frame[['test_azimuth','true_angle','rep']], original[['test_azimuth','true_angle','rep']])
        max_diff = float(np.max(np.abs(frame[METHODS].to_numpy()-original[METHODS].to_numpy())))
        assert max_diff < 1e-10
        assert np.allclose(original.arm_calibration, arm, atol=1e-12, rtol=0)
        assert np.allclose(original.trunk_calibration, trunk, atol=1e-12, rtol=0)
        g = frame[frame.test_azimuth <= 60]
        mae = float(abs(g.bonelength-g.true_angle).mean())
        assert abs(mae-saved_summary.loc[delta, 'bonelength']) < 1e-12
        details.append(dict(calibration_azimuth=delta, arm_length=float(arm), trunk_length=float(trunk),
                            corrected_mae=mae, max_saved_trial_difference=max_diff,
                            mean_signed_error=float((g.bonelength-g.true_angle).mean())))
    # Isolate length changes: original estimator is monotone in the resulting
    # angle only locally, so inspect predictions rather than assert a global rule.
    base = saved[(saved.calibration_azimuth==0)&(saved.test_azimuth<=60)].reset_index(drop=True)
    for row in details:
        alt = saved[(saved.calibration_azimuth==row['calibration_azimuth'])&(saved.test_azimuth<=60)].reset_index(drop=True)
        row['mean_angle_shift_vs_zero'] = float((alt.bonelength-base.bonelength).mean())
        row['lower_absolute_error_trials_vs_zero'] = int((abs(alt.bonelength-alt.true_angle)<abs(base.bonelength-base.true_angle)-1e-12).sum())
    return dict(all_30000_trials_match=True, details=details,
                direction='For body X>0, positive phi gives rotated Z=-X*sin(phi); '
                'the pinhole denominator is 2-Z. The modeled right arm moves farther from the camera. '
                'Positive offsets reduce calibrated arm length; trunk calibration remains 0.25 image units.',
                estimator='Same angles.abduction_bonelength function and positive depth signs as real analysis; '
                'synthetic coordinates use normalized image units consistently, real coordinates use pixels.')


if __name__ == '__main__':
    result = {'depth_sign': sign_audit(), 'calibration': calibration_audit()}
    (OUT/'sensitivity_audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
