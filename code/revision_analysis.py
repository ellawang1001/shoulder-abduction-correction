"""Reproduce revised tables, statistics and Figures 2–4 from packaged CSVs.
Run: python code/revision_analysis.py (working directory does not matter).
Original observations are never rewritten. Seed 42; 10,000 cluster bootstraps.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import icc21, bland_altman, paired_mae_reduction, bootstrap_ci
from sim_proof import build_scene, render, estimate_all, METHODS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
FIG = ROOT / 'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / 'dataset_per_clip.csv')
syn = pd.read_csv(ROOT / 'synthetic_dataset.csv')
assert len(df) == 270 and df.person.nunique() == 8
assert not df[METHODS + ['true_angle']].isna().any().any()
labels = ['Nose baseline', 'Raw 2D', 'Bone-length correction', '3D reference']
colors = ['#777777', '#D55E00', '#0072B2', '#009E73']
for m in METHODS:
    df['err_'+m] = abs(df[m]-df.true_angle)
    syn['err_'+m] = abs(syn[m]-syn.true_angle)
main = df[df.azimuth <= 60]
people = sorted(df.person.unique())

def stats(frame, m):
    groups = [g for _, g in frame.groupby('person')]
    sums = np.array([g['err_'+m].sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    idx = np.random.default_rng(42).integers(0, len(groups), (10000, len(groups)))
    boots = sums[idx].sum(axis=1)/sizes[idx].sum(axis=1)
    # Same participant bootstrap for ICC; all clips from a selected person retained.
    matrices = [g[['true_angle',m]].values for g in groups]
    iccs = [icc21(np.concatenate([matrices[i] for i in row])) for row in idx]
    return dict(n=len(frame), mae=float(frame['err_'+m].mean()),
                ci=np.percentile(boots,[2.5,97.5]).tolist(),
                rmse=float(np.sqrt(np.mean((frame[m]-frame.true_angle)**2))),
                icc=icc21(frame[['true_angle',m]].values),
                icc_ci=np.percentile(iccs,[2.5,97.5]).tolist())

summary = {'main': {m: stats(main,m) for m in METHODS},
           'heldout': {m: stats(df[df.azimuth.isin([30,45,60])],m) for m in METHODS[1:]},
           'all': {m: stats(df,m) for m in METHODS[1:]},
           'bland_altman': bland_altman(main.bonelength,main.true_angle)}
person = main.groupby('person')[['err_'+m for m in METHODS]].mean()
person.columns = METHODS
person['reduction'] = person['2d_trunk']-person.bonelength
person.to_csv(OUT/'participant_mae.csv')
summary['person_reduction'] = dict(mean=float(person.reduction.mean()), ci=bootstrap_ci(person.reduction), sign_p=2/2**8)
for name, frame in [('main',main),('heldout',df[df.azimuth.isin([30,45,60])]),('all',df)]:
    summary[name+'_paired_clip'] = paired_mae_reduction(frame['2d_trunk'],frame.bonelength,frame.true_angle)
zero = df[df.true_angle == 0].groupby('person')[METHODS].mean()
zero.to_csv(OUT/'zero_target_mean_estimates.csv')
frontzero = df[(df.true_angle == 0)&(df.azimuth == 0)].groupby('person')[METHODS].mean()
frontzero.to_csv(OUT/'frontal_zero_target_estimates.csv')
offsets = df[df.azimuth == 0].assign(signed=lambda x:x.bonelength-x.true_angle).groupby('person').signed.mean()
after = main.assign(adjusted=lambda x:abs(x.bonelength-x.person.map(offsets)-x.true_angle)).groupby('person').adjusted.mean()
summary['offset_analysis'] = {'frontal_offsets':offsets.to_dict(),'equal_person_mae_before':float(person.bonelength.mean()),'equal_person_mae_after':float(after.mean()), 'after_per_person':after.to_dict()}
summary['azimuth_pooled'] = df.groupby('azimuth')[['err_'+m for m in METHODS]].mean().to_dict()
composition = df.groupby(['person','subject_id','round','studied_side']).size().reset_index(name='clips')
composition.to_csv(OUT/'data_composition.csv',index=False)
repeat = df[df.person == 'A'].pivot(index=['true_angle','azimuth'],columns='round',values=METHODS)
summary['repeatability'] = {m: {'mean_abs_difference':float(abs(repeat[m][1]-repeat[m][2]).mean()),'icc':icc21(repeat[m].values),'n':len(repeat)} for m in METHODS[1:]}

# Calibration sensitivity: same six target postures as the real protocol,
# median noiseless projected segment lengths at the calibration azimuth.
# All test observations and random errors are held fixed across offsets.
trials=[]
rng=np.random.default_rng(42)
for phi in [0,30,45,60,90]:
    for theta in [0,30,60,90,120,150]:
        for rep in range(200):
            lm2,lm3=render(build_scene(theta),phi,'persp',.004,.01,rng)
            trials.append((phi,theta,rep,lm2,lm3))
sens=[]
for delta in [-10,-5,0,5,10]:
    lengths=[]
    for theta in [0,30,60,90,120,150]:
        lm,_=render(build_scene(theta),delta,'persp')
        lengths.append([np.linalg.norm(lm['elbow']-lm['shoulder']),np.linalg.norm(lm['hip']-lm['shoulder'])])
    arm,trunk=np.median(lengths,axis=0)
    for phi,theta,rep,lm2,lm3 in trials:
        estimates=estimate_all(lm2,lm3,arm,trunk)
        sens.append(dict(calibration_azimuth=delta,test_azimuth=phi,true_angle=theta,rep=rep,arm_calibration=arm,trunk_calibration=trunk,**estimates))
sens=pd.DataFrame(sens)
sens.to_csv(OUT/'calibration_sensitivity_trials.csv',index=False)
for m in METHODS: sens['err_'+m]=abs(sens[m]-sens.true_angle)
stable=sens[sens.test_azimuth<=60].groupby('calibration_azimuth')[['err_'+m for m in METHODS]].mean()
stable.columns=METHODS
stable['change_from_zero']=stable.bonelength-stable.loc[0,'bonelength']
stable.to_csv(OUT/'calibration_sensitivity_summary.csv')
summary['calibration_sensitivity']=stable.to_dict(orient='index')
(OUT/'statistics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

g=df.groupby(['person','azimuth'])[['err_'+m for m in METHODS]].mean()
g.to_csv(OUT/'figure3_participant_points.csv')

# Render existing results at the manuscript width; numerical analysis above is unchanged.
import runpy
runpy.run_path(str(ROOT/'code/render_figures.py'), run_name='__main__')
