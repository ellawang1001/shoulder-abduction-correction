"""Validate the public deposit without reading any files outside it."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
checks={}
def check(name,passed):
    checks[name]=bool(passed)
    if not passed:print('FAILED:',name)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
expected={
    'dataset_per_clip.csv':'1db2ddb663a3eefcceff31ae3b4ba84761e0530580ea9241f16f7a5747dfdea2',
    'synthetic_dataset.csv':'6358cc97c5bfcce30138cb5864a5d767039132de069347f06c1d3f47b42f24a9'}
for name,digest in expected.items():check(name+' unchanged',sha(ROOT/name)==digest)
d=pd.read_csv(ROOT/'dataset_per_clip.csv')
check('8 participants, 9 sessions, 270 clips',d.person.nunique()==8 and d.subject_id.nunique()==9 and len(d)==270)
main=d[d.azimuth<=60]
check('216 clips at azimuth <= 60',len(main)==216)
s=json.loads((OUT/'statistics.json').read_text(encoding='utf-8'))
for method,expected_mae in [('2d_trunk',14.551029065290594),('bonelength',11.155076326391749),('3d_world',10.732944873004957)]:
    value=float(np.mean(abs(main[method]-main.true_angle)))
    check(method+' MAE matches reference and released observations',np.isclose(value,expected_mae,rtol=0,atol=1e-10) and np.isclose(value,s['main'][method]['mae'],rtol=0,atol=1e-10))
check('Primary sign test',s['person_reduction']['sign_p']==0.0078125)
points=pd.read_csv(OUT/'figure3_participant_points.csv')
check('Eight participants at each plotted azimuth',len(points)==40 and points.groupby('azimuth').person.nunique().eq(8).all())
trials=pd.read_csv(OUT/'calibration_sensitivity_trials.csv')
check('Complete sensitivity grid',len(trials)==30000 and len(trials[trials.test_azimuth<=60])==24000)
for method in ['baseline_nose','2d_trunk','3d_world']:
    check('Identical test landmarks across calibration offsets: '+method,trials.groupby(['test_azimuth','true_angle','rep'])[method].nunique().eq(1).all())
summary=pd.read_csv(OUT/'calibration_sensitivity_summary.csv',index_col=0)
for delta,target in [(-10,5.035963701903152),(-5,4.920497789958384),(0,4.698185599455196),(5,4.426388400325988),(10,4.226894462841639)]:
    check('Sensitivity result '+str(delta),np.isclose(summary.loc[delta,'bonelength'],target,rtol=0,atol=1e-10))
for n in range(1,5):
    png=ROOT/f'figures/fig{n}.png';svg=ROOT/f'figures/fig{n}.svg'
    check('Figure '+str(n)+' PNG and SVG available',png.exists() and svg.exists() and png.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n')
report={'all_passed':all(checks.values()),'checks':checks,
        'scope':'CSV-based statistics and figure reproduction; no raw-video re-extraction or online upload.'}
(OUT/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
manifest=OUT/'file_manifest.csv'
with manifest.open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['relative_path','bytes','sha256'])
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p!=manifest and not any(part in {'.git','__pycache__','.venv','venv'} for part in p.relative_to(ROOT).parts):
            w.writerow([p.relative_to(ROOT).as_posix(),p.stat().st_size,sha(p)])
print('Deposit validation:', 'PASS' if report['all_passed'] else 'FAIL',f'({len(checks)} checks)')
if not report['all_passed']:raise SystemExit(1)
