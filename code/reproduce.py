"""Single entry point for the public CSV-based revision analysis."""
from pathlib import Path
import contextlib
import importlib.metadata
import json
import platform
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
for name in ['revision_analysis.py', 'make_fig1.py']:
    subprocess.run([sys.executable, '-B', str(ROOT/'code'/name)], check=True)

# Preserve the original demonstration's condition-mean-error convention.
import sim_proof as S
with (OUT/'synthetic_demonstration.txt').open('w', encoding='utf-8') as log:
    with contextlib.redirect_stdout(log):
        clean = S.run_block('Orthographic, no noise', 'ortho')
        perspective = S.run_block('Perspective, no noise', 'persp')
        noisy = S.run_block('Perspective with noise; average estimates before absolute error',
                            'persp', noise2d=.004, noise3d=.01, reps=200)
# A degenerate projected vector gives undefined raw/baseline angles in clean runs.
# The text log retains nan; JSON uses null, not nonstandard NaN.
import math
demo = {label:{method:value if math.isfinite(value) else None for method,value in block.items()}
        for label,block in [('orthographic',clean),('perspective',perspective),('noisy_condition_mean_error',noisy)]}
(OUT/'synthetic_demonstration.json').write_text(json.dumps(demo, indent=2, allow_nan=False), encoding='utf-8')
environment = {'python':platform.python_version(), 'packages':{
    name:importlib.metadata.version(name) for name in ['numpy','pandas','matplotlib']}}
(OUT/'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
subprocess.run([sys.executable, '-B', str(ROOT/'code/validate_deposit.py')], check=True)
print('CSV-based reproduction and validation completed. No raw videos or external files were used.')
