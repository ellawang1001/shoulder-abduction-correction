"""Compatibility entry point: use the released merged CSV, not private session files."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        raise SystemExit('Usage: python -B code/combine_8subj.py (no arguments)')
    print('Using dataset_per_clip.csv through the revised reproduction workflow.')
    runpy.run_path(str(Path(__file__).with_name('reproduce.py')), run_name='__main__')
