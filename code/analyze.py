"""Compatibility entry point for the current revised statistics and Figures 1-4."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        raise SystemExit('The legacy CLI is retired. Run: python -B code/reproduce.py')
    runpy.run_path(str(Path(__file__).with_name('reproduce.py')), run_name='__main__')
