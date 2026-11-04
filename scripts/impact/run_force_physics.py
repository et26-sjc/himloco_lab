"""Paired 128-robot physics grid, separate from any model/test selection."""
import json
from pathlib import Path
import subprocess
import sys

root = Path('datasets/go1_impact/force_study_v1/physics')
logs = Path('logs/impact_force_study_v1/physics')
logs.mkdir(parents=True, exist_ok=True)
runs = [(dt, 4, 0) for dt in (.005, .0025, .00125, .000625)]
runs += [(.0025, 8, 4), (.00125, 8, 4)]
for dt, pos, vel in runs:
    name = f'dt{round(dt*1e6)}_p{pos}_v{vel}'+('_global' if pos==8 else '')
    out = root / name
    if (out / 'report.json').exists():
        continue
    command = [sys.executable, 'scripts/impact/collect.py', '--headless', '--device', 'cuda:0',
               '--num_envs', '128', '--steps', '64', '--scenario', 'drop', '--seed', '47',
               '--drop-height-range', '.4', '.9', '--physics-dt', str(dt),
               '--solver-position-iterations', str(pos), '--solver-velocity-iterations', str(vel),
               '--output', str(out)]
    print('RUN_PHYSICS', name, flush=True)
    with (logs / f'{name}.log').open('x') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
print('PHYSICS_GRID_COMPLETE', flush=True)
