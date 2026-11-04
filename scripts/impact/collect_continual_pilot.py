"""Collect fresh compact continual-learning pilot data.

Raw shards are immediately converted to I0-only compact caches and removed.
This intentionally does not write research-observation/high-frequency arrays.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from himloco_lab.impact.force_study import prepare_force

ROOT = Path('datasets/go1_impact/continual_replay_v1')
LOG = Path('logs/impact_continual_replay_v1/collection')
ROOT.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
spec = json.loads(Path('configs/impact/continual_replay_v1_plan.json').read_text())
jobs = [('train', 901, 2500), ('monitor', 902, 600), ('calibration', 903, 600)]
for role, seed, steps in jobs:
    raw = ROOT/'raw'/role
    cache = ROOT/'cache'/role
    if not (cache/'metadata.json').exists():
        if not (raw/'report.json').exists():
            cmd=[sys.executable,'scripts/impact/collect.py','--headless','--device','cuda:0',
                 '--num_envs',str(spec['num_envs']),'--steps',str(steps),'--scenario','suite_walk',
                 '--seed',str(seed),'--physics-dt',str(spec['physics_dt']),'--observation-noise',
                 '--output',str(raw)]
            print('COLLECT',role,flush=True)
            with (LOG/f'{role}.log').open('w') as stream:
                subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('PREPARE',role,flush=True)
        prepare_force(raw,cache)
        # I0 training only needs causal pre/post frames and targets. Delete the
        # high-rate research-observation array from this pilot cache.
        hf=cache/'hf.npy'
        if hf.exists():hf.unlink()
        metadata=json.loads((cache/'metadata.json').read_text())
        metadata['pilot_compact']='I0 only; hf.npy intentionally removed after preparation'
        metadata['files_sha256'].pop('hf',None)
        (cache/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        provenance=Path('artifacts/impact/continual_replay_v1/data_provenance')/role
        provenance.mkdir(parents=True,exist_ok=True)
        # Keep the audit evidence, not the bulky force shards.
        for item in raw.iterdir():
            if item.suffix in ('.json','.yaml','.yml'):
                shutil.copy2(item,provenance/item.name)
        shutil.copy2(cache/'metadata.json',provenance/'cache_metadata.json')
        shutil.rmtree(raw)
    print('PILOT_ROLE_COMPLETE',role,flush=True)
print('CONTINUAL_PILOT_COLLECTION_COMPLETE',flush=True)
