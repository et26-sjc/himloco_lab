"""Collect five independently seeded 4096-env replications; final test not inspected."""
import json
from pathlib import Path
import subprocess
import sys

from himloco_lab.impact.force_study import prepare_force

spec = json.loads(Path('configs/impact/force_study_v1.json').read_text())
root = Path('datasets/go1_impact/force_study_v1')
logs = Path('logs/impact_force_study_v1/collection')
logs.mkdir(parents=True,exist_ok=True)
for rep in range(5):
    for split in ('train','validation','test'):
        name=f'rep{rep+1}/{split}'
        raw,cache = root/'raw'/name,root/'cache'/name
        if not (raw/'report.json').exists():
            command=[sys.executable,'scripts/impact/collect.py','--headless','--device','cuda:0',
                     '--num_envs',str(spec['num_envs']),'--steps',str(spec['steps'][split]),
                     '--scenario','suite_walk','--seed',str(spec[split+'_seeds'][rep]),
                     '--physics-dt',str(spec['physics_dt']),'--observation-noise',
                     '--research-observations','--output',str(raw)]
            print('COLLECT',name,flush=True)
            with (logs/f'rep{rep+1}_{split}.log').open('x') as stream:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if split!='test' and not (cache/'metadata.json').exists():
            print('PREPARE',name,flush=True)
            prepare_force(raw,cache)
    print('REPLICATION_COLLECTION_COMPLETE',rep+1,flush=True)
print('FORCE_COLLECTION_COMPLETE',flush=True)
