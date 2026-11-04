"""Bounded staging for the preregistered event observability experiment."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from himloco_lab.impact.observability import prepare_events


def collect(rep, role):
    spec=json.loads(Path('configs/impact/observability_v1.json').read_text())
    name='development' if role=='development' else f'rep{rep}/{role}'
    root=Path('datasets/go1_impact/observability_v1')/name
    raw,cache=root/'raw',root/'cache'
    evidence=Path('artifacts/impact/observability_v1')/name/'data'
    evidence.mkdir(parents=True,exist_ok=True)
    logs=Path('logs/impact_observability_v1')/name;logs.mkdir(parents=True,exist_ok=True)
    if (cache/'metadata.json').exists():return
    if not (raw/'report.json').exists():
        if raw.exists():raise RuntimeError(f'Incomplete raw collection: {raw}')
        if shutil.disk_usage('.').free<25*2**30:raise RuntimeError('Insufficient staging space')
        velocity=spec['high_velocity_range'] if role=='high_speed' else spec['velocity_range']
        seed=spec['development_seed'] if role=='development' else spec['data_seeds'][role][rep-1]
        command=[sys.executable,'scripts/impact/collect.py','--headless','--device','cuda:0',
                 '--num_envs',str(spec['num_envs']),'--steps',str(spec['steps'][role]),'--scenario','suite_walk',
                 '--seed',str(seed),'--physics-dt',str(spec['physics_dt']),'--observation-noise',
                 '--velocity-range-x',*map(str,velocity),'--episode-length-s',str(spec['episode_length_s']),
                 '--output',str(raw)]
        with (logs/'collection.log').open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    meta=prepare_events(raw,cache)
    for filename in ('report.json','env.yaml','plan.json'):
        shutil.copy2(raw/filename,evidence/filename)
    shutil.copy2(cache/'metadata.json',evidence/'metadata.json')
    shutil.rmtree(raw)
    print('EVENT_COLLECTION_COMPLETE',name,json.dumps(meta['domains']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rep',type=int,default=1)
    p.add_argument('--roles',nargs='+',default=['train','validation','test','high_speed'])
    a=p.parse_args()
    for role in a.roles:collect(a.rep,role)
