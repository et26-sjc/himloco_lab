"""Bounded collection for causal continuous landing monitoring."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from himloco_lab.impact.causal_stream import prepare_stream


def collect(rep, role):
    spec=json.loads(Path('configs/impact/causal_stream_v1.json').read_text())
    name='development' if role=='development' else f'rep{rep}/{role}'
    root=Path('datasets/go1_impact/causal_stream_v1')/name
    raw,cache=root/'raw',root/'cache'
    logs=Path('logs/impact_causal_stream_v1')/name;logs.mkdir(parents=True,exist_ok=True)
    if (cache/'stream_metadata.json').exists():return
    if not (raw/'report.json').exists():
        if raw.exists():raise RuntimeError(f'Incomplete raw collection: {raw}')
        if shutil.disk_usage('.').free<30*2**30:raise RuntimeError('Insufficient staging space')
        velocity=spec['high_velocity_range'] if role=='high_speed' else spec['velocity_range']
        seed=spec['development_seed'] if role=='development' else spec['data_seeds'][role][rep-1]
        command=[sys.executable,'scripts/impact/collect.py','--headless','--device','cuda:0',
                 '--num_envs',str(spec['num_envs']),'--steps',str(spec['steps'][role]),'--scenario','suite_walk',
                 '--seed',str(seed),'--physics-dt',str(spec['physics_dt']),'--observation-noise',
                 '--velocity-range-x',*map(str,velocity),'--episode-length-s',str(spec['episode_length_s']),
                 '--output',str(raw)]
        with (logs/'collection.log').open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    meta=prepare_stream(raw,cache)
    shutil.rmtree(raw)
    evidence=Path('artifacts/impact/causal_stream_v1')/name
    evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'collection.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('STREAM_COLLECTED',name,meta['events'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rep',type=int,default=1)
    p.add_argument('--roles',nargs='+',default=['development']);a=p.parse_args()
    for role in a.roles:collect(a.rep,role)
