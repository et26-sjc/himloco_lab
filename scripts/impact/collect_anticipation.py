"""One-role-at-a-time frozen-policy collection, including exact original HIM targets."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from himloco_lab.impact.anticipation import prepare_anticipation
from himloco_lab.impact.dataset import file_hash


ROOT=Path('datasets/go1_impact/shared_anticipation_v1')
ART=Path('artifacts/impact/shared_anticipation_v1')
LOG=Path('logs/impact_shared_anticipation_v1')
SPEC=Path('configs/impact/shared_anticipation_v1.json')


def collect(rep,role,development=False):
    spec=json.loads(SPEC.read_text())
    name=f'development/{role}' if development else f'rep{rep}/{role}'
    root=ROOT/name;raw,cache=root/'raw',root/'cache'
    logs=LOG/name;logs.mkdir(parents=True,exist_ok=True)
    prepared=(cache/'anticipation_metadata.json').exists()
    if prepared and (ART/name/'audit.json').exists() and not raw.exists():return
    if not prepared and not (raw/'report.json').exists():
        if raw.exists():raise RuntimeError(f'Incomplete raw collection: {raw}')
        if shutil.disk_usage('.').free<35*2**30:raise RuntimeError('Insufficient role staging space')
        velocity=spec['high_velocity_range'] if role=='high_speed' else spec['velocity_range']
        seed=(spec['development_seeds'][role] if development else spec['data_seeds'][role][rep-1])
        steps=spec['steps'][f'development_{role}' if development else role]
        command=[sys.executable,'scripts/impact/collect.py','--headless','--device','cuda:0',
                 '--num_envs',str(spec['num_envs']),'--steps',str(steps),'--scenario','suite_walk',
                 '--seed',str(seed),'--physics-dt',str(spec['physics_dt']),'--observation-noise',
                 '--velocity-range-x',*map(str,velocity),'--episode-length-s',str(spec['episode_length_s']),
                 '--him-supervision','--output',str(raw)]
        with (logs/'collection.log').open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    meta=(json.loads((cache/'anticipation_metadata.json').read_text()) if prepared
          else prepare_anticipation(raw,cache))
    evidence=ART/name;evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'collection.json').write_text(json.dumps(meta,indent=2)+'\n')
    # Audit before deleting raw shards, including terminal next48 slices.
    audit=validate_cache(cache,raw if raw.exists() else None)
    (evidence/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if raw.exists():shutil.rmtree(raw)
    print('ANTICIPATION_COLLECTED',name,meta['events'],flush=True)


def validate_cache(cache,raw=None):
    import numpy as np
    cache=Path(cache);meta=json.loads((cache/'metadata.json').read_text())
    data=dict(np.load(cache/'future.npz'));events=dict(np.load(cache/'events.npz'))
    native=dict(np.load(cache/'native_audit.npz'));fixture=dict(np.load(cache/'fixture.npz'))
    checked=0;max_error=0.
    for col,env in enumerate(native['env_id']):
        force=np.linalg.norm(native['normal_forces_w'][:,col,-8:].astype('float64'),axis=-1).reshape(-1,4)
        rows=np.flatnonzero(events['env']==env)
        for row in rows:
            k=int(events['native_onset'][row]);f=int(events['foot'][row])
            assert (force[k-16:k,f]<=5).all() and (force[k:k+2,f]>5).all()
            rates=[max((force[j-7:j+1,f].mean()-force[j-15:j-7,f].mean())/.02,0) for j in range(k,k+16)]
            max_error=max(max_error,abs(max(rates)-events['severity'][row]));checked+=1
        # Independent exhaustive first-future-event search on all audit inputs.
        for f in range(4):
            erows=rows[events['foot'][rows]==f]
            erows=erows[np.argsort(events['native_onset'][erows])]
            for t in range(meta['steps']):
                candidates=erows[(events['native_onset'][erows]+1>t*8)&
                                 (events['native_onset'][erows]+1<=t*8+32)]
                candidates=[r for r in candidates if data['age'][events['base'][r],env]==
                            data['age'][t,env]+events['base'][r]-t]
                expected=candidates[0] if candidates else -1
                assert data['event_id'][t,env,f]==expected
    assert max_error<.01
    terminal_rows=0
    if raw is not None:
        raw=Path(raw);report=json.loads((raw/'report.json').read_text());cursor=0
        next48=np.load(cache/'next48.npy',mmap_mode='r')
        for item in report['shards']:
            with np.load(raw/item['file']) as d:
                count=len(d['policy_step'])
                np.testing.assert_array_equal(next48[cursor:cursor+count],d['him_next_critic48'])
                terminal_rows+=int(d['done'].sum());cursor+=count
    return dict(status='passed',native_envs=len(native['env_id']),events_recomputed=checked,
                max_severity_error_n_per_s=float(max_error),terminal_rows_preserved=terminal_rows,
                first_future_event_audit='exhaustive over audit trajectories, all feet and boundaries',
                fixture_sha256=file_hash(cache/'fixture.npz'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rep',type=int,default=1)
    p.add_argument('--roles',nargs='+',default=['train','validation'])
    p.add_argument('--development',action='store_true');a=p.parse_args()
    for role in a.roles:collect(a.rep,role,a.development)
