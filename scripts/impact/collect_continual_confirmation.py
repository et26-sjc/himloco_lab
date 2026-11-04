"""Fresh confirmation data, with bounded raw staging and retained native audits.

Reuses the already validated collector/label converter. This is deliberately
not a new simulator-side streaming writer: one role is staged at a time.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.force_study import force_targets,prepare_force


def collect(rep,role):
    spec=json.loads(Path('configs/impact/continual_replay_v1_plan.json').read_text())
    seed=spec['data_roles'][f'confirmatory_{role}_seeds'][rep-1]
    steps=spec['data_roles']['steps'][role]
    root=Path('datasets/go1_impact/continual_replay_v1/confirmation')/f'rep{rep}'
    raw=root/'raw'/role;cache=root/'cache'/role
    evidence=Path('artifacts/impact/continual_replay_v1/confirmation')/f'rep{rep}'/'data'/role
    logs=Path('logs/impact_continual_replay_v1/confirmation')/f'rep{rep}'
    evidence.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    if (evidence/'audit.json').exists() and (cache/'metadata.json').exists():return
    if not (raw/'report.json').exists():
        if raw.exists():raise RuntimeError(f'Incomplete raw collection needs inspection: {raw}')
        if shutil.disk_usage(Path('.')).free<40*2**30:raise RuntimeError('Less than 40 GiB free before role staging')
        cmd=[sys.executable,'scripts/impact/collect.py','--headless','--device','cuda:0',
             '--num_envs','4096','--steps',str(steps),'--scenario','suite_walk','--seed',str(seed),
             '--physics-dt','0.0025','--velocity','0.7','0','0','--observation-noise','--output',str(raw)]
        with (logs/f'collect_{role}.log').open('w') as stream:
            subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
    report=json.loads((raw/'report.json').read_text())
    if not (cache/'metadata.json').exists():prepare_force(raw,cache)
    meta=json.loads((cache/'metadata.json').read_text())
    assert report['reference_contact_max_error']<=1e-5 and report['history_overlap_max_error']==0
    assert report['reset_count']==report['terminal_traces_captured']
    audit_env=np.arange(0,4096,200)  # Predeclared 0.513%; independent of label/model.
    native=[];ages=[]
    for shard in report['shards']:
        with np.load(raw/shard['file']) as data:
            native.append(data['normal_forces_w'][:,audit_env,-8:].copy())
            ages.append(data['episode_step'][:,audit_env].copy())
    native=np.concatenate(native);ages=np.concatenate(ages)
    previous=np.zeros((len(audit_env),16,4),np.float32)
    targets=np.load(cache/'targets.npy',mmap_mode='r');maximum=0.
    for t in range(steps):
        previous[ages[t]==0]=0
        current=np.linalg.norm(native[t].astype(np.float64),axis=-1)
        reconstructed=force_targets(previous,current,.0025)
        maximum=max(maximum,float(abs(reconstructed-targets[t,audit_env]).max()))
        previous=np.concatenate((previous,current),axis=1)[:,-16:]
    assert maximum==0.,maximum
    # Audit traces stay locally, small reports and hashes are versioned.
    native_path=cache/'native_audit.npz'
    np.savez_compressed(native_path,normal_forces_w=native,age=ages,env_id=audit_env)
    for item in raw.iterdir():
        if item.suffix in ('.json','.yaml','.yml'):shutil.copy2(item,evidence/item.name)
    shutil.copy2(cache/'metadata.json',evidence/'cache_metadata.json')
    age=np.load(cache/'age.npy',mmap_mode='r');done=np.load(cache/'done.npy',mmap_mode='r')
    term=np.load(cache/'terminated.npy',mmap_mode='r');valid=age>=25
    assert np.all(age[1:][done[:-1]]==0)
    for lag in range(1,6):assert not np.any(valid[lag:]&done[:-lag])
    domains={};family=np.asarray(meta['terrain']['family_id'])
    for name,fams in [('A',[0,1]),('B',[2,3]),('C',[4,5]),('U',[6])]:
        env=np.isin(family,fams);mask=valid[:,env];s=np.asarray(targets[:,env,7])
        domains[name]={'robot_intervals':int(mask.sum()),'foot_prevalence':float((s[mask]>2000.).mean()),
                       'terminations':int(term[:,env].sum()),'resets':int(done[:,env].sum())}
    audit={'status':'passed','seed':seed,'num_envs':4096,'steps':steps,'domains':domains,
           'native_audit_envs':audit_env.tolist(),'native_target_max_error':maximum,
           'native_audit_sha256':file_hash(native_path),
           'cache_bytes':sum(p.stat().st_size for p in cache.iterdir() if p.is_file()),
           'raw_report_sha256':file_hash(raw/'report.json'),
           'format_note':'Validated collector + converter; raw staged one role at a time, then removed.'}
    (evidence/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    shutil.rmtree(raw)
    print('CONFIRMATION_ROLE_COMPLETE',rep,role,seed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reps',nargs='+',type=int,default=[1,2,3,4,5])
    parser.add_argument('--roles',nargs='+',default=['train','monitor','calibration','test'])
    args=parser.parse_args()
    for rep in args.reps:
        for role in args.roles:collect(rep,role)
