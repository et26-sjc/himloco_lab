"""Verify compact pilot cache integrity without invoking Isaac Sim."""
import json
from pathlib import Path
import re

import numpy as np

from himloco_lab.impact.dataset import file_hash


def main():
    root=Path('datasets/go1_impact/continual_replay_v1/cache')
    out=Path('artifacts/impact/continual_replay_v1');out.mkdir(parents=True,exist_ok=True)
    records={}
    for role in ('train','monitor','calibration'):
        path=root/role;meta=json.loads((path/'metadata.json').read_text())
        files={k:path/f'{k}.npy' for k in meta['files_sha256']}
        for k,p in files.items():
            assert file_hash(p)==meta['files_sha256'][k],(role,k)
        age=np.load(files['age'],mmap_mode='r');done=np.load(files['done'],mmap_mode='r')
        term=np.load(files['terminated'],mmap_mode='r');y=np.load(files['targets'],mmap_mode='r')
        assert np.isfinite(y).all()
        assert np.all(age[1:][done[:-1]]==0)
        valid=age>=25
        for lag in range(1,6):
            assert not np.any(valid[lag:] & done[:-lag])
        family=np.asarray(meta['terrain']['family_id']);domains={}
        for name,fams in [('A',[0,1]),('B',[2,3]),('C',[4,5]),('U',[6])]:
            env=np.isin(family,fams);v=valid[:,env];s=np.asarray(y[:,env,7])
            domains[name]={'valid_robot_intervals':int(v.sum()),'valid_robot_minutes':float(v.sum()*.02/60),
                           'terminations':int(term[:,env].sum()),'resets_including_timeouts':int(done[:,env].sum()),
                           'positive_foot_intervals':int(((s>2000.)&v[...,None]).sum()),
                           'stratum_robot_intervals':[int(((s.max(2)<=2000.)&v).sum()),int((((s.max(2)>2000.)&(s.max(2)<=4000.))&v).sum()),int(((s.max(2)>4000.)&v).sum())]}
        lines=Path(f'logs/impact_continual_replay_v1/collection/{role}.log').read_text()
        matches=re.findall(r'IMPACT_COLLECTION_OK (\{[^\n]+\})',lines)
        assert len(matches)==1
        summary=json.loads(matches[0]);assert summary['reference_contact_max_error']<=1e-5
        assert summary['history_overlap_max_error']==0 and summary['history_reconstruction_max_error']==0
        records[role]={'seed':meta['seed'],'steps':meta['steps'],'num_envs':meta['num_envs'],
                       'files_sha256':meta['files_sha256'],'bytes':sum(p.stat().st_size for p in files.values()),
                       'domains':domains,'collector_checks':summary,'max_target_by_kind':y.max((0,1,3)).tolist()}
        dest=out/'data_provenance'/role;dest.mkdir(parents=True,exist_ok=True)
        (dest/'cache_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    report={'status':'passed','roles':records,'new_cache_bytes':sum(r['bytes'] for r in records.values()),
            'limitations':['Pilot used validated existing collector followed by I0 cache conversion and raw deletion; not the proposed new streaming writer.',
                          'Original pilot raw report/trace were removed before provenance copy was added. Cache hashes and collector log checks retained; no claim of a newly retained native-trace audit.']}
    (out/'data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'new_cache_GiB':report['new_cache_bytes']/2**30}))


if __name__=='__main__':main()
