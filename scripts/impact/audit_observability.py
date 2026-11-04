"""Recompute event labels independently on retained native audit trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np

from himloco_lab.impact.dataset import file_hash


def audit(path):
    path=Path(path);meta=json.loads((path/'metadata.json').read_text())
    data=dict(np.load(path/'native_audit.npz'));event=dict(np.load(path/'events.npz'))
    ids=data['env_id'];steps=meta['steps'];checked=0;max_error=0.
    actual_min=np.full(6,np.inf);actual_max=np.full(6,-np.inf)
    for column,env in enumerate(ids):
        force=np.linalg.norm(data['normal_forces_w'][:,column,-8:].astype(np.float64),axis=-1).reshape(steps*8,4)
        rows=np.flatnonzero(event['env']==env)
        for row in rows:
            onset=int(event['native_onset'][row]);foot=int(event['foot'][row]);base=int(event['base'][row])
            assert np.all(force[onset-16:onset,foot]<=5.)
            assert np.all(force[onset:onset+2,foot]>5.)
            values=[]
            for k in range(onset,onset+16):
                current=force[k-7:k+1,foot].mean()
                old=force[k-15:k-7,foot].mean()
                values.append(max((current-old)/.02,0))
            error=abs(max(values)-float(event['severity'][row]));max_error=max(max_error,error)
            if error>.01:raise AssertionError(f'Native target error {error}')
            assert base*.02<(onset+1)*.0025
            for h,offset in enumerate((-2,-1,0,1,2,3)):
                cutoff=base+offset
                if not event['valid_cutoff'][row,h]:continue
                assert data['age'][cutoff,column]==data['age'][base,column]+offset
                assert not np.any(data['done'][cutoff-5:cutoff,column])
                relative=cutoff*.02-(onset+1)*.0025
                actual_min[h]=min(actual_min[h],relative*1000);actual_max[h]=max(actual_max[h],relative*1000)
            end=(onset+15)//8
            assert data['age'][end,column]==data['age'][base,column]+end-base
            checked+=1
    report={'status':'passed','audited_envs':ids.tolist(),'events_recomputed':checked,
            'max_abs_label_error_n_per_s':max_error,'tolerance_n_per_s':.01,
            'cutoff_ms_min':actual_min.tolist(),'cutoff_ms_max':actual_max.tolist(),
            'metadata_sha256':file_hash(path/'metadata.json'),
            'native_audit_sha256':file_hash(path/'native_audit.npz'),'events_sha256':file_hash(path/'events.npz')}
    destination=Path('artifacts/impact/observability_v1')/path.parent.relative_to('datasets/go1_impact/observability_v1')/'data'/'native_audit.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print('OBS_AUDIT',str(path),checked,max_error,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+');a=p.parse_args()
    for path in a.paths:audit(path)
