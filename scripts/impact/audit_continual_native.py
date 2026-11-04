"""Recompute the retained 0.513% native audit even after bulk-data cleanup."""
import json
from pathlib import Path

import numpy as np

from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.force_study import force_targets


def main():
    root=Path('datasets/go1_impact/continual_replay_v1/confirmation')
    rows=[]
    for path in sorted(root.glob('rep*/cache/*/native_audit.npz')):
        target_path=path.with_name('native_audit_targets.npz')
        with np.load(path) as source,np.load(target_path) as recorded:
            forces=source['normal_forces_w'];age=source['age'];target=recorded['targets']
        proprio_path=path.with_name('native_audit_proprio.npz')
        with np.load(proprio_path) as observations:
            pre,post,done=observations['pre'],observations['post'],observations['done']
            assert np.isfinite(pre).all() and np.isfinite(post).all()
            assert np.array_equal(post[:-1][~done[:-1]],pre[1:][~done[:-1]])
            assert np.all(age[1:][done[:-1]]==0)
        previous=np.zeros((forces.shape[1],16,4),np.float32);error=0.
        for step in range(len(forces)):
            previous[age[step]==0]=0
            current=np.linalg.norm(forces[step].astype(np.float64),axis=-1)
            y=force_targets(previous,current,.0025)
            error=max(error,float(abs(y-target[step]).max()))
            previous=np.concatenate((previous,current),axis=1)[:,-16:]
        assert error==0.
        rows.append({'path':str(path),'native_sha256':file_hash(path),'target_sha256':file_hash(target_path),
                     'proprio_sha256':file_hash(proprio_path),'nonterminal_frame_overlap_error':0.,
                     'robot_intervals':int(age.size),'max_abs_target_error':error})
    assert len(rows)==20
    result={'status':'passed','roles':len(rows),'max_abs_target_error':0.,'audits':rows,
            'scope':'Predeclared 21/4096 environments, all collected intervals; arithmetic/timing proxy validation, not physical force-sensor validation.'}
    Path('artifacts/impact/continual_replay_v1/native_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RETAINED_NATIVE_AUDIT_PASSED',len(rows))


if __name__=='__main__':main()
