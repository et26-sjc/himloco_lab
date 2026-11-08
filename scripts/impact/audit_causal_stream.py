"""Independently audit native labels and confirmation timing on retained traces."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(path):
    path=Path(path);meta=json.loads((path/'metadata.json').read_text())
    trace=dict(np.load(path/'native_audit.npz'));stream=dict(np.load(path/'stream.npz'))
    events=dict(np.load(path/'events.npz'));checked=0;max_error=0.
    for column,env in enumerate(trace['env_id']):
        force=np.linalg.norm(trace['normal_forces_w'][:,column,-8:].astype('float64'),axis=-1).reshape(-1,4)
        state=(force.reshape(meta['steps'],8,4)[:-1,-2:]>5).all(1)
        state[trace['age'][1:,column]==0]=False
        np.testing.assert_array_equal(state,stream['state'][1:,env])
        rows=np.flatnonzero(events['env']==env)
        expected_pulse=np.zeros((meta['steps'],4),bool)
        expected_id=np.full((meta['steps'],4),-1,np.int32)
        for row in rows:
            k=int(events['native_onset'][row]);foot=int(events['foot'][row])
            assert (force[k-16:k,foot]<=5).all() and (force[k:k+2,foot]>5).all()
            confirmation=next(t for t in range(k//8, k//8+3) if t*8>=k+2)
            assert confirmation==events['confirm'][row]
            expected_pulse[confirmation,foot]=True
            for t in range(k//8,k//8+4):
                if t<meta['steps'] and k+1<=t*8<k+17:expected_id[t,foot]=row
            rates=[max((force[j-7:j+1,foot].mean()-force[j-15:j-7,foot].mean())/.02,0)
                   for j in range(k,k+16)]
            max_error=max(max_error,abs(max(rates)-events['severity'][row]))
            assert int(np.argmax(force[k:k+16,foot]))==events['native_peak_offset'][row]
            checked+=1
        np.testing.assert_array_equal(expected_pulse,stream['pulse'][:,env])
        np.testing.assert_array_equal(expected_id,stream['event_id'][:,env])
    assert max_error<.01
    report=dict(status='passed',native_envs=len(trace['env_id']),events=checked,max_label_error_n_per_s=float(max_error))
    (path/'stream_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('STREAM_AUDIT',path,checked,max_error,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path');a=p.parse_args();audit(a.path)
