"""Check time boundaries, terminal retention and clean/noisy sensor agreement."""
import argparse
import json
from pathlib import Path

import numpy as np
from himloco_lab.impact.dataset import file_hash


def audit(path):
    meta=json.loads((path/'metadata.json').read_text())
    a={k:np.load(path/f'{k}.npy',mmap_mode='r') for k in ('pre','post','hf','age','done','targets')}
    noise=np.array([0]*3+[.05]*3+[.05]*3+[.01]*12+[.075]*12+[0]*12)
    maximum=np.zeros(45);continuity=0.;identity=0.;count=0
    for start in range(0,meta['steps'],64):
        end=min(start+64,meta['steps'])
        clean=a['hf'][start:end,:,-1,:45].astype('float32')
        maximum=np.maximum(maximum,abs(clean-a['post'][start:end]).max(axis=(0,1)))
        lo=max(1,start)
        if lo<end:
            old=a['post'][lo-1:end-1];new=a['pre'][lo:end]
            valid=~a['done'][lo-1:end-1]
            continuity=max(continuity,float(abs(old-new)[valid].max()))
            f=a['targets'][lo-1:end-1,:,5];g=a['targets'][lo:end,:,5]
            valid=a['age'][lo:end]>=26
            expected=np.maximum(g-f,0)/.02
            identity=max(identity,float(abs(expected-a['targets'][lo:end,:,6])[valid].max()))
        count+=int((a['age'][start:end]>=25).sum())
    # Float16 caching of clean features can add small quantization error.
    quant=np.asarray(meta['hf_quantization_max_abs_by_channel'])[:45]
    if np.any(maximum>noise+quant+2e-5):
        raise AssertionError(f'Unexpected sensor mismatch: {maximum-noise-quant}')
    if continuity!=0 or identity>.02:raise AssertionError((continuity,identity))
    result={'status':'passed','audit_version':'all_transition_pairs_v2','metadata_sha256':file_hash(path/'metadata.json'),
            'clean_noisy_max_abs_difference':maximum.tolist(),'expected_noise_bound':noise.tolist(),
            'post_next_pre_max_abs_difference':continuity,'force20_rate20_identity_max_error':identity,
            'valid_transitions':count,'terminal_intervals':int(a['done'].sum())}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.cache);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print('CACHE_AUDIT_PASS',args.cache)
