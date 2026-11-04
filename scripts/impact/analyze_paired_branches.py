"""Measure prefix equality before asserting any future-prediction lower bound."""
import json
from pathlib import Path

import numpy as np


root=Path('datasets/go1_impact/force_study_v1/paired_branches')
report=json.loads((root/'report.json').read_text())
data=np.load(root/'branches.npz')
h=data['history'];dt=report['physics_dt'];dec=round(.02/dt)
v=data['forces'].transpose(1,0,2,3,4).reshape(4096,-1,4,3)
f=np.linalg.norm(v.astype('float64'),axis=-1)
lag=round(.02/dt);length=f.shape[1]
padded=np.pad(f,((0,0),(2*lag,0),(0,0)))
avg=sum(padded[:,2*lag-k:2*lag-k+length] for k in range(lag))/lag
old=np.pad(avg,((0,0),(lag,0),(0,0)))[:,:length]
rate=np.maximum(avg-old,0)/.02
onset=np.argmax((f>1).any(-1),axis=1)
rows=[];bucket_rows=[];noise_rows=[]
noise=np.tile(np.array([0]*3+[.05]*3+[.05]*3+[.01]*12+[.075]*12+[0]*12),6)
for group in range(128):
    ids=np.arange(group*32,(group+1)*32)
    cutoff=int(onset[ids].min()//dec)
    if cutoff<5:raise AssertionError('Insufficient common history')
    if np.any(f[ids,:cutoff*dec]>1):raise AssertionError('Prefix already contacted')
    prefix=h[cutoff,ids]
    row={'group':group,'cutoff_ms':cutoff*20,'max_prefix_abs_difference':float(abs(prefix-prefix[:1]).max()),'horizons':{}}
    for horizon in (20,200):
        stop=cutoff+horizon//20
        if np.any(data['done'][cutoff:stop,ids]):
            row['horizons'][str(horizon)]={'censored':True};continue
        y=rate[ids,cutoff*dec:stop*dec].max(1)
        z=y>2000
        # Reveal a fixed mirrored-height pair ID to an oracle. Its Bayes error
        # lower-bounds the error without that extra information. Equal-volume
        # independent uniform noise boxes overlap by gamma; on the overlap the
        # two branches have posterior 1/2 and MSE delta_y^2/4.
        left,right=np.arange(16),np.arange(31,15,-1)
        delta=abs(prefix[left]-prefix[right])
        noiseless_equal=(delta[:,noise==0]==0).all(1)
        for multiplier in (.01,.1,1.):
            gamma=np.maximum(1-delta[:,noise>0]/(2*multiplier*noise[noise>0]),0).prod(1)*noiseless_equal
            noise_rows.append({'group':group,'horizon_ms':horizon,'noise_multiplier':multiplier,
                               'overlap_mean':float(gamma.mean()),'overlap_min':float(gamma.min()),
                               'MSE_lower_bound':float(np.mean(gamma[:,None]*(y[left]-y[right])**2)/4),
                               'Brier_lower_bound':float(np.mean(gamma[:,None]*(z[left].astype(float)-z[right])**2)/4)})
        row['horizons'][str(horizon)]={'censored':False,'mean_conditional_variance_if_prefix_identical':float(y.var(0).mean()),
                                     'per_foot_event_probability':z.mean(0).tolist(),
                                     'amplitude_range':(y.max(0)-y.min(0)).tolist()}
        for kind,visible in [('raw_float32',prefix),('quantized_1e-3',np.round(prefix,3))]:
            _,assignment=np.unique(visible,axis=0,return_inverse=True)
            for bucket in np.unique(assignment):
                member=assignment==bucket
                count=int(member.sum())
                if count<2:continue
                yy,zz=y[member],z[member]
                p=zz.mean(0)
                bucket_rows.append({'group':group,'horizon_ms':horizon,'input':kind,'branches':count,
                                    'MSE_lower_bound':float(yy.var(0).mean()),
                                    'event_Brier_lower_bound':float((p*(1-p)).mean()),
                                    'deterministic_event_feet':int(((p==0)|(p==1)).sum()),
                                    'same_class_with_amplitude_variance_feet':int((((p==0)|(p==1))&(yy.var(0)>1)).sum()),
                                    'ambiguous_event_feet':int(((p>0)&(p<1)).sum())})
    rows.append(row)
summary={}
for kind in ('raw_float32','quantized_1e-3'):
    for horizon in (20,200):
        records=[r for r in bucket_rows if r['input']==kind and r['horizon_ms']==horizon]
        total=sum(r['branches'] for r in records)
        summary[f'{kind}_{horizon}ms']={'covered_branches':total,'fraction_of_4096':total/4096,
             'weighted_MSE_lower_bound':sum(r['MSE_lower_bound']*r['branches'] for r in records)/total if total else None,
             'weighted_Brier_lower_bound':sum(r['event_Brier_lower_bound']*r['branches'] for r in records)/total if total else None,
             'same_class_variable_amplitude_foot_buckets':sum(r['same_class_with_amplitude_variance_feet'] for r in records),
             'ambiguous_event_foot_buckets':sum(r['ambiguous_event_feet'] for r in records)}
result={'status':'completed','scope':report['scope'],
        'warning':'Bounds apply only to exactly equal stated inputs; 1e-3 quantization is an explicit different sensor model. Not a bound on unrestricted floating-point I0 walking input.',
        'prefix_difference_max':max(r['max_prefix_abs_difference'] for r in rows),
        'summary':summary,'groups':rows,'buckets':bucket_rows,
        'noise_model':'Independent continuous uniform additive noise in six actor frames, half-widths matched to configured policy noise; no clipping active. This is an explicit statistical sensor model, not a bitwise claim about finite-precision PRNG outputs.',
        'noise_bound_proof':'With mirrored pair identity revealed, uniform noise-box overlap gamma has posterior 1/2. Average pair risk gamma*(ya-yb)^2/4 is a lower bound for the full 32-way problem without pair identity.',
        'noise_bounds':[]}
for horizon in (20,200):
    for multiplier in (.01,.1,1.):
        records=[r for r in noise_rows if r['horizon_ms']==horizon and r['noise_multiplier']==multiplier]
        result['noise_bounds'].append({'horizon_ms':horizon,'noise_multiplier':multiplier,'groups':len(records),
              **{key:float(np.mean([r[key] for r in records])) for key in ('overlap_mean','overlap_min','MSE_lower_bound','Brier_lower_bound')}})
out=Path('artifacts/impact/force_study_v1/paired_branches.json')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print(json.dumps(result['noise_bounds'],indent=2))
