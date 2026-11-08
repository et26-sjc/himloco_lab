"""POST-HOC test-label FROC diagnostic; never a deployable operating point.

Added after model/operating-point lock. A normal validation-selected MSE point
may leave more FAR budget unused than BCE; diagnose that without changing a
single model, gate or locked threshold. Results cannot confirm the primary.
"""
import json
from pathlib import Path

import numpy as np


ART=Path('artifacts/impact/causal_stream_v1')
ROOT=Path('datasets/go1_impact/causal_stream_v1')


def fixed_threshold_delay_audit():
    """Retain the zero-delay validation point, then attribute delayed alarms."""
    rows=[]
    for rep in range(1,6):
        points=json.loads((ART/f'rep{rep}/operating_points.json').read_text())['points']
        for role in ('test','high_speed'):
            cache=ROOT/f'rep{rep}/{role}/cache'
            meta=json.loads((cache/'metadata.json').read_text());ev=dict(np.load(cache/'events.npz'))
            context=np.load(cache/'evaluation_context.npz');family=np.asarray(meta['terrain']['family_id'])
            # Search past strong events without assuming arrays are time ordered.
            span=int(meta['steps']*8+1);group=ev['env'].astype('int64')*4+ev['foot']
            strong_ids=np.flatnonzero(ev['severity']>2000)
            key=group[strong_ids]*span+ev['native_onset'][strong_ids]+1
            order=np.argsort(key);key=key[order];ordered=strong_ids[order]
            for seed in (1,2):
                for method in ('direct_warm_bce','state_gate_adapted_bce'):
                    threshold=points[f'{seed}/0/{method}']['threshold']
                    record=dict(np.load(ART/f'rep{rep}/{role}/seed{seed}_delay1_{method}.npz'))
                    chosen=(record['lower']<threshold)&(record['upper']>=threshold)
                    env=record['env'][chosen];foot=record['foot'][chosen];time=record['time'][chosen]
                    ids=record['event_id'][chosen];tp=np.zeros(len(ids),bool);known=ids>=0
                    tp[known]=ev['severity'][ids[known]]>2000
                    query=(env.astype('int64')*4+foot)*span+time*8
                    index=np.searchsorted(key,query,side='right')-1;safe=index.clip(0,len(key)-1)
                    nearest=ordered[safe];delta=time*8-ev['native_onset'][nearest]-1
                    late=(~known)&(index>=0)&(group[nearest]==env*4+foot)&(delta>=16)&(delta<32)
                    for domain in ('seen','heldout'):
                        mask=family<6 if domain=='seen' else family==6
                        selected=mask[env];minutes=float(context['valid'][:,mask].sum()*.02/60)
                        target=mask[ev['env']]&(ev['severity']>2000)&context['valid'][ev['base']+1,ev['env']]
                        true=int((tp&selected).sum());false=int((~tp&selected).sum())
                        rows.append(dict(rep=rep,seed=seed,method=method,domain=f'{role}_{domain}',
                                         recall=true/max(int(target.sum()),1),far=false/minutes,
                                         fp=false,late_strong_40_to_80ms=int((late&selected).sum()),
                                         weak_qualified=int((known&~tp&selected).sum()),
                                         other_background=int((~known&~late&selected).sum())))
    groups={}
    for domain in ('test_seen','test_heldout','high_speed_seen','high_speed_heldout'):
        for method in ('direct_warm_bce','state_gate_adapted_bce'):
            part=[r for r in rows if r['domain']==domain and r['method']==method]
            groups[f'{domain}/{method}']=dict(recall=float(np.mean([r['recall'] for r in part])),
                far=float(np.mean([r['far'] for r in part])),
                late_strong_fraction_of_fp=sum(r['late_strong_40_to_80ms'] for r in part)/max(sum(r['fp'] for r in part),1))
    result=dict(status='posthoc_fixed_zero_delay_operating_point_diagnostic',rows=rows,groups=groups,
                warning='Zero-delay validation threshold applied to 20ms delayed packets. No retraining. Late strong attribution uses same-foot40-80ms window; remaining background may contain impacts outside the qualified-landing task.')
    (ART/'posthoc_delay_attribution.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def diagnostic(records,events,valid,family,domain):
    mask=family<6 if domain=='seen' else family==6
    keep=mask[records['env']];lower=records['lower'][keep];upper=records['upper'][keep]
    ids=records['event_id'][keep];positive=np.zeros(len(ids),bool);ok=ids>=0
    positive[ok]=events['severity'][ids[ok]]>2000
    thresholds=np.unique(np.r_[np.quantile(upper,np.linspace(0,1,1024)).astype('float32'),
                               np.nextafter(upper.max(),np.float32(np.inf))])
    def count(subset):
        return np.searchsorted(np.sort(lower[subset]),thresholds,side='left')-np.searchsorted(np.sort(upper[subset]),thresholds,side='left')
    tp=count(positive);fp=count(~positive)
    minutes=float(valid[:,mask].sum()*.02/60)
    eligible=mask[events['env']]&valid[events['base']+1,events['env']]&(events['severity']>2000)
    total=int(eligible.sum());far=fp/minutes
    output={}
    for budget in (.25,.5,1.):
        choices=np.flatnonzero(far<=budget);best=max(choices,key=lambda i:(tp[i],-far[i],thresholds[i]))
        output[str(budget)]=dict(recall=float(tp[best]/max(total,1)),far=float(far[best]),
                                threshold=float(thresholds[best]),tp=int(tp[best]),fp=int(fp[best]))
    return output


def main():
    rows=[]
    for rep in range(1,6):
        for role in ('test','high_speed'):
            cache=ROOT/f'rep{rep}/{role}/cache'
            meta=json.loads((cache/'metadata.json').read_text());events=dict(np.load(cache/'events.npz'))
            path=cache/'evaluation_context.npz'
            context=np.load(path if path.exists() else cache/'stream.npz')
            family=np.asarray(meta['terrain']['family_id'])
            for seed in (1,2):
                for method in ('direct_warm_bce','direct_warm_mse','state_gate_adapted_bce','state_gate_bce'):
                    record=dict(np.load(ART/f'rep{rep}/{role}/seed{seed}_delay0_{method}.npz'))
                    for domain in ('seen','heldout'):
                        rows.append(dict(rep=rep,seed=seed,method=method,domain=f'{role}_{domain}',
                                         diagnostic=diagnostic(record,events,context['valid'],family,domain)))
        print('POSTHOC_FROC_REP',rep,flush=True)
    groups={}
    for domain in ('test_seen','test_heldout','high_speed_seen','high_speed_heldout'):
        for method in ('direct_warm_bce','direct_warm_mse','state_gate_adapted_bce','state_gate_bce'):
            part=[r for r in rows if r['domain']==domain and r['method']==method]
            groups[f'{domain}/{method}']={str(b):float(np.mean([r['diagnostic'][str(b)]['recall'] for r in part]))
                                         for b in (.25,.5,1.)}
    result=dict(status='posthoc_diagnostic_only',
                warning='Uses test labels to choose thresholds. No deployability, calibration transfer, or confirmatory claim; locked primary is unchanged.',
                grid='1024 upper-score quantiles plus disabled; empirical FAR only, no bootstrap constraint',groups=groups,rows=rows)
    (ART/'posthoc_froc.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fixed_threshold_delay_audit()


if __name__=='__main__':main()
