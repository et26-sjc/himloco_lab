"""Descriptive target statistics and training-only constant prediction controls.

Run after all models are locked and final caches exist. No model fitting or test
selection takes place here. Constants use only valid, seen-family training rows.
"""
import json
from pathlib import Path

import numpy as np

from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.force_study import TARGETS,THRESHOLDS


def constant_error(y,point,threshold):
    absolute=abs(y-point)
    active=y>threshold
    return {'MAE':float(absolute.mean()),'RMSE':float(np.sqrt(np.mean(absolute**2))),
            'active_MAE':float(absolute[active].mean()) if active.any() else None,
            'p95_absolute_error':float(np.quantile(absolute,.95))}


def prior_ap(positive,n,prior):
    """Exact AP of four foot-specific constant scores, aggregating score ties."""
    cumulative_tp=0.;cumulative_n=0;total=positive.sum();ap=0.
    if not total:return None
    for score in sorted(set(prior),reverse=True):
        feet=prior==score;tp=positive[feet].sum()
        cumulative_tp+=tp;cumulative_n+=int(feet.sum())*n
        ap+=tp/total*cumulative_tp/cumulative_n
    return float(ap)


def main():
    if not Path('logs/impact_force_study_v1/ALL_MODELS_LOCKED.json').exists():
        raise ValueError('Wait until all model arms are locked')
    root=Path('datasets/go1_impact/force_study_v1/cache')
    result={'status':'completed','scope':'Same target rows for D and P20; training constants exclude family6. No final-test fitting.',
            'replications':[]}
    targets=(5,6,7)
    for rep in range(1,6):
        train,test=root/f'rep{rep}'/'train',root/f'rep{rep}'/'test'
        tm=json.loads((train/'metadata.json').read_text());vm=json.loads((test/'metadata.json').read_text())
        ty=np.load(train/'targets.npy',mmap_mode='r');ta=np.load(train/'age.npy',mmap_mode='r')
        vy=np.load(test/'targets.npy',mmap_mode='r');va=np.load(test/'age.npy',mmap_mode='r')
        tf=np.array(tm['terrain']['family_id']);vf=np.array(vm['terrain']['family_id'])
        row={'rep':rep,'train_metadata_sha256':file_hash(train/'metadata.json'),
             'test_metadata_sha256':file_hash(test/'metadata.json'),'targets':{}}
        for target in targets:
            seen=tf<6;tr=ty[:,seen,target,:][ta[:,seen]>=25]
            mean=tr.mean(0,dtype=np.float64);prior=(tr>THRESHOLDS[target]).mean(0)
            entry={'training_mean_per_foot':mean.tolist(),'training_event_prior_per_foot':prior.tolist(),'domains':{},'families':{}}
            for domain,env in [('seen',vf<6),('heldout',vf==6)]:
                yy=vy[:,env,target,:][va[:,env]>=25]
                positive=(yy>THRESHOLDS[target]).sum(0);frequency=positive/len(yy)
                entry['domains'][domain]={'valid_robot_intervals':len(yy),'robot_minutes':len(yy)*.02/60,
                    'zero':constant_error(yy,0.,THRESHOLDS[target]),
                    'training_mean':constant_error(yy,mean,THRESHOLDS[target]),
                    'training_prior_AP':prior_ap(positive,len(yy),prior),
                    'training_prior_Brier':float(np.mean(prior**2-2*prior*frequency+frequency)),
                    'prevalence':float(frequency.mean()),'zero_fraction':float((yy==0).mean())}
            for family,name in enumerate(vm['terrain']['families']):
                env=vf==family;yy=vy[:,env,target,:][va[:,env]>=25]
                entry['families'][name]={'valid_robot_intervals':len(yy),'prevalence':float((yy>THRESHOLDS[target]).mean()),
                    'zero_fraction':float((yy==0).mean()),'mean':float(yy.mean()),
                    'quantiles':dict(zip(('p50','p90','p99','p999'),np.quantile(yy,[.5,.9,.99,.999]).tolist()))}
            row['targets'][TARGETS[target]]=entry
        result['replications'].append(row)
        print('REFERENCE_COMPLETE',rep,flush=True)
    out=Path('artifacts/impact/force_study_v1/constant_references.json')
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
