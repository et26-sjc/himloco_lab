"""Validation-locked future alarms; inference reads causal history only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import t as student_t

from himloco_lab.impact.anticipation import (AnticipationData,SharedBank,PhaseBank,phase_features,
    eligible_future_events,future_alarm_records,retime_future_records,future_summary,select_future_threshold)
from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.metrics import average_precision
from collect_anticipation import ROOT,ART,LOG,SPEC
from train_anticipation import key


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@torch.no_grad()
def predictions(rep,role,development=False):
    torch.backends.cuda.matmul.allow_tf32=False
    name='development' if development else f'rep{rep}'
    cache=ROOT/name/role/'cache';dest=cache/'prediction_staging.npy'
    models=LOG/name;payload=torch.load(models/'models.pt',map_location='cpu',weights_only=False)
    pp=torch.load(models/'phase.pt',map_location='cpu',weights_only=False)
    spec=json.loads(SPEC.read_text())
    manifest=dict(models_sha256=file_hash(models/'models.pt'),phase_sha256=file_hash(models/'phase.pt'),
                  input_sha256=file_hash(cache/'pre.npy'))
    entries=payload['entries']+pp['entries']
    if dest.exists() and (cache/'prediction_staging.json').exists():
        assert json.loads((cache/'prediction_staging.json').read_text())==manifest
        return np.load(dest,mmap_mode='r'),entries
    checkpoint=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    bank=SharedBank(payload['entries'],checkpoint);bank.load_state_dict(payload['state']);bank.cuda().eval()
    phase=PhaseBank([e['seed'] for e in pp['entries']]);phase.load_state_dict(pp['state']);phase.cuda().eval()
    data=AnticipationData(cache,supervision=False);t,n=data.pre.shape[:2]
    values=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(entries),t*n,4))
    for start in range(0,t*n,4096):
        frames=data.frames(torch.arange(start,min(t*n,start+4096),device='cuda'))
        raw,_=bank(frames.flatten(1))
        pr=phase(phase_features(frames,pp['hip'],pp['moments']))
        values[:,start:start+len(frames)]=torch.cat((raw,pr)).cpu().numpy()
    values.flush()
    shaped=values.reshape(len(entries),t,n,4)
    fixture_ids=np.arange(0,n,64)
    np.savez_compressed(cache/'fixture_predictions.npz',raw=shaped[:,:,fixture_ids],
                        entries_json=np.array(json.dumps(entries)))
    dump(cache/'prediction_staging.json',manifest)
    return values,entries


def context(rep,role,development=False):
    name='development' if development else f'rep{rep}'
    cache=ROOT/name/role/'cache';meta=json.loads((cache/'metadata.json').read_text())
    data=dict(np.load(cache/'future.npz'));events=dict(np.load(cache/'events.npz'))
    return cache,data,events,np.asarray(meta['terrain']['family_id']),np.asarray(meta['terrain']['tile_id'])


def combinations(spec):
    return [(lead,delay) for lead in spec['lead_bins_ms'] for delay in spec['additional_link_delays_ms']]


def calibrate(rep,development=False):
    spec=json.loads(SPEC.read_text());name='development' if development else f'rep{rep}'
    output=ART/name/'operating_points.json'
    if output.exists():return
    values,entries=predictions(rep,'validation',development)
    cache,data,events,family,tile=context(rep,'validation',development)
    t,n=data['age'].shape;points={};verify={}
    inference=spec['inference_allowance_ms']
    eligibility={(lead,delay):eligible_future_events(events,data,lead,delay+inference) for lead,delay in combinations(spec)}
    for i,entry in enumerate(entries):
        score=values[i].reshape(t,n,4)
        base=future_alarm_records(score,data,events,0.,0.)
        for lead,delay in combinations(spec):
            records=retime_future_records(base,lead,delay+inference)
            ident=f"{entry['seed']}/{key(entry)}/{lead}/{delay}"
            point=select_future_threshold(records,data,family,tile,spec)
            check=future_summary(records,point['threshold'],events,data,family,tile,role=1,
                 minimum_lead_ms=lead,delay_ms=delay+inference,draws=spec['bootstrap_draws'],eligible=eligibility[lead,delay])
            assert check['tp']==point['selection_tp'] and check['fp']==point['selection_fp']
            points[ident]=point
            verify[ident]=future_summary(records,point['threshold'],events,data,family,tile,role=2,
                 minimum_lead_ms=lead,delay_ms=delay+inference,draws=spec['bootstrap_draws'],eligible=eligibility[lead,delay])
        print('ANTICIPATION_CALIBRATE_MODEL',name,i+1,len(entries),flush=True)
    dump(output,dict(points=points,verification_role2=verify,protocol_sha256=file_hash(SPEC),
                     models_sha256=file_hash(LOG/name/'models.pt')))
    del values
    (cache/'prediction_staging.npy').unlink();(cache/'prediction_staging.json').unlink()
    print('ANTICIPATION_CALIBRATED',name,flush=True)


def ranking_metrics(score,entry,events,data,family,tile,training_mean=None):
    rows=[]
    for domain in ('seen','heldout'):
        mask=data['valid']&((family<6) if domain=='seen' else (family==6))[None]
        ids=np.flatnonzero(mask)
        selected=np.random.default_rng(9861).choice(ids,min(100000,len(ids)),replace=False)
        flat=score.reshape(-1,4)[selected]
        eid=data['event_id'].reshape(-1,4)[selected]
        severity=np.where(eid>=0,events['severity'][eid.clip(0)],0.)
        label=severity>2000
        result=dict(domain=domain,robot_intervals=len(selected),AP=float(average_precision(label,flat)),
                    positive_fraction=float(label.mean()),sample_seed=9861)
        if entry['loss']=='mse':
            error=flat*2000-severity;tail=severity>4000
            result.update(RMSE_n_per_s=float(np.sqrt(np.mean(error**2))),MAE_n_per_s=float(np.mean(abs(error))),
                zero_baseline_RMSE_n_per_s=float(np.sqrt(np.mean(severity**2))),
                tail_gt4000_MAE_n_per_s=float(np.mean(abs(error[tail]))) if tail.any() else None,
                tail_gt4000_count=int(tail.sum()))
            if training_mean is not None:
                result['training_mean_baseline_RMSE_n_per_s']=float(np.sqrt(np.mean((severity-np.array(training_mean))**2)))
        else:
            result['Brier']=float(np.mean((expit(flat)-label)**2))
        rows.append(result)
    return rows


def evaluate(rep,development=False):
    spec=json.loads(SPEC.read_text());name='development' if development else f'rep{rep}'
    if not development:
        lock=json.loads((ART/'model_lock.json').read_text())
        for path,sha in {**lock['models'],**lock['operating_points']}.items():assert file_hash(path)==sha
    points=json.loads((ART/name/'operating_points.json').read_text())['points'];rows=[];ranks=[]
    training_mean=json.loads((LOG/name/'training.json').read_text()).get('training_target_mean_n_per_s')
    for role in (['validation'] if development else ['test','high_speed']):
        values,entries=predictions(rep,role,development)
        cache,data,events,family,tile=context(rep,role,development)
        t,n=data['age'].shape
        inference=spec['inference_allowance_ms']
        eligibility={(lead,delay):eligible_future_events(events,data,lead,delay+inference) for lead,delay in combinations(spec)}
        for i,entry in enumerate(entries):
            score=values[i].reshape(t,n,4);base=future_alarm_records(score,data,events,0.,0.)
            for row in ranking_metrics(score,entry,events,data,family,tile,training_mean):
                ranks.append(dict(rep=rep,seed=entry['seed'],method=key(entry),role=role,**row))
            for lead,delay in combinations(spec):
                ident=f"{entry['seed']}/{key(entry)}/{lead}/{delay}"
                threshold=points[ident]['threshold'];records=retime_future_records(base,lead,delay+inference)
                selected=(records['lower']<threshold)&(records['upper']>=threshold)
                dest=ART/name/role/f"{entry['seed']}_{key(entry)}_lead{lead}_delay{delay}.npz"
                dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,**{k:v[selected] for k,v in records.items()},threshold=np.array(threshold),
                                    lead_ms_required=np.array(lead),delay_ms=np.array(delay),inference_ms=np.array(inference))
                for domain in ('seen','heldout'):
                    result=future_summary(records,threshold,events,data,family,tile,domain,
                        role=2 if development else None,minimum_lead_ms=lead,delay_ms=delay+inference,
                        draws=spec['bootstrap_draws'],eligible=eligibility[lead,delay])
                    rows.append(dict(rep=rep,seed=entry['seed'],method=key(entry),lead_ms=lead,delay_ms=delay,
                                     inference_allowance_ms=inference,domain=f'{role}_{domain}',threshold=threshold,**result))
            print('ANTICIPATION_EVALUATE_MODEL',name,role,i+1,len(entries),flush=True)
        del values
        (cache/'prediction_staging.npy').unlink();(cache/'prediction_staging.json').unlink()
    dump(ART/name/'results.json',rows);dump(ART/name/'ranking.json',ranks)
    print('ANTICIPATION_EVALUATED',name,flush=True)


def ci(values):
    x=np.asarray(values,float);mean=float(x.mean());half=float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return dict(mean=mean,lower=mean-half,upper=mean+half,replication_values=x.tolist())


def summarize():
    spec=json.loads(SPEC.read_text())
    rows=[r for rep in range(1,6) for r in json.loads((ART/f'rep{rep}/results.json').read_text())]
    ranking=[r for rep in range(1,6) for r in json.loads((ART/f'rep{rep}/ranking.json').read_text())]
    groups={}
    for domain in ('test_seen','test_heldout','high_speed_seen','high_speed_heldout'):
        for lead,delay in combinations(spec):
            for method in sorted({r['method'] for r in rows}):
                chosen=[r for r in rows if r['domain']==domain and r['lead_ms']==lead and r['delay_ms']==delay and r['method']==method]
                metrics={}
                for metric in ('recall','far','lead_ms_median','false_background','false_ordinary','false_late_strong'):
                    values=[np.mean([r[metric] for r in chosen if r['rep']==rep and r[metric] is not None]) for rep in range(1,6)]
                    if np.isfinite(values).all():metrics[metric]=ci(values)
                metrics['far_upper_conservative']=max(metrics['far']['upper'],max(r['far_tile_upper95'] for r in chosen),
                                                       max(r['zero_count_poisson_upper95'] for r in chosen))
                groups[f'{domain}/{lead}/{delay}/{method}']=metrics
    a=groups['test_seen/20/0/scratch_bce'];b=groups['test_seen/20/0/phase_bce']
    difference=ci(np.array(a['recall']['replication_values'])-b['recall']['replication_values'])
    comparisons={}
    selected=json.loads((ART/'development/decision.json').read_text())['selected_joint_lambda']
    for loss in spec['losses']:
        c=f'joint_{loss}_lambda{selected[loss]:g}'
        for label,left,right in [('C-A',c,f'dedicated_{loss}'),('D-B',f'moving_{loss}',f'frozen_{loss}'),('C-D',c,f'moving_{loss}')]:
            l=groups[f'test_seen/20/0/{left}'];r=groups[f'test_seen/20/0/{right}']
            comparisons[f'{loss}/{label}']=dict(difference=ci(np.array(l['recall']['replication_values'])-r['recall']['replication_values']),
                both_far_pass=max(l['far_upper_conservative'],r['far_upper_conservative'])<=1.)
    # Descriptive interaction, no post-hoc promotion to the single primary endpoint.
    cb=groups[f"test_seen/20/0/joint_bce_lambda{selected['bce']:g}"]['recall']['replication_values']
    cm=groups[f"test_seen/20/0/joint_mse_lambda{selected['mse']:g}"]['recall']['replication_values']
    ab=groups['test_seen/20/0/dedicated_bce']['recall']['replication_values']
    am=groups['test_seen/20/0/dedicated_mse']['recall']['replication_values']
    interaction=ci(np.array(cb)-cm-np.array(ab)+am)
    rankgroups={}
    for role in ('test','high_speed'):
        for domain in ('seen','heldout'):
            for method in sorted({r['method'] for r in ranking}):
                chosen=[r for r in ranking if r['role']==role and r['domain']==domain and r['method']==method]
                rankgroups[f'{role}_{domain}/{method}']={metric:ci([np.mean([r[metric] for r in chosen if r['rep']==rep]) for rep in range(1,6)])
                    for metric in ('AP','Brier','RMSE_n_per_s','MAE_n_per_s','tail_gt4000_MAE_n_per_s',
                                   'zero_baseline_RMSE_n_per_s','training_mean_baseline_RMSE_n_per_s') if metric in chosen[0]}
    dump(ART/'summary.json',dict(groups=groups,ranking=rankgroups,primary_difference=difference,
        primary_pass=difference['lower']>.03 and max(a['far_upper_conservative'],b['far_upper_conservative'])<=1.,
        shared_comparisons= comparisons,sharing_by_target_interaction=interaction,selected_joint_lambda=selected,
        physical_utility_verified=False,geometric_contact_audited=False,
        metric_scope='Force-threshold future risk,20ms minimum research lead; no demonstrated intervention benefit.',
        uncertainty='Five independent data replications after averaging two initializations; secondary mechanism comparisons descriptive.'))
    print('ANTICIPATION_SUMMARIZED',difference,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rep',type=int,default=1)
    p.add_argument('--development',action='store_true');p.add_argument('--calibrate',action='store_true')
    p.add_argument('--summarize',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    if a.summarize:summarize()
    elif a.calibrate:calibrate(a.rep,a.development)
    else:evaluate(a.rep,a.development)
