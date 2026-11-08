"""Validation-only operating points and locked causal stream test evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit,logit
from scipy.stats import t as student_t
import torch

from himloco_lab.impact.causal_stream import (StreamData,StreamBank,PDBank,state_gate,
    alarm_records,delayed,summarize_records)
from himloco_lab.impact.dataset import file_hash


ART=Path('artifacts/impact/causal_stream_v1')
ROOT=Path('datasets/go1_impact/causal_stream_v1')
LOG=Path('logs/impact_causal_stream_v1')


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@torch.no_grad()
def predictions(rep,role,development=False):
    name='development' if development else f'rep{rep}/{role}'
    root=ROOT/name/'cache';out=root/'predictions.npz'
    if out.exists():return dict(np.load(out))
    modeldir=LOG/('development' if development else f'rep{rep}')
    data=StreamData(root);t,n=data.pre.shape[:2];result={}
    fixture_ids=np.arange(0,n,64);fixture={}
    for kind in ('models','frozen','pd'):
        payload=torch.load(modeldir/f'{kind}.pt',map_location='cpu',weights_only=False)
        entries=[tuple(v) for v in payload['entries']]
        bank=PDBank([s for s,_ in entries]) if kind=='pd' else StreamBank(entries)
        bank.load_state_dict(payload['state']);bank=bank.cuda().eval()
        values=np.empty((len(entries),t*n,4),np.float32)
        for start in range(0,t*n,4096):
            frames=data.frames(torch.arange(start,min(t*n,start+4096),device='cuda'))
            raw=bank(frames) if kind=='pd' else bank(frames.flatten(1))
            values[:,start:start+raw.shape[1]]=raw.cpu().numpy()
        values=values.reshape(len(entries),t,n,4)
        for i,(seed,method) in enumerate(entries):
            key=f'{seed}_{kind}_{method}';result[key]=values[i];fixture[key]=values[i,:,fixture_ids].transpose(1,0,2)
        del bank,values
    np.savez_compressed(out,**result)
    # Complete-input fixture already retained by collection, with exact raw outputs.
    np.savez_compressed(root/'fixture_predictions.npz',**fixture)
    return result


def context(rep,role,development=False):
    path=ROOT/('development' if development else f'rep{rep}/{role}')/'cache'
    meta=json.loads((path/'metadata.json').read_text())
    data=dict(np.load(path/'stream.npz'));event=dict(np.load(path/'events.npz'))
    family=np.asarray(meta['terrain']['family_id']);tile=np.asarray(meta['terrain']['tile_id'])
    return data,event,family,tile


def pulse_trigger(score,age,threshold):
    previous=np.full_like(score,-np.inf)
    for lag in range(1,6):
        previous[lag:]=np.maximum(previous[lag:],np.where((age[lag:]>=lag)[:,:,None],score[:-lag],-np.inf))
    return (score>=threshold)&(previous<threshold)


def onset_f1(gate,data,event,family,tile):
    env=(family<6)&(tile%3==0)
    select=gate&data['valid'][:,:,None]&env[None,:,None]
    ids=data['event_id'][select];hits=ids[ids>=0]
    hits=np.unique(hits)
    eligible=env[event['env']]&data['valid'][event['base']+1,event['env']]
    total=int(eligible.sum());tp=len(hits);alarms=int(select.sum())
    return dict(f1=2*tp/max(total+alarms,1),recall=tp/max(total,1),precision=tp/max(alarms,1),
                events=total,tp=tp,alarms=alarms)


def choose_gates(pred,data,event,family,tile,spec):
    selected={}
    for seed in spec['optimization_seeds']:
        selected[str(seed)]={}
        for kind,key in [('state',f'{seed}_models_state'),('pd',f'{seed}_pd_pd'),('pulse',f'{seed}_models_pulse')]:
            grid=spec['pulse_gate_thresholds'] if kind=='pulse' else spec['state_gate_thresholds']
            trials=[]
            for high in grid:
                gate=pulse_trigger(pred[key],data['age'],logit(high)) if kind=='pulse' else state_gate(expit(pred[key]),data['age'],high)
                trials.append(dict(threshold=high,**onset_f1(gate,data,event,family,tile)))
            best=max(trials,key=lambda r:(r['f1'],r['precision'],r['threshold']))
            selected[str(seed)][kind]=dict(threshold=best['threshold'],validation_role0=trials)
    return selected


def methods(pred,data,event,gates,seed,delay):
    age=data['age'];state=delayed(expit(pred[f'{seed}_models_state']),age,delay)
    pd=delayed(expit(pred[f'{seed}_pd_pd']),age,delay)
    pulse=delayed(pred[f'{seed}_models_pulse'],age,delay)
    pulse_gate=pulse_trigger(pulse,age,logit(gates['pulse']['threshold']))
    state_pulse=state_gate(state,age,gates['state']['threshold'])
    pd_pulse=state_gate(pd,age,gates['pd']['threshold'])
    oracle=np.zeros_like(state_pulse)
    confirmation=event['confirm']+delay;ok=confirmation<len(oracle)
    oracle[confirmation[ok],event['env'][ok],event['foot'][ok]]=True
    # Oracle's delayed packet must also belong to the current episode.
    oracle[age<delay]=False
    for loss in ('bce','mse'):
        raw=delayed(pred[f'{seed}_frozen_{loss}'],age,delay)
        for kind,mask in [('oracle',oracle),('state',state_pulse),('pulse',pulse_gate),('pd',pd_pulse)]:
            yield f'{kind}_gate_{loss}',np.where(mask,raw,-np.inf),mask
        yield f'ungated_frozen_{loss}',raw,None
    for kind in ('direct_scratch_bce','direct_warm_bce','direct_warm_mse'):
        raw=delayed(pred[f'{seed}_models_{kind}'],age,delay)
        yield kind,raw,None
        if 'warm' in kind:
            loss=kind.rsplit('_',1)[-1]
            for gate_name,mask in [('state',state_pulse),('oracle',oracle)]:
                yield f'{gate_name}_gate_adapted_{loss}',np.where(mask,raw,-np.inf),mask


def threshold_choice(records,event,data,family,tile,spec):
    env=(family<6)&(tile%3==1);keep=env[records['env']]
    upper=records['upper'][keep];lower=records['lower'][keep]
    eid=records['event_id'][keep];e=records['env'][keep]
    strong=np.zeros(len(eid),bool);ok=eid>=0;strong[ok]=event['severity'][eid[ok]]>2000
    if len(upper):
        # Representable float32 thresholds make scalar comparisons and sorted
        # interval counts exactly agree at ties, including the disabled point.
        thresholds=np.unique(np.r_[np.quantile(upper,np.linspace(0,1,128)).astype('float32'),
                                   np.nextafter(upper.max(),np.float32(np.inf))])
    else:thresholds=np.array([1e30])
    def counts(mask):
        return np.searchsorted(np.sort(lower[mask]),thresholds,side='left')-np.searchsorted(np.sort(upper[mask]),thresholds,side='left')
    tp=counts(strong);tiles=np.unique(tile[env]);false=np.array([counts((~strong)&(tile[e]==u)) for u in tiles])
    minutes=np.array([data['valid'][:,env&(tile==u)].sum()*.02/60 for u in tiles])
    rng=np.random.default_rng(8401);w=rng.multinomial(len(tiles),np.ones(len(tiles))/len(tiles),spec['bootstrap_draws'])
    rate=(w@false)/np.maximum((w@minutes)[:,None],1e-12)
    upper95=np.quantile(rate,.95,axis=0);far=false.sum(0)/minutes.sum()
    accepted=np.flatnonzero(upper95<=spec['selection_far_upper_per_minute'])
    assert len(accepted)>0
    best=max(accepted,key=lambda i:(tp[i],-far[i],thresholds[i]))
    return dict(threshold=float(thresholds[best]),selection_tp=int(tp[best]),selection_fp=int(false[:,best].sum()),
                far=float(far[best]),far_tile_upper95=float(upper95[best]),thresholds_considered=len(thresholds),
                disabled=bool(tp[best]==0 and false[:,best].sum()==0))


def calibrate(rep,development=False):
    spec=json.loads(Path('configs/impact/causal_stream_v1.json').read_text())
    name='development' if development else f'rep{rep}'
    out=ART/name/'operating_points.json'
    if out.exists():return
    pred=predictions(rep,'validation',development);data,event,family,tile=context(rep,'validation',development)
    gates=choose_gates(pred,data,event,family,tile,spec)
    points={};verification={}
    for seed in spec['optimization_seeds']:
        for delay in spec['delays_policy_steps']:
            for method,score,gate in methods(pred,data,event,gates[str(seed)],seed,delay):
                key=f'{seed}/{delay}/{method}'
                records=alarm_records(score,data['age'],data['valid'],data['event_id'])
                points[key]=threshold_choice(records,event,data,family,tile,spec)
                selected=summarize_records(records,points[key]['threshold'],event,data['valid'],family,tile,'seen',role=1)
                assert selected['tp']==points[key]['selection_tp'] and selected['fp']==points[key]['selection_fp']
                verification[key]=summarize_records(records,points[key]['threshold'],event,data['valid'],family,tile,'seen',role=2)
    dump(out,dict(gates=gates,points=points,verification_role2=verification,
                  protocol_sha256=file_hash('configs/impact/causal_stream_v1.json')))
    print('STREAM_CALIBRATED',name,flush=True)


def evaluate(rep,development=False):
    spec=json.loads(Path('configs/impact/causal_stream_v1.json').read_text())
    name='development' if development else f'rep{rep}'
    points=json.loads((ART/name/'operating_points.json').read_text());rows=[]
    for role in (['validation'] if development else ['test','high_speed']):
        pred=predictions(rep,role,development);data,event,family,tile=context(rep,role,development)
        for seed in spec['optimization_seeds']:
            for delay in spec['delays_policy_steps']:
                for method,score,gate in methods(pred,data,event,points['gates'][str(seed)],seed,delay):
                    key=f'{seed}/{delay}/{method}';threshold=points['points'][key]['threshold']
                    records=alarm_records(score,data['age'],data['valid'],data['event_id'])
                    dest=ART/name/role/f'seed{seed}_delay{delay}_{method}.npz'
                    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**records)
                    for domain in ('seen','heldout'):
                        row=dict(rep=rep,seed=seed,delay=delay,method=method,domain=f'{role}_{domain}',threshold=threshold,
                                 **summarize_records(records,threshold,event,data['valid'],family,tile,domain))
                        if gate is not None:
                            ids=data['event_id'][gate&data['valid'][:,:,None]];ids=np.unique(ids[ids>=0])
                            eligible=(family[event['env']]<6) if domain=='seen' else (family[event['env']]==6)
                            target=eligible&(event['severity']>2000)&data['valid'][event['base']+1,event['env']]
                            phase_hits=int(target[ids].sum())
                            row['phase_timely_recall']=phase_hits/max(int(target.sum()),1)
                            row['miss_no_timely_phase']=int(target.sum())-phase_hits
                            row['miss_severity_or_alarm_rule']=phase_hits-row['tp']
                        rows.append(row)
        print('STREAM_SCORED',rep,role,flush=True)
    dump(ART/name/'results.json',rows)


def ci(values):
    x=np.asarray(values,float);mean=float(x.mean())
    half=float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return dict(mean=mean,lower=mean-half,upper=mean+half,replication_values=x.tolist())


def summarize():
    rows=[v for rep in range(1,6) for v in json.loads((ART/f'rep{rep}/results.json').read_text())]
    groups={}
    for domain in ('test_seen','test_heldout','high_speed_seen','high_speed_heldout'):
        for delay in (0,1):
            for method in sorted(set(r['method'] for r in rows)):
                selected=[r for r in rows if r['domain']==domain and r['delay']==delay and r['method']==method]
                metrics={}
                for metric in ('recall','far','far_tile_upper95','phase_timely_recall','delay_ms_median','before_native_peak_fraction'):
                    if metric not in selected[0]:continue
                    values=[np.mean([r[metric] for r in selected if r['rep']==rep and r[metric] is not None]) for rep in range(1,6)]
                    if np.isfinite(values).all():metrics[metric]=ci(values)
                metrics['far_upper_conservative']=max(metrics['far']['upper'],max(r['far_tile_upper95'] for r in selected),max(r['zero_count_poisson_upper95'] for r in selected))
                groups[f'{domain}/{delay}/{method}']=metrics
    a=groups['test_seen/0/state_gate_bce'];b=groups['test_seen/0/direct_warm_bce']
    difference=ci(np.array(a['recall']['replication_values'])-b['recall']['replication_values'])
    result=dict(groups=groups,primary_difference=difference,
                primary_pass=difference['lower']>.03 and max(a['far_upper_conservative'],b['far_upper_conservative'])<=1,
                practical_pass=a['recall']['lower']>=.7 and all(groups[f'{d}/0/state_gate_bce']['far_upper_conservative']<=1 for d in ('test_seen','test_heldout','high_speed_seen','high_speed_heldout')),
                uncertainty='Across five data seeds after two-initialization average; FAR upper also includes maximum per-run tile upper, not a distribution-free bound.')
    dump(ART/'summary.json',result);print('STREAM_SUMMARY',json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rep',type=int,default=1)
    p.add_argument('--calibrate',action='store_true');p.add_argument('--summarize',action='store_true')
    p.add_argument('--development',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    if a.summarize:summarize()
    elif a.calibrate:calibrate(a.rep,a.development)
    else:evaluate(a.rep,a.development)
