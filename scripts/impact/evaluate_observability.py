"""Locked event-level evaluation, including within-phase/velocity comparisons."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr,t as student_t
import torch

from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.metrics import average_precision,fit_calibration,calibrate
from himloco_lab.impact.observability import EventData,EventBank,matched_auc
from train_observability import predict


def load_bank(path):
    item=torch.load(path,map_location='cpu',weights_only=False)
    bank=EventBank(item['input']['dim'],item['entries'],item['input']['hidden']).cuda()
    bank.load_state_dict(item['state']);bank.eval()
    return bank,item


def frame_stats(y,score,probability,strata,point=None,threshold=None,tail=None):
    positive=y>2000
    row={'events':len(y),'positives':int(positive.sum()),'prevalence':float(positive.mean()),
         'AP':average_precision(positive,score),'matched':matched_auc(positive,score,strata),
         'Brier':float(np.mean((probability-positive)**2))}
    if threshold is not None:
        alarm=score>threshold
        row.update(weak_contact_FPR=float(alarm[~positive].mean()),strong_contact_recall=float(alarm[positive].mean()))
    if point is not None:
        error=point-y
        row.update(RMSE=float(np.sqrt((error**2).mean())),MAE=float(abs(error).mean()),
                   negative_prediction_fraction=float((point<0).mean()),
                   severity_spearman=float(spearmanr(y,point).statistic),
                   tail_RMSE=float(np.sqrt(np.mean(error[y>=tail]**2))) if np.any(y>=tail) else None)
    return row


def evaluate(rep):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    root=Path('datasets/go1_impact/observability_v1')/f'rep{rep}'
    log=Path('logs/impact_observability_v1')/f'rep{rep}'
    out=Path('artifacts/impact/observability_v1')/f'rep{rep}';out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(Path('configs/impact/observability_v1.json').read_text())
    lock=json.loads(Path('artifacts/impact/observability_v1/model_lock.json').read_text())
    for kind in spec['inputs']:
        path=log/kind/'models.pt'
        assert file_hash(path)==lock['models'][str(path)]
    if (out/'results.json').exists():return
    validation=EventData(root/'validation/cache')
    tail=float(np.quantile(np.load(root/'train/cache/events.npz')['severity'][
               np.asarray(json.loads((root/'train/cache/metadata.json').read_text())['terrain']['family_id'])[
                   np.load(root/'train/cache/events.npz')['env']]<6],.95))
    calibrators={};thresholds={};val_results={}
    for kind in spec['inputs']:
        bank,item=load_bank(log/kind/'models.pt')
        ids=validation.ids();scores,_=predict(bank,validation,ids,kind,item['moments'])
        strata=validation.strata(item['moments'])
        np.savez_compressed(out/f'validation_{kind}_predictions.npz',event_id=ids,score=scores)
        for i,(seed,method,h) in enumerate(bank.entries):
            key=f'{kind}/{method}/{seed}/{h}'
            valid=validation.cpu['valid_cutoff'][ids,h]
            y=validation.cpu['severity'][ids]>2000
            cal=valid & (validation.tile[ids]%3==0)
            choose=valid & (validation.tile[ids]%3==1)
            verify=valid & (validation.tile[ids]%3==2)
            c=fit_calibration(scores[i,cal],y[cal]);calibrators[key]=c
            threshold=float(np.quantile(scores[i,choose & ~y],1-spec['event_risk_fpr_selection']))
            thresholds[key]=threshold
            selected=ids[verify]
            val_results[key]=frame_stats(validation.cpu['severity'][selected],scores[i,verify],
                calibrate(scores[i,verify],c),strata[selected],threshold=threshold)
        del bank;gc.collect();torch.cuda.empty_cache()
    (out/'validation_lock.json').write_text(json.dumps({'calibrators':calibrators,'thresholds':thresholds,
        'verification':val_results,'train_tail95':tail},indent=2)+'\n')
    del validation;gc.collect();torch.cuda.empty_cache()
    results={}
    for role in ('test','high_speed'):
        data=EventData(root/role/'cache');ids=data.ids('all')
        for kind in spec['inputs']:
            bank,item=load_bank(log/kind/'models.pt')
            scores,_=predict(bank,data,ids,kind,item['moments'])
            strata=data.strata(item['moments'])
            np.savez_compressed(out/f'{role}_{kind}_predictions.npz',event_id=ids,score=scores)
            for i,(seed,method,h) in enumerate(bank.entries):
                key=f'{kind}/{method}/{seed}/{h}'
                for domain in ('seen','heldout'):
                    mask=(data.family[ids]<6 if domain=='seen' else data.family[ids]==6)&data.cpu['valid_cutoff'][ids,h]
                    selected=ids[mask];score=scores[i,mask]
                    row=frame_stats(data.cpu['severity'][selected],score,calibrate(score,calibrators[key]),
                          strata[selected],point=score*2000 if method=='mse' else None,
                          threshold=thresholds[key],tail=tail)
                    results[f'{role}/{domain}/{key}']=row
            del bank;gc.collect();torch.cuda.empty_cache()
        del data;gc.collect();torch.cuda.empty_cache()
    (out/'results.json').write_text(json.dumps({'rep':rep,'results':results},indent=2)+'\n')
    print('OBS_EVALUATED',rep,flush=True)


def interval(x):
    a=np.asarray(x,float);mean=float(a.mean());sd=float(a.std(ddof=1)) if len(a)>1 else 0.
    delta=float(student_t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a))) if len(a)>1 else 0.
    return {'mean':mean,'CI95':[mean-delta,mean+delta],'replications':a.tolist()}


def summarize():
    root=Path('artifacts/impact/observability_v1')
    spec=json.loads(Path('configs/impact/observability_v1.json').read_text())
    collected=[json.loads((root/f'rep{rep}/results.json').read_text())['results'] for rep in range(1,6)]
    summary={}
    for role in ('test','high_speed'):
        for domain in ('seen','heldout'):
            for kind,methods in spec['methods'].items():
                for method in methods:
                    for h in range(6):
                        key=f'{role}/{domain}/{kind}/{method}/{h}'
                        rows=[[r[f'{role}/{domain}/{kind}/{method}/{seed}/{h}'] for seed in (1,2)] for r in collected]
                        metrics={'AP':lambda x:x['AP'],'matched_auc':lambda x:x['matched']['auc'],
                                 'coverage':lambda x:x['matched']['coverage_fraction'],'Brier':lambda x:x['Brier'],
                                 'FPR':lambda x:x['weak_contact_FPR'],'recall':lambda x:x['strong_contact_recall']}
                        if method=='mse':metrics.update(RMSE=lambda x:x['RMSE'],tail_RMSE=lambda x:x['tail_RMSE'])
                        summary[key]={name:interval([np.mean([fn(x) for x in row]) for row in rows]) for name,fn in metrics.items()}
    differences={}
    for role in ('test','high_speed'):
        for domain in ('seen','heldout'):
            for reference,method in [('phase','bce'),('kinematics','bce'),('full','mse')]:
                for h in range(6):
                    a=summary[f'{role}/{domain}/full/bce/{h}']['matched_auc']['replications']
                    b=summary[f'{role}/{domain}/{reference}/{method}/{h}']['matched_auc']['replications']
                    differences[f'{role}/{domain}/full_bce_minus_{reference}_{method}/{h}']=interval(np.asarray(a)-b)
    primary=differences['test/heldout/full_bce_minus_phase_bce/3']
    result={'summary':summary,'paired_differences':differences,'primary':primary,
            'primary_passed':primary['CI95'][0]>spec['primary']['practical_margin'],
            'protocol_sha256':file_hash('configs/impact/observability_v1.json')}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('OBS_SUMMARY',json.dumps(result['primary']),result['primary_passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rep',type=int,default=1)
    p.add_argument('--summarize',action='store_true');a=p.parse_args()
    summarize() if a.summarize else evaluate(a.rep)
