"""Locked final test and paired replication statistics; no model selection."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import chi2,t,ttest_1samp
import torch

from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.force_study import ForceBank,ForceData
from himloco_lab.impact.metrics import average_precision,fit_calibration,calibrate,truth_event_ids,event_metrics
from run_continual_pilot import ids_for
from train_continual_confirmation import METHODS

ROOT=Path('artifacts/impact/continual_replay_v1/confirmation')
LOG=Path('logs/impact_continual_replay_v1/confirmation')


def assemble(paths):
    states=[torch.load(p,map_location='cpu',weights_only=False)['model'] for p in paths]
    bank=ForceBank(270,[(1,'bce',7)]*len(states),(128,64)).cuda()
    for s in states[1:]:
        assert torch.equal(s['mean'],states[0]['mean']) and torch.equal(s['std'],states[0]['std'])
    bank.load_state_dict({k:states[0][k] if k in ('mean','std') else torch.cat([s[k] for s in states]) for k in states[0]})
    bank.eval();return bank


@torch.no_grad()
def stream(bank,data,families):
    env=np.flatnonzero(np.isin(data.family,families));ids=ids_for(data,families)
    lookup=np.full(data.n,-1);lookup[env]=np.arange(len(env))
    score=np.full((len(bank.entries),data.t,len(env),4),-80.,np.float32)
    for start in range(0,len(ids),4096):
        rows=ids[start:start+4096];x,_=data.batch(rows,'detection')
        s=bank(x)[:,:,:4].cpu().numpy()
        score[:,rows//data.n,lookup[rows%data.n]]=s
    valid=np.asarray(data.cpu['age'])[:,env]>=25
    severity=np.asarray(data.cpu['targets'])[:,env,7]
    return score,valid,severity,data.tile[env]


def frame(score,valid,severity,calibration=None):
    s=score[valid];z=severity[valid]>2000.;p=expit(s)
    result={'AP':average_precision(z,s),'Brier':float(np.mean((p-z)**2)),
            'log_loss':float(np.mean(np.logaddexp(0,s)-z*s)),
            'robot_intervals':int(valid.sum()),'foot_prevalence':float(z.mean())}
    if calibration is not None:result['frozen_A_calibrated_Brier']=float(np.mean((calibrate(s,calibration)-z)**2))
    return result


def poisson_upper(false,minutes):
    return float(.5*chi2.ppf(.95,2*(false+1))/minutes) if minutes else None


def alarms(score,threshold,valid,severity,tiles,by_tile=False):
    truth,starts,_=truth_event_ids(severity,valid,2000.,merge_gap=2)
    result=event_metrics(score,threshold,valid,truth,starts,quiet_steps=5,task='detection')
    result['score_threshold']=result.pop('probability_threshold')
    result['poisson_FAR_upper_one_sided_95']=poisson_upper(result['false_alarms'],result['robot_minutes'])
    if by_tile:
        result['tiles']={}
        for tile in np.unique(tiles):
            mask=tiles==tile
            item=event_metrics(score[:,mask],threshold,valid[:,mask],truth[:,mask],starts[:,mask],quiet_steps=5,task='detection')
            item.pop('probability_threshold')
            result['tiles'][str(int(tile))]=item
    return result


def select_alarm(score,valid,severity,tiles):
    # Fully predetermined candidates and tie-break, no verification feedback.
    values=score[valid];candidates=np.unique(np.r_[np.quantile(values,1-np.geomspace(1e-5,.5,128)),float(values.max())+1.])
    truth,starts,_=truth_event_ids(severity,valid,2000.,merge_gap=2)
    best=None
    for threshold in candidates:
        row=event_metrics(score,float(threshold),valid,truth,starts,quiet_steps=5,task='detection')
        if row['false_alarms_per_robot_minute']>.5:continue
        key=(row['event_recall'] or 0.,-row['false_alarms_per_robot_minute'],float(threshold))
        if best is None or key>best[0]:best=(key,float(threshold))
    assert best is not None
    return best[1]


def calibration_roles(bank,data):
    scores,valid,severity,tiles=stream(bank,data,[0,1]);z=severity>2000.
    # Whole terrain tiles, not random neighboring frames or shared-tile robots.
    roles=(tiles//14)%3
    fit=valid&(roles[None,:]==0);selection=valid&(roles[None,:]==1);verification=valid&(roles[None,:]==2)
    result=[]
    for score in scores:
        mapping=fit_calibration(score[fit],z[fit])
        threshold=select_alarm(score,selection,severity,tiles)
        verify=alarms(score,threshold,verification,severity,tiles)
        result.append({'calibrator':mapping,'threshold':threshold,'selection':alarms(score,threshold,selection,severity,tiles),
                       'verification':verify,'verification_pass_poisson_upper_1':verify['poisson_FAR_upper_one_sided_95']<=1.})
    return result


def evaluate_rep(rep,model_lock):
    out=ROOT/f'rep{rep}'/'test.json'
    if out.exists():return json.loads(out.read_text())
    logs=LOG/f'rep{rep}'
    names=[(s,m) for s in (1,2) for m in ['freeze']+METHODS]
    paths=[logs/f'seed{s}_{m}_{"A0" if m=="freeze" else "C5"}.pt' for s,m in names]
    for p in paths:assert model_lock['weights_sha256'][str(p)]==file_hash(p)
    root=Path('datasets/go1_impact/continual_replay_v1/confirmation')/f'rep{rep}'/'cache'
    cal=ForceData(root/'calibration');bank=assemble(paths)
    # Calibrate/choose/verify before opening any test arrays for this replication.
    calibration=calibration_roles(bank,cal)
    (ROOT/f'rep{rep}'/'calibration_lock.json').write_text(json.dumps({'names':names,'models_sha256':[file_hash(p) for p in paths],'calibration':calibration},indent=2)+'\n')
    del cal;torch.cuda.empty_cache()
    test=ForceData(root/'test')
    records=[{'seed':s,'method':m,'domains':{},'calibration':c} for (s,m),c in zip(names,calibration)]
    for domain,families in [('A',[0,1]),('B',[2,3]),('C',[4,5]),('U',[6])]:
        scores,valid,severity,tiles=stream(bank,test,families)
        for i,(name,c,score) in enumerate(zip(names,calibration,scores)):
            seed,method=name;initial=calibration[(seed-1)*(len(METHODS)+1)]
            r=frame(score,valid,severity,initial['calibrator'])
            if domain=='A':
                r['fixed_initial_alarm']=alarms(score,initial['threshold'],valid,severity,tiles,by_tile=True)
                r['oracle_recalibrated_alarm']=alarms(score,c['threshold'],valid,severity,tiles)
                r['oracle_recalibrated_Brier']=float(np.mean((calibrate(score[valid],c['calibrator'])-(severity[valid]>2000.))**2))
            records[i]['domains'][domain]=r
        print('TEST_DOMAIN_COMPLETE',rep,domain,flush=True)
    del bank;torch.cuda.empty_cache()
    reference_paths=[logs/f'seed{s}_{m}_B5.pt' for s,m in names if m!='freeze']
    for p in reference_paths:assert model_lock['weights_sha256'][str(p)]==file_hash(p)
    reference=assemble(reference_paths)
    scores,valid,severity,_=stream(reference,test,[2,3]);j=0
    for i,(seed,method) in enumerate(names):
        if method=='freeze':records[i]['B_after_first_learning']=records[i]['domains']['B']
        else:records[i]['B_after_first_learning']=frame(scores[j],valid,severity);j+=1
        initial=records[(seed-1)*(len(METHODS)+1)]
        records[i]['A_initial']=initial['domains']['A']
        final=records[i]['domains']
        records[i]['old_macro_AP']=(final['A']['AP']+final['B']['AP'])/2
        records[i]['mean_forgetting_AP']=((records[i]['A_initial']['AP']-final['A']['AP'])+(records[i]['B_after_first_learning']['AP']-final['B']['AP']))/2
    result={'status':'completed','rep':rep,'records':records,'test_metadata_sha256':file_hash(root/'test'/'metadata.json'),
            'model_lock_sha256':file_hash(LOG/'ALL_MODELS_LOCKED.json'),'evaluation_source_sha256':file_hash(Path(__file__))}
    out.write_text(json.dumps(result,indent=2)+'\n');return result


def mean_ci(x):
    x=np.asarray(x,float);half=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return {'mean':float(x.mean()),'CI95':[float(x.mean()-half),float(x.mean()+half)],'per_replication':x.tolist()}


def summarize(reps):
    methods=['freeze']+METHODS
    aggregates={}
    for method in methods:
        by_rep=[[r for r in data['records'] if r['method']==method] for data in reps]
        a={}
        getters={'old_macro_AP':lambda r:r['old_macro_AP'],'forgetting_AP':lambda r:r['mean_forgetting_AP']}
        for domain in ('A','B','C','U'):
            getters[f'{domain}_AP']=lambda r,d=domain:r['domains'][d]['AP']
            getters[f'{domain}_Brier']=lambda r,d=domain:r['domains'][d]['Brier']
        getters['A_fixed_FAR']=lambda r:r['domains']['A']['fixed_initial_alarm']['false_alarms_per_robot_minute']
        getters['A_fixed_recall']=lambda r:r['domains']['A']['fixed_initial_alarm']['event_recall']
        getters['A_recalibrated_FAR']=lambda r:r['domains']['A']['oracle_recalibrated_alarm']['false_alarms_per_robot_minute']
        getters['A_recalibrated_recall']=lambda r:r['domains']['A']['oracle_recalibrated_alarm']['event_recall']
        for key,get in getters.items():a[key]=mean_ci([np.mean([get(r) for r in rows]) for rows in by_rep])
        aggregates[method]=a
    comparisons={};pvals=[]
    for other in ('uniform','binary'):
        d=np.asarray(aggregates['severity']['old_macro_AP']['per_replication'])-aggregates[other]['old_macro_AP']['per_replication']
        c=np.asarray(aggregates['severity']['C_AP']['per_replication'])-aggregates[other]['C_AP']['per_replication']
        p=float(ttest_1samp(d,.01,alternative='greater').pvalue);pvals.append(p)
        comparisons[f'severity_minus_{other}']={'old_AP_gain':mean_ci(d),'new_C_AP_difference':mean_ci(c),
            'p_superiority_over_0.01':p,'new_C_noninferiority_pass':mean_ci(c)['CI95'][0]>-.01}
    order=np.argsort(pvals);adjusted=np.empty(2);running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(2-rank)*pvals[i]));adjusted[i]=running
    for i,other in enumerate(('uniform','binary')):
        comparisons[f'severity_minus_{other}']['Holm_p']=float(adjusted[i])
        comparisons[f'severity_minus_{other}']['old_AP_superiority_pass']=bool(adjusted[i]<.05)
    for a,b in [('uniform','finetune'),('all_past','finetune'),('finetune','stationary_A'),('severity_unweighted','severity')]:
        comparisons[f'{a}_minus_{b}']={key:mean_ci(np.asarray(aggregates[a][key]['per_replication'])-aggregates[b][key]['per_replication']) for key in ('old_macro_AP','C_AP','A_Brier')}
    result={'status':'completed','independent_data_replications':5,'initializations_per_replication':2,
            'methods':aggregates,'paired_comparisons':comparisons,
            'primary_claim_supported':all(comparisons[f'severity_minus_{b}']['old_AP_superiority_pass'] and comparisons[f'severity_minus_{b}']['new_C_noninferiority_pass'] for b in ('uniform','binary')),
            'inference_unit':'Five collection seeds; two initializations averaged first. Frames and feet are not independent replications.',
            'alarm_scope':'A initial threshold and explicit calibration-label oracle. Per-tile counts retained. Poisson upper bounds are diagnostics; no operational utility or unseen-scene FAR guarantee claimed by this summary.'}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    torch.set_num_threads(4)
    lock=json.loads((LOG/'ALL_MODELS_LOCKED.json').read_text())
    assert lock['status']=='all_models_locked_before_final_test_scoring'
    reps=[evaluate_rep(rep,lock) for rep in range(1,6)]
    summary=summarize(reps)
    print('CONFIRMATION_EVALUATION_COMPLETE',summary['primary_claim_supported'],flush=True)


if __name__=='__main__':main()
