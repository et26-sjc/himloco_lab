"""Locked natural-stream evaluation with separate probability/alarm roles."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.special import expit
import torch

from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.force_study import ForceData,ForceBank,TARGETS,SCALES,THRESHOLDS,prepare_force
from himloco_lab.impact.metrics import average_precision,fit_calibration,calibrate,truth_event_ids,event_metrics
from train_force_study import predict


def fit_isotonic(x,y):
    x,y=np.asarray(x).ravel(),np.asarray(y,float).ravel()
    order=np.argsort(x,kind='stable');x,y=x[order],y[order]
    unique,start,count=np.unique(x,return_index=True,return_counts=True)
    sums=np.add.reduceat(y,start)
    result=isotonic_regression(sums/count,weights=count)
    changes=np.r_[0,np.flatnonzero(np.diff(result.x)!=0)+1]
    return {'x':unique[changes].tolist(),'p':result.x[changes].tolist()}


def apply_cal(x,c):
    if c['kind']=='logistic':return calibrate(x,c['parameters'])
    model=c['parameters']
    ids=np.searchsorted(model['x'],x,side='right')-1
    return np.asarray(model['p'])[np.maximum(ids,0)]


def fit_calibrators(scores,labels):
    half=len(labels)//2
    target=labels[:half]
    logistic={'kind':'logistic','parameters':fit_calibration(scores[:half],target)}
    iso={'kind':'isotonic','parameters':fit_isotonic(scores[:half],target)}
    errors=[float(np.mean((apply_cal(scores[half:],c)-labels[half:])**2)) for c in (logistic,iso)]
    result=[logistic,iso][int(np.argmin(errors))]
    result['selection_Brier']=errors
    return result


def continuous(y,p,target):
    result={}
    if not np.isfinite(p).all():return result
    scale=SCALES[target]
    for name,point in [('raw',p),('nonnegative',np.maximum(p,0))]:
        error=point-y;absolute=abs(error)
        rows=[]
        for lo,hi in zip([0,.25,.5,1,2,4],[.25,.5,1,2,4,np.inf]):
            mask=(y>=lo*scale)&(y<hi*scale)
            rows.append({'range':[lo*scale,hi*scale if np.isfinite(hi) else None],'count':int(mask.sum()),
                         'MAE':float(absolute[mask].mean()) if mask.any() else None,
                         'bias':float(error[mask].mean()) if mask.any() else None,
                         'p95_absolute_error':float(np.quantile(absolute[mask],.95)) if mask.any() else None})
        tolerances=[1,2,5,10,20] if target in (0,3,5) else [100,250,500,1000,2000]
        active=y>THRESHOLDS[target]
        resolution=1. if target in (0,3,5) else 100.
        subgroups={}
        for group,mask in [('exact_zero',y==0),('positive_below_resolution',(y>0)&(y<=resolution)),
                           ('weak_resolved',(y>resolution)&(y<.25*scale)),('above_event_threshold',active)]:
            subgroups[group]={'count':int(mask.sum())}
            if mask.any():
                subgroups[group].update(MAE=float(absolute[mask].mean()),bias=float(error[mask].mean()),
                    RMSE=float(np.sqrt(np.mean(error[mask]**2))),negative_fraction=float((point[mask]<0).mean()),
                    near_zero_prediction_fraction=float((abs(point[mask])<=resolution).mean()),
                    target_median=float(np.median(y[mask])),prediction_median=float(np.median(point[mask])))
        result[name]={'MAE':float(absolute.mean()),'RMSE':float(np.sqrt(np.mean(error**2))),
                      'bias':float(error.mean()),'p95_absolute_error':float(np.quantile(absolute,.95)),
                      'negative_fraction':float((point<0).mean()),'bins':rows,
                      'within_tolerance':{str(e):float((absolute<=e).mean()) for e in tolerances},
                      'resolution':resolution,'subgroups':subgroups,
                      'active_MAE':float(absolute[active].mean()) if active.any() else None,
                      'active_p95_absolute_error':float(np.quantile(absolute[active],.95)) if active.any() else None}
    return result


def frame_stats(y,s,p,point,target):
    z=y>THRESHOLDS[target]
    p=np.clip(p,1e-7,1-1e-7)
    bins=[]
    for lo,hi in zip([0,.001,.005,.01,.02,.05,.1,.2,.5,.8],[.001,.005,.01,.02,.05,.1,.2,.5,.8,1.000001]):
        mask=(p>=lo)&(p<hi)
        if mask.any():bins.append({'range':[lo,hi],'count':int(mask.sum()),
                                  'mean_probability':float(p[mask].mean()),'event_frequency':float(z[mask].mean())})
    return {'AP':average_precision(z,s),'point_AP':average_precision(z,point) if np.isfinite(point).all() else None,
            'prevalence':float(z.mean()),'rows':len(y),'calibration_bins':bins,
            'Brier':float(np.mean((p-z)**2)),
            'log_loss':float(np.mean(-(z*np.log(p)+(~z)*np.log1p(-p)))),
            'continuous':continuous(y,point,target)}


def stream_predictions(bank,data,task):
    ids=data.indices(families=tuple(range(7)))
    result=predict(bank,data,ids,task,details=True)
    return ids,result


def dense_stream(values,ids,t,n):
    result=np.zeros((t*n,4),dtype='float32');result[ids]=values
    return result.reshape(t,n,4)


def select_alarm(probability,valid,truth,starts,budgets,spec,task):
    columns=valid.any(axis=0)
    probability,valid=probability[:,columns],valid[:,columns]
    truth,starts=truth[:,columns],starts[:,columns]
    values=probability[valid]
    quantiles=1-np.geomspace(1e-5,.5,256)
    disabled=float(values.max())+max(1.,abs(float(values.max()))*.01)
    candidates=np.unique(np.r_[np.quantile(values,quantiles),disabled])
    best={str(b):None for b in budgets}
    for threshold in candidates:
        row=event_metrics(probability,float(threshold),valid,truth,starts,task=task)
        row['score_threshold']=row.pop('probability_threshold')
        for budget in budgets:
            old=best[str(budget)]
            if row['false_alarms_per_robot_minute']<=budget and (old is None or (row['event_recall'] or 0)>(old['event_recall'] or 0)):
                best[str(budget)]=row
    return best


def lock_validation(bank,val,task,spec,model_path,lock_path):
    """Fits only seen-family validation roles 1/2; never accesses final test."""
    model_hash=file_hash(model_path)
    if lock_path.exists():
        lock=json.loads(lock_path.read_text())
        if lock['model_hash']!=model_hash:raise ValueError('Lock/model mismatch')
        return lock,None,None
    rng=np.random.default_rng(9182)
    cids=val.indices(role=1);cids=rng.choice(cids,min(len(cids),spec['calibration_samples']),replace=False)
    calibration=predict(bank,val,cids,task,details=True)
    ycal=np.asarray(val.cpu['targets']).reshape(-1,len(TARGETS),4)[cids]
    calibrators=[fit_calibrators(calibration['score'][i],ycal[:,entry[2]]>THRESHOLDS[entry[2]])
                 for i,entry in enumerate(bank.entries)]
    print('CALIBRATION_LOCKED',str(model_path),flush=True)
    has_alarms=any(entry[2]==7 for entry in bank.entries)
    vids,vpred=stream_predictions(bank,val,task) if has_alarms else (None,None)
    valid=np.asarray(val.cpu['age'])>=25
    selections={}
    for i,(seed,method,target) in enumerate(bank.entries):
        if target!=7:continue
        pv=dense_stream(vpred['score'][i],vids,val.t,val.n)
        choose=valid & (np.arange(val.n)[None,:]%4==2) & (val.family[None,:]<6)
        truth,starts,_=truth_event_ids(np.asarray(val.cpu['targets'])[:,:,target],choose,THRESHOLDS[target])
        selections[str(i)]=select_alarm(pv,choose,truth,starts,spec['alarm_far_budgets'],spec,task)
        print('ALARM_LOCKED',str(model_path),seed,method,flush=True)
    best_regression={}
    for seed in sorted(set(e[0] for e in bank.entries)):
        candidates=[(m,selections[str(i)]['1.0']['event_recall'] or 0.)
                    for i,(s,m,t) in enumerate(bank.entries) if s==seed and t==7 and m in ('mse','mixture')]
        if candidates:best_regression[str(seed)]=max(candidates,key=lambda pair:pair[1])[0]
    lock={'model_hash':model_hash,'calibrators':calibrators,'alarms':selections,
          'validation_selected_regression':best_regression,
          'alarm_space':'raw ranking score; monotonic calibration does not change attainable rank decisions; avoids isotonic ties'}
    temporary=lock_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(lock,indent=2)+'\n');temporary.replace(lock_path)
    return lock,vids,vpred


def evaluate(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    spec=json.loads(Path('configs/impact/force_study_v1.json').read_text())
    root=Path('datasets/go1_impact/force_study_v1')
    runroot=Path('logs/impact_force_study_v1')/args.group/f'rep{args.rep}'/args.input
    resultroot=Path('artifacts/impact/force_study_v1')/args.group/f'rep{args.rep}'/args.input
    resultroot.mkdir(parents=True,exist_ok=True)
    if all((resultroot/f'{task}.json').exists() for task in args.tasks):
        return
    test_path=root/'cache'/f'rep{args.rep}'/'test'
    # All requested models must exist before opening the final test data.
    for task in args.tasks:
        if not (runroot/task/'report.json').exists():raise ValueError('Training not locked')
    val=ForceData(root/'cache'/f'rep{args.rep}'/'validation',input_kind=args.input)
    test=None
    for task in args.tasks:
        output=resultroot/f'{task}.json'
        if output.exists():continue
        saved=torch.load(runroot/task/'models.pt',map_location='cuda',weights_only=False)
        bank=ForceBank(saved['dim'],saved['entries'],saved['hidden']).cuda();bank.load_state_dict(saved['state'])
        bank.eval()
        lock_path=runroot/task/'operating_points.json'
        lock,vids,vpred=lock_validation(bank,val,task,spec,runroot/task/'models.pt',lock_path)
        if args.calibration_only:
            del bank
            torch.cuda.empty_cache()
            print('VALIDATION_ONLY_COMPLETE',str(lock_path),flush=True)
            continue
        calibrators,selections=lock['calibrators'],lock['alarms']
        if any(e[2]==7 for e in bank.entries) and vpred is None:
            vids,vpred=stream_predictions(bank,val,task)
        valid_val=np.asarray(val.cpu['age'])>=25
        if test is None:
            # Different input arms share this cache. Serialize construction.
            with (test_path.parent/'.test_cache.lock').open('a+') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX)
                if not (test_path/'metadata.json').exists():
                    prepare_force(root/'raw'/f'rep{args.rep}'/'test',test_path)
            test=ForceData(test_path,input_kind=args.input)
        tids,tpred=stream_predictions(bank,test,task)
        result={'status':'completed','rep':args.rep,'input':args.input,'task':task,'models':[],
                'model_hash':file_hash(runroot/task/'models.pt'),
                'spec_hash':file_hash(Path('configs/impact/force_study_v1.json')),
                'operating_points_hash':file_hash(lock_path),
                'validation_selected_regression':lock['validation_selected_regression'],
                'validation_hash':file_hash(val.path/'metadata.json'),'test_hash':file_hash(test.path/'metadata.json')}
        ytest=np.asarray(test.cpu['targets']).reshape(-1,len(TARGETS),4)[tids]
        valid_test=np.asarray(test.cpu['age'])>=25
        # These row assignments do not depend on the model. Reusing their
        # ordered indices preserves the exact samples and tie handling while
        # avoiding a full-array modulo/mask pass for every model and tile.
        row_family=test.family[tids%test.n]
        row_tile=test.tile[tids%test.n]
        domain_masks={'seen':row_family<6,'heldout':row_family==6}
        tile_indices={int(tile):np.flatnonzero(row_tile==tile) for tile in np.unique(row_tile)}
        family_indices={family:np.flatnonzero(row_family==family) for family in range(7)}
        event_context={}
        if any(entry[2]==7 for entry in bank.entries):
            # Ground-truth grouping is identical for all models and alarm
            # budgets. Cache it without changing the grouping or valid masks.
            for domain,families in [('seen',[0,1,2,3,4,5]),('heldout',[6])]:
                for stage,data,valid in [('verification',val,valid_val),('test',test,valid_test)]:
                    mask=valid & np.isin(data.family,families)[None,:]
                    if stage=='verification':mask &= np.arange(data.n)[None,:]%4==3
                    truth,starts,_=truth_event_ids(np.asarray(data.cpu['targets'])[:,:,7],mask,THRESHOLDS[7])
                    event_context[stage,domain]=(mask,truth,starts)
        for i,(seed,method,target) in enumerate(bank.entries):
            score=tpred['score'][i];point=tpred['point'][i];prob=apply_cal(score,calibrators[i])
            record={'seed':seed,'method':method,'target':TARGETS[target],'calibration':calibrators[i],
                    'domains':{},'best_step':saved['best_steps'][i]}
            for domain,families in [('seen',[0,1,2,3,4,5]),('heldout',[6])]:
                mask=domain_masks[domain]
                record['domains'][domain]=frame_stats(ytest[mask,target],score[mask],prob[mask],point[mask],target)
                tile_rows={}
                for tile,indices in tile_indices.items():
                    if (tile%14)//2 not in families:continue
                    tile_rows[str(tile)]=average_precision(ytest[indices,target]>THRESHOLDS[target],score[indices])
                record['domains'][domain]['tile_AP']=tile_rows
            if target in (5,6,7):
                record['terrain_families']={}
                for family,name in enumerate(test.meta['terrain']['families']):
                    indices=family_indices[family]
                    yy,pp=ytest[indices,target],point[indices]
                    tiles={key:value for domain in record['domains'].values() for key,value in domain['tile_AP'].items()
                           if (int(key)%14)//2==family and value is not None}
                    row={'foot_intervals':int(yy.size),'prevalence':float((yy>THRESHOLDS[target]).mean()),
                         'target_mean':float(yy.mean()),'mean_tile_AP':float(np.mean(list(tiles.values()))) if tiles else None,
                         'mean_tile_AP_note':'Equal-weight tile AP, not pooled family AP'}
                    if np.isfinite(pp).all():
                        absolute=abs(np.maximum(pp,0)-yy);active=yy>THRESHOLDS[target]
                        row.update(MAE=float(absolute.mean()),RMSE=float(np.sqrt(np.mean(absolute**2))),
                                   active_MAE=float(absolute[active].mean()) if active.any() else None)
                    record['terrain_families'][name]=row
            # Alarm endpoint is preregistered impact20 only; all other targets use frame metrics.
            if target==7:
                pv=dense_stream(vpred['score'][i],vids,val.t,val.n)
                pt=dense_stream(score,tids,test.t,test.n)
                selected=selections[str(i)]
                record['alarms']={}
                for budget,chosen in selected.items():
                    threshold=chosen['score_threshold']
                    row={'selection':chosen,'verification':{},'test':{}}
                    for domain,families in [('seen',[0,1,2,3,4,5]),('heldout',[6])]:
                        for stage,data,probs,valid in [('verification',val,pv,valid_val),('test',test,pt,valid_test)]:
                            mask,truth,starts=event_context[stage,domain]
                            metrics=event_metrics(probs,threshold,mask,truth,starts,task=task)
                            metrics['score_threshold']=metrics.pop('probability_threshold')
                            tiles={}
                            for tile in np.unique(data.tile[np.isin(data.family,families)]):
                                envmask=data.tile==tile
                                tiles[str(int(tile))]=event_metrics(probs[:,envmask],threshold,mask[:,envmask],truth[:,envmask],starts[:,envmask],task=task)
                            metrics['tiles']=tiles
                            row[stage][domain]=metrics
                    record['alarms'][budget]=row
                if np.isfinite(point).all():
                    physical=dense_stream(point,tids,test.t,test.n)
                    record['physical_threshold_alarms']={}
                    for domain,families in [('seen',[0,1,2,3,4,5]),('heldout',[6])]:
                        mask,truth,starts=event_context['test',domain]
                        metric=event_metrics(physical,THRESHOLDS[target],mask,truth,starts,task=task)
                        metric['physical_threshold_n_per_s']=metric.pop('probability_threshold')
                        record['physical_threshold_alarms'][domain]=metric
            result['models'].append(record)
            print('EVAL_MODEL',args.rep,args.input,task,seed,method,TARGETS[target],flush=True)
        result['derived_loading20_from_force20']=[]
        for i,(seed,method,target) in enumerate(bank.entries):
            if target!=5 or not np.isfinite(tpred['point'][i]).all():continue
            predicted_force=dense_stream(np.maximum(tpred['point'][i],0),tids,test.t,test.n)
            derived=np.maximum(predicted_force-np.concatenate((np.zeros_like(predicted_force[:1]),predicted_force[:-1])),0)/.02
            true_force=np.asarray(test.cpu['targets'])[:,:,5]
            oracle=np.maximum(true_force-np.concatenate((np.zeros_like(true_force[:1]),true_force[:-1])),0)/.02
            valid=np.asarray(test.cpu['age'])>=26
            true_loading=np.asarray(test.cpu['targets'])[:,:,6]
            if np.max(abs(oracle[valid]-true_loading[valid]))>.02:
                raise AssertionError('Force20/difference20 target identity failed')
            row={'seed':seed,'method':method,'domains':{}}
            direct_index=next((j for j,e in enumerate(bank.entries) if e==(seed,method,6)),None)
            direct=dense_stream(tpred['point'][direct_index],tids,test.t,test.n) if direct_index is not None else None
            for domain,families in [('seen',[0,1,2,3,4,5]),('heldout',[6])]:
                mask=valid & np.isin(test.family,families)[None]
                row['domains'][domain]={'derived':continuous(true_loading[mask],derived[mask],6),
                                         'direct':continuous(true_loading[mask],direct[mask],6) if direct is not None else {}}
            result['derived_loading20_from_force20'].append(row)
        # Write once after all models; unchanged failed executions may rerun.
        serialized=json.dumps(result,indent=2,allow_nan=False)
        temporary=output.with_suffix('.json.tmp')
        temporary.write_text(serialized+'\n')
        temporary.replace(output)
        del bank
        torch.cuda.empty_cache()
        print('FORCE_EVALUATION_COMPLETE',str(output),flush=True)


def run(args):
    logs=Path('logs/impact_force_study_v1')
    if not args.calibration_only and not (logs/'ALL_MODELS_LOCKED.json').exists():
        raise ValueError('All model arms must be locked before final evaluation')
    runroot=logs/args.group/f'rep{args.rep}'/args.input
    # A serial driver and the optional parallel driver may submit the same run.
    # Exactly one process evaluates it. At most two runs allocate score streams,
    # keeping host-memory peaks below the available RAM on this workstation.
    with (runroot/'.evaluation.lock').open('a+') as run_lock:
        fcntl.flock(run_lock,fcntl.LOCK_EX)
        slot=None
        while slot is None:
            for number in range(2):
                candidate=(logs/f'.evaluation_slot{number}').open('a+')
                try:
                    fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    slot=candidate;break
                except BlockingIOError:
                    candidate.close()
            if slot is None:time.sleep(5)
        try:evaluate(args)
        finally:slot.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rep',type=int,required=True);p.add_argument('--input',default='I0')
    p.add_argument('--group',default='main');p.add_argument('--tasks',nargs='+',default=['detection','forecast'])
    p.add_argument('--calibration-only',action='store_true',help='Fit validation roles1/2 only; never prepare or open final test')
    run(p.parse_args())
