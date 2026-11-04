"""Aggregate independent collection seeds, paired comparisons and scientific figures."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from himloco_lab.impact.dataset import file_hash


ROOT=Path('artifacts/impact/force_study_v1')


def interval(values):
    a=np.asarray(values,float)
    mean=float(a.mean());sd=float(a.std(ddof=1)) if len(a)>1 else 0.
    half=float(student_t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a))) if len(a)>1 else None
    return {'mean':mean,'sd_across_data_seeds':sd,'n':len(a),'CI95':[mean-half,mean+half] if half is not None else None,'per_replication':a.tolist()}


def model(result,method,seed,target='impact20'):
    return next(r for r in result['models'] if r['method']==method and r['seed']==seed and r['target']==target)


def clustered_alarm_intervals(runs,method,stage,domain,budget,draws=4000):
    """Resample data replications, then shared terrain tiles within replication.

    Two initializations share data and exposure. Average their counts before
    resampling; never count the same robot minutes twice as independent data.
    This quantifies variation in these sampled geometries, not new terrain types.
    """
    clusters=[]
    for run in runs:
        records=[model(run,method,seed)['alarms'][str(budget)][stage][domain]['tiles'] for seed in (1,2)]
        keys=sorted(records[0])
        if keys!=sorted(records[1]):raise ValueError('Initialization tile mismatch')
        rows=np.asarray([[np.mean([record[key][field] for record in records])
                          for field in ('false_alarms','robot_minutes','matched_events','events')] for key in keys])
        families=np.array([(int(key)%14)//2 for key in keys])
        clusters.append((rows,families))
    rng=np.random.default_rng(3781)
    estimates=np.zeros((draws,2))
    selected_reps=rng.integers(len(clusters),size=(draws,len(clusters)))
    # Vectorize across bootstrap draws. Repeated copies of a replication get
    # independent tile resamples, preserving the hierarchical sampling law.
    for position in range(len(clusters)):
        for rep,(rows,families) in enumerate(clusters):
            selected=np.flatnonzero(selected_reps[:,position]==rep)
            counts=np.zeros((len(selected),4))
            for family in np.unique(families):
                members=np.flatnonzero(families==family)
                samples=rng.choice(members,size=(len(selected),len(members)),replace=True)
                counts+=rows[samples].sum(1)
            estimates[selected,0]+=counts[:,0]/counts[:,1]/len(clusters)
            estimates[selected,1]+=counts[:,2]/counts[:,3]/len(clusters)
    rates=[];recalls=[]
    for rows,_ in clusters:
        counts=rows.sum(0);rates.append(counts[0]/counts[1]);recalls.append(counts[2]/counts[3])
    ci=np.quantile(estimates,[.025,.975],axis=0)
    return {'FAR':interval(rates),'event_recall':interval(recalls),
            'hierarchical_percentile_CI95':{'FAR':ci[:,0].tolist(),'event_recall':ci[:,1].tolist()},
            'FAR_upper_CI_within_budget':bool(ci[1,0]<=budget),'bootstrap_draws':draws}


def training_convergence():
    rows=[]
    for path in sorted(Path('logs/impact_force_study_v1').glob('*/*/*/*/report.json')):
        if path.parts[-5] not in ('main','alternatives'):continue
        report=json.loads(path.read_text())
        curve=json.loads(path.with_name('curve.json').read_text())
        before=np.asarray(next(c['loss'] for c in curve if c['step']==50000))
        final=np.asarray(curve[-1]['loss'])
        for i,(seed,method,target) in enumerate(report['entries']):
            rows.append({'path':str(path.parent),'seed':seed,'method':method,'target_index':target,
                         'updates':report['updates'],'best_step':report['best_steps'][i],
                         'last_10000_relative_validation_improvement':float((before[i]-final[i])/max(abs(before[i]),1e-8)),
                         'unique_exposure_fraction':report['unique_exposed']/report['available']})
    return rows


def aggregate():
    files=sorted(ROOT.glob('*/*/*/*.json'))
    loaded=[]
    for path in files:
        d=json.loads(path.read_text())
        if d.get('status')=='completed' and 'models' in d:
            d['_group']=path.parts[-4];loaded.append(d)
    expected=10+15+10
    if len(loaded)!=expected:raise ValueError(f'Expected all {expected} task runs, found {len(loaded)}')
    groups={};flat=[];family_groups={}
    for d in loaded:
        for r in d['models']:
            for domain,v in r['domains'].items():
                base={'group':d['_group'],'rep':d['rep'],'input':d['input'],'task':d['task'],
                      'seed':r['seed'],'method':r['method'],'target':r['target'],'domain':domain}
                row={**base,'AP':v['AP'],'point_AP':v['point_AP'],'Brier':v['Brier'],'log_loss':v['log_loss'],'prevalence':v['prevalence']}
                if v['continuous']:
                    for key in ('MAE','RMSE','active_MAE','p95_absolute_error','active_p95_absolute_error','bias'):
                        row[key]=v['continuous']['nonnegative'][key]
                    for tolerance,value in v['continuous']['nonnegative']['within_tolerance'].items():
                        row['within_'+tolerance]=value
                    row['raw_negative_fraction']=v['continuous']['raw']['negative_fraction']
                    for subgroup,values in v['continuous']['nonnegative']['subgroups'].items():
                        for metric in ('MAE','bias','near_zero_prediction_fraction','prediction_median','target_median'):
                            if metric in values:row[subgroup+'_'+metric]=values[metric]
                    weak=v['continuous']['raw']['subgroups']['weak_resolved']
                    if weak['count']:row['weak_resolved_raw_negative_fraction']=weak['negative_fraction']
                if 'alarms' in r:
                    a=r['alarms']['1.0']['test'][domain]
                    row.update(event_recall=a['event_recall'],FAR=a['false_alarms_per_robot_minute'])
                flat.append(row)
                group=tuple(base[k] for k in ('group','input','task','method','target','domain'))
                groups.setdefault(group,[]).append(row)
            for family,v in r.get('terrain_families',{}).items():
                key=(d['_group'],d['input'],d['task'],r['method'],r['target'],family)
                family_groups.setdefault(key,[]).append({'rep':d['rep'],**v})
    summaries=[]
    for key,rows in groups.items():
        entry=dict(zip(('group','input','task','method','target','domain'),key));entry['metrics']={}
        fields=set().union(*(row.keys() for row in rows))-set(('group','rep','input','task','seed','method','target','domain'))
        for field in sorted(fields):
            per_rep=[np.mean([r[field] for r in rows if r['rep']==rep and r.get(field) is not None])
                     for rep in range(1,6) if any(r['rep']==rep and r.get(field) is not None for r in rows)]
            if per_rep:entry['metrics'][field]=interval(per_rep)
        summaries.append(entry)
    family_summaries=[]
    for key,rows in family_groups.items():
        entry=dict(zip(('group','input','task','method','target','family'),key));entry['metrics']={}
        for field in ('prevalence','target_mean','mean_tile_AP','MAE','RMSE','active_MAE'):
            values=[np.mean([r[field] for r in rows if r['rep']==rep and r.get(field) is not None])
                    for rep in range(1,6) if any(r['rep']==rep and r.get(field) is not None for r in rows)]
            if values:entry['metrics'][field]=interval(values)
        family_summaries.append(entry)
    comparisons=[];distribution_compression=[]
    for task in ('detection','forecast'):
        runs=sorted([r for r in loaded if r['_group']=='main' and r['input']=='I0' and r['task']==task],key=lambda d:d['rep'])
        for domain in ('seen','heldout'):
            values=[np.mean([model(run,'mixture',seed)['domains'][domain]['AP']-
                             model(run,'mixture',seed)['domains'][domain]['point_AP'] for seed in (1,2)]) for run in runs]
            distribution_compression.append({'task':task,'domain':domain,
                'contrast':'Same mixture model: survival-risk AP minus mean-point AP',**interval(values)})
            for a,b in [('bce','mse'),('brier','mse'),('brier','bce'),('mixture','bce'),('bce','validation_selected_regression')]:
                for endpoint in ('AP','event_recall'):
                    values=[]
                    for run in runs:
                        delta=[]
                        for seed in (1,2):
                            bb=run['validation_selected_regression'][str(seed)] if b=='validation_selected_regression' else b
                            ra,rb=model(run,a,seed),model(run,bb,seed)
                            if endpoint=='AP':va,vb=ra['domains'][domain]['AP'],rb['domains'][domain]['AP']
                            else:va,vb=ra['alarms']['1.0']['test'][domain]['event_recall'],rb['alarms']['1.0']['test'][domain]['event_recall']
                            delta.append(va-vb)
                        values.append(np.mean(delta))
                    margin=.01 if endpoint=='AP' else .03
                    comparison={'task':task,'domain':domain,'a':a,'b':b,'endpoint':endpoint,**interval(values)}
                    comparison['equivalence_margin']=margin
                    comparison['CI95_within_equivalence_margin']=bool(comparison['CI95'][0]>=-margin and comparison['CI95'][1]<=margin)
                    comparison['CI95_above_upper_equivalence_margin']=bool(comparison['CI95'][0]>margin)
                    if endpoint=='event_recall':
                        comparison['CI95_exceeds_minimum_event_gain']=bool(comparison['CI95'][0]>.03)
                    comparisons.append(comparison)
    alarms=[]
    for group in ('main','alternatives'):
        methods=('mse','brier','bce','mixture') if group=='main' else ('huber','logmse','cdf')
        for task in ('detection','forecast'):
            runs=sorted([r for r in loaded if r['_group']==group and r['input']=='I0' and r['task']==task],key=lambda d:d['rep'])
            for method in methods:
                for stage in ('verification','test'):
                    for domain in ('seen','heldout'):
                        for budget in (.1,1.,5.):
                            alarms.append({'group':group,'task':task,'method':method,'stage':stage,'domain':domain,'budget':budget,
                                           **clustered_alarm_intervals(runs,method,stage,domain,budget)})
    derived=[]
    for group in ('main','alternatives'):
        for task in ('detection','forecast'):
            runs=sorted([r for r in loaded if r['_group']==group and r['input']=='I0' and r['task']==task],key=lambda d:d['rep'])
            methods=sorted(set(row['method'] for row in runs[0]['derived_loading20_from_force20']))
            for method in methods:
                for domain in ('seen','heldout'):
                    entry={'group':group,'task':task,'method':method,'domain':domain,'metrics':{}}
                    for way in ('direct','derived'):
                        for metric in ('MAE','RMSE','active_MAE','p95_absolute_error'):
                            values=[np.mean([r['domains'][domain][way]['nonnegative'][metric]
                                             for r in run['derived_loading20_from_force20'] if r['method']==method]) for run in runs]
                            entry['metrics'][way+'_'+metric]=interval(values)
                    derived.append(entry)
    result={'status':'completed','independent_collection_seeds':5,'initializations_per_collection':2,
            'statistics':'Each initialization pair is averaged before t intervals across five data seeds; not ten independent data replications.',
            'summaries':summaries,'terrain_family_summaries':family_summaries,'paired_comparisons':comparisons,
            'distribution_mean_compression':distribution_compression,
            'alarm_uncertainty':alarms,
            'alarm_bootstrap':'Resample five collection seeds, then terrain tiles within family strata in each; average initialization counts before sampling. Conditional on the selected terrain families and frozen policy.',
            'derived_loading':derived,'training_convergence':training_convergence(),
            'input_files':{str(p):file_hash(p) for p in files}}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=sorted(set().union(*(r.keys() for r in flat)))
    with (ROOT/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    return result,loaded


def plot_physics():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    physics=json.loads((ROOT/'physics.json').read_text())
    x=[400,800,1600]
    keys=['dt5000_p4_v0__dt2500_p4_v0','dt2500_p4_v0__dt1250_p4_v0','dt1250_p4_v0__dt625_p4_v0']
    for ax,prefix,title in zip(axes,['force','loading'],['Force peak','Loading-rate peak']):
        for suffix,label in [('0ms','Raw solver sample'),('10ms','10 ms average/difference'),('20ms','20 ms average/difference'),('40ms','40 ms average/difference')]:
            y=[physics['comparisons'][k]['metrics'][prefix+'_'+suffix]['median_symmetric_relative_difference']*100 for k in keys]
            ax.plot(x,y,'o-',label=label)
        ax.set(xlabel='Finer simulation rate in adjacent pair (Hz)',ylabel='Median symmetric difference (%)',title=title)
        ax.set_xticks(x);ax.set_yscale('log');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.savefig(ROOT/'physical_time_scales.png',dpi=180);fig.savefig(ROOT/'physical_time_scales.pdf');plt.close(fig)


def plots(result,loaded):
    plot_physics()
    colors={'mse':'#657b83','brier':'#268bd2','bce':'#cb4b16','mixture':'#859900','huber':'#6c71c4','logmse':'#b58900','cdf':'#2aa198'}
    methods=['mse','brier','bce','mixture','huber','logmse','cdf']
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    for ax,task in zip(axes,['detection','forecast']):
        for j,method in enumerate(methods):
            for domain,x0 in [('seen',0),('heldout',1)]:
                item=next(r for r in result['summaries'] if r['input']=='I0' and r['task']==task and r['method']==method and r['target']=='impact20' and r['domain']==domain)
                v=item['metrics']['AP'];x0=x0+(j-3)*.105
                ax.bar(x0,v['mean'],width=.095,color=colors[method],label=method if domain=='seen' else None,
                       yerr=v['sd_across_data_seeds'],capsize=2)
        ax.set(xticks=[0,1],xticklabels=['New instances','Held-out blocks'],ylabel='Natural-prevalence AP',title=task,ylim=(0,1))
    axes[1].legend(ncol=2,fontsize=8)
    fig.suptitle('Mean ± SD across five independent data collections',fontsize=10)
    fig.savefig(ROOT/'event_ranking.png',dpi=180);fig.savefig(ROOT/'event_ranking.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    for ax,target,unit in zip(axes,['force20','loading20'],['N','N/s']):
        for j,domain in enumerate(['seen','heldout']):
            means=[];sds=[]
            for kind in ['I0','I0clean','I1','I2']:
                item=next(r for r in result['summaries'] if r['input']==kind and r['task']=='detection' and r['method']=='mse' and r['target']==target and r['domain']==domain)
                means.append(item['metrics']['RMSE']['mean']);sds.append(item['metrics']['RMSE']['sd_across_data_seeds'])
            ax.errorbar(np.arange(4),means,yerr=sds,marker='o',label=domain,capsize=3)
        ax.set(xticks=np.arange(4),xticklabels=['50Hz noisy','50Hz clean','200Hz clean','200Hz + IMU/effort'],ylabel=f'RMSE ({unit})',title=target)
        ax.tick_params(axis='x',labelrotation=12);ax.grid(alpha=.2);ax.legend()
    fig.savefig(ROOT/'input_information.png',dpi=180);fig.savefig(ROOT/'input_information.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.1),constrained_layout=True)
    for ax,(task,domain) in zip(axes,[('detection','seen'),('detection','heldout'),('forecast','heldout')]):
        for method in methods:
            rows=[next(r for r in result['alarm_uncertainty'] if r['task']==task and r['method']==method
                       and r['domain']==domain and r['stage']=='test' and r['budget']==b) for b in (.1,1.,5.)]
            x=np.array([r['FAR']['mean'] for r in rows]);y=np.array([r['event_recall']['mean'] for r in rows])
            ax.plot(x,y,'o-',color=colors[method],label=method,markersize=4)
        ax.axvline(1,color='black',linestyle=':',alpha=.5)
        ax.set_xscale('symlog',linthresh=.1)
        ax.set_xlim(left=0)
        ax.set(xlabel='Actual test false alarms / robot-minute',ylabel='Event recall',
               title=f'{task}, {domain}',ylim=(0,1))
        ax.grid(alpha=.2)
    axes[-1].legend(ncol=2,fontsize=8)
    fig.suptitle('Three validation-selected budgets (0.1, 1, 5); thresholds unchanged on test',fontsize=10)
    fig.savefig(ROOT/'alarm_transfer.png',dpi=180);fig.savefig(ROOT/'alarm_transfer.pdf');plt.close(fig)


if __name__=='__main__':
    result,loaded=aggregate();plots(result,loaded)
    print('FORCE_STUDY_SUMMARY_COMPLETE',len(result['summaries']))
