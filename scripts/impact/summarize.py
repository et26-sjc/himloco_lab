"""Produce compact reviewable evidence, tables and standalone figures."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from himloco_lab.impact.dataset import file_hash


def save_figure(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def summarize(args):
    args.output.mkdir(parents=True,exist_ok=args.refresh)
    spec=json.loads((args.training/'spec.json').read_text())
    results={task:json.loads((args.evaluation/f'{task}_results.json').read_text()) for task in spec['tasks']}
    if json.loads((args.evaluation/'report.json').read_text())['status']!='completed':
        raise ValueError('Incomplete evaluation')
    for source in [args.training/'report.json',args.evaluation/'report.json',args.diagnostics]:
        name=('training_report.json' if source.parent==args.training else
              'evaluation_report.json' if source.parent==args.evaluation else 'physical_diagnostics.json')
        shutil.copyfile(source,args.output/name)
    shutil.copyfile(args.training/'spec.json',args.output/'spec.json')
    fields=['per_foot_AP','Brier','prevalence','touchdown_conditioned_AP','touchdown_conditioned_prevalence',
            'MAE_n','RMSE_n']
    rows=[]
    for task,result in results.items():
        shutil.copyfile(args.evaluation/f'{task}_results.json',args.output/f'{task}_results.json')
        shutil.copyfile(args.evaluation/f'{task}_locked_thresholds.json',args.output/f'{task}_locked_thresholds.json')
        shutil.copyfile(args.training/task/'learning_curve.json',args.output/f'{task}_learning_curve.json')
        for domain in ['all_seen','unseen_boxes']:
            for method in spec['models']:
                selected=[r['domains'][domain] for r in result['models'] if r['method']==method]
                row={'task':task,'domain':domain,'method':method}
                for field in fields:
                    vals=[r[field] for r in selected if r.get(field) is not None]
                    if vals:
                        row[field+'_mean']=float(np.mean(vals));row[field+'_std']=float(np.std(vals,ddof=1))
                for field in ['event_recall','false_alarms_per_robot_minute','strict_precontact_recall']:
                    vals=[r['events_at_validation_operating_point'][field] for r in selected
                          if r['events_at_validation_operating_point'].get(field) is not None]
                    if vals:
                        row[field+'_mean']=float(np.mean(vals));row[field+'_std']=float(np.std(vals,ddof=1))
                rows.append(row)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (args.output/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    labels=spec['models']; ypos=np.arange(len(labels))
    for ax,task in zip(axes,spec['tasks']):
        for shift,domain,color in [(-.16,'all_seen','#27649b'),(.16,'unseen_boxes','#c06928')]:
            group=[next(r for r in rows if (r['task'],r['domain'],r['method'])==(task,domain,m)) for m in labels]
            ax.barh(ypos+shift,[r['per_foot_AP_mean'] for r in group],height=.3,color=color,
                    xerr=[r['per_foot_AP_std'] for r in group],label=domain,capsize=2)
            ax.axvline(group[0]['prevalence_mean'],color=color,ls=':',alpha=.6)
        ax.set_title(task);ax.set_xlabel('Natural-distribution per-foot AP');ax.set_xlim(0,1)
        ax.set_yticks(ypos,labels);ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.5,1.01))
    fig.suptitle('5 initializations; independent test terrain seed; dotted lines = prevalence',y=1.06)
    fig.tight_layout();save_figure(fig,args.output/'ranking')
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for task,axs in zip(spec['tasks'],axes):
        curve=json.loads((args.training/task/'learning_curve.json').read_text())
        steps=np.array([r['step'] for r in curve]);loss=np.array([r['selection_losses'] for r in curve])
        for ax,method in zip(axs,['bce','mse','ordinal']):
            j=labels.index(method);v=loss[:,j::len(labels)]
            ax.plot(steps,v.mean(1),color='#27649b');ax.fill_between(steps,v.min(1),v.max(1),alpha=.15)
            ax.set_title(f'{task}: {method}');ax.set_xlabel('Optimizer steps');ax.set_ylabel('Selection objective')
            ax.set_yscale('log');ax.grid(alpha=.15)
    fig.tight_layout();save_figure(fig,args.output/'learning_curves')
    physical=json.loads(args.diagnostics.read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,key in zip(axes,['raw_rise_n','smoothed_rise_n','first_40ms_integral_ns']):
        x=np.array(physical['runs'][0]['per_foot'][key]);y=np.array(physical['runs'][1]['per_foot'][key])
        heights=np.array(physical['runs'][0]['initial_root_state'])[:,2]
        ax.scatter(x.ravel(),y.ravel(),c=np.repeat(heights,4),cmap='viridis',s=9,alpha=.6)
        lim=max(x.max(),y.max())*1.04;ax.plot([0,lim],[0,lim],ls='--',color='gray');ax.set_xlim(0,lim);ax.set_ylim(0,lim)
        relative=physical['paired_comparison'][key]['median_symmetric_relative_difference']
        ax.set_title(f'{key}\nmedian relative difference {relative:.1%}');ax.set_xlabel('200 Hz');ax.set_ylabel('400 Hz')
    fig.text(.5,-.02,'Color: initial base height, purple = lower, yellow = higher. Each robot contributes four feet.',ha='center',fontsize=9)
    fig.tight_layout();save_figure(fig,args.output/'physics_dt_diagnostics')
    evidence={'status':'completed','scope':spec['stage'],'source_training':str(args.training.resolve()),
              'source_evaluation':str(args.evaluation.resolve()),'optimization_initializations':5,
              'independent_training_data_seeds':1,'spec_sha256':file_hash(args.training/'spec.json'),
              'summary':'summary.json','figures':['ranking.png','learning_curves.png','physics_dt_diagnostics.png'],
              'files_sha256':{p.name:file_hash(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='evidence.json'}}
    (args.output/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--diagnostics',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refresh',action='store_true',help='Regenerate review artifacts in an existing output directory')
    summarize(p.parse_args())
