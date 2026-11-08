"""Compact secondary comparisons and standalone research figures."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


ART=Path('artifacts/impact/causal_stream_v1')


def interval(values):
    x=np.asarray(values);half=student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return dict(mean=float(x.mean()),lower=float(x.mean()-half),upper=float(x.mean()+half))


def main():
    result=json.loads((ART/'summary.json').read_text());g=result['groups']
    contrasts={}
    for domain in ('test_seen','test_heldout','high_speed_seen','high_speed_heldout'):
        for delay in (0,1):
            pairs=[('state_gate_bce','oracle_gate_bce'),('state_gate_bce','direct_warm_bce'),
                   ('direct_warm_bce','direct_warm_mse'),('direct_warm_bce','direct_scratch_bce'),
                   ('pd_gate_bce','state_gate_bce'),('state_gate_adapted_bce','direct_warm_bce'),
                   ('state_gate_adapted_bce','state_gate_bce'),('state_gate_adapted_bce','oracle_gate_adapted_bce')]
            for a,b in pairs:
                x=np.asarray(g[f'{domain}/{delay}/{a}']['recall']['replication_values'])
                y=np.asarray(g[f'{domain}/{delay}/{b}']['recall']['replication_values'])
                contrasts[f'{domain}/{delay}/{a}-{b}']=interval(x-y)
    (ART/'secondary_contrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n')
    with (ART/'operating_results.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['domain','delay_ms','method','recall','recall_ci_low','recall_ci_high',
                                                  'far_per_robot_min','far_upper_conservative','phase_timely_recall'])
        for key,v in g.items():
            domain,delay,method=key.split('/')
            writer.writerow([domain,int(delay)*20,method,v['recall']['mean'],v['recall']['lower'],v['recall']['upper'],
                             v['far']['mean'],v['far_upper_conservative'],v.get('phase_timely_recall',{}).get('mean','')])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['oracle_gate_bce','state_gate_bce','state_gate_adapted_bce','direct_warm_bce','direct_warm_mse','direct_scratch_bce','pd_gate_bce']
    labels=['Oracle + frozen','State + frozen','State + adapted','Direct warm BCE','Direct warm MSE','Direct fresh BCE','PD + frozen']
    colors=['#7c7c7c','#2979b9','#55a868','#c44e52','#8172b2','#ccb974','#64b5cd']
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True,layout='constrained')
    domains=['test_seen','test_heldout','high_speed_seen','high_speed_heldout']
    titles=['Normal speed / seen families','Normal speed / held-out boxes','High speed / seen families','High speed / held-out boxes']
    for ax,domain,title in zip(axes.flat,domains,titles):
        for i,(method,color) in enumerate(zip(methods,colors)):
            v=g[f'{domain}/0/{method}'];r=v['recall'];mean=r['mean']
            ax.bar(i,mean,color=color,alpha=.85)
            ax.errorbar(i,mean,yerr=[[mean-r['lower']],[r['upper']-mean]],fmt='none',color='black',capsize=3)
            far=v['far']['mean'];upper=v['far_upper_conservative']
            ax.text(i,.025,f'FAR\n{far:.2f}',ha='center',va='bottom',fontsize=8,color='white' if mean>.15 else 'black',fontweight='bold')
            if upper>1:ax.text(i,min(mean+.055,1.04),'*',ha='center',fontsize=15)
        ax.set_title(title);ax.set_xticks(range(len(methods)),labels,rotation=25,ha='right',fontsize=8)
        ax.set_ylim(0,1.12);ax.set_ylabel('Strong landing recall within 40 ms');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Causal continuous monitoring at validation-locked operating points\n95% data-seed CI; FAR = alarms / robot-minute; * FAR upper > 1/min',fontsize=12)
    fig.savefig(ART/'continuous_monitoring.png',dpi=180);fig.savefig(ART/'continuous_monitoring.svg')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for method,label,color in zip(methods[1:5],labels[1:5],colors[1:5]):
        means=[g[f'{domain}/{delay}/{method}']['recall']['mean'] for delay in (0,1) for domain in domains]
        base=np.asarray(means[:4]);lag=np.asarray(means[4:])
        ax.plot(np.arange(4),lag-base,'o-',label=label,color=color)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),['Normal / seen','Normal / boxes','Fast / seen','Fast / boxes'])
    ax.set_ylabel('Recall change with 20 ms packet delay');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(ART/'packet_delay.png',dpi=180);fig.savefig(ART/'packet_delay.svg');plt.close(fig)
    print('STREAM_ANALYSIS_WRITTEN',flush=True)


if __name__=='__main__':main()
