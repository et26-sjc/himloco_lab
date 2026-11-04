"""Export paired-replication results with fixed endpoint and learning curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('artifacts/impact/continual_replay_v1/confirmation')
    summary=json.loads((root/'summary.json').read_text())
    methods=['freeze','stationary_A','finetune','fifo','uniform','binary','severity','severity_unweighted','all_past']
    labels=['Freeze','Stationary A','Finetune','FIFO','Uniform','Binary','Severity (w)','Severity (no w)','All past']
    colors=['#909090','#c4c4c4','#d55e00','#e69f00','#0072b2','#56b4e9','#009e73','#cc79a7','#443388']
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for ax,key,title in zip(axes[0],['old_macro_AP','C_AP'],['Final AP on old domains A/B','Final AP on new domain C']):
        means=[summary['methods'][m][key]['mean'] for m in methods]
        low=[v-summary['methods'][m][key]['CI95'][0] for m,v in zip(methods,means)]
        high=[summary['methods'][m][key]['CI95'][1]-v for m,v in zip(methods,means)]
        ax.bar(np.arange(len(methods)),means,color=colors,yerr=np.array([low,high]),capsize=3)
        ax.set_xticks(np.arange(len(methods)),labels,rotation=45,ha='right');ax.set_title(title);ax.set_ylim(min(means)-.025,1.)
        ax.set_ylabel('AP (mean and 95% CI over 5 collections)')
    reps=[json.loads((root/f'rep{r}'/'training.json').read_text()) for r in range(1,6)]
    for domain,ax in zip(['A','C'],axes[1]):
        for method,color in zip(methods,colors):
            if method in ('freeze','fifo','severity_unweighted','binary'):continue
            curves=[]
            for rep in reps:
                indices=[i for i,(_,m) in enumerate(rep['names']) if m==method]
                curves.append([np.mean([node['metrics'][i][domain]['AP'] for i in indices]) for node in rep['history']])
            curves=np.array(curves);x=np.arange(11)*4000
            ax.plot(x,curves.mean(0),label=labels[methods.index(method)],color=color)
            ax.fill_between(x,curves.min(0),curves.max(0),color=color,alpha=.1)
        ax.axvline(20000,color='black',ls=':',lw=1);ax.set_title(f'Fixed monitor: {domain}');ax.set_xlabel('Adaptation updates (B then C)');ax.set_ylabel('AP')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Go1 impact detection: fixed policy, 4096 envs; 5 collections x 2 initializations\nEndpoint bars: final held-out test; curves: fixed monitor, range across collections')
    for suffix in ('png','svg','pdf'):fig.savefig(root/f'continual_results.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
