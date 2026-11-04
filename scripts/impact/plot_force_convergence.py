"""Plot checkpoint-selection curves; these are validation, never final test."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
for ax,target,scale,label in zip(axes,(5,6),(40.,2000.),('F20 (N)','L20 (N/s)')):
    for task,color in [('detection','#268bd2'),('forecast','#cb4b16')]:
        values=[]
        for rep in range(1,6):
            path=Path(f'logs/impact_force_study_v1/main/rep{rep}/I0/{task}')
            report=json.loads((path/'report.json').read_text());curve=json.loads((path/'curve.json').read_text())
            indices=[i for i,entry in enumerate(report['entries']) if entry[1]=='mse' and entry[2]==target]
            values.append([np.mean([np.sqrt(row['loss'][i])*scale for i in indices]) for row in curve])
        values=np.array(values);steps=np.array([row['step'] for row in curve])
        mean,sd=values.mean(0),values.std(0,ddof=1)
        ax.plot(steps,mean,color=color,label=task)
        ax.fill_between(steps,np.maximum(mean-sd,1e-9),mean+sd,color=color,alpha=.15)
    ax.set(xlabel='Independent Adam updates',ylabel=f'Checkpoint-selection RMSE, {label}',yscale='log')
    ax.set_yticks([1,2,5,10,20,50] if target==5 else [50,100,200,500])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(alpha=.2);ax.legend()
fig.suptitle('I0 MSE: five data seeds, initialization pair averaged; shading = SD',fontsize=10)
out=Path('artifacts/impact/force_study_v1')
fig.savefig(out/'regression_convergence.png',dpi=180);fig.savefig(out/'regression_convergence.pdf')
print('REGRESSION_CONVERGENCE_PLOT_COMPLETE')
