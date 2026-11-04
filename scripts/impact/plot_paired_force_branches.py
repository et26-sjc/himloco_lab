"""Illustrate fixed group0, choosing its first-loaded foot by a stated rule."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from himloco_lab.impact.dataset import file_hash


root=Path('datasets/go1_impact/force_study_v1/paired_branches')
report=json.loads((root/'report.json').read_text())
if file_hash(root/'branches.npz')!=report['sha256']:raise ValueError('Branch archive changed')
data=np.load(root/'branches.npz')
prefix=data['history'][:,:32]
dt=report['physics_dt'];dec=round(.02/dt)
raw=np.linalg.norm(data['forces'][:,:32].transpose(1,0,2,3,4).reshape(32,-1,4,3),axis=-1)
lag=round(.02/dt);length=raw.shape[1]
pad=np.pad(raw,((0,0),(2*lag,0),(0,0)))
force=sum(pad[:,2*lag-k:2*lag-k+length] for k in range(lag))/lag
old=np.pad(force,((0,0),(lag,0),(0,0)))[:,:length]
rate=np.maximum(force-old,0)/.02
onset_by_foot=np.argmax(raw>1,axis=1)
foot=int(np.unravel_index(np.argmin(onset_by_foot),onset_by_foot.shape)[1])
onset=onset_by_foot.min(1)
cutoff=int(onset.min()//dec)*.02
time=(np.arange(length)+1)*dt
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
delta=abs(prefix-prefix[:,:1]).max(axis=(1,2))
axes[0].semilogy(np.arange(len(prefix))*.02,delta+1e-9,color='#657b83')
axes[0].set(ylabel='Largest actor-history difference + 1e-9',title='Observable prefix, all 32 heights')
for branch,color in zip((0,10,21,31),('#268bd2','#859900','#b58900','#cb4b16')):
    label=f'{float(data["height"][branch])*100:.1f} cm'
    axes[1].plot(time,force[branch,:,foot],color=color,label=label)
    axes[2].plot(time,rate[branch,:,foot],color=color,label=label)
axes[1].set(ylabel='F20 (N)',title=f'Foot{foot} finite-time normal load')
axes[2].set(ylabel='L20 (N/s)',title=f'Foot{foot} finite-time loading-rate')
for ax in axes:
    ax.axvline(cutoff,color='black',linestyle='--',linewidth=1)
    ax.axvspan(cutoff,cutoff+.02,color='#268bd2',alpha=.12)
    ax.set(xlim=(0,.4),xlabel='Time since release (s)');ax.grid(alpha=.2)
axes[2].legend(title='Hidden patch height',fontsize=8,title_fontsize=8)
fig.suptitle('Fixed group0, first-loaded foot; dashed line = causal cutoff, shaded = next 20 ms',fontsize=10)
out=Path('artifacts/impact/force_study_v1')
fig.savefig(out/'paired_branches.png',dpi=180);fig.savefig(out/'paired_branches.pdf')
print('PAIRED_BRANCH_PLOT_COMPLETE',cutoff)
