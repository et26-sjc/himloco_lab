"""Train independent stream heads; never read final tests."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from himloco_lab.impact.causal_stream import StreamData,StreamBank,PDBank,pd_features
from himloco_lab.impact.dataset import file_hash
from train_force_study import IndependentAdam


def frozen_bank(rep,device='cuda'):
    path=Path(f'logs/impact_observability_v1/rep{rep}/full/models.pt')
    old=torch.load(path,map_location='cpu',weights_only=False)
    entries=[(s,m) for s in (1,2) for m in ('bce','mse')]
    bank=StreamBank(entries)
    with torch.no_grad():
        for i,(s,m) in enumerate(entries):
            j=[tuple(x) for x in old['entries']].index((s,m,3))
            for key in ('w1','b1','w2','b2','w3','b3'):getattr(bank,key)[i].copy_(old['state'][key][j])
            bank.mean[i].copy_(old['state']['mean']);bank.std[i].copy_(old['state']['std'])
    return bank.to(device),file_hash(path)


def save_bank(bank,path,**metadata):
    torch.save(dict(state={k:v.detach().cpu() for k,v in bank.state_dict().items()},
                    entries=bank.entries,**metadata),path)


def train(rep,steps=None,development=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    spec=json.loads(Path('configs/impact/causal_stream_v1.json').read_text())
    name='development' if development else f'rep{rep}'
    root=Path('datasets/go1_impact/causal_stream_v1')/name
    source=root/'cache' if development else root/'train/cache'
    validation=source if development else root/'validation/cache'
    dest=Path('logs/impact_causal_stream_v1')/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'training.json').exists():return
    data=StreamData(source);val=StreamData(validation)
    entries=[(s,k) for s in spec['optimization_seeds'] for k in spec['new_heads']]
    bank=StreamBank(entries).cuda();pd=PDBank(spec['optimization_seeds']).cuda()
    frozen,old_hash=frozen_bank(rep)
    ids=torch.as_tensor(data.indices(),device='cuda');rng=torch.Generator(device='cuda').manual_seed(7300+rep)
    normids=ids[torch.randint(len(ids),(32768,),device='cuda',generator=rng)]
    with torch.no_grad():
        norm=data.frames(normids);x=norm.flatten(1)
        bank.mean.copy_(x.mean(0));bank.std.copy_(x.std(0).clamp_min(.01))
        p=pd_features(norm);pd.mean.copy_(p.mean(0));pd.std.copy_(p.std(0).clamp_min(.01))
        for i,(seed,kind) in enumerate(entries):
            if 'warm' not in kind:continue
            method=kind.rsplit('_',1)[-1];j=frozen.entries.index((seed,method))
            for key in ('w1','b1','w2','b2','w3','b3','mean','std'):
                getattr(bank,key)[i].copy_(getattr(frozen,key)[j])
        raw=bank(x[:128]);reference=frozen(x[:128]);init_error=0.
        for i,(seed,kind) in enumerate(entries):
            if 'warm' in kind:
                j=frozen.entries.index((seed,kind.rsplit('_',1)[-1]))
                init_error=max(init_error,float(abs(raw[i]-reference[j]).max()))
        assert init_error<1e-4,init_error
    save_bank(frozen,dest/'frozen.pt',source_sha256=old_hash)
    configs=[{'lr':spec['learning_rate'],'wd':0.} for _ in entries]
    optimizer=IndependentAdam(bank,configs)
    pdopt=IndependentAdam(pd,[{'lr':spec['learning_rate'],'wd':0.} for _ in pd.entries])
    valid_ids=val.indices(role=2)
    monitor=torch.as_tensor(np.random.default_rng(8300+rep).choice(valid_ids,min(16384,len(valid_ids)),replace=False),device='cuda')
    curve=[];start=time.perf_counter();total=steps or spec['updates']
    for step in range(1,total+1):
        part=ids[torch.randint(len(ids),(spec['batch_size'],),device='cuda',generator=rng)]
        frames=data.frames(part);target=data.targets(part)
        bank.zero_grad(set_to_none=True);pd.zero_grad(set_to_none=True)
        loss=bank.objective(bank(frames.flatten(1)),target)
        pd_loss=F.binary_cross_entropy_with_logits(pd(frames),target[0][None].expand(len(pd.entries),-1,-1),reduction='none').mean((1,2))
        (loss.sum()+pd_loss.sum()).backward()
        multiplier=1.
        for milestone,mult in spec['lr_milestones']:
            if step>=milestone:multiplier=mult
        optimizer.step(multiplier);pdopt.step(multiplier)
        if step==1 or step%spec['monitor_every']==0 or step==total:
            with torch.no_grad():
                monitor_loss=bank.objective(bank(val.frames(monitor).flatten(1)),val.targets(monitor)).cpu().tolist()
            assert np.isfinite(monitor_loss).all()
            curve.append(dict(step=step,monitor_loss=monitor_loss,seconds=time.perf_counter()-start))
            (dest/'curve.json').write_text(json.dumps(curve,indent=2)+'\n')
            print('STREAM_TRAIN',name,step,round(curve[-1]['seconds'],1),flush=True)
    save_bank(bank,dest/'models.pt',updates=total);save_bank(pd,dest/'pd.pt',updates=total)
    report=dict(status='completed',updates=total,training_robot_intervals=len(ids),entries=entries,
                parameters_per_MLP=sum(p.numel() for p in bank.parameters())//len(entries),
                parameters_per_PD=sum(p.numel() for p in pd.parameters())//len(pd.entries),
                warm_start_max_error=init_error,optimizer='fresh independent Adam for every model',
                old_source_sha256=old_hash,model_sha256={p.name:file_hash(p) for p in dest.glob('*.pt')},
                protocol_sha256=file_hash('configs/impact/causal_stream_v1.json'),seconds=time.perf_counter()-start,
                development_only=development)
    (dest/'training.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rep',type=int,default=1)
    p.add_argument('--steps',type=int);p.add_argument('--development',action='store_true')
    a=p.parse_args();train(a.rep,a.steps,a.development)
