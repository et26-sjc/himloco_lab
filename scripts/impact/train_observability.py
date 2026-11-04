"""Fixed-budget event severity fits. Never opens either final test role."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.observability import EventData,EventBank
from train_force_study import IndependentAdam


@torch.no_grad()
def predict(bank, data, ids, kind, moments, batch=2048):
    result=np.empty((len(bank.entries),len(ids)),np.float32)
    losses=np.zeros(len(bank.entries))
    for start in range(0,len(ids),batch):
        part=ids[start:start+batch]
        f=data.features(part,kind,moments)
        loss,score=bank.objective(bank(f),data.ev['foot'][part],data.ev['severity'][part],data.ev['valid_cutoff'][part])
        result[:,start:start+len(part)]=score.cpu().numpy()
        losses+=loss.cpu().numpy()*len(part)
    return result,losses/max(1,len(ids))


def train(rep, steps=None, development=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    spec=json.loads(Path('configs/impact/observability_v1.json').read_text())
    root=Path('datasets/go1_impact/observability_v1')
    name='development' if development else f'rep{rep}'
    source=root/'development/cache' if development else root/name/'train/cache'
    validation=source if development else root/name/'validation/cache'
    outroot=Path('logs/impact_observability_v1')/name
    outroot.mkdir(parents=True,exist_ok=True)
    data=EventData(source);val=EventData(validation)
    moments=data.hip_moments()
    (outroot/'hip_moments.json').write_text(json.dumps(moments,indent=2)+'\n')
    rng=np.random.default_rng(6100+rep)
    train_ids=data.ids();monitor=val.ids(role=2)
    monitor=rng.choice(monitor,min(len(monitor),spec['monitor_events']),replace=False)
    norm_pool=train_ids[data.cpu['valid_cutoff'][train_ids].all(1)]
    norm_ids=rng.choice(norm_pool,min(32768,len(norm_pool)),replace=False)
    indices=torch.tensor(train_ids,device='cuda')
    for kind in spec['inputs']:
        dest=outroot/kind;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'report.json').exists():continue
        entries=[(seed,method,h) for seed in spec['optimization_seeds'] for method in spec['methods'][kind] for h in range(6)]
        cfg=spec['inputs'][kind]
        bank=EventBank(cfg['dim'],entries,cfg['hidden']).cuda()
        with torch.no_grad():
            samples=data.features(norm_ids,kind,moments).reshape(-1,cfg['dim'])
            bank.mean.copy_(samples.mean(0));bank.std.copy_(samples.std(0).clamp_min(.01))
            del samples
        optimizer=IndependentAdam(bank,[{'lr':spec['learning_rate'],'wd':0.} for _ in entries])
        generator=torch.Generator(device='cuda').manual_seed(7100+rep)
        total=steps or spec['updates'];curve=[];start=time.perf_counter()
        for step in range(1,total+1):
            ids=indices[torch.randint(len(indices),(spec['batch_size'],),device='cuda',generator=generator)]
            features=data.features(ids,kind,moments)
            bank.zero_grad(set_to_none=True)
            losses,_=bank.objective(bank(features),data.ev['foot'][ids],data.ev['severity'][ids],data.ev['valid_cutoff'][ids])
            losses.sum().backward()
            multiplier=1.
            for milestone,mult in spec['lr_milestones']:
                if step>=milestone:multiplier=mult
            optimizer.step(multiplier)
            if step==1 or step%spec['monitor_every']==0 or step==total:
                _,vloss=predict(bank,val,monitor,kind,moments)
                if not np.isfinite(vloss).all():raise FloatingPointError(vloss)
                row={'step':step,'validation_loss':vloss.tolist(),'seconds':time.perf_counter()-start}
                curve.append(row);(dest/'curve.json').write_text(json.dumps(curve,indent=2)+'\n')
                print('OBS_TRAIN',name,kind,step,round(row['seconds'],1),flush=True)
        payload={'state':{k:v.detach().cpu() for k,v in bank.state_dict().items()},
                 'entries':entries,'input':cfg,'kind':kind,'moments':moments,'updates':total}
        torch.save(payload,dest/'models.pt')
        report={'status':'completed','updates':total,'entries':entries,'training_events':len(train_ids),
                'parameters_per_model':sum(p.numel() for p in bank.parameters())//len(entries),
                'seconds':time.perf_counter()-start,'model_sha256':file_hash(dest/'models.pt'),
                'train_metadata_sha256':file_hash(source/'metadata.json'),
                'validation_metadata_sha256':file_hash(validation/'metadata.json'),
                'protocol_sha256':file_hash('configs/impact/observability_v1.json'),
                'development_only':development}
        (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        del bank,optimizer,features,losses;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rep',type=int,default=1);p.add_argument('--steps',type=int)
    p.add_argument('--development',action='store_true')
    a=p.parse_args();train(a.rep,a.steps,a.development)
