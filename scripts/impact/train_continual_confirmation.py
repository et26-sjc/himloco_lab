"""Compute-matched continual learning, independent models batched on the GPU.

Each model has separate parameters and Adam moments. Sharing a batched tensor
does not share gradients or data between methods. Test data are never opened.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.force_study import ForceBank,ForceData
from himloco_lab.impact.metrics import average_precision
from run_continual_pilot import Replay,ids_for

METHODS=['stationary_A','finetune','fifo','uniform','binary','severity','severity_unweighted','all_past']


def state_one(bank,i):
    return {k:(v if k in ('mean','std') else v[i:i+1]).detach().cpu().clone() for k,v in bank.state_dict().items()}


@torch.no_grad()
def metrics(bank,data,ids):
    scores=[];labels=[]
    for start in range(0,len(ids),4096):
        x,y=data.batch(ids[start:start+4096],'detection')
        scores.append(bank(x)[:,:,:4].cpu().numpy())
        labels.append((y[:,7,:]>2000.).cpu().numpy())
    s=np.concatenate(scores,1);z=np.concatenate(labels)
    p=1/(1+np.exp(-np.clip(s,-80,80)))
    return [{'AP':average_precision(z,ss),'Brier':float(np.mean((pp-z)**2)),
             'log_loss':float(np.mean(np.logaddexp(0,ss)-z*ss)),'rows':len(z),'prevalence':float(z.mean())}
            for ss,pp in zip(s,p)]


def evaluate(bank,data,indices):
    result=[{} for _ in bank.entries]
    for domain,ids in indices.items():
        for i,m in enumerate(metrics(bank,data,ids)):result[i][domain]=m
    return result


def memory_gpu(mem):
    q=np.bincount(mem.stratum,minlength=3)/len(mem.x);p=mem.count/mem.seen
    w=np.ones(len(mem.x),np.float32)
    if mem.mode not in ('uniform','fifo'):
        w=(p[mem.stratum]/q[mem.stratum]).astype(np.float32)
    return (torch.as_tensor(mem.x,device='cuda'),torch.as_tensor(mem.y,device='cuda'),torch.as_tensor(w,device='cuda'))


def seed_memory(mem,data,ids):
    for start in range(0,len(ids),8192):
        x,y=data.batch(ids[start:start+8192],'detection');mem.add_batch(x,y)


def train(rep):
    torch.set_num_threads(4)
    lock_path=Path('configs/impact/continual_replay_v1_lock.json')
    lock=json.loads(lock_path.read_text());assert lock['status']=='locked_before_confirmation'
    spec=json.loads(Path('configs/impact/continual_replay_v1_plan.json').read_text())
    out=Path('artifacts/impact/continual_replay_v1/confirmation')/f'rep{rep}'
    logs=Path('logs/impact_continual_replay_v1/confirmation')/f'rep{rep}'
    logs.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    if (out/'training.json').exists():return
    root=Path('datasets/go1_impact/continual_replay_v1/confirmation')/f'rep{rep}'/'cache'
    train=ForceData(root/'train');monitor=ForceData(root/'monitor')
    a=ids_for(train,[0,1]);a_gpu=torch.as_tensor(a,device='cuda')
    full={d:ids_for(monitor,f) for d,f in [('A',[0,1]),('B',[2,3]),('C',[4,5]),('U',[6])]}
    rng=np.random.default_rng(9000+rep)
    probes={d:rng.choice(ids,min(16384,len(ids)),replace=False) for d,ids in full.items() if d!='U'}
    initial=ForceBank(270,[(s,'bce',7) for s in (1,2)],(128,64)).cuda()
    norm=a[rng.choice(len(a),min(65536,len(a)),replace=False)]
    xx,_=train.batch(norm,'detection');initial.mean.copy_(xx.mean(0));initial.std.copy_(xx.std(0).clamp_min(.01))
    opt=torch.optim.Adam(initial.parameters(),lr=.001,weight_decay=0.)
    generator=torch.Generator(device='cuda').manual_seed(6100+rep)
    stage0=[];start=time.perf_counter()
    initial_file=logs/'stage0.pt'
    if initial_file.exists():
        saved=torch.load(initial_file,weights_only=False);initial.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);stage0=saved['curve'];generator.set_state(saved['generator'])
    else:
        for step in range(1,60001):
            ids=a_gpu[torch.randint(len(a_gpu),(2048,),device='cuda',generator=generator)]
            x,y=train.batch(ids,'detection');raw=initial(x)[:,:,:4]
            losses=F.binary_cross_entropy_with_logits(raw,(y[:,7,:]>2000.).float()[None].expand_as(raw),reduction='none').mean((1,2))
            opt.zero_grad(set_to_none=True);losses.sum().backward();opt.step()
            if step%10000==0:
                stage0.append({'step':step,'metrics':evaluate(initial,monitor,{'A':probes['A']}),'seconds':time.perf_counter()-start})
                print('STAGE0',rep,step,stage0[-1]['metrics'],flush=True)
        torch.save({'model':initial.state_dict(),'optimizer':opt.state_dict(),'curve':stage0,'generator':generator.get_state()},initial_file)
    initial_metrics=evaluate(initial,monitor,full)
    entries=[(seed,'bce',7) for seed in (1,2) for _ in METHODS]
    names=[(seed,method) for seed in (1,2) for method in METHODS]
    bank=ForceBank(270,entries,(128,64)).cuda()
    seed_indices=torch.tensor([s-1 for s,_ in names],device='cuda')
    state=bank.state_dict()
    for key,value in initial.state_dict().items():state[key].copy_(value if key in ('mean','std') else value[seed_indices])
    bank.load_state_dict(state)
    optimizer=torch.optim.Adam(bank.parameters(),lr=.0001,weight_decay=0.)
    state=copy.deepcopy(opt.state_dict())
    for v in state['state'].values():
        for key in ('exp_avg','exp_avg_sq'):v[key]=v[key][seed_indices].clone()
    for group in state['param_groups']:group['lr']=.0001
    optimizer.load_state_dict(state)
    del opt,initial
    capacity=(8388608-4096)//1217
    memories={};mem_gpu={}
    for i,(seed,method) in enumerate(names):
        if method not in ('stationary_A','finetune','all_past'):
            mode='severity' if method=='severity_unweighted' else method
            memories[i]=Replay(capacity,mode,12000+rep*10+seed)
            seed_memory(memories[i],train,a);mem_gpu[i]=memory_gpu(memories[i])
    history=[{'stage':'A0','metrics':[initial_metrics[s-1] for s,_ in names]}]
    for seed in (1,2):
        torch.save({'model':state_one(bank,(seed-1)*len(METHODS))},logs/f'seed{seed}_freeze_A0.pt')
    past=a.copy();completed=0
    resume=logs/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,weights_only=False)
        bank.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        memories=saved['memories'];mem_gpu={i:memory_gpu(mem) for i,mem in memories.items()}
        history=saved['history'];past=saved['past'];completed=saved['completed'];generator.set_state(saved['generator'])
    ordinal=0
    for domain,families in [('B',[2,3]),('C',[4,5])]:
        for block,t0 in enumerate((0,500,1000,1500,2000),1):
            ordinal+=1
            if ordinal<=completed:continue
            ids=ids_for(train,families,t0,t0+500);current=torch.as_tensor(ids,device='cuda')
            past_gpu=torch.as_tensor(past,device='cuda')
            incoming=metrics(bank,train,ids)
            curve=[]
            for step in range(1,4001):
                xs=[];ys=[];weights=[]
                # Identical current draw prefix for all replay methods/init seeds.
                draw=torch.randint(len(current),(2048,),device='cuda',generator=generator)
                current_x,current_y=train.batch(current[draw],'detection')
                stationary_ids=a_gpu[torch.randint(len(a_gpu),(2048,),device='cuda',generator=generator)]
                stationary_x,stationary_y=train.batch(stationary_ids,'detection')
                reference_ids=past_gpu[torch.randint(len(past_gpu),(1024,),device='cuda',generator=generator)]
                reference_x,reference_y=train.batch(reference_ids,'detection')
                for i,(seed,method) in enumerate(names):
                    w=torch.ones(2048,device='cuda')
                    if method=='stationary_A':
                        x,y=stationary_x,stationary_y
                    elif method=='finetune':x,y=current_x,current_y
                    else:
                        x,y=current_x[:1024],current_y[:1024]
                        if method=='all_past':
                            ox,oy=reference_x,reference_y;ow=torch.ones(1024,device='cuda')
                        else:
                            mx,my,mw=mem_gpu[i];mi=torch.randint(len(mx),(1024,),device='cuda',generator=generator)
                            ox,oy,ow=mx[mi],my[mi],mw[mi]
                            if method=='severity_unweighted':ow=torch.ones_like(ow)
                        x=torch.cat((x,ox));y=torch.cat((y,oy));w[1024:]=ow
                    xs.append(x);ys.append(y);weights.append(w)
                x=torch.stack(xs);y=torch.stack(ys);w=torch.stack(weights)
                raw=bank(x)[:,:,:4]
                losses=(F.binary_cross_entropy_with_logits(raw,(y[:,:,7,:]>2000.).float(),reduction='none').mean(2)*w).mean(1)
                optimizer.zero_grad(set_to_none=True);losses.sum().backward();optimizer.step()
                if step%1000==0:
                    assert torch.isfinite(losses).all()
                    curve.append({'step':step,'monitor_probe':evaluate(bank,monitor,probes)})
                    print('ADAPT',rep,domain,block,step,round(time.perf_counter()-start,1),flush=True)
            for i,mem in memories.items():
                seed_memory(mem,train,ids);mem_gpu[i]=memory_gpu(mem)
                assert mem.serialized_bytes()<=8388608
            post=evaluate(bank,monitor,full)
            history.append({'stage':domain,'block':block,'metrics':post,'incoming_prequential':incoming,'curve':curve,
                            'memory':{str(i):{'rows':len(m.x),'payload_bytes':m.bytes(),'serialized_bytes':m.serialized_bytes(),'seen':m.seen,'counts':m.count.tolist()} for i,m in memories.items()}})
            (out/'progress.json').write_text(json.dumps({'names':names,'history':history},indent=2)+'\n')
            # Fixed block endpoints, never best-of-monitor selection.
            for i,(seed,method) in enumerate(names):
                torch.save({'model':state_one(bank,i)},logs/f'seed{seed}_{method}_{domain}{block}.pt')
            past=np.concatenate((past,ids))
            torch.save({'model':bank.state_dict(),'optimizer':optimizer.state_dict(),'memories':memories,
                        'history':history,'past':past,'completed':ordinal,'generator':generator.get_state()},resume)
    report={'status':'completed','rep':rep,'names':names,'stage0_curve':stage0,'history':history,
            'num_envs':4096,'stage0_updates':60000,'adaptation_updates':40000,'total_batch':2048,
            'current_rows_per_update':{'finetune':2048,'replay':1024},'memory_budget_bytes':8388608,
            'data_metadata_sha256':{role:file_hash(root/role/'metadata.json') for role in ('train','monitor')},
            'lock_sha256':file_hash(lock_path),'source_sha256':{str(p):file_hash(p) for p in [Path(__file__),Path('scripts/impact/run_continual_pilot.py')]},
            'elapsed_s':time.perf_counter()-start}
    (out/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CONFIRMATION_TRAIN_COMPLETE',rep,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rep',type=int,required=True)
    train(p.parse_args().rep)
