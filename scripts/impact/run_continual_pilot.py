"""Run the fresh compact A->B->C continual-learning pilot.

This pilot is a protocol check and an effect-size diagnostic, not the planned
five-replication confirmation. It uses one collection and one initialization;
all methods receive the same rows, model, optimizer state, and update budget.
"""
import copy
import pickle
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from himloco_lab.impact.force_study import ForceData, ForceBank, THRESHOLDS
from himloco_lab.impact.metrics import average_precision

ROOT=Path('datasets/go1_impact/continual_replay_v1/cache')
OUT=Path('artifacts/impact/continual_replay_v1')
LOG=Path('logs/impact_continual_replay_v1')
OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
SPEC=json.loads(Path('configs/impact/continual_replay_v1_plan.json').read_text())
torch.set_num_threads(4)

def ids_for(data, families, t0=0, t1=None):
    t1=data.t if t1 is None else min(t1,data.t)
    env=np.isin(data.family,families)
    age=np.asarray(data.cpu['age'])[t0:t1]
    return np.flatnonzero(((age>=25)&env[None]).reshape(-1)) + t0*data.n

def make_opt(bank, lr):
    # A one-model Adam state is explicit and easy to clone for each method.
    return torch.optim.Adam(bank.parameters(),lr=lr,weight_decay=0.)

@torch.no_grad()
def predict(bank,data,ids):
    scores=[];labels=[]
    for start in range(0,len(ids),2048):
        x,y=data.batch(ids[start:start+2048],'detection')
        raw=bank(x);_,score,_=bank.loss_output(raw,y)
        scores.append(score[0].sigmoid().cpu().numpy())
        labels.append((y[:,7,:]>THRESHOLDS[7]).cpu().numpy())
    return np.concatenate(scores),np.concatenate(labels)

def evaluate(bank,data):
    result={}
    for name,families in [('A',[0,1]),('B',[2,3]),('C',[4,5]),('U',[6])]:
        ids=ids_for(data,families)
        if not len(ids):continue
        p,y=predict(bank,data,ids)
        result[name]={'rows':len(y),'prevalence':float(y.mean()),'AP':average_precision(y,p),
                      'Brier':float(np.mean((p-y)**2))}
    return result

class Replay:
    def __init__(self, capacity, mode, seed):
        self.capacity=capacity;self.mode=mode;self.rng=np.random.default_rng(seed)
        self.x=np.empty((0,270),dtype=np.float32);self.y=np.empty((0,8,4),dtype=np.float32)
        self.stratum=np.empty(0,dtype=np.int8);self.key_values=np.empty(0,dtype=np.float64)
        self.seen=0;self.count=np.zeros(3,dtype=np.int64)
    def add_batch(self,x,y):
        # Vectorized priority-key reservoir: keeping the largest uniform keys
        # is exactly equivalent to reservoir sampling for each fixed stratum.
        x=x.detach().cpu().numpy().astype(np.float32);y=y.detach().cpu().numpy().astype(np.float32)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        score=y[:,7,:].max(1).astype(np.float32)
        if self.mode in ('uniform','fifo'):s=np.zeros(len(y),dtype=np.int8);quotas=[1.]
        elif self.mode=='binary':s=(score>2000.).astype(np.int8);quotas=[.5,.5]
        else:s=np.where(score<=2000.,0,np.where(score<=4000.,1,2)).astype(np.int8);quotas=[.5,.25,.25]
        self.seen+=len(y)
        for value in np.unique(s):self.count[int(value)]+=int((s==value).sum())
        keys=self.rng.random(len(y));allx=np.concatenate((self.x,x));ally=np.concatenate((self.y,y))
        alls=np.concatenate((self.stratum,s));allk=np.concatenate((self.key_values,keys))
        if self.mode=='fifo':
            self.x,self.y,self.stratum,self.key_values=allx[-self.capacity:],ally[-self.capacity:],alls[-self.capacity:],allk[-self.capacity:]
            return
        keep=[]
        for value,quota in enumerate(quotas):
            candidates=np.flatnonzero(alls==value);q=min(int(self.capacity*quota),len(candidates))
            if q:keep.append(candidates[np.argpartition(allk[candidates],-q)[-q:]])
        keep=np.concatenate(keep) if keep else np.empty(0,dtype=np.int64)
        self.x,self.y,self.stratum,self.key_values=allx[keep],ally[keep],alls[keep],allk[keep]
    def sample(self,n,device='cuda:0',weighted=True):
        if len(self.x)==0:raise ValueError('Cannot sample empty replay')
        idx=self.rng.integers(len(self.x),size=n)
        q=np.bincount(self.stratum,minlength=3)/len(self.x)
        p=self.count/self.seen
        if np.any((p>0)&(q==0)):raise ValueError('Missing positive-mass replay stratum')
        weights=p[self.stratum[idx]]/q[self.stratum[idx]] if weighted else np.ones(n)
        return (torch.from_numpy(self.x[idx]).to(device),torch.from_numpy(self.y[idx]).to(device),
                torch.as_tensor(weights,dtype=torch.float32,device=device))
    def bytes(self):return int(self.x.nbytes+self.y.nbytes+self.stratum.nbytes+self.key_values.nbytes)
    def serialized_bytes(self):return len(pickle.dumps(self,protocol=5))

def train_step(bank,opt,x,y,weights=None):
    opt.zero_grad(set_to_none=True)
    per_row=F.binary_cross_entropy_with_logits(bank(x)[0,:,:4],(y[:,7,:]>2000.).float(),reduction='none').mean(1)
    loss=(per_row*weights).mean() if weights is not None else per_row.mean()
    loss.backward();opt.step();return float(loss.item())

def train_stage(bank,opt,data,ids,updates,replay=None,weighted=True,probe=None):
    rng=np.random.default_rng(8801+updates+len(ids));history=[]
    for step in range(updates):
        current=ids[rng.integers(len(ids),size=1024 if replay is not None else 2048)]
        x,y=data.batch(current,'detection')
        weights=None
        if replay is None:
            xx,yy=x,y
        else:
            oldx,oldy,oldw=replay.sample(1024,weighted=weighted)
            xx=torch.cat((x,oldx),0);yy=torch.cat((y,oldy),0)
            weights=torch.cat((torch.ones(1024,device=x.device),oldw))
        loss=train_step(bank,opt,xx,yy,weights)
        if not np.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        if step==0 or (step+1)%1000==0:
            item={'step':step+1,'loss':loss,'memory_bytes':replay.bytes() if replay else 0}
            if probe is not None and (step+1)%10000==0:
                pd,pi=probe;p,z=predict(bank,pd,pi)
                item['A_monitor_probe']={'AP':average_precision(z,p),'Brier':float(np.mean((p-z)**2))}
            history.append(item)
            print('UPDATE',step+1,loss,flush=True)
    if replay is not None:
        # Insert each incoming block row once after learning, in chunks so the
        # GPU never holds the full block's expanded history.
        for start in range(0,len(ids),8192):
            x,y=data.batch(ids[start:start+8192],'detection');replay.add_batch(x,y)
    return history

def seed_memory(replay,data,ids):
    for start in range(0,len(ids),8192):
        x,y=data.batch(ids[start:start+8192],'detection');replay.add_batch(x,y)

def main():
    train=ForceData(ROOT/'train',input_kind='I0');monitor=ForceData(ROOT/'monitor',input_kind='I0')
    a=ids_for(train,[0,1]);blocks={name:[ids_for(train,fam,t0,t0+500) for t0 in (0,500,1000,1500,2000)]
                              for name,fam in [('B',[2,3]),('C',[4,5])]}
    # Only use first B/C block sequence; each domain contributes five blocks.
    primary_capacity=(SPEC['memory']['primary_bytes']-4096)//(270*4+32*4+9)
    entries=[(1,'bce',7)]
    base=ForceBank(270,entries,SPEC['training']['hidden']).cuda()
    norm=a[np.random.default_rng(901).choice(len(a),min(65536,len(a)),replace=False)]
    xx,_=train.batch(norm,'detection');base.mean.copy_(xx.mean(0));base.std.copy_(xx.std(0).clamp_min(.01))
    opt=make_opt(base,SPEC['training']['stage0_learning_rate_candidate'])
    initial_file=LOG/'pilot_stage0.pt'
    if initial_file.exists():
        saved=torch.load(initial_file,weights_only=False);base.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);stage0=saved['curve']
    else:
        aid=ids_for(monitor,[0,1]);probe_ids=aid[np.random.default_rng(902).choice(len(aid),min(65536,len(aid)),replace=False)]
        stage0=train_stage(base,opt,train,a,SPEC['training']['stage0_updates'],probe=(monitor,probe_ids))
        torch.save({'model':base.state_dict(),'optimizer':opt.state_dict(),'curve':stage0},initial_file)
    initial=copy.deepcopy(base.state_dict())
    optimizer_initial=copy.deepcopy(opt.state_dict())
    methods=['freeze','stationary_A','finetune','uniform','binary','severity','severity_unweighted']
    all_results=[]
    for method in methods:
        bank=ForceBank(270,entries,SPEC['training']['hidden']).cuda();bank.load_state_dict(initial)
        opt=make_opt(bank,SPEC['training']['adaptation_learning_rate_candidate'])
        opt.load_state_dict(copy.deepcopy(optimizer_initial))
        for group in opt.param_groups:group['lr']=SPEC['training']['adaptation_learning_rate_candidate']
        memory=None if method in ('freeze','stationary_A','finetune') else Replay(primary_capacity,'severity' if method=='severity_unweighted' else method,1200)
        if memory is not None:seed_memory(memory,train,a)
        previous=evaluate(bank,monitor)
        rows=[{'stage':'A0','method':method,'metrics':previous}]
        if method!='freeze':
            for domain in ('B','C'):
                for block,ids in enumerate(blocks[domain],1):
                    if method=='stationary_A':ids=a
                    pre=previous
                    incoming_p,incoming_y=predict(bank,train,ids)
                    incoming={'AP':average_precision(incoming_y,incoming_p),'Brier':float(np.mean((incoming_p-incoming_y)**2)),'rows':len(ids)}
                    hist=train_stage(bank,opt,train,ids,SPEC['training']['updates_per_block'],memory,weighted=method!='severity_unweighted')
                    post=evaluate(bank,monitor)
                    previous=post
                    rows.append({'stage':domain,'block':block,'method':method,'pre':pre,'post':post,'history':hist,
                                 'incoming_prequential':incoming,
                                 'memory_rows':len(memory.x) if memory else 0,'memory_bytes':memory.bytes() if memory else 0,
                                 'memory_serialized_bytes':memory.serialized_bytes() if memory else 0,
                                 'memory_counts':memory.count.tolist() if memory else None})
                    if memory is not None:assert memory.serialized_bytes()<=SPEC['memory']['primary_bytes']
                    (OUT/f'pilot_{method}.json').write_text(json.dumps(rows,indent=2)+'\n')
                    torch.save({'model':bank.state_dict(),'optimizer':opt.state_dict(),'stage':domain,'block':block},LOG/f'pilot_{method}_{domain}{block}.pt')
                    print('PILOT_BLOCK_COMPLETE',method,domain,block,{k:round(v['AP'],4) for k,v in post.items()},flush=True)
        all_results.append(rows)
        (OUT/f'pilot_{method}.json').write_text(json.dumps(rows,indent=2)+'\n')
        torch.save({'model':bank.state_dict(),'optimizer':opt.state_dict()},LOG/f'pilot_{method}.pt')
        print('PILOT_METHOD_COMPLETE',method,flush=True)
        del bank,opt,memory;torch.cuda.empty_cache()
    out={'status':'completed','pilot':'one collection, one initialization; diagnostic only','num_envs':4096,
         'stage0_updates':SPEC['training']['stage0_updates'],'updates_per_block':SPEC['training']['updates_per_block'],
         'memory_capacity_rows':primary_capacity,'methods':all_results,'stage0_curve':stage0,
         'data_metadata':{role:json.loads((ROOT/role/'metadata.json').read_text())['raw_report_sha256'] for role in ('train','monitor','calibration')}}
    (OUT/'pilot_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print('CONTINUAL_PILOT_COMPLETE',flush=True)

if __name__=='__main__':main()
