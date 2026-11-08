"""Causal landing monitoring: boundary-aligned inputs, labels and alarm accounting.

Privileged labels never enter a deployable network or its phase state machine.
Native onset index k has physical time (k+1)*2.5ms; pre[t] is at t*20ms.
"""
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import file_hash
from .observability import prepare_events


def prepare_stream(raw, output):
    raw,output=Path(raw),Path(output)
    if not (output/'metadata.json').exists():prepare_events(raw,output)
    meta=json.loads((output/'metadata.json').read_text())
    report=json.loads((raw/'report.json').read_text())
    t,n=meta['steps'],meta['num_envs']
    age=np.empty((t,n),np.int16);done=np.empty((t,n),bool)
    force=np.lib.format.open_memmap(output/'native_staging.npy',mode='w+',dtype='float32',shape=(t,n,8,4))
    state=np.zeros((t,n,4),bool);root=np.empty((t,n,2),np.float32)
    cursor=0
    for item in report['shards']:
        path=raw/item['file'];assert file_hash(path)==item['sha256']
        with np.load(path) as d:
            count=len(d['policy_step']);sl=slice(cursor,cursor+count)
            force[sl]=np.linalg.norm(d['normal_forces_w'][:,:,-8:].astype('float64'),axis=-1)
            age[sl]=d['episode_step'];done[sl]=d['done'];root[sl]=d['root_xy_from_origin']
            cursor+=count
    assert cursor==t
    state[1:]=(force[:-1,:,-2:]>5).all(2)
    state[age==0]=False
    event=dict(np.load(output/'events.npz'));num=len(event['env'])
    eid=np.full((t,n,4),-1,np.int32);pulse=np.zeros_like(eid,dtype=bool)
    confirm=(event['native_onset']+9)//8
    for delta in (0,1):
        boundary=event['base']+1+delta
        # The last native sample of an interval can be exactly on a boundary.
        ok=(boundary*8>=event['native_onset']+1)&(boundary*8<event['native_onset']+17)&(boundary<t)
        eid[boundary[ok],event['env'][ok],event['foot'][ok]]=np.flatnonzero(ok)
    pulse[confirm,event['env'],event['foot']]=True
    peak_offset=np.empty(num,np.int8)
    for e0 in range(0,n,64):
        e1=min(n,e0+64);rows=np.flatnonzero((event['env']>=e0)&(event['env']<e1))
        f=np.asarray(force[:,e0:e1]).transpose(0,2,1,3).reshape(t*8,e1-e0,4)
        values=f[event['native_onset'][rows,None]+np.arange(16),
                 event['env'][rows,None]-e0,event['foot'][rows,None]]
        peak_offset[rows]=values.argmax(1)
    event['confirm']=confirm;event['native_peak_offset']=peak_offset
    np.savez_compressed(output/'events.npz',**event)
    valid=age>=25
    valid[-2:]=False
    valid[:-2]&=age[2:]==age[:-2]+2
    np.savez_compressed(output/'stream.npz',age=age,done=done,state=state,pulse=pulse,event_id=eid,valid=valid)
    fixture_ids=np.arange(0,n,64)
    pre=np.load(output/'pre.npy',mmap_mode='r')
    np.savez_compressed(output/'fixture.npz',env_id=fixture_ids,pre=pre[:,fixture_ids],
                        age=age[:,fixture_ids],done=done[:,fixture_ids],state=state[:,fixture_ids],
                        pulse=pulse[:,fixture_ids],event_id=eid[:,fixture_ids],valid=valid[:,fixture_ids],
                        root_xy=root[:,fixture_ids])
    evidence=output/'provenance';evidence.mkdir(exist_ok=True)
    for filename in ('report.json','env.yaml','plan.json'):shutil.copy2(raw/filename,evidence/filename)
    del force
    (output/'native_staging.npy').unlink()
    compact={k:meta[k] for k in ('steps','num_envs','seed','domains','counts','max_xy','terminations','resets')}
    compact.update(status='completed',events=num,valid_robot_minutes=float(valid.sum()*.02/60),
                   artifacts_sha256={p.name:file_hash(p) for p in (output/'pre.npy',output/'events.npz',
                                     output/'stream.npz',output/'fixture.npz',output/'native_audit.npz')},
                   raw_report_sha256=meta['raw_report_sha256'])
    (output/'stream_metadata.json').write_text(json.dumps(compact,indent=2)+'\n')
    return compact


class StreamData:
    def __init__(self,path,device='cuda'):
        self.path=Path(path);self.device=device
        self.meta=json.loads((self.path/'metadata.json').read_text())
        self.cpu=dict(np.load(self.path/'stream.npz'));self.events=dict(np.load(self.path/'events.npz'))
        self.pre=torch.tensor(np.load(self.path/'pre.npy',mmap_mode='r'),device=device)
        self.age=torch.as_tensor(self.cpu['age'],device=device)
        self.labels={k:torch.as_tensor(self.cpu[k],device=device) for k in ('state','pulse','event_id')}
        self.severity=torch.as_tensor(self.events['severity'],device=device)
        self.family=np.array(self.meta['terrain']['family_id']);self.tile=np.array(self.meta['terrain']['tile_id'])

    def indices(self,role=None):
        mask=self.cpu['valid']&(self.family[None]<6)
        if role is not None:mask&=self.tile[None]%3==role
        return np.flatnonzero(mask)

    def frames(self,ids):
        ids=torch.as_tensor(ids,device=self.device);n=self.pre.shape[1]
        t,e=ids//n,ids%n
        lag=torch.arange(6,device=self.device)
        # Reset history is filled with the known first frame, never an old episode.
        past=t[:,None]-torch.minimum(lag[None],self.age[t,e,None].long())
        return self.pre[past,e[:,None]]

    def targets(self,ids):
        n=self.pre.shape[1];t,e=ids//n,ids%n;event=self.labels['event_id'][t,e].long()
        severity=torch.where(event>=0,self.severity[event.clamp_min(0)]/2000,0.)
        return self.labels['state'][t,e].float(),self.labels['pulse'][t,e].float(),severity


def pd_features(frames):
    """Per-foot q,dq, unsaturated PD response proxy, dq finite difference."""
    q=frames[:,0,9:21].reshape(-1,3,4).transpose(1,2)
    v=frames[:,0,21:33].reshape(-1,3,4).transpose(1,2)/.05
    old=frames[:,1,21:33].reshape(-1,3,4).transpose(1,2)/.05
    action=frames[:,0,33:45].reshape(-1,3,4).transpose(1,2)
    return torch.cat((q,v,40*(.25*action-q)-v,(v-old)/.02),-1)


class StreamBank(nn.Module):
    def __init__(self,entries,hidden=(128,64)):
        super().__init__();self.entries=entries;m=len(entries)
        self.w1=nn.Parameter(torch.empty(m,270,hidden[0]));self.b1=nn.Parameter(torch.zeros(m,1,hidden[0]))
        self.w2=nn.Parameter(torch.empty(m,hidden[0],hidden[1]));self.b2=nn.Parameter(torch.zeros(m,1,hidden[1]))
        self.w3=nn.Parameter(torch.empty(m,hidden[1],4));self.b3=nn.Parameter(torch.zeros(m,1,4))
        for i,(seed,_) in enumerate(entries):
            g=torch.Generator().manual_seed(seed)
            for w in (self.w1,self.w2,self.w3):
                with torch.no_grad():w[i].uniform_(-1/np.sqrt(w.shape[1]),1/np.sqrt(w.shape[1]),generator=g)
        self.register_buffer('mean',torch.zeros(m,1,270));self.register_buffer('std',torch.ones(m,1,270))

    def forward(self,x):
        x=((x.reshape(1,-1,270)-self.mean)/self.std).clamp(-30,30)
        h=F.silu(torch.bmm(x,self.w1)+self.b1);h=F.silu(torch.bmm(h,self.w2)+self.b2)
        return torch.bmm(h,self.w3)+self.b3

    def objective(self,raw,targets):
        state,pulse,severity=targets
        losses=[]
        for i,(_,kind) in enumerate(self.entries):
            if 'mse' in kind:loss=(raw[i]-severity).square().mean()
            else:
                y=state if kind=='state' else pulse if kind=='pulse' else (severity>1).float()
                loss=F.binary_cross_entropy_with_logits(raw[i],y)
            losses.append(loss)
        return torch.stack(losses)


class PDBank(nn.Module):
    def __init__(self,seeds=(1,2)):
        super().__init__();self.entries=[(s,'pd') for s in seeds];m=len(seeds)
        self.weight=nn.Parameter(torch.zeros(m,1,4,12));self.bias=nn.Parameter(torch.zeros(m,1,4))
        with torch.no_grad():
            for i,seed in enumerate(seeds):
                self.weight[i].normal_(0,.01,generator=torch.Generator().manual_seed(seed))
        self.register_buffer('mean',torch.zeros(1,1,4,12));self.register_buffer('std',torch.ones(1,1,4,12))

    def forward(self,frames):
        x=((pd_features(frames)[None]-self.mean)/self.std).clamp(-30,30)
        return (x*self.weight).sum(-1)+self.bias


def state_gate(prob,age,high):
    """Only estimated contact and known reset age drive this state machine."""
    t,n,f=prob.shape;low_count=np.zeros((n,f),np.int8);armed=np.zeros((n,f),bool)
    last=np.full((n,f),-100,np.int32);out=np.zeros_like(prob,dtype=bool)
    for step in range(t):
        reset=age[step]==0;armed[reset]=False;low_count[reset]=0;last[reset]=-100
        low_count=np.where(np.isfinite(prob[step])&(prob[step]<high/2),np.minimum(low_count+1,2),0)
        armed|=low_count>=2
        fire=armed&(prob[step]>=high)&(step-last>=5)
        out[step]=fire;armed[fire]=False;last[fire]=step
    return out


def alarm_records(score,age,valid,event_id):
    """Sparse records of intervals (previous_max, score] that can alarm.

    Exact for the versioned five-packet quiet rule; no truth enters triggering.
    """
    previous=np.full_like(score,-np.inf)
    for lag in range(1,6):
        same=age[lag:]>=lag
        previous[lag:]=np.maximum(previous[lag:],np.where(same[:,:,None],score[:-lag],-np.inf))
    keep=(score>previous)&valid[:,:,None]&np.isfinite(score)
    t,e,f=np.where(keep)
    return dict(time=t.astype('int32'),env=e.astype('int16'),foot=f.astype('int8'),
                lower=previous[keep],upper=score[keep],event_id=event_id[keep])


def delayed(value,age,steps):
    if not steps:return value
    result=np.zeros_like(value)
    if value.dtype.kind=='f':result.fill(-np.inf)
    result[steps:]=value[:-steps]
    result[age<steps]=False if value.dtype.kind=='b' else -np.inf
    return result


def summarize_records(records,threshold,events,valid,family,tile,domain,role=None,draws=2000):
    env_mask=family<6 if domain=='seen' else family==6
    if role is not None:env_mask&=tile%3==role
    selected=(records['lower']<threshold)&(records['upper']>=threshold)&env_mask[records['env']]
    ids=records['event_id'][selected];strong=np.zeros(len(ids),bool);ok=ids>=0
    strong[ok]=events['severity'][ids[ok]]>2000
    detected=ids[strong]
    assert len(np.unique(detected))==len(detected),'Quiet rule must give <=1 alarm per qualified event'
    b=events['base']+1;e=events['env'];f=events['foot']
    eligible=env_mask[e]&valid[b,e]
    target=eligible&(events['severity']>2000)
    # Validity is constant across the two event decision boundaries except censoring;
    # the event denominator only includes onsets whose first boundary is evaluable.
    total=int(target.sum());tp=int(strong.sum());fp=int((~strong).sum())
    assert np.all(target[detected]),'Matched alarm outside eligible event denominator'
    exposure=float(valid[:,env_mask].sum()*.02/60)
    env=records['env'][selected];false_env=env[~strong]
    tiles=np.unique(tile[env_mask]);counts=np.array([(tile[false_env]==u).sum() for u in tiles])
    minutes=np.array([valid[:,env_mask&(tile==u)].sum()*.02/60 for u in tiles])
    rng=np.random.default_rng(8401);weights=rng.multinomial(len(tiles),np.ones(len(tiles))/len(tiles),draws)
    rates=(weights@counts)/np.maximum(weights@minutes,1e-12)
    poisson_floor=2.995732/max(exposure,1e-12) if fp==0 else 0.
    delay=(records['time'][selected][strong]*8-events['native_onset'][detected]-1)*2.5
    peak_ms=events['native_peak_offset'][detected]*2.5
    return dict(events=total,tp=tp,fp=fp,recall=tp/max(total,1),robot_minutes=exposure,
                far=fp/max(exposure,1e-12),far_tile_upper95=float(np.quantile(rates,.95)),
                zero_count_poisson_upper95=poisson_floor,
                delay_ms_median=float(np.median(delay)) if tp else None,
                delay_ms_p90=float(np.quantile(delay,.9)) if tp else None,
                before_native_peak_fraction=float((delay<peak_ms).mean()) if tp else None,
                per_foot=[dict(events=int((target&(f==j)).sum()),tp=int((f[detected]==j).sum()),
                               fp=int((records['foot'][selected][~strong]==j).sum())) for j in range(4)],
                per_tile=[dict(tile=int(u),fp=int(c),robot_minutes=float(m),
                               events=int((target&(tile[e]==u)).sum()),tp=int((tile[e[detected]]==u).sum()))
                          for u,c,m in zip(tiles,counts,minutes)])
