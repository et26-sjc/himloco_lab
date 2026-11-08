"""Future landing labels and controlled HIM sharing; no simulator imports.

All shadow encoders are distinct parameter slices. Privileged next48 is used
only in the original HIM objective, never in inference or alarm generation.
"""
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import file_hash
from .observability import prepare_events,native_rate


GROUPS = ('dedicated', 'frozen', 'joint', 'moving')
LOSSES = ('bce', 'brier', 'mse')


def future_event_ids(events, age, horizon_steps=4):
    """First strictly future qualified landing, not the next strong landing.

    Native sample k occurs at (k+1)*2.5ms; boundary t at t*20ms.
    Event indices need not be chronologically sorted.
    """
    t, n = age.shape
    sentinel = np.iinfo(np.int32).max
    rank = np.argsort(events['native_onset'], kind='stable')
    order = np.empty(len(rank), np.int32)
    order[rank] = np.arange(len(rank))
    ids = np.full((t, n, 4), sentinel, np.int32)
    b = events['native_onset']//8
    for lag in range(horizon_steps):
        boundary = b-lag
        safe = boundary.clip(0, t-1)
        delta = events['native_onset']+1-boundary*8
        ok = ((boundary >= 0) & (boundary < t) & (delta > 0) &
              (delta <= horizon_steps*8) &
              (age[safe, events['env']] == age[b, events['env']]-lag))
        np.minimum.at(ids, (boundary[ok], events['env'][ok], events['foot'][ok]), order[ok])
    present = ids != sentinel
    ids[~present] = -1
    ids[present] = rank[ids[present]]
    return ids


def prediction_valid(age, future_steps=6, burnin=25):
    valid = age >= burnin
    valid[-future_steps:] = False
    valid[:-future_steps] &= age[future_steps:] == age[:-future_steps]+future_steps
    return valid


def prepare_anticipation(raw, output):
    raw, output = Path(raw), Path(output)
    if not (output/'metadata.json').exists():
        prepare_events(raw, output)
    meta = json.loads((output/'metadata.json').read_text())
    report = json.loads((raw/'report.json').read_text())
    assert report['him_supervision'] and report['status'] == 'completed'
    t, n = meta['steps'], meta['num_envs']
    age = np.empty((t,n), np.int16)
    done = np.empty((t,n), bool)
    next48 = np.lib.format.open_memmap(output/'next48.npy', mode='w+', dtype='float32', shape=(t,n,48))
    post = np.lib.format.open_memmap(output/'post.npy', mode='w+', dtype='float32', shape=(t,n,45))
    native = np.lib.format.open_memmap(output/'current_force_staging.npy',mode='w+',dtype='float32',shape=(t,n,8,4))
    cursor = 0
    for shard in report['shards']:
        path = raw/shard['file']
        assert file_hash(path) == shard['sha256']
        with np.load(path) as d:
            count = len(d['policy_step']); sl = slice(cursor,cursor+count)
            age[sl], done[sl] = d['episode_step'], d['done']
            next48[sl], post[sl] = d['him_next_critic48'], d['post_frame']
            native[sl] = np.linalg.norm(d['normal_forces_w'][:,:,-8:].astype('float64'),axis=-1)
            cursor += count
    assert cursor == t
    next48.flush(); post.flush()
    current=np.lib.format.open_memmap(output/'current_targets.npy',mode='w+',dtype='float32',shape=(t,n,4,3))
    current[0]=0
    for e0 in range(0,n,64):
        e1=min(n,e0+64)
        f=np.asarray(native[:,e0:e1]).transpose(0,2,1,3).reshape(t*8,e1-e0,4)
        rate=native_rate(f).reshape(t,8,e1-e0,4)
        # At input boundary t, these quantities belong to completed interval t-1.
        current[1:,e0:e1,:,0]=f.reshape(t,8,e1-e0,4)[:-1].mean(1)
        current[1:,e0:e1,:,1]=rate[:-1,-1]
        current[1:,e0:e1,:,2]=rate[:-1].max(1)
    current.flush();del native
    (output/'current_force_staging.npy').unlink()
    event = dict(np.load(output/'events.npz'))
    eid = future_event_ids(event,age)
    valid = prediction_valid(age)
    np.savez_compressed(output/'future.npz', age=age, done=done, event_id=eid, valid=valid)
    fixture_ids = np.arange(0,n,64)
    pre = np.load(output/'pre.npy',mmap_mode='r')
    np.savez_compressed(output/'fixture.npz',env_id=fixture_ids,pre=pre[:,fixture_ids],
                        post=post[:,fixture_ids],next48=next48[:,fixture_ids],age=age[:,fixture_ids],
                        done=done[:,fixture_ids],event_id=eid[:,fixture_ids],valid=valid[:,fixture_ids],
                        current_targets=current[:,fixture_ids])
    provenance = output/'provenance'; provenance.mkdir(exist_ok=True)
    for name in ('report.json','env.yaml','plan.json'):
        shutil.copy2(raw/name,provenance/name)
    present = eid >= 0
    strong = np.zeros_like(present)
    strong[present] = event['severity'][eid[present]] > 2000
    summary = dict(status='completed',steps=t,num_envs=n,seed=report['seed'],
                   valid_robot_intervals=int(valid.sum()),events=len(event['env']),
                   positive_per_foot_fraction=float(strong[valid].mean()),
                   future_landing_per_foot_fraction=float(present[valid].mean()),
                   input='pre[t] and five older frames only; newest first',
                   target='first qualified landing strictly after input, within80ms; severity after40ms',
                   onset_semantics='5N force threshold; no strict geometric contact or actuator utility claim',
                   hashes={p.name:file_hash(p) for p in (output/'pre.npy',output/'next48.npy',
                       output/'post.npy',output/'events.npz',output/'future.npz',output/'fixture.npz',
                       output/'native_audit.npz',output/'current_targets.npy')})
    (output/'anticipation_metadata.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


class AnticipationData:
    def __init__(self,path,device='cuda',supervision=True,current_diagnostic=False):
        self.path,self.device = Path(path),device
        self.meta = json.loads((self.path/'metadata.json').read_text())
        self.cpu = dict(np.load(self.path/'future.npz'))
        self.events = dict(np.load(self.path/'events.npz'))
        self.pre = torch.tensor(np.load(self.path/'pre.npy',mmap_mode='r'),device=device)
        self.age = torch.as_tensor(self.cpu['age'],device=device)
        self.eid = torch.as_tensor(self.cpu['event_id'],device=device)
        self.severity = torch.as_tensor(self.events['severity'],device=device)
        self.next48 = (torch.tensor(np.load(self.path/'next48.npy',mmap_mode='r'),device=device)
                       if supervision else None)
        self.current=(torch.tensor(np.load(self.path/'current_targets.npy',mmap_mode='r')[...,2],device=device)
                      if current_diagnostic else None)
        self.family = np.asarray(self.meta['terrain']['family_id'])
        self.tile = np.asarray(self.meta['terrain']['tile_id'])
        self.hip = [self.meta['joint_names'].index(f'{leg}_thigh_joint') for leg in ('FL','FR','RL','RR')]

    def indices(self,role=None):
        mask = self.cpu['valid'] & (self.family[None] < 6)
        if role is not None:
            mask &= self.tile[None]%3 == role
        return np.flatnonzero(mask)

    def frames(self,ids):
        ids = torch.as_tensor(ids,device=self.device); n = self.pre.shape[1]
        t,e = ids//n,ids%n
        lag = torch.arange(6,device=self.device)[None]
        result = self.pre[(t[:,None]-lag).clamp_min(0),e[:,None]]
        return result*(lag<=self.age[t,e,None])[...,None]

    def targets(self,ids):
        n = self.pre.shape[1]; t,e = ids//n,ids%n
        if self.current is not None:return self.current[t,e]/2000.
        idx = self.eid[t,e].long()
        return torch.where(idx>=0,self.severity[idx.clamp_min(0)]/2000.,0.)

    def original_targets(self,ids):
        n = self.pre.shape[1]
        return self.next48[ids//n,ids%n]


def batched_sinkhorn(score,eps=.05,iters=3):
    with torch.no_grad():
        q = torch.exp(score/eps).transpose(1,2)
        k,b = q.shape[1:]
        q /= q.sum((1,2),keepdim=True)
        for _ in range(iters):
            q /= q.sum(2,keepdim=True); q /= k
            q /= q.sum(1,keepdim=True); q /= b
        return (q*b).transpose(1,2)


class SharedBank(nn.Module):
    """Paired distinct copies of HIM; leading dimension never mixes gradients."""
    def __init__(self,entries,checkpoint):
        super().__init__(); self.entries = entries
        m = len(entries)
        s = checkpoint['model_state_dict']
        for name, key in [('w1','encoder.0.weight'),('b1','encoder.0.bias'),
                          ('w2','encoder.2.weight'),('b2','encoder.2.bias'),
                          ('wo','encoder.4.weight'),('bo','encoder.4.bias'),
                          ('t1','target.0.weight'),('tb1','target.0.bias'),
                          ('t2','target.2.weight'),('tb2','target.2.bias'),
                          ('twout','target.4.weight'),('tbo','target.4.bias'),('proto','proto.weight')]:
            value = s['estimator.'+key].clone()
            if name.startswith(('w','t')) and value.ndim == 2:
                value = value.T.contiguous()
            setattr(self,name,nn.Parameter(value[None].repeat(m,*([1]*value.ndim))))
        self.h1 = nn.Parameter(torch.empty(m,64,64)); self.hb1 = nn.Parameter(torch.zeros(m,64))
        self.h2 = nn.Parameter(torch.empty(m,64,4)); self.hb2 = nn.Parameter(torch.zeros(m,4))
        for i,entry in enumerate(entries):
            g = torch.Generator().manual_seed(19000+entry['seed'])
            with torch.no_grad():
                self.h1[i].uniform_(-1/8,1/8,generator=g)
                self.h2[i].uniform_(-1/8,1/8,generator=g)
                if entry['group'] == 'scratch':
                    for w,b in ((self.w1,self.b1),(self.w2,self.b2)):
                        bound = 1/np.sqrt(w.shape[1]); w[i].uniform_(-bound,bound,generator=g)
                        b[i].uniform_(-bound,bound,generator=g)
        self.register_buffer('base_mask',torch.tensor([e['group'] in ('joint','moving') for e in entries]))
        self.register_buffer('stop_mask',torch.tensor([e['group'] in ('frozen','moving','probe19') for e in entries]))
        self.register_buffer('probe_mask',torch.tensor([e['group']=='probe19' for e in entries]))
        self.register_buffer('lambdas',torch.tensor([e.get('lambda',1.) if e['group']=='joint' else 1. for e in entries]))
        self.register_buffer('mse_mask',torch.tensor([e['loss']=='mse' for e in entries]))
        self.register_buffer('brier_mask',torch.tensor([e['loss']=='brier' for e in entries]))
        self.has_probe = any(e['group']=='probe19' for e in entries)

    def trunk(self,x):
        h = F.elu(torch.matmul(x,self.w1)+self.b1[:,None])
        return F.elu(torch.bmm(h,self.w2)+self.b2[:,None])

    def original(self,h):
        parts = torch.bmm(h,self.wo)+self.bo[:,None]
        return parts[...,:3],F.normalize(parts[...,3:],dim=-1,p=2)

    def forward(self,x):
        h = self.trunk(x)
        h_read = torch.where(self.stop_mask[:,None,None],h.detach(),h)
        if self.has_probe:
            vel,z = self.original(h)
            probe = F.pad(torch.cat((vel,z),-1),(0,45)).detach()
            h_read = torch.where(self.probe_mask[:,None,None],probe,h_read)
        out = torch.bmm(F.elu(torch.bmm(h_read,self.h1)+self.hb1[:,None]),self.h2)+self.hb2[:,None]
        return out,h

    def impact_losses(self,raw,severity):
        label = (severity>1).float()[None].expand_as(raw)
        mse = (raw-severity).square().mean((1,2))
        brier = (raw.sigmoid()-label).square().mean((1,2))
        bce = F.binary_cross_entropy_with_logits(raw,label,reduction='none').mean((1,2))
        return torch.where(self.mse_mask,mse,torch.where(self.brier_mask,brier,bce))

    @torch.no_grad()
    def normalize_prototypes(self):
        normalized = F.normalize(self.proto,dim=-1,p=2)
        self.proto.copy_(torch.where(self.base_mask[:,None,None],normalized,self.proto))

    def original_losses(self,h,next48):
        vel,z = self.original(h)
        target = F.elu(torch.matmul(next48[:,3:48],self.t1)+self.tb1[:,None])
        target = F.elu(torch.bmm(target,self.t2)+self.tb2[:,None])
        target = F.normalize(torch.bmm(target,self.twout)+self.tbo[:,None],dim=-1,p=2)
        score_s = z@self.proto.transpose(1,2)
        score_t = target@self.proto.transpose(1,2)
        qs,qt = batched_sinkhorn(score_s),batched_sinkhorn(score_t)
        swap = -.5*(qs*F.log_softmax(score_t/3.,-1)+qt*F.log_softmax(score_s/3.,-1)).mean((1,2))
        velocity = (vel-next48[None,:,45:48]).square().mean((1,2))
        return velocity,swap

    def objective(self,x,severity,next48):
        raw,h = self(x)
        impact = self.impact_losses(raw,severity)
        velocity,swap = self.original_losses(h,next48)
        loss = impact*self.lambdas+(velocity+swap)*self.base_mask
        return loss,dict(impact=impact,velocity=velocity,swap=swap),raw


def phase_features(frames,hip,moments):
    device = frames.device
    mom = {k:torch.as_tensor(v,device=device) for k,v in moments.items()}
    q = (frames[:,:,9+np.asarray(hip)]-mom['qmean'])/mom['qstd']
    v = -(frames[:,:,21+np.asarray(hip)]-mom['vmean'])/mom['vstd']
    mag = (q.square()+v.square()).sqrt().clamp_min(1e-5)
    return torch.cat((frames[:,0,:3],torch.stack((q/mag,v/mag),-1).flatten(1)),1)


class PhaseBank(nn.Module):
    def __init__(self,seeds=(1,2)):
        super().__init__(); self.entries = [dict(seed=s,group='phase',loss='bce') for s in seeds]
        m = len(seeds)
        for name,shape in [('w1',(m,51,352)),('b1',(m,352)),('w2',(m,352,64)),
                           ('b2',(m,64)),('w3',(m,64,4)),('b3',(m,4))]:
            setattr(self,name,nn.Parameter(torch.zeros(shape)))
        with torch.no_grad():
            for i,seed in enumerate(seeds):
                g = torch.Generator().manual_seed(19000+seed)
                for w in (self.w1,self.w2,self.w3):
                    w[i].uniform_(-1/np.sqrt(w.shape[1]),1/np.sqrt(w.shape[1]),generator=g)

    def forward(self,x):
        h = F.elu(x@self.w1+self.b1[:,None])
        h = F.elu(torch.bmm(h,self.w2)+self.b2[:,None])
        return torch.bmm(h,self.w3)+self.b3[:,None]


def clip_models(bank,maximum=10.):
    """Clip each model separately; include zero-gradient frozen parameters."""
    squares = [p.grad.flatten(1).square().sum(1) for p in bank.parameters() if p.grad is not None]
    norm = torch.stack(squares).sum(0).sqrt()
    scale = (maximum/(norm+1e-6)).clamp(max=1.)
    for p in bank.parameters():
        if p.grad is not None:
            p.grad.mul_(scale.reshape((-1,)+(1,)*(p.ndim-1)))
    return norm


def eligible_future_events(events,data,minimum_lead_ms=20.,delay_ms=0.):
    """Offline denominator only; never a runtime inference/trigger gate."""
    eligible=np.zeros(len(events['env']),bool)
    eid=data['event_id'];valid=data['valid']
    for t in range(len(eid)):
        rows=eid[t][valid[t]]
        rows=rows[rows>=0]
        lead=(events['native_onset'][rows]+1-t*8)*2.5-delay_ms
        eligible[rows[lead>=minimum_lead_ms]]=True
    return eligible


def future_alarm_records(score,data,events,minimum_lead_ms=20.,delay_ms=0.):
    from .causal_stream import alarm_records
    records=alarm_records(score,data['age'],data['valid'],data['event_id'])
    target=records['event_id'].copy();present=target>=0
    lead=np.full(len(target),-np.inf,np.float32)
    lead[present]=(events['native_onset'][target[present]]+1-records['time'][present]*8)*2.5-delay_ms
    strong=np.zeros(len(target),bool);strong[present]=events['severity'][target[present]]>2000
    good=strong&(lead>=minimum_lead_ms)
    records['target_event_id']=target
    records['event_id']=np.where(good,target,-1).astype('int32')
    records['lead_ms']=lead
    # 0=no next landing in80ms, 1=ordinary next landing, 2=strong but too late, 3=timely strong.
    records['kind']=np.where(good,3,np.where(strong,2,np.where(present,1,0))).astype('int8')
    return records


def retime_future_records(base,minimum_lead_ms,delay_ms):
    records=dict(base)
    lead=base['lead_ms']-delay_ms
    strong=base['kind']>=2;good=strong&(lead>=minimum_lead_ms)
    records['event_id']=np.where(good,base['target_event_id'],-1).astype('int32')
    records['lead_ms']=lead
    records['kind']=np.where(good,3,np.where(strong,2,np.where(base['target_event_id']>=0,1,0))).astype('int8')
    return records


def future_summary(records,threshold,events,data,family,tile,domain='seen',role=None,
                   minimum_lead_ms=20.,delay_ms=0.,draws=2000,eligible=None):
    env_mask=family<6 if domain=='seen' else family==6
    if role is not None:env_mask&=tile%3==role
    selected=(records['lower']<threshold)&(records['upper']>=threshold)&env_mask[records['env']]
    matched=records['event_id'][selected];good=matched>=0;hits=matched[good]
    assert len(np.unique(hits))==len(hits),'One-to-one matching violated'
    if eligible is None:eligible=eligible_future_events(events,data,minimum_lead_ms,delay_ms)
    target=eligible&env_mask[events['env']]&(events['severity']>2000)
    assert target[hits].all()
    tp,fp=len(hits),int((~good).sum());exposure=float(data['valid'][:,env_mask].sum()*.02/60)
    kinds=records['kind'][selected];false_env=records['env'][selected][~good]
    tiles=np.unique(tile[env_mask]);counts=np.array([(tile[false_env]==u).sum() for u in tiles])
    minutes=np.array([data['valid'][:,env_mask&(tile==u)].sum()*.02/60 for u in tiles])
    weights=np.random.default_rng(9841).multinomial(len(tiles),np.ones(len(tiles))/len(tiles),draws)
    rates=(weights@counts)/np.maximum(weights@minutes,1e-12)
    lead=records['lead_ms'][selected][good]
    return dict(events=int(target.sum()),tp=int(tp),fp=fp,recall=tp/max(int(target.sum()),1),
        robot_minutes=exposure,far=fp/max(exposure,1e-12),
        far_tile_upper95=float(np.quantile(rates,.95)),
        zero_count_poisson_upper95=2.995732/max(exposure,1e-12) if fp==0 else 0.,
        lead_ms_median=float(np.median(lead)) if tp else None,
        lead_ms_p10=float(np.quantile(lead,.1)) if tp else None,
        false_background=int((kinds==0).sum()),false_ordinary=int((kinds==1).sum()),
        false_late_strong=int((kinds==2).sum()),
        strong_without_valid_opportunity=int((~eligible&env_mask[events['env']]&(events['severity']>2000)).sum()),
        per_tile=[dict(tile=int(u),fp=int(c),robot_minutes=float(m),
                  events=int((target&(tile[events['env']]==u)).sum()),tp=int((tile[events['env'][hits]]==u).sum()))
                  for u,c,m in zip(tiles,counts,minutes)],
        per_foot=[dict(events=int((target&(events['foot']==f)).sum()),
                  tp=int((events['foot'][hits]==f).sum()),fp=int((records['foot'][selected][~good]==f).sum())) for f in range(4)])


def select_future_threshold(records,data,family,tile,spec):
    env=(family<6)&(tile%3==1);keep=env[records['env']]
    upper,lower=records['upper'][keep],records['lower'][keep]
    matched=records['event_id'][keep];strong=matched>=0;e=records['env'][keep]
    quantiles=np.unique(np.r_[np.linspace(0,1,128),1-np.logspace(-6,-2,128)])
    if len(upper):
        thresholds=np.unique(np.r_[np.quantile(upper,quantiles).astype('float32'),
                                   np.nextafter(upper.max(),np.float32(np.inf))])
    else:thresholds=np.array([1e30],np.float32)
    def counts(mask):
        return (np.searchsorted(np.sort(lower[mask]),thresholds,side='left')-
                np.searchsorted(np.sort(upper[mask]),thresholds,side='left'))
    tp=counts(strong);tiles=np.unique(tile[env]);false=np.array([counts((~strong)&(tile[e]==u)) for u in tiles])
    minutes=np.array([data['valid'][:,env&(tile==u)].sum()*.02/60 for u in tiles])
    w=np.random.default_rng(9841).multinomial(len(tiles),np.ones(len(tiles))/len(tiles),spec['bootstrap_draws'])
    rate=(w@false)/np.maximum((w@minutes)[:,None],1e-12)
    upper95=np.quantile(rate,.95,axis=0)
    # A zero bootstrap rate is not interpreted as certainty of zero false alarms.
    upper95=np.maximum(upper95,np.where(false.sum(0)==0,2.995732/max(minutes.sum(),1e-12),0.))
    far=false.sum(0)/minutes.sum()
    accepted=np.flatnonzero(upper95<=spec['selection_far_upper_per_minute'])
    assert len(accepted)>0,'Insufficient exposure even for disabled alarm bound'
    best=max(accepted,key=lambda i:(tp[i],-far[i],thresholds[i]))
    return dict(threshold=float(thresholds[best]),selection_tp=int(tp[best]),selection_fp=int(false[:,best].sum()),
                far=float(far[best]),far_tile_upper95=float(upper95[best]),thresholds_considered=len(thresholds),
                disabled=bool(tp[best]==0 and false[:,best].sum()==0))
