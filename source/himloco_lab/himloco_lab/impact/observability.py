"""Event-aligned observability diagnostics; no Isaac imports or policy changes."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import file_hash


OFFSETS = np.array([-2, -1, 0, 1, 2, 3])


def native_rate(force, lag=8, dt=.0025):
    """Causal lag-sample moving mean, then positive lag-sample difference."""
    cumulative = np.concatenate((np.zeros_like(force[:1], dtype=np.float64),
                                 np.cumsum(force, axis=0, dtype=np.float64)))
    mean = np.zeros_like(force)
    mean[lag-1:] = (cumulative[lag:] - cumulative[:-lag]) / lag
    rate = np.zeros_like(force)
    rate[2*lag-1:] = np.maximum((mean[2*lag-1:] - mean[lag-1:-lag]) / (lag*dt), 0)
    return rate


def contact_candidates(force, threshold=5., air_samples=16):
    """Offline onset = first of two high samples after a full low-force gap."""
    high = force > threshold
    cumulative = np.concatenate((np.zeros_like(high[:1], dtype=np.int32),
                                 np.cumsum(high, axis=0, dtype=np.int32)))
    clear = np.zeros_like(high)
    clear[air_samples:] = (cumulative[air_samples:-1] - cumulative[:-air_samples-1]) == 0
    sustained = np.zeros_like(high)
    sustained[:-1] = high[:-1] & high[1:]
    return np.where(clear & sustained)


def prepare_events(raw, output):
    """Stage one raw role, retain only pre frames/event records/small audits."""
    raw, output = Path(raw), Path(output)
    report = json.loads((raw/'report.json').read_text())
    if report['status'] != 'completed' or (raw/'INCOMPLETE').exists():
        raise ValueError('Incomplete raw collection')
    output.mkdir(parents=True, exist_ok=False)
    (output/'INCOMPLETE').touch()
    t, n, dec = report['steps'], report['num_envs'], 8
    assert abs(report['window']['physics_dt']-.0025) < 1e-9
    assert report['history_reconstruction_max_error'] == 0
    assert report['reference_contact_max_error'] <= 1e-5
    assert report['history_overlap_max_error'] == 0
    assert report['reset_count'] == report['terminal_traces_captured']
    arrays = {
        'pre': np.lib.format.open_memmap(output/'pre.npy', mode='w+', dtype='float32', shape=(t,n,45)),
        'force': np.lib.format.open_memmap(output/'force_staging.npy', mode='w+', dtype='float32', shape=(t,n,dec,4)),
        'age': np.empty((t,n), np.int16), 'done': np.empty((t,n), bool),
        'terminated': np.empty((t,n), bool)}
    audit_ids = np.arange(0,n,200)
    audit = {k:[] for k in ('normal_forces_w','pre','post','age','done')}
    cursor = 0
    max_xy = 0.
    for shard in report['shards']:
        path = raw/shard['file']
        if file_hash(path) != shard['sha256']:
            raise ValueError('Raw hash mismatch')
        with np.load(path) as d:
            count = len(d['policy_step']); sl = slice(cursor,cursor+count)
            np.testing.assert_array_equal(d['policy_step'],np.arange(cursor,cursor+count))
            arrays['pre'][sl] = d['pre_frame']
            arrays['force'][sl] = np.linalg.norm(d['normal_forces_w'][:,:,-dec:].astype('float64'),axis=-1)
            for key, source in [('age','episode_step'),('done','done'),('terminated','terminated')]:
                arrays[key][sl] = d[source]
            for key,source in [('normal_forces_w','normal_forces_w'),('pre','pre_frame'),('post','post_frame'),
                               ('age','episode_step'),('done','done')]:
                audit[key].append(d[source][:,audit_ids].copy())
            max_xy = max(max_xy,float(abs(d['root_xy_from_origin']).max()))
            cursor += count
    if cursor != t or max_xy >= 5.5:
        raise ValueError(f'Incomplete or tile boundary crossing: {cursor}, {max_xy}')
    assert np.all(arrays['age'][1:][arrays['done'][:-1]] == 0)
    for key in ('pre','force'): arrays[key].flush()
    columns = {k:[] for k in ('base','env','foot','native_onset','severity','force_peak','valid_cutoff')}
    counts = {'all_force_rises':0,'air_gap_candidates':0,'initial_excluded':0,
              'outcome_censored':0,'kept':0}
    for e0 in range(0,n,64):
        e1 = min(n,e0+64)
        f = np.asarray(arrays['force'][:,e0:e1]).transpose(0,2,1,3).reshape(t*dec,e1-e0,4)
        rate = native_rate(f)
        high = f>5.
        counts['all_force_rises'] += int((high[1:] & ~high[:-1]).sum())
        k, e, foot = contact_candidates(f)
        counts['air_gap_candidates'] += len(k)
        b, end = k//dec, (k+15)//dec
        initial = arrays['age'][b,e+e0] >= 25
        counts['initial_excluded'] += int((~initial).sum())
        end_safe = end.clip(0,t-1)
        complete = ((end<t) & (arrays['age'][end_safe,e+e0] == arrays['age'][b,e+e0]+end-b))
        counts['outcome_censored'] += int((initial & ~complete).sum())
        keep = initial & complete
        k,e,foot,b = k[keep],e[keep],foot[keep],b[keep]
        counts['kept'] += len(k)
        if not len(k): continue
        samples = k[:,None]+np.arange(16)[None]
        severity = rate[samples,e[:,None],foot[:,None]].max(1)
        peak = f[samples,e[:,None],foot[:,None]].max(1)
        cuts = b[:,None]+OFFSETS
        safe = cuts.clip(0,t-1)
        valid = ((cuts>=5)&(cuts<t)&(arrays['age'][safe,(e+e0)[:,None]] ==
                 arrays['age'][b,e+e0,None]+OFFSETS))
        for key,value in [('base',b),('env',e+e0),('foot',foot),('native_onset',k),
                           ('severity',severity),('force_peak',peak),('valid_cutoff',valid)]:
            columns[key].append(value)
    events = {key:np.concatenate(value) for key,value in columns.items()}
    for key in ('base','env','foot','native_onset'): events[key] = events[key].astype('int32')
    np.savez_compressed(output/'events.npz',**events)
    np.savez_compressed(output/'native_audit.npz',env_id=audit_ids,
                        **{k:np.concatenate(v) for k,v in audit.items()})
    family = np.asarray(report['terrain']['family_id'])
    domains = {}
    for name,mask in [('seen',family[events['env']]<6),('heldout',family[events['env']]==6)]:
        y=events['severity'][mask]
        domains[name] = {'events':len(y), 'positive_fraction':float((y>2000).mean()),
                         'severity_quantiles':np.quantile(y,[0,.25,.5,.75,.9,.95,.99,1]).tolist(),
                         'threshold_positive_fractions':{str(u):float((y>u).mean()) for u in (1000,2000,4000)}}
    actual = (events['base'][:,None]+OFFSETS)*.02-(events['native_onset'][:,None]+1)*.0025
    meta = {'status':'completed','steps':t,'num_envs':n,'seed':report['seed'],
            'terrain':report['terrain'],'joint_names':report['joint_names'],
            'velocity_range_x':report['velocity_range_x'],'counts':counts,'domains':domains,
            'valid_cutoff_counts':events['valid_cutoff'].sum(0).tolist(),
            'actual_cutoff_ms_min_median_max':np.quantile(actual*1000,[0,.5,1],axis=0).T.tolist(),
            'max_xy':max_xy,'terminations':int(arrays['terminated'].sum()),'resets':int(arrays['done'].sum()),
            'raw_report_sha256':file_hash(raw/'report.json'),
            'pre_sha256':file_hash(output/'pre.npy'), 'events_sha256':file_hash(output/'events.npz'),
            'native_audit_sha256':file_hash(output/'native_audit.npz'),
            'bytes':sum(p.stat().st_size for p in output.iterdir() if p.is_file())}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    del arrays
    (output/'force_staging.npy').unlink()
    (output/'INCOMPLETE').unlink()
    return meta


class EventData:
    def __init__(self, path, device='cuda'):
        self.path, self.device = Path(path), device
        self.meta = json.loads((self.path/'metadata.json').read_text())
        if (self.path/'INCOMPLETE').exists(): raise ValueError('Incomplete event data')
        self.cpu = dict(np.load(self.path/'events.npz'))
        self.ev = {k:torch.as_tensor(v,device=device) for k,v in self.cpu.items()}
        self.pre = torch.tensor(np.load(self.path/'pre.npy',mmap_mode='r'),device=device)
        self.hip = [self.meta['joint_names'].index(f'{leg}_thigh_joint') for leg in ('FL','FR','RL','RR')]
        self.family = np.asarray(self.meta['terrain']['family_id'])[self.cpu['env']]
        self.tile = np.asarray(self.meta['terrain']['tile_id'])[self.cpu['env']]

    def ids(self, family='seen', role=None):
        mask = self.family<6 if family=='seen' else self.family==6 if family=='heldout' else np.ones(len(self.family),bool)
        if role is not None: mask &= self.tile%3 == role
        return np.flatnonzero(mask)

    def frames(self, ids):
        ids = torch.as_tensor(ids,device=self.device)
        boundary = self.ev['base'][ids].long()[None,:]+torch.as_tensor(OFFSETS,device=self.device)[:,None]
        steps = (boundary[:,:,None]-torch.arange(6,device=self.device)).clamp(0,len(self.pre)-1)
        env = self.ev['env'][ids].long()[None,:,None]
        return self.pre[steps,env]

    def hip_moments(self):
        rng = np.random.default_rng(821)
        t = rng.integers(25,len(self.pre),65536)
        eligible = np.flatnonzero(np.asarray(self.meta['terrain']['family_id'])<6)
        env = rng.choice(eligible,65536)
        # All-cycle samples, not only touchdown-aligned q distributions.
        x = self.pre[t,env]
        q,v = x[:,9+np.array(self.hip)],x[:,21+np.array(self.hip)]
        return {'qmean':q.mean(0).cpu().tolist(),'qstd':q.std(0).clamp_min(.01).cpu().tolist(),
                'vmean':v.mean(0).cpu().tolist(),'vstd':v.std(0).clamp_min(.01).cpu().tolist()}

    def features(self, ids, kind, moments):
        frames = self.frames(ids)
        if kind=='full': return frames.flatten(2)
        if kind=='kinematics': return torch.cat((frames[:,:,0,:3],frames[:,:,:,9:33].flatten(2)),2)
        moment = {k:torch.as_tensor(v,device=self.device) for k,v in moments.items()}
        q = (frames[:,:,:,9+np.array(self.hip)]-moment['qmean'])/moment['qstd']
        v = -(frames[:,:,:,21+np.array(self.hip)]-moment['vmean'])/moment['vstd']
        magnitude = (q.square()+v.square()).sqrt().clamp_min(1e-5)
        return torch.cat((frames[:,:,0,:3],torch.stack((q/magnitude,v/magnitude),-1).flatten(2)),2)

    def strata(self, moments):
        # Fixed pre-contact proxy for all horizons; family is evaluation-only.
        b,e,f = self.cpu['base'],self.cpu['env'],self.cpu['foot']
        with torch.no_grad():
            x = self.pre[b,e].cpu().numpy()
        q = (x[:,9+np.asarray(self.hip)]-moments['qmean'])/moments['qstd']
        v = -(x[:,21+np.asarray(self.hip)]-moments['vmean'])/moments['vstd']
        angle = np.arctan2(v[np.arange(len(f)),f],q[np.arange(len(f)),f])
        phase = np.minimum(((angle+np.pi)/(2*np.pi)*8).astype(int),7)
        speed = np.clip(np.searchsorted([.525,.65,.775],x[:,0],side='right'),0,3)
        return (((self.family*4+f)*8+phase)*4+speed).astype('int32')


class EventBank(nn.Module):
    """Independent MLPs for each initialization/method/input cutoff."""
    def __init__(self, dim, entries, hidden):
        super().__init__()
        self.entries=entries
        count=len(entries)
        self.w1=nn.Parameter(torch.empty(count,dim,hidden[0]));self.b1=nn.Parameter(torch.zeros(count,1,hidden[0]))
        self.w2=nn.Parameter(torch.empty(count,hidden[0],hidden[1]));self.b2=nn.Parameter(torch.zeros(count,1,hidden[1]))
        self.w3=nn.Parameter(torch.empty(count,hidden[1],4));self.b3=nn.Parameter(torch.zeros(count,1,4))
        for i,(seed,_,_) in enumerate(entries):
            generator=torch.Generator().manual_seed(seed)
            for w in (self.w1,self.w2,self.w3):
                with torch.no_grad(): w[i].uniform_(-1/np.sqrt(w.shape[1]),1/np.sqrt(w.shape[1]),generator=generator)
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('std',torch.ones(dim))
        self.register_buffer('horizons',torch.tensor([e[2] for e in entries]))
        self.register_buffer('is_mse',torch.tensor([e[1]=='mse' for e in entries]))

    def forward(self, features):
        x=((features[self.horizons]-self.mean)/self.std).clamp(-30,30)
        h=F.silu(torch.bmm(x,self.w1)+self.b1)
        h=F.silu(torch.bmm(h,self.w2)+self.b2)
        return torch.bmm(h,self.w3)+self.b3

    def objective(self, raw, foot, severity, valid):
        score=raw.gather(2,foot.long()[None,:,None].expand(len(self.entries),-1,1)).squeeze(2)
        z=(severity>2000).float()[None].expand_as(score)
        losses=torch.where(self.is_mse[:,None],(score-severity[None]/2000).square(),
                           F.binary_cross_entropy_with_logits(score,z,reduction='none'))
        mask=valid[:,self.horizons].T.float()
        return (losses*mask).sum(1)/mask.sum(1).clamp_min(1),score


def matched_auc(y, score, strata, min_count=5):
    """Only within-stratum pairs; no cross-phase or cross-domain comparisons."""
    y=np.asarray(y,bool);score=np.asarray(score)
    numerator=denominator=covered=groups=0
    order=np.argsort(strata,kind='stable')
    ordered=np.asarray(strata)[order]
    ends=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1,len(order)]
    for start,end in zip(ends[:-1],ends[1:]):
        select=order[start:end];z=y[select];p=int(z.sum());n=len(z)-p
        if min(p,n)<min_count: continue
        rank=rankdata(score[select],method='average')
        numerator+=float(rank[z].sum()-p*(p+1)/2)
        denominator+=p*n;covered+=len(z);groups+=1
    return {'auc':numerator/denominator if denominator else None,'covered_events':covered,
            'coverage_fraction':covered/len(y) if len(y) else 0.,'strata':groups,'pairs':denominator}
