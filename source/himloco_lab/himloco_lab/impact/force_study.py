"""Causal force/rate targets and independent supervised estimators (no simulator)."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import file_hash


TARGETS = ('force10', 'loading10', 'impact10', 'force_raw', 'rate_raw', 'force20', 'loading20', 'impact20')
SCALES = (40., 4000., 4000., 40., 20000., 40., 2000., 2000.)
THRESHOLDS = (80., 4000., 4000., 80., 20000., 80., 2000., 2000.)


def force_targets(previous, current, dt):
    """Physical-time 10ms averaging/difference; current is [env,substep,foot]."""
    size = previous.shape[1]
    if size < 2 * round(.02/dt):
        raise ValueError('Need 40ms preceding load history')
    x = np.concatenate((previous, current), axis=1).astype(np.float64)
    n = current.shape[1]
    def filtered(b):
        lag = round(b/dt)
        avg = sum(x[:,size-k:size-k+n] for k in range(lag))/lag
        old = sum(x[:,size-lag-k:size-lag-k+n] for k in range(lag))/lag
        return avg,np.maximum((avg-old)/b,0)
    avg,rate = filtered(.01)
    avg20,rate20 = filtered(.02)
    raw = np.maximum(np.diff(x[:, size - 1:], axis=1) / dt, 0)
    return np.stack((avg[:, -1], rate[:, -1], rate.max(1), current[:, -1], raw.max(1),
                     avg20[:,-1],rate20[:,-1],rate20.max(1)), axis=1).astype('float32')


def prepare_force(root, output):
    root, output = Path(root), Path(output)
    report = json.loads((root / 'report.json').read_text())
    if report['status'] != 'completed' or (root / 'INCOMPLETE').exists():
        raise ValueError('Incomplete collection')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'INCOMPLETE').touch()
    t, n = report['steps'], report['num_envs']
    dec = round(report['window']['policy_dt']/report['window']['physics_dt'])
    has_hf = bool(report.get('physics_observation_schema'))
    hf_dec = round(.02 / report['physics_observation_schema']['sample_dt']) if has_hf else 4
    shapes = {'pre': ((t,n,45),'float32'), 'post': ((t,n,45),'float32'),
              'targets': ((t,n,len(TARGETS),4),'float32'), 'age': ((t,n),'int16'),
              'done': ((t,n),'bool'), 'terminated': ((t,n),'bool'),
              'hf': ((t,n,hf_dec,60),'float16')}
    if not has_hf:
        del shapes['hf']
    arrays = {k: np.lib.format.open_memmap(output / f'{k}.npy', mode='w+', dtype=d, shape=s)
              for k,(s,d) in shapes.items()}
    previous = np.zeros((n, 2*round(.02/report['window']['physics_dt']),4),np.float32)
    cursor, max_quant = 0, np.zeros(60)
    max_xy = 0.
    for shard in report['shards']:
        path = root / shard['file']
        if file_hash(path) != shard['sha256']:
            raise ValueError('Raw data hash mismatch')
        with np.load(path) as d:
            count = len(d['policy_step'])
            if not np.array_equal(d['policy_step'], np.arange(cursor,cursor+count)):
                raise ValueError('Missing interval')
            sl = slice(cursor,cursor+count)
            for key,src in [('pre','pre_frame'),('post','post_frame'),('age','episode_step'),
                            ('done','done'),('terminated','terminated')]:
                arrays[key][sl] = d[src]
            if has_hf:
                hf = d['physics_observations']
                if not np.isfinite(hf).all():
                    raise ValueError('Nonfinite physics observation')
                hf16 = hf.astype('float16')
                max_quant = np.maximum(max_quant, np.abs(hf - hf16.astype('float32')).max(axis=(0,1,2)))
                arrays['hf'][sl] = hf16
            max_xy = max(max_xy,float(np.abs(d['root_xy_from_origin']).max()))
            for k in range(count):
                previous[d['episode_step'][k] == 0] = 0
                current = np.linalg.norm(d['normal_forces_w'][k,:,-dec:].astype('float64'),axis=-1)
                arrays['targets'][cursor+k] = force_targets(previous,current,report['window']['physics_dt'])
                previous = np.concatenate((previous,current),axis=1)[:,-previous.shape[1]:]
            cursor += count
    if cursor != t or max_xy >= 5.5:
        raise ValueError('Incomplete or tile boundary crossing')
    for a in arrays.values():
        a.flush()
    meta = {'status':'completed','steps':t,'num_envs':n,'decimation':hf_dec,'physics_decimation':dec,
            'physics_dt':report['window']['physics_dt'],'policy_dt':.02,
            'seed':report['seed'],'terrain':report['terrain'],'targets':TARGETS,
            'scales':SCALES,'thresholds':THRESHOLDS,'max_xy':max_xy,
            'hf_quantization_max_abs_by_channel':max_quant.tolist(),
            'raw_report_sha256':file_hash(root/'report.json'),
            'files_sha256':{k:file_hash(output/f'{k}.npy') for k in shapes}}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (output/'INCOMPLETE').unlink()
    return meta


class ForceData:
    def __init__(self,path,device='cuda:0',input_kind='I0'):
        self.path,self.device,self.kind = Path(path),device,input_kind
        self.meta = json.loads((self.path/'metadata.json').read_text())
        if (self.path/'INCOMPLETE').exists():
            raise ValueError('Incomplete cache')
        self.t,self.n,self.dec = self.meta['steps'],self.meta['num_envs'],self.meta['decimation']
        self.cpu = {k:np.load(self.path/f'{k}.npy',mmap_mode='r') for k in ('age','targets','done','terminated')}
        self.family = np.asarray(self.meta['terrain']['family_id'])
        self.tile = np.asarray(self.meta['terrain']['tile_id'])
        self.y = torch.tensor(np.asarray(self.cpu['targets']),device=device)
        self.age = torch.tensor(np.asarray(self.cpu['age']).astype('int64'),device=device)
        if input_kind == 'I0':
            self.pre = torch.tensor(np.load(self.path/'pre.npy',mmap_mode='r'),device=device)
            self.post = torch.tensor(np.load(self.path/'post.npy',mmap_mode='r'),device=device)
            self.dim = 270
        else:
            self.hf = torch.tensor(np.load(self.path/'hf.npy',mmap_mode='r'),device=device)
            self.dim = {'I0clean':270,'I1':945,'I2':1260}[input_kind]

    def indices(self,families=(0,1,2,3,4,5),role=None):
        env = np.isin(self.family,families)
        if role is not None:
            env &= np.arange(self.n)%4 == role
        return np.flatnonzero(((self.cpu['age']>=25)&env[None]).reshape(-1))

    def batch(self,ids,task):
        ids = torch.as_tensor(ids,device=self.device,dtype=torch.long)
        t,e = ids//self.n,ids%self.n
        if self.kind == 'I0':
            offsets = torch.arange(5 if task=='detection' else 6,device=self.device)
            frames = self.pre[(t[:,None]-offsets).clamp_min(0),e[:,None]]
            if task=='detection':
                frames = torch.cat((self.post[t,e,None],frames),1)
        else:
            # Samples are post-physics, before reset. At pre-action cutoff use last
            # sample of the preceding interval. Never include a future substep.
            end = (t+int(task=='detection'))*self.dec-1
            stride = self.dec if self.kind=='I0clean' else 1
            offsets = torch.arange(6 if self.kind=='I0clean' else 21,device=self.device)*stride
            sample = end[:,None]-offsets
            pt,sub = sample.clamp_min(0)//self.dec,sample.clamp_min(0)%self.dec
            frames = self.hf[pt,e[:,None],sub].float()
            if self.kind!='I2':
                frames = frames[...,:45]
        return frames.reshape(len(ids),-1),self.y[t,e]


class ForceBank(nn.Module):
    """Independent model axis, including target: no auxiliary-task competition."""
    def __init__(self,dim,entries,hidden=(128,64)):
        super().__init__()
        self.entries = entries
        m = len(entries)
        self.w1 = nn.Parameter(torch.empty(m,dim,hidden[0]))
        self.b1 = nn.Parameter(torch.zeros(m,1,hidden[0]))
        self.w2 = nn.Parameter(torch.empty(m,hidden[0],hidden[1]))
        self.b2 = nn.Parameter(torch.zeros(m,1,hidden[1]))
        self.w3 = nn.Parameter(torch.empty(m,hidden[1],40))
        self.b3 = nn.Parameter(torch.zeros(m,1,40))
        for i,(seed,method,target) in enumerate(entries):
            gen = torch.Generator().manual_seed(seed)
            for weight in (self.w1,self.w2,self.w3):
                with torch.no_grad():
                    weight[i].uniform_(-1/np.sqrt(weight.shape[1]),1/np.sqrt(weight.shape[1]),generator=gen)
        self.register_buffer('mean',torch.zeros(dim))
        self.register_buffer('std',torch.ones(dim))
        self.groups = {method:[i for i,e in enumerate(entries) if e[1]==method] for method in set(e[1] for e in entries)}
        self.register_buffer('target_ids',torch.tensor([e[2] for e in entries]))
        self.register_buffer('scales',torch.tensor([SCALES[e[2]] for e in entries])[:,None,None])
        self.register_buffer('thresholds',torch.tensor([THRESHOLDS[e[2]]/SCALES[e[2]] for e in entries])[:,None,None])

    def forward(self,x):
        x = ((x-self.mean)/self.std).clamp(-30,30)
        h = F.silu(torch.matmul(x,self.w1)+self.b1)
        h = F.silu(torch.bmm(h,self.w2)+self.b2)
        return torch.bmm(h,self.w3)+self.b3

    def loss_output(self,raw,y):
        target = y[:,self.target_ids,:].permute(1,0,2)/self.scales
        label = (target>self.thresholds).float()
        losses = torch.empty(len(self.entries),device=raw.device)
        score = torch.empty_like(target)
        point = torch.full_like(target,float('nan'))
        for method,indices in self.groups.items():
            r,t,z,u = raw[indices],target[indices],label[indices],self.thresholds[indices]
            if method in ('mse','huber','logmse'):
                p = r[...,:4]
                if method=='huber':
                    p = F.softplus(p)
                    loss = F.huber_loss(p,t,reduction='none')
                elif method=='logmse':
                    loss = (p-t.log1p()).square()
                    p = p.clamp(max=10).expm1().clamp_min(0)
                else:
                    loss = (p-t).square()
                s = p
            elif method in ('bce','brier'):
                s = r[...,:4]
                loss = (F.binary_cross_entropy_with_logits(s,z,reduction='none') if method=='bce'
                        else (s.sigmoid()-z).square())
                p = torch.full_like(s,float('nan'))
            elif method=='mixture':
                v = r.reshape(r.shape[0],r.shape[1],4,10)
                zero_logit,weights,mu,sigma = v[...,0],v[...,1:4],v[...,4:7],F.softplus(v[...,7:10])+.08
                logt = t.clamp_min(1e-6).log()[...,None]
                density = -.5*((logt-mu)/sigma).square()-sigma.log()-logt-.5*np.log(2*np.pi)
                positive_nll = -F.logsigmoid(-zero_logit)-torch.logsumexp(weights.log_softmax(-1)+density,-1)
                loss = torch.where(t<=1e-6,-F.logsigmoid(zero_logit),positive_nll)
                survival = (.5*torch.erfc((u.log()[...,None]-mu)/(sigma*np.sqrt(2)))*weights.softmax(-1)).sum(-1)
                risk = zero_logit.neg().sigmoid()*survival
                s = torch.logit(risk.clamp(1e-7,1-1e-7))
                p = zero_logit.neg().sigmoid()*(weights.softmax(-1)*torch.exp((mu+.5*sigma.square()).clamp(max=10))).sum(-1)
            elif method=='cdf':
                # Nine fixed thresholds include primary exactly; interval distribution.
                mult = torch.tensor([.125,.25,.5,.75,1.,1.5,2.,4.,8.],device=r.device)
                v = r[...,:36].reshape(r.shape[0],r.shape[1],4,9)
                logits = torch.cat((v[...,:1],v[...,:1]-F.softplus(v[...,1:]).cumsum(-1)),-1)
                labels = (t[...,None] > u[...,None]*mult).float()
                loss = F.binary_cross_entropy_with_logits(logits,labels,reduction='none').mean(-1)
                s = logits[...,4]
                p = torch.full_like(s,float('nan'))
            else:
                raise ValueError(method)
            losses[indices] = loss.mean((1,2))
            score[indices],point[indices] = s,p*self.scales[indices]
        return losses,score,point
