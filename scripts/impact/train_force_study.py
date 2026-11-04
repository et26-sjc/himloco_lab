"""Independent force/rate fits, including binary MSE control; never reads test."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.force_study import ForceData,ForceBank,TARGETS,prepare_force


class IndependentAdam:
    """Leading-axis learning rates/weight decay; states never cross models."""
    def __init__(self,bank,configs):
        self.params=list(bank.parameters())
        self.m=[torch.zeros_like(p) for p in self.params]
        self.v=[torch.zeros_like(p) for p in self.params]
        self.lr=torch.tensor([c['lr'] for c in configs],device=self.params[0].device)
        self.wd=torch.tensor([c['wd'] for c in configs],device=self.params[0].device)
        self.step_number=0

    @torch.no_grad()
    def step(self,multiplier=1.):
        self.step_number+=1
        for p,m,v in zip(self.params,self.m,self.v):
            shape=(-1,)+(1,)*(p.ndim-1)
            grad=p.grad+self.wd.reshape(shape)*p
            m.mul_(.9).add_(grad,alpha=.1)
            v.mul_(.999).addcmul_(grad,grad,value=.001)
            update=(m/(1-.9**self.step_number))/(torch.sqrt(v/(1-.999**self.step_number))+1e-8)
            p.add_(-self.lr.reshape(shape)*multiplier*update)


@torch.no_grad()
def predict(bank,data,ids,task,batch=2048,details=False):
    losses=np.zeros(len(bank.entries))
    scores,points=[],[]
    for start in range(0,len(ids),batch):
        part=ids[start:start+batch]
        x,y=data.batch(part,task)
        loss,score,point=bank.loss_output(bank(x),y)
        losses+=loss.cpu().numpy()*len(part)
        if details:
            scores.append(score.cpu().numpy());points.append(point.cpu().numpy())
    result={'loss':losses/len(ids)}
    if details:result.update(score=np.concatenate(scores,1),point=np.concatenate(points,1))
    return result


def fit(train,val,out,entries,configs,task,steps,spec,rep=0):
    out.mkdir(parents=True,exist_ok=False)
    (out/'INCOMPLETE').touch()
    bank=ForceBank(train.dim,entries,spec['hidden']).cuda()
    indices_np=train.indices()
    indices=torch.tensor(indices_np,device='cuda')
    rng=np.random.default_rng(913+rep)
    norm=rng.choice(indices_np,min(65536,len(indices_np)),replace=False)
    x,_=train.batch(norm,'forecast')
    bank.mean.copy_(x.mean(0));bank.std.copy_(x.std(0).clamp_min(.01))
    select=val.indices(role=0)
    select=rng.choice(select,min(spec['selection_samples'],len(select)),replace=False)
    opt=IndependentAdam(bank,configs)
    best=copy.deepcopy(bank.state_dict())
    best_loss=np.full(len(entries),np.inf);best_step=np.zeros(len(entries),int)
    generator=torch.Generator(device='cuda').manual_seed(12345+rep)
    history=[];unique=np.zeros(train.t*train.n,bool)
    start=time.perf_counter()
    for step in range(1,steps+1):
        ids=indices[torch.randint(len(indices),(spec['batch_size'],),device='cuda',generator=generator)]
        x,y=train.batch(ids,task)
        bank.zero_grad(set_to_none=True)
        loss,_,_=bank.loss_output(bank(x),y)
        loss.sum().backward()
        multiplier=1.
        for milestone,mult in spec['lr_multipliers']:
            if step>=milestone:multiplier=mult
        opt.step(multiplier)
        unique[ids.cpu().numpy()]=True
        if step==1 or step%spec['validation_every']==0 or step==steps:
            current=predict(bank,val,select,task)['loss']
            if not np.isfinite(current).all():
                raise FloatingPointError(f'Non-finite objective {current}')
            improved=current<best_loss
            best_loss[improved]=current[improved];best_step[improved]=step
            mask=torch.tensor(improved,device='cuda')
            for key,value in bank.state_dict().items():
                if key in ('mean','std'):
                    best[key].copy_(value)
                else:best[key][mask]=value[mask]
            history.append({'step':step,'loss':current.tolist(),'elapsed_s':time.perf_counter()-start})
            (out/'curve.json').write_text(json.dumps(history,indent=2)+'\n')
            print('FORCE_TRAIN',str(out),step,round(time.perf_counter()-start,1),flush=True)
    bank.load_state_dict(best)
    torch.save({'state':best,'entries':entries,'dim':train.dim,'hidden':spec['hidden'],
                'best_steps':best_step.tolist(),'configs':configs},out/'models.pt')
    result={'status':'completed','entries':entries,'best_loss':best_loss.tolist(),'best_steps':best_step.tolist(),
            'allocated_parameters_per_model':sum(p.numel() for p in bank.parameters())//len(entries),
            'updates':steps,'exposures_per_model':steps*spec['batch_size'],'unique_exposed':int(unique.sum()),
            'available':len(indices_np),'seconds':time.perf_counter()-start,'task':task,'input':train.kind,
            'train_metadata_sha256':file_hash(train.path/'metadata.json'),
            'validation_metadata_sha256':file_hash(val.path/'metadata.json'),
            'models_sha256':file_hash(out/'models.pt'),
            'source_sha256':{str(p):file_hash(p) for p in [Path(__file__),Path('source/himloco_lab/himloco_lab/impact/force_study.py')]}}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'INCOMPLETE').unlink()
    return result


def main(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    spec=json.loads(Path('configs/impact/force_study_v1.json').read_text())
    root=Path('datasets/go1_impact/force_study_v1')
    logs=Path('logs/impact_force_study_v1')
    if args.hpo or args.hpo_secondary:
        for split in ('train','validation'):
            dest=root/'development'/split
            if not (dest/'metadata.json').exists():
                prepare_force(Path('datasets/go1_impact/step1')/split,dest)
        train=ForceData(root/'development/train');val=ForceData(root/'development/validation')
        entries=[(1,m,t) for m in (spec['secondary_methods_after_main'] if args.hpo_secondary else spec['primary_methods'])
                 for t in ([5,6,7] if args.hpo_secondary else range(len(TARGETS)))]
        hpo_root=logs/('hpo_alternatives' if args.hpo_secondary else 'hpo')
        candidates=[]
        for trial,(lr,wd) in enumerate(itertools.product(spec['development_hpo']['learning_rates'],spec['development_hpo']['weight_decays'])):
            out=hpo_root/f'trial{trial}'
            configs=[{'lr':lr,'wd':wd} for _ in entries]
            result=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else fit(train,val,out,entries,configs,'detection',spec['development_hpo']['updates_per_trial'],spec)
            candidates.append(result['best_loss'])
        choice=np.asarray(candidates).argmin(0)
        grid=list(itertools.product(spec['development_hpo']['learning_rates'],spec['development_hpo']['weight_decays']))
        selected={f'{m}/{TARGETS[t]}':{'lr':grid[k][0],'wd':grid[k][1],'trial':int(k)} for (_,m,t),k in zip(entries,choice)}
        (hpo_root/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
        if args.hpo_secondary:
            combined=json.loads((logs/'hpo'/'selected.json').read_text());combined.update(selected)
            (logs/'hpo'/'selected.json').write_text(json.dumps(combined,indent=2)+'\n')
        print('HPO_COMPLETE',flush=True)
        return
    rep=args.rep
    spec['hidden']=spec['input_hidden'][args.input]
    train=ForceData(root/'cache'/f'rep{rep}'/'train',input_kind=args.input)
    val=ForceData(root/'cache'/f'rep{rep}'/'validation',input_kind=args.input)
    chosen=json.loads((logs/'hpo'/'selected.json').read_text())
    methods=args.methods or (spec['primary_methods'] if args.input=='I0' else ['mse'])
    for task in args.tasks:
        targets=(list(range(len(TARGETS))) if task=='detection' else [2,5,6,7]) if args.input=='I0' else [0,1,5,6]
        if args.targets:targets=[TARGETS.index(name) for name in args.targets]
        entries=[(seed,m,t) for seed in spec['optimization_seeds'] for m in methods for t in targets]
        configs=[chosen.get(f'{m}/{TARGETS[t]}',{'lr':.001,'wd':0.}) for _,m,t in entries]
        out=logs/args.group/f'rep{rep}'/args.input/task
        if (out/'report.json').exists():continue
        fit(train,val,out,entries,configs,task,args.steps or spec['steps_per_model'],spec,rep)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hpo',action='store_true');p.add_argument('--rep',type=int,default=1)
    p.add_argument('--hpo-secondary',action='store_true')
    p.add_argument('--input',choices=['I0','I0clean','I1','I2'],default='I0')
    p.add_argument('--tasks',nargs='+',default=['detection','forecast'])
    p.add_argument('--methods',nargs='+');p.add_argument('--steps',type=int)
    p.add_argument('--targets',nargs='+',choices=TARGETS)
    p.add_argument('--group',default='main')
    main(p.parse_args())
