"""Train paired independent/shared HIM shadows; never open final test roles."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from himloco_lab.impact.anticipation import (AnticipationData,SharedBank,PhaseBank,
    phase_features,clip_models)
from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.metrics import average_precision
from collect_anticipation import ROOT,ART,LOG,SPEC
from train_force_study import IndependentAdam


def entries(spec):
    result=[]
    for seed in spec['optimization_seeds']:
        for group in (*spec['groups'],'probe19'):
            for loss in spec['losses']:
                for weight in spec['lambda_candidates'] if group=='joint' else [1.]:
                    result.append(dict(seed=seed,group=group,loss=loss,**({'lambda':weight} if group=='joint' else {})))
        result.append(dict(seed=seed,group='scratch',loss='bce'))
    return result


def key(entry):
    name=f"{entry['group']}_{entry['loss']}"
    return name+(f"_lambda{entry['lambda']:g}" if entry['group']=='joint' else '')


def actor(checkpoint,device='cuda'):
    layers=[]
    for i,(a,b) in enumerate(((64,512),(512,256),(256,128),(128,12))):
        layer=nn.Linear(a,b)
        layer.weight.data.copy_(checkpoint['model_state_dict'][f'actor.{i*2}.weight'])
        layer.bias.data.copy_(checkpoint['model_state_dict'][f'actor.{i*2}.bias'])
        layers.append(layer)
        if i<3:layers.append(nn.ELU())
    model=nn.Sequential(*layers).to(device).eval()
    model.requires_grad_(False)
    return model


def teacher(x,checkpoint):
    s=checkpoint['model_state_dict']
    h=F.elu(F.linear(x,s['estimator.encoder.0.weight'],s['estimator.encoder.0.bias']))
    h=F.elu(F.linear(h,s['estimator.encoder.2.weight'],s['estimator.encoder.2.bias']))
    out=F.linear(h,s['estimator.encoder.4.weight'],s['estimator.encoder.4.bias'])
    return out[:,:3],F.normalize(out[:,3:],dim=-1)


@torch.no_grad()
def monitor(bank,phase,data,ids,moments,behavior,checkpoint):
    metrics=[];totals={name:np.zeros(len(bank.entries)) for name in ('impact','velocity','swap')}
    phase_loss=np.zeros(len(phase.entries));score=[];truth=[];action_square=np.zeros(len(bank.entries))
    for start in range(0,len(ids),512):
        part=ids[start:start+512];frames=data.frames(part);x=frames.flatten(1);y=data.targets(part)
        _,pieces,raw=bank.objective(x,y,data.original_targets(part))
        for name,value in pieces.items():totals[name]+=value.cpu().numpy()*len(part)
        p=phase(phase_features(frames,data.hip,moments))
        phase_loss+=F.binary_cross_entropy_with_logits(p,(y>1).float()[None].expand_as(p),
                                                      reduction='none').mean((1,2)).cpu().numpy()*len(part)
        score.append(raw.cpu().numpy());truth.append((y>1).cpu().numpy())
        if start<1024:
            v,z=bank.original(bank.trunk(x));rv,rz=teacher(x,checkpoint)
            reference=behavior(torch.cat((x[:,:45],rv,rz),-1))
            inputs=torch.cat((x[None,:,:45].expand(len(bank.entries),-1,-1),v,z),-1)
            actions=behavior(inputs)
            action_square+=(actions-reference).square().mean((1,2)).cpu().numpy()*len(part)
    raw=np.concatenate(score,1);y=np.concatenate(truth).reshape(-1)
    for i,entry in enumerate(bank.entries):
        probability=1/(1+np.exp(-np.clip(raw[i],-60,60)))
        metrics.append(dict(entry=entry,impact_loss=float(totals['impact'][i]/len(ids)),
            velocity_rmse=float(np.sqrt(totals['velocity'][i]/len(ids))),swap_loss=float(totals['swap'][i]/len(ids)),
            AP=float(average_precision(y,raw[i].reshape(-1))),
            Brier=None if entry['loss']=='mse' else float(((probability.reshape(-1)-y)**2).mean()),
            severity_RMSE_n_per_s=(float(np.sqrt(totals['impact'][i]/len(ids))*2000) if entry['loss']=='mse' else None),
            actor_action_RMS_shift=float(np.sqrt(action_square[i]/min(len(ids),1024)))))
    return dict(models=metrics,phase_loss=(phase_loss/len(ids)).tolist())


def gradients(bank,x,y,next48):
    _,pieces,_=bank.objective(x,y,next48)
    items={}
    for name,value in [('impact',pieces['impact']*bank.lambdas),
                       ('velocity',pieces['velocity']*bank.base_mask),('swap',pieces['swap']*bank.base_mask)]:
        values=torch.autograd.grad(value.sum(),(bank.w1,bank.w2),retain_graph=True,allow_unused=True)
        items[name]=[v.detach() for v in values]
    result=[]
    for layer in range(2):
        a=items['impact'][layer].flatten(1);v=items['velocity'][layer].flatten(1);s=items['swap'][layer].flatten(1)
        base=v+s;an=a.norm(dim=1);bn=base.norm(dim=1)
        cos=(a*base).sum(1)/(an*bn).clamp_min(1e-20)
        for i,e in enumerate(bank.entries):
            result.append(dict(entry=e,layer=layer,impact_norm=float(an[i]),base_norm=float(bn[i]),
                               velocity_norm=float(v[i].norm()),swap_norm=float(s[i].norm()),
                               cosine=float(cos[i]) if float(an[i]*bn[i])>1e-20 else None))
    return result


def train(rep,development=False,total_override=None,current_diagnostic=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    spec=json.loads(SPEC.read_text());name='development' if development else f'rep{rep}'
    dest=LOG/name/('current_diagnostic' if current_diagnostic else '');dest.mkdir(parents=True,exist_ok=True)
    requested=total_override or (spec['development_updates'] if development else spec['updates'])
    if (dest/'training.json').exists():
        previous=json.loads((dest/'training.json').read_text())
        if previous['updates']>=requested:return
        (dest/f"training_{previous['updates']}.json").write_text(json.dumps(previous,indent=2)+'\n')
    root=ROOT/name
    data=AnticipationData(root/'train/cache',current_diagnostic=current_diagnostic)
    val=AnticipationData(root/'validation/cache',current_diagnostic=current_diagnostic)
    assert file_hash(spec['checkpoint'])==spec['checkpoint_sha256']
    checkpoint=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    bank=SharedBank(entries(spec),checkpoint).cuda();phase=PhaseBank(spec['optimization_seeds']).cuda()
    behavior=actor(checkpoint)
    gpu_checkpoint=dict(model_state_dict={k:v.cuda() for k,v in checkpoint['model_state_dict'].items() if k.startswith('estimator.')})
    ids=torch.as_tensor(data.indices(),device='cuda');generator=torch.Generator(device='cuda').manual_seed(9800+rep)
    normids=ids[torch.randint(len(ids),(32768,),device='cuda',generator=generator)]
    with torch.no_grad():
        frames=data.frames(normids);q=frames[:,0,9+np.asarray(data.hip)];v=frames[:,0,21+np.asarray(data.hip)]
        target_mean=(data.targets(normids).mean(0)*2000).cpu().tolist()
        moments=dict(qmean=q.mean(0).cpu().tolist(),qstd=q.std(0).clamp_min(.01).cpu().tolist(),
                     vmean=v.mean(0).cpu().tolist(),vstd=v.std(0).clamp_min(.01).cpu().tolist())
        x=frames[:128].flatten(1);bv,bz=bank.original(bank.trunk(x));rv,rz=teacher(x,gpu_checkpoint)
        warm=[i for i,e in enumerate(bank.entries) if e['group']!='scratch']
        initial_error=max(float((bv[warm]-rv).abs().max()),float((bz[warm]-rz).abs().max()))
        assert initial_error<.005,initial_error
    configs=[dict(lr=spec['learning_rate'],wd=0.) for _ in bank.entries]
    opt=IndependentAdam(bank,configs)
    popt=IndependentAdam(phase,[dict(lr=spec['learning_rate'],wd=0.) for _ in phase.entries])
    monitor_ids=val.indices(role=0)
    monitor_ids=torch.as_tensor(np.random.default_rng(9820+rep).choice(monitor_ids,min(8192,len(monitor_ids)),replace=False),device='cuda')
    curve=[];elapsed=0.;begin_step=1
    clip_count=torch.zeros(len(bank.entries),device='cuda')
    resume=dest/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cuda',weights_only=False)
        assert saved['protocol_sha256']==file_hash(SPEC)
        bank.load_state_dict(saved['state']);phase.load_state_dict(saved['phase_state'])
        for optimizer,record in [(opt,saved['optimizer']),(popt,saved['phase_optimizer'])]:
            optimizer.m=record['m'];optimizer.v=record['v'];optimizer.step_number=record['step_number']
        generator.set_state(saved['rng'].cpu());curve=saved['curve'];begin_step=saved['step']+1;elapsed=saved['seconds']
        if 'clip_count' in saved:clip_count=saved['clip_count']
    total=requested
    start=time.perf_counter()
    diagnostic=json.loads((dest/'gradient_diagnostics.json').read_text()) if (dest/'gradient_diagnostics.json').exists() else []
    for step in range(begin_step,total+1):
        part=ids[torch.randint(len(ids),(spec['batch_size'],),device='cuda',generator=generator)]
        frames=data.frames(part);x=frames.flatten(1);y=data.targets(part);next48=data.original_targets(part)
        bank.normalize_prototypes();bank.zero_grad(set_to_none=True);phase.zero_grad(set_to_none=True)
        loss,_,_=bank.objective(x,y,next48)
        p=phase(phase_features(frames,data.hip,moments))
        ploss=F.binary_cross_entropy_with_logits(p,(y>1).float()[None].expand_as(p),reduction='none').mean((1,2))
        (loss.sum()+ploss.sum()).backward();norm=clip_models(bank,spec['max_grad_norm']);clip_models(phase,spec['max_grad_norm'])
        clip_count+=(norm>spec['max_grad_norm']).float()
        multiplier=1.
        for milestone,value in spec['lr_milestones']:
            if step>=milestone:multiplier=value
        opt.step(multiplier);popt.step(multiplier)
        if step==1 or step%spec['monitor_every']==0 or step==total:
            result=monitor(bank,phase,val,monitor_ids,moments,behavior,gpu_checkpoint)
            assert all(np.isfinite(m['impact_loss']) for m in result['models'])
            curve.append(dict(step=step,seconds=elapsed+time.perf_counter()-start,
                              gradient_norm_before_clip=norm.detach().cpu().tolist(),**result))
            (dest/'curve.json').write_text(json.dumps(curve,indent=2,allow_nan=False)+'\n')
            diagnostic.append(dict(step=step,layers=gradients(bank,x[:512],y[:512],next48[:512])))
            (dest/'gradient_diagnostics.json').write_text(json.dumps(diagnostic,indent=2,allow_nan=False)+'\n')
            state=dict(state=bank.state_dict(),phase_state=phase.state_dict(),entries=bank.entries,moments=moments,
                optimizer=dict(m=opt.m,v=opt.v,step_number=opt.step_number),
                phase_optimizer=dict(m=popt.m,v=popt.v,step_number=popt.step_number),
                rng=generator.get_state(),curve=curve,step=step,clip_count=clip_count,
                seconds=curve[-1]['seconds'],protocol_sha256=file_hash(SPEC))
            torch.save(state,resume)
            print('ANTICIPATION_TRAIN',name,step,round(curve[-1]['seconds'],1),flush=True)
    state={k:v.detach().cpu() for k,v in bank.state_dict().items()}
    torch.save(dict(state=state,entries=bank.entries,moments=moments,hip=data.hip,updates=total,
                    training_target_mean_n_per_s=target_mean),dest/'models.pt')
    torch.save(dict(state={k:v.detach().cpu() for k,v in phase.state_dict().items()},entries=phase.entries,
                    moments=moments,hip=data.hip,updates=total),dest/'phase.pt')
    # Verify truly frozen encoders and paired distinct copies at completion.
    frozen=SharedBank(bank.entries,checkpoint)
    for i,e in enumerate(bank.entries):
        if e['group'] in ('frozen','probe19'):
            for k in ('w1','b1','w2','b2','wo','bo'):
                assert torch.equal(state[k][i],getattr(frozen,k)[i].detach()),(e,k)
    report=dict(status='completed',updates=total,entries=bank.entries,training_robot_intervals=len(ids),
                exposure_per_model=total*spec['batch_size'],initial_him_max_error=initial_error,
                independent_parameters_and_optimizer=True,behavior_updated=False,
                seconds=elapsed+time.perf_counter()-start,development_only=development,
                current_diagnostic=current_diagnostic,
                training_target_mean_n_per_s=target_mean,
                parameters_per_shadow=sum(p.numel() for p in bank.parameters())//len(bank.entries),
                phase_parameters=sum(p.numel() for p in phase.parameters())//len(phase.entries),
                gradient_clipping_fraction=(clip_count/total).cpu().tolist(),
                effective_trainable_parameters={
                    'dedicated_or_scratch':sum(getattr(bank,k)[0].numel() for k in ('w1','b1','w2','b2','h1','hb1','h2','hb2')),
                    'frozen64_head':sum(getattr(bank,k)[0].numel() for k in ('h1','hb1','h2','hb2')),
                    'probe19_head':19*64+64+64*4+4,
                    'joint_or_moving':sum(p.numel() for p in bank.parameters())//len(bank.entries)},
                protocol_sha256=file_hash(SPEC),models_sha256=file_hash(dest/'models.pt'),
                phase_sha256=file_hash(dest/'phase.pt'))
    (dest/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ANTICIPATION_TRAIN_COMPLETE',name,round(report['seconds'],1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rep',type=int,default=1)
    p.add_argument('--development',action='store_true');p.add_argument('--updates',type=int)
    p.add_argument('--current-diagnostic',action='store_true')
    a=p.parse_args();train(a.rep,a.development,a.updates,a.current_diagnostic)
