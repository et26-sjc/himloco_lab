"""Re-infer complete causal fixtures and independently count retained alarms."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from himloco_lab.impact.anticipation import (SharedBank,PhaseBank,phase_features,eligible_future_events,future_summary)
from himloco_lab.impact.dataset import file_hash
from collect_anticipation import ROOT,ART,LOG,SPEC
from evaluate_anticipation import dump


@torch.no_grad()
def verify(rep,role,development=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    name='development' if development else f'rep{rep}'
    cache=ROOT/name/role/'cache';out=ART/name/role/'verification.json'
    if out.exists():return
    spec=json.loads(SPEC.read_text());meta=json.loads((cache/'metadata.json').read_text())
    fixture=dict(np.load(cache/'fixture.npz'));expected=np.load(cache/'fixture_predictions.npz')['raw']
    models=LOG/name;payload=torch.load(models/'models.pt',map_location='cpu',weights_only=False)
    pp=torch.load(models/'phase.pt',map_location='cpu',weights_only=False)
    ckpt=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    bank=SharedBank(payload['entries'],ckpt);bank.load_state_dict(payload['state']);bank.cuda().eval()
    phase=PhaseBank([e['seed'] for e in pp['entries']]);phase.load_state_dict(pp['state']);phase.cuda().eval()
    t,n=fixture['age'].shape
    x=np.zeros((t,n,6,45),np.float32)
    for lag in range(6):
        ts=np.arange(lag,t);source=fixture['pre'][ts-lag]
        x[ts,:,lag]=np.where((fixture['age'][ts]>=lag)[...,None],source,0.)
    maximum=0.
    flat=x.reshape(t*n,6,45);reference=expected.reshape(len(expected),t*n,4)
    for start in range(0,len(flat),512):
        frames=torch.as_tensor(flat[start:start+512],device='cuda')
        a,_=bank(frames.flatten(1));b=phase(phase_features(frames,pp['hip'],pp['moments']))
        actual=torch.cat((a,b)).cpu().numpy()
        maximum=max(maximum,float(abs(actual-reference[:,start:start+len(frames)]).max()))
    assert maximum<1e-4,maximum
    model_count=len(expected);del bank,phase
    data=dict(np.load(cache/'future.npz'));events=dict(np.load(cache/'events.npz'))
    family=np.asarray(meta['terrain']['family_id']);tile=np.asarray(meta['terrain']['tile_id'])
    # Recompute retained final counts from sparse selected alarms, not dense scores.
    checked=0;max_count_error=0.
    if not development and role in ('test','high_speed'):
        result=json.loads((ART/name/'results.json').read_text())
        for path in (ART/name/role).glob('*.npz'):
            d=dict(np.load(path));threshold=float(d.pop('threshold'));lead=float(d.pop('lead_ms_required'));delay=float(d.pop('delay_ms'))
            inference=float(d.pop('inference_ms'))
            assert ((d['lower']<threshold)&(d['upper']>=threshold)).all()
            for domain in ('seen','heldout'):
                stem=path.stem;seed=int(stem.split('_',1)[0]);method=stem.split('_',1)[1].rsplit('_lead',1)[0]
                row=next(r for r in result if r['seed']==seed and r['method']==method and r['lead_ms']==lead and
                         r['delay_ms']==delay and r['domain']==f'{role}_{domain}')
                recounted=future_summary(d,threshold,events,data,family,tile,domain,minimum_lead_ms=lead,delay_ms=delay+inference,
                                         draws=spec['bootstrap_draws'])
                for k in ('tp','fp','events','far','recall','false_background','false_ordinary','false_late_strong'):
                    max_count_error=max(max_count_error,abs(recounted[k]-row[k]))
                checked+=1
        assert max_count_error==0.
    # Current diagnostic targets, independently using raw physical samples.
    current_error=0.;current_checked=0
    if (cache/'current_targets.npy').exists():
        native=dict(np.load(cache/'native_audit.npz'))
        targets=np.load(cache/'current_targets.npy',mmap_mode='r')
        for column,env in enumerate(native['env_id']):
            force=np.linalg.norm(native['normal_forces_w'][:,column,-8:].astype('float64'),axis=-1).reshape(-1,4)
            for boundary in np.arange(25,t,17):
                if native['age'][boundary,column]<25:continue
                end=boundary*8
                rates=np.stack([np.maximum((force[j-7:j+1].mean(0)-force[j-15:j-7].mean(0))/.02,0)
                                for j in range(end-8,end)])
                truth=np.stack((force[end-8:end].mean(0),rates[-1],rates.max(0)),-1)
                current_error=max(current_error,float(abs(truth-targets[boundary,env]).max()));current_checked+=1
        assert current_error<.01
    dump(out,dict(status='passed',fixture_envs=n,fixture_steps=t,models=model_count,
                  max_reinference_error=maximum,operating_rows_recounted=checked,max_count_error=max_count_error,
                  current_diagnostic_samples=current_checked,current_target_max_error=current_error,
                  fixture_sha256=file_hash(cache/'fixture.npz'),prediction_sha256=file_hash(cache/'fixture_predictions.npz')))
    print('ANTICIPATION_VERIFIED',name,role,maximum,checked,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rep',type=int,default=1)
    p.add_argument('--development',action='store_true');p.add_argument('--roles',nargs='+',default=['test','high_speed'])
    a=p.parse_args()
    for role in a.roles:verify(a.rep,role,a.development)
