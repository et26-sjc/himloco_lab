"""Host batch1 latency with synchronous CUDA completion; not onboard latency."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from himloco_lab.impact.anticipation import SharedBank,PhaseBank
from collect_anticipation import LOG,ART,SPEC
from evaluate_anticipation import dump


class ImpactOnly(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model(x)[0]


def benchmark(rep=1,development=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    name='development' if development else f'rep{rep}';dest=ART/name/'latency.json'
    if dest.exists():return json.loads(dest.read_text())
    spec=json.loads(SPEC.read_text());payload=torch.load(LOG/name/'models.pt',map_location='cpu',weights_only=False)
    pp=torch.load(LOG/name/'phase.pt',map_location='cpu',weights_only=False)
    ckpt=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    models={}
    for group in ('dedicated','probe19'):
        i=next(i for i,e in enumerate(payload['entries']) if e['group']==group and e['loss']=='bce')
        model=SharedBank([payload['entries'][i]],ckpt)
        model.load_state_dict({k:v[i:i+1] for k,v in payload['state'].items()})
        models[group]=(ImpactOnly(model).cuda().eval(),torch.zeros(1,270,device='cuda'))
    model=PhaseBank([pp['entries'][0]['seed']]);model.load_state_dict({k:v[:1] for k,v in pp['state'].items()})
    models['phase']=(model.cuda().eval(),torch.zeros(1,51,device='cuda'))
    result={}
    export=LOG/name/'exports';export.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for label,(model,x) in models.items():
            trace=torch.jit.freeze(torch.jit.trace(model,x))
            for _ in range(30):trace(x)
            torch.cuda.synchronize();samples=[]
            for _ in range(300):
                start=time.perf_counter_ns();trace(x);torch.cuda.synchronize()
                samples.append((time.perf_counter_ns()-start)/1e6)
            result[label]=dict(batch=1,repetitions=300,p50_ms=float(np.quantile(samples,.5)),
                p95_ms=float(np.quantile(samples,.95)),p99_ms=float(np.quantile(samples,.99)),max_ms=float(max(samples)))
            trace.save(str(export/f'latency_fixture_{label}.pt'))
    report=dict(status='measured',device=torch.cuda.get_device_name(),scope='Current host only; includes Python dispatch and synchronous CUDA completion. No robot communication/actuator measurement.',
                architectures=result,maximum_p99_ms=max(r['p99_ms'] for r in result.values()))
    dump(dest,report);print('ANTICIPATION_LATENCY',name,report['maximum_p99_ms'],flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rep',type=int,default=1)
    p.add_argument('--development',action='store_true');a=p.parse_args();benchmark(a.rep,a.development)
