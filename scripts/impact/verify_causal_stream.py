"""Readback checks requiring only retained weights, small fixtures and alarm records."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from himloco_lab.impact.causal_stream import StreamBank,PDBank,summarize_records
from himloco_lab.impact.dataset import file_hash


ROOT=Path('datasets/go1_impact/causal_stream_v1')
ART=Path('artifacts/impact/causal_stream_v1')
LOG=Path('logs/impact_causal_stream_v1')


@torch.no_grad()
def fixture_check(rep,role):
    cache=ROOT/f'rep{rep}/{role}/cache';fixture=dict(np.load(cache/'fixture.npz'))
    expected=dict(np.load(cache/'fixture_predictions.npz'))
    pre=torch.tensor(fixture['pre'],device='cuda');age=torch.tensor(fixture['age'],device='cuda')
    t,n=pre.shape[:2];errors={}
    for kind in ('models','frozen','pd'):
        payload=torch.load(LOG/f'rep{rep}/{kind}.pt',map_location='cpu',weights_only=False)
        entries=[tuple(v) for v in payload['entries']]
        bank=PDBank([s for s,_ in entries]) if kind=='pd' else StreamBank(entries)
        bank.load_state_dict(payload['state']);bank=bank.cuda().eval()
        got=np.empty((len(entries),t*n,4),np.float32)
        for start in range(0,t*n,4096):
            ids=torch.arange(start,min(start+4096,t*n),device='cuda');boundary,e=ids//n,ids%n
            lag=torch.minimum(torch.arange(6,device='cuda')[None],age[boundary,e,None].long())
            frames=pre[boundary[:,None]-lag,e[:,None]]
            raw=bank(frames) if kind=='pd' else bank(frames.flatten(1))
            got[:,start:start+len(ids)]=raw.cpu().numpy()
        got=got.reshape(len(entries),t,n,4)
        for i,(seed,method) in enumerate(entries):
            key=f'{seed}_{kind}_{method}';error=float(abs(got[i]-expected[key]).max())
            errors[key]=error
            assert error<.002,(rep,role,key,error)
    return errors


def verify():
    lock=json.loads((ART/'model_lock.json').read_text())
    for group in ('models','operating_points','source_sha256'):
        for name,sha in lock[group].items():assert file_hash(name)==sha,name
    fixtures={};metric_rows=0;max_error=0.
    for rep in range(1,6):
        results=json.loads((ART/f'rep{rep}/results.json').read_text())
        for role in ('validation','test','high_speed'):
            fixtures[f'{rep}/{role}']=fixture_check(rep,role)
        for role in ('test','high_speed'):
            cache=ROOT/f'rep{rep}/{role}/cache'
            meta=json.loads((cache/'metadata.json').read_text());events=dict(np.load(cache/'events.npz'))
            context_path=cache/'evaluation_context.npz'
            data=dict(np.load(context_path if context_path.exists() else cache/'stream.npz'))
            family=np.asarray(meta['terrain']['family_id']);tile=np.asarray(meta['terrain']['tile_id'])
            for row in (r for r in results if r['domain'].startswith(role+'_')):
                path=ART/f'rep{rep}/{role}'/f"seed{row['seed']}_delay{row['delay']}_{row['method']}.npz"
                records=dict(np.load(path));domain=row['domain'][len(role)+1:]
                check=summarize_records(records,row['threshold'],events,data['valid'],family,tile,domain)
                for metric in ('events','tp','fp','recall','far','far_tile_upper95','delay_ms_median','before_native_peak_fraction'):
                    if check[metric] is None:assert row[metric] is None
                    else:
                        error=abs(check[metric]-row[metric]);max_error=max(max_error,error)
                        assert error<1e-10,(row,metric,error)
                # Independently check the sparse record's claimed event matches.
                known=records['event_id']>=0;ids=records['event_id'][known]
                assert np.all(records['env'][known]==events['env'][ids])
                assert np.all(records['foot'][known]==events['foot'][ids])
                delta=records['time'][known]*8-events['native_onset'][ids]-1
                assert np.all((delta>=0)&(delta<16))
                metric_rows+=1
        print('STREAM_VERIFIED_REP',rep,flush=True)
    report=dict(status='passed',fixture_max_raw_error=max(v for row in fixtures.values() for v in row.values()),
                fixtures=fixtures,metrics_recomputed=metric_rows,max_metric_error=max_error,
                model_lock_sha256=file_hash(ART/'model_lock.json'))
    (ART/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('STREAM_VERIFIED',metric_rows,report['fixture_max_raw_error'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    verify()
