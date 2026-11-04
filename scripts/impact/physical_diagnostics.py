"""A0 controlled falls: first-contact short-window loading, not whole-run support.

Matches environment IDs only after verifying identical initial root states at
both time steps. Comparisons change integration as well as sampling, and do not
establish hardware ground truth or short-window numerical convergence.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from himloco_lab.impact.dataset import file_hash


def summarize(root):
    root=Path(root)
    report=json.loads((root/'report.json').read_text())
    if report['status']!='completed' or (root/'INCOMPLETE').exists():
        raise ValueError('Incomplete drop')
    dt=report['window']['physics_dt']; lag=round(.01/dt); decimation=round(.02/dt)
    forces=[]; episode=[]
    for shard in report['shards']:
        if file_hash(root/shard['file'])!=shard['sha256']:
            raise ValueError('Drop shard hash mismatch')
        with np.load(root/shard['file']) as d:
            forces.append(d['normal_forces_w'][:,:,lag:])
            episode.append(d['episode_id'])
    vectors=np.concatenate(forces).transpose(1,0,2,3,4).reshape(report['num_envs'],-1,4,3)
    first_episode=np.repeat(np.concatenate(episode).T==0,decimation,axis=1)
    magnitude=np.linalg.norm(vectors.astype(np.float64),axis=-1)
    padded=np.pad(magnitude,((0,0),(2*lag,0),(0,0)))
    length=magnitude.shape[1]
    smooth=sum(padded[:,2*lag-k:2*lag-k+length] for k in range(lag))/lag
    delayed=np.pad(smooth,((0,0),(lag,0),(0,0)))[:,:length]
    filtered_rise=np.maximum(smooth-delayed,0)
    raw_delayed=padded[:,lag:lag+length]
    raw_rise=np.maximum(magnitude-raw_delayed,0)
    one_step=np.maximum(magnitude-padded[:,2*lag-1:2*lag-1+length],0)/dt
    results={key:np.zeros((report['num_envs'],4)) for key in
             ('peak_n','raw_rise_n','smoothed_rise_n','one_step_rate_n_per_s','first_40ms_integral_ns')}
    onset_ms=np.full((report['num_envs'],4),np.nan)
    for env in range(report['num_envs']):
        for foot in range(4):
            contacts=np.flatnonzero((magnitude[env,:,foot]>1)&first_episode[env])
            if not len(contacts):
                raise ValueError('Foot never made first-episode contact')
            onset=int(contacts[0]); end=min(onset+round(.04/dt),length)
            onset_ms[env,foot]=(onset+1)*dt*1000
            window=slice(onset,end)
            results['peak_n'][env,foot]=magnitude[env,window,foot].max()
            results['raw_rise_n'][env,foot]=raw_rise[env,window,foot].max()
            results['smoothed_rise_n'][env,foot]=filtered_rise[env,window,foot].max()
            results['one_step_rate_n_per_s'][env,foot]=one_step[env,window,foot].max()
            results['first_40ms_integral_ns'][env,foot]=magnitude[env,window,foot].sum()*dt
    initial=np.load(root/'initial_root_state.npz')['root_state_w']
    heights=initial[:,2]
    summary={key:{'mean':float(value.mean()),'max':float(value.max()),
                  'height_vs_per_robot_mean_spearman':float(spearmanr(heights,value.mean(1)).statistic)}
             for key,value in results.items()}
    return {'root':str(root),'report_sha256':file_hash(root/'report.json'),'physics_dt':dt,
            'robots':report['num_envs'],'height_range_m':[float(heights.min()),float(heights.max())],
            'summary':summary,'initial_root_state':initial.tolist(),
            'first_contact_ms':onset_ms.tolist(),'per_foot':{key:value.tolist() for key,value in results.items()}}


def compare(a,b):
    if not np.array_equal(a['initial_root_state'],b['initial_root_state']):
        raise ValueError('Cannot claim paired falls: initial root states differ')
    comparison={}
    for key in a['per_foot']:
        x=np.array(a['per_foot'][key]); y=np.array(b['per_foot'][key])
        relative=np.abs(y-x)/np.maximum((np.abs(x)+np.abs(y))/2,1e-6)
        comparison[key]={'paired_spearman':float(spearmanr(x.ravel(),y.ravel()).statistic),
                         'median_symmetric_relative_difference':float(np.median(relative)),
                         'p90_symmetric_relative_difference':float(np.quantile(relative,.9))}
    return {'status':'completed','runs':[a,b],'paired_comparison':comparison,
            'scope':'controlled first-contact 40ms diagnostics; solver and trajectory both change with dt',
            'interpretation':'smoothing robustness is a candidate, not assumed physical equivalence; all quantities remain simulation normal-loading proxies'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('a',type=Path);p.add_argument('b',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=compare(summarize(args.a),summarize(args.b))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['paired_comparison'],indent=2))
