"""Fixed physical-time diagnostics over the complete paired dt/solver grid."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

from himloco_lab.impact.dataset import file_hash


def analyze(root):
    report = json.loads((root/'report.json').read_text())
    dt = report['window']['physics_dt']
    dec = round(report['window']['policy_dt']/dt)
    traces = []
    for item in report['shards']:
        if file_hash(root/item['file'])!=item['sha256']:
            raise ValueError('Hash mismatch')
        with np.load(root/item['file']) as d:
            traces.append(d['normal_forces_w'][:,:,-dec:])
    f = np.linalg.norm(np.concatenate(traces).transpose(1,0,2,3,4).reshape(report['num_envs'],-1,4,3).astype('float64'),axis=-1)
    n,t,_ = f.shape
    if not (f>1).any(axis=1).all():
        raise ValueError('A paired foot never established contact')
    onset = (f>1).argmax(1)
    measurements = {}
    for b in (.0,.005,.01,.02,.04):
        lag = max(1,round(b/dt))
        padded = np.pad(f,((0,0),(2*lag,0),(0,0)))
        avg = sum(padded[:,2*lag-k:2*lag-k+t] for k in range(lag))/lag
        old = np.pad(avg,((0,0),(lag,0),(0,0)))[:,:t]
        rate = np.maximum(avg-old,0)/(b if b else dt)
        peak,loading,integral = np.zeros((n,4)),np.zeros((n,4)),np.zeros((n,4))
        for e in range(n):
            for foot in range(4):
                start = onset[e,foot]
                window = slice(start,start+round(.06/dt))
                peak[e,foot]=avg[e,window,foot].max()
                loading[e,foot]=rate[e,window,foot].max()
                integral[e,foot]=f[e,start:start+round(.04/dt),foot].sum()*dt
        measurements[f'force_{round(b*1000)}ms'] = peak
        measurements[f'loading_{round(b*1000)}ms'] = loading
        measurements['first40ms_integral'] = integral
    return report,measurements,np.load(root/'initial_root_state.npz')['root_state_w'],onset*dt*1000


def main():
    root=Path('datasets/go1_impact/force_study_v1/physics')
    output=Path('artifacts/impact/force_study_v1')
    output.mkdir(parents=True,exist_ok=True)
    runs={p.name:analyze(p) for p in sorted(root.iterdir()) if (p/'report.json').exists()}
    result={'status':'completed','scope':'paired 128 initial states; rerunning changes integration and trajectories; no hardware truth','runs':{},'comparisons':{}}
    for name,(r,m,s,o) in runs.items():
        result['runs'][name]={'dt':r['window']['physics_dt'],'solver':r['solver_iterations'],
                              'solver_global_bounds':r.get('solver_global_bounds'),
                              'report_hash':file_hash(root/name/'report.json'),
                              'mean':{k:float(v.mean()) for k,v in m.items()}}
    names=['dt5000_p4_v0','dt2500_p4_v0','dt1250_p4_v0','dt625_p4_v0']
    pairs=list(zip(names[:-1],names[1:]))
    pairs += [(f'dt{dt}_p4_v0',f'dt{dt}_p8_v4_global') for dt in (2500,1250)
              if f'dt{dt}_p8_v4_global' in runs]
    result['excluded_solver_comparisons'] = 'Original p8/v4 requests were capped by global max4/v0. Preserved as implementation diagnostics; only enforced reruns establish the solver contrast.'
    for a,b in pairs:
        ra,ma,sa,oa=runs[a];rb,mb,sb,ob=runs[b]
        if not np.array_equal(sa,sb):raise ValueError('Unpaired initial conditions')
        row={'onset_median_abs_difference_ms':float(np.median(abs(oa-ob))),'metrics':{}}
        for key,x in ma.items():
            y=mb[key]
            rel=abs(x-y)/np.maximum((abs(x)+abs(y))/2,1e-6)
            thresholds=[2000.,4000.,8000.] if 'loading' in key else [40.,80.,160.]
            row['metrics'][key]={'median_symmetric_relative_difference':float(np.median(rel)),
                                 'median_ratio_b_over_a':float(np.median(y/np.maximum(x,1e-9))),
                                 'p90_relative_difference':float(np.quantile(rel,.9)),
                                 'paired_spearman':float(spearmanr(x.ravel(),y.ravel()).statistic),
                                 'threshold_prevalence':{str(u):[float((x>u).mean()),float((y>u).mean())] for u in thresholds},
                                 'label_disagreement':{str(u):float(((x>u)!=(y>u)).mean()) for u in thresholds}}
        result['comparisons'][a+'__'+b]=row
    (output/'physics.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,row in result['comparisons'].items():
        print(name,{k:round(v['median_symmetric_relative_difference'],4) for k,v in row['metrics'].items()})


if __name__=='__main__':main()
