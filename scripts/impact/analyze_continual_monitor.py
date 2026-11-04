"""Descriptive old-domain curves; never choose checkpoints or hyperparameters."""
import json
from pathlib import Path

import numpy as np

from evaluate_continual_confirmation import mean_ci
from train_continual_confirmation import METHODS


def main():
    root=Path('artifacts/impact/continual_replay_v1/confirmation')
    values={m:[] for m in METHODS}
    for rep in range(1,6):
        data=json.loads((root/f'rep{rep}/training.json').read_text())
        for method in METHODS:
            init=[]
            for i,(_,m) in enumerate(data['names']):
                if m!=method:continue
                a=np.array([h['metrics'][i]['A']['AP'] for h in data['history']])
                b=np.array([h['metrics'][i]['B']['AP'] for h in data['history'][5:]])
                init.append({'worst_A_AP_drop':float(np.max(a[0]-a)),
                             'worst_B_AP_drop_after_learning':float(np.max(b[0]-b)),
                             'mean_old_retention_curve_change':float((np.trapezoid(a-a[0],dx=1)/(len(a)-1)+np.trapezoid(b-b[0],dx=1)/(len(b)-1))/2)})
            values[method].append({k:float(np.mean([r[k] for r in init])) for k in init[0]})
    result={m:{key:mean_ci([r[key] for r in rows]) for key in rows[0]} for m,rows in values.items()}
    (root/'monitor_curve_summary.json').write_text(json.dumps({'scope':'Descriptive fixed monitor endpoints, 4000 update spacing; no best-node model selection. B retention starts after B learning.',
                                                            'methods':result},indent=2)+'\n')
    print('CONTINUAL_MONITOR_CURVES_ANALYZED')


if __name__=='__main__':main()
