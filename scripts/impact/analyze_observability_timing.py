"""Exploratory timing of target maxima, using only retained native audit envs."""
import json
from pathlib import Path

import numpy as np


def main():
    result={}
    for rep in range(1,6):
        for role in ('train','validation','test','high_speed'):
            path=Path('datasets/go1_impact/observability_v1')/f'rep{rep}'/role/'cache'
            trace=dict(np.load(path/'native_audit.npz'));event=dict(np.load(path/'events.npz'))
            phase=[];lags=[];positive=[]
            for column,env in enumerate(trace['env_id']):
                force=np.linalg.norm(trace['normal_forces_w'][:,column,-8:].astype(np.float64),axis=-1).reshape(-1,4)
                for row in np.flatnonzero(event['env']==env):
                    k=int(event['native_onset'][row]);foot=int(event['foot'][row]);values=[]
                    for sample in range(k,k+16):
                        values.append(max((force[sample-7:sample+1,foot].mean()-force[sample-15:sample-7,foot].mean())/.02,0))
                    lag=int(np.argmax(values));lags.append(lag*.0025*1000)
                    cutoff=(event['base'][row]+np.array([-2,-1,0,1,2,3]))*.02
                    phase.append((k+lag+1)*.0025<=cutoff+1e-12)
                    positive.append(event['severity'][row]>2000)
            phase=np.asarray(phase);positive=np.asarray(positive)
            result[f'{rep}/{role}']={'audited_events':len(lags),'strong_events':int(positive.sum()),
                'peak_lag_ms_quantiles':np.quantile(lags,[0,.25,.5,.75,1]).tolist(),
                'fraction_peak_already_occurred_by_cutoff':phase.mean(0).tolist(),
                'strong_fraction_peak_already_occurred_by_cutoff':phase[positive].mean(0).tolist()}
    payload={'scope':'Exploratory timing diagnostic on fixed 21/4096 env audit traces; no models or thresholds changed. Earliest argmax used if ties.',
             'roles':result}
    Path('artifacts/impact/observability_v1/native_peak_timing.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('OBS_PEAK_TIMING_COMPLETE',flush=True)


if __name__=='__main__':main()
