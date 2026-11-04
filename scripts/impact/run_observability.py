"""Collect/train all replications, lock every model, then open final tests."""
import json
from pathlib import Path
import subprocess
import sys

from himloco_lab.impact.dataset import file_hash
from collect_observability import collect
from train_observability import train
from audit_observability import audit


def main():
    spec=json.loads(Path('configs/impact/observability_v1.json').read_text())
    artifact=Path('artifacts/impact/observability_v1')
    logs=Path('logs/impact_observability_v1')
    for rep in range(1,6):
        for role in ('train','validation'):
            collect(rep,role)
            audit(Path('datasets/go1_impact/observability_v1')/f'rep{rep}'/role/'cache')
        # Separate process releases all training CUDA tensors before next simulation.
        with (logs/f'rep{rep}_train.log').open('w') as stream:
            subprocess.run([sys.executable,'scripts/impact/train_observability.py','--rep',str(rep)],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('OBS_REP_TRAINED',rep,flush=True)
    models={}
    for rep in range(1,6):
        for kind in spec['inputs']:
            path=logs/f'rep{rep}'/kind/'models.pt'
            assert (path.parent/'report.json').exists()
            models[str(path)]=file_hash(path)
    sources=[Path('configs/impact/observability_v1.json'),Path('docs/GO1_IMPACT_OBSERVABILITY_PLAN.md'),
             Path('source/himloco_lab/himloco_lab/impact/observability.py'),Path('scripts/impact/collect.py')]
    sources+=list(Path('scripts/impact').glob('*observability*.py'))
    lock={'models':models,'protocol_sha256':file_hash('configs/impact/observability_v1.json'),
          'source_sha256':{str(p):file_hash(p) for p in sources},
          'test_policy':'All 240 independent models locked before any final test scoring.'}
    lockpath=artifact/'model_lock.json'
    if lockpath.exists():assert json.loads(lockpath.read_text())==lock
    else:lockpath.write_text(json.dumps(lock,indent=2)+'\n')
    print('OBS_ALL_MODELS_LOCKED',flush=True)
    for rep in range(1,6):
        for role in ('test','high_speed'):
            collect(rep,role)
            audit(Path('datasets/go1_impact/observability_v1')/f'rep{rep}'/role/'cache')
        with (logs/f'rep{rep}_evaluation.log').open('w') as stream:
            subprocess.run([sys.executable,'scripts/impact/evaluate_observability.py','--rep',str(rep)],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('OBS_REP_EVALUATED',rep,flush=True)
    subprocess.run([sys.executable,'scripts/impact/evaluate_observability.py','--summarize'],check=True)


if __name__=='__main__':main()
