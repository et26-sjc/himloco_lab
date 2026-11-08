"""Collect/train/calibrate all replications before opening final test streams."""
import json
from pathlib import Path
import subprocess
import sys

from himloco_lab.impact.dataset import file_hash
from collect_causal_stream import collect
from audit_causal_stream import audit


def run(args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        subprocess.run([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    artifact=Path('artifacts/impact/causal_stream_v1');artifact.mkdir(parents=True,exist_ok=True)
    logs=Path('logs/impact_causal_stream_v1');logs.mkdir(parents=True,exist_ok=True)
    for rep in range(1,6):
        for role in ('train','validation'):
            collect(rep,role)
            audit(Path(f'datasets/go1_impact/causal_stream_v1/rep{rep}/{role}/cache'))
        run(['scripts/impact/train_causal_stream.py','--rep',str(rep)],logs/f'rep{rep}_training.log')
        run(['scripts/impact/evaluate_causal_stream.py','--rep',str(rep),'--calibrate'],logs/f'rep{rep}_calibration.log')
        print('STREAM_REP_LOCK_READY',rep,flush=True)
    sources=[Path('configs/impact/causal_stream_v1.json'),Path('docs/GO1_IMPACT_CAUSAL_STREAM_PLAN.md'),
             Path('source/himloco_lab/himloco_lab/impact/causal_stream.py'),
             Path('source/himloco_lab/himloco_lab/impact/observability.py'),Path('scripts/impact/collect.py')]
    sources+=list(Path('scripts/impact').glob('*causal_stream*.py'))
    lock=dict(protocol_sha256=file_hash('configs/impact/causal_stream_v1.json'),
              source_sha256={str(p):file_hash(p) for p in sources},models={},operating_points={})
    for rep in range(1,6):
        for name in ('models','frozen','pd'):
            path=logs/f'rep{rep}/{name}.pt';lock['models'][str(path)]=file_hash(path)
        path=artifact/f'rep{rep}/operating_points.json';lock['operating_points'][str(path)]=file_hash(path)
    path=artifact/'model_lock.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:path.write_text(json.dumps(lock,indent=2)+'\n')
    print('STREAM_ALL_MODELS_AND_OPERATING_POINTS_LOCKED',flush=True)
    for rep in range(1,6):
        for role in ('test','high_speed'):
            collect(rep,role)
            audit(Path(f'datasets/go1_impact/causal_stream_v1/rep{rep}/{role}/cache'))
        run(['scripts/impact/evaluate_causal_stream.py','--rep',str(rep)],logs/f'rep{rep}_evaluation.log')
        print('STREAM_REP_EVALUATED',rep,flush=True)
    run(['scripts/impact/evaluate_causal_stream.py','--summarize'],logs/'summary.log')
    print('STREAM_COMPLETE',flush=True)


if __name__=='__main__':main()
