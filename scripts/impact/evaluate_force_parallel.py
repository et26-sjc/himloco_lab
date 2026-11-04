"""Evaluate locked arms with two independent processes; never changes models."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from pathlib import Path
import subprocess
import sys
import time


def job(item):
    group,rep,kind,tasks,calibration_only=item
    phase='validation_calibration' if calibration_only else 'parallel_evaluation'
    name=f'{phase}_{group}_{kind}_rep{rep}'
    log=Path('logs/impact_force_study_v1')/f'{name}.console.log'
    model_root=Path('logs/impact_force_study_v1')/group/f'rep{rep}'/kind
    while not all((model_root/task/'report.json').exists() for task in tasks):
        time.sleep(10)
    command=[sys.executable,'scripts/impact/evaluate_force_study.py','--rep',str(rep),
             '--group',group,'--input',kind,'--tasks',*tasks]
    if calibration_only:command.append('--calibration-only')
    with log.open('a') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('PARALLEL_EVALUATION_COMPLETE',group,rep,kind,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-only',action='store_true')
    args=parser.parse_args()
    while not args.calibration_only and not Path('logs/impact_force_study_v1/ALL_MODELS_LOCKED.json').exists():
        time.sleep(10)
    jobs=[('main',rep,'I0',['detection','forecast'],args.calibration_only) for rep in range(1,6)]
    jobs += [('main',rep,kind,['detection'],args.calibration_only) for kind in ('I0clean','I1','I2') for rep in range(1,6)]
    jobs += [('alternatives',rep,'I0',['detection','forecast'],args.calibration_only) for rep in range(1,6)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(job,jobs))
    print('ALL_FORCE_EVALUATIONS_COMPLETE',flush=True)
