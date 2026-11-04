"""Reproduce A0/A1 from scratch with isolated data and run directories.

All stages refuse to overwrite their result directories. --stage allows a
completed earlier stage to be reused explicitly. This never starts PPO.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run(command, log):
    print('RUN', ' '.join(str(x) for x in command), flush=True)
    with Path(log).open('x') as stream:
        subprocess.run([str(x) for x in command],stdout=stream,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--spec',type=Path,default=Path('configs/impact/step1.json'))
    p.add_argument('--stage',choices=['all','collect','prepare','train','evaluate','export','summarize'],default='all')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    spec=json.loads(args.spec.read_text())
    args.run_root.mkdir(parents=True,exist_ok=True)
    py=sys.executable
    if args.stage in ('all','collect'):
        for split,seed in spec['collection_seeds'].items():
            run([py,'scripts/impact/collect.py','--headless','--device',args.device,
                 '--num_envs',spec['num_envs'],'--steps',spec['collection_steps'][split],
                 '--scenario','suite_walk','--seed',seed,'--observation-noise',
                 '--output',args.data_root/'raw'/split],args.run_root/f'{split}_collect.console.log')
        for dt,name in [('.005','drop_200hz'),('.0025','drop_400hz')]:
            run([py,'scripts/impact/collect.py','--headless','--device',args.device,
                 '--num_envs',64,'--steps',64,'--scenario','drop','--seed',19,
                 '--drop-height-range',.4,.9,'--physics-dt',dt,'--output',args.data_root/name],
                args.run_root/f'{name}.console.log')
    if args.stage in ('all','prepare'):
        for split in spec['collection_seeds']:
            run([py,'scripts/impact/audit_collection.py',args.data_root/'raw'/split,
                 '--output',args.run_root/f'{split}_audit.json'],args.run_root/f'{split}_audit.console.log')
            run([py,'scripts/impact/prepare_dataset.py',args.data_root/'raw'/split,
                 '--output',args.data_root/'cache'/split,'--spec',args.spec],args.run_root/f'{split}_prepare.console.log')
        run([py,'scripts/impact/physical_diagnostics.py',args.data_root/'drop_200hz',args.data_root/'drop_400hz',
             '--output',args.run_root/'physical_diagnostics.json'],args.run_root/'physical_diagnostics.console.log')
    if args.stage in ('all','train'):
        run([py,'scripts/impact/train.py','--train',args.data_root/'cache'/'train',
             '--validation',args.data_root/'cache'/'validation','--output',args.run_root/'training',
             '--spec',args.spec,'--device',args.device],args.run_root/'training.console.log')
    if args.stage in ('all','evaluate'):
        run([py,'scripts/impact/evaluate.py','--training',args.run_root/'training',
             '--validation',args.data_root/'cache'/'validation','--test',args.data_root/'cache'/'test',
             '--output',args.run_root/'evaluation','--device',args.device],args.run_root/'evaluation.console.log')
    if args.stage in ('all','export'):
        run([py,'scripts/impact/export.py',args.run_root/'training','--output',args.run_root/'exports'],
            args.run_root/'export.console.log')
    if args.stage in ('all','summarize'):
        run([py,'scripts/impact/summarize.py','--training',args.run_root/'training',
             '--evaluation',args.run_root/'evaluation','--diagnostics',args.run_root/'physical_diagnostics.json',
             '--output',args.run_root/'review'],args.run_root/'summary.console.log')
