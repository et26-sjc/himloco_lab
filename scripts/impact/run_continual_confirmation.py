"""Sequential GPU orchestration: collect one replication, train, then evaluate all.

Do not run concurrently with an Isaac collection or another training process.
"""
import json
from pathlib import Path
import subprocess
import sys

from himloco_lab.impact.dataset import file_hash


def main():
    logs=Path('logs/impact_continual_replay_v1/confirmation');logs.mkdir(parents=True,exist_ok=True)
    for rep in range(1,6):
        for name,command in [
            ('collection',[sys.executable,'scripts/impact/collect_continual_confirmation.py','--reps',str(rep)]),
            ('training',[sys.executable,'scripts/impact/train_continual_confirmation.py','--rep',str(rep)])]:
            print('START',rep,name,flush=True)
            with (logs/f'rep{rep}_{name}_driver.log').open('a') as stream:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
            print('FINISH',rep,name,flush=True)
    paths=sorted(logs.glob('rep*/seed*.pt'))
    lock={'status':'all_models_locked_before_final_test_scoring',
          'protocol_sha256':file_hash(Path('configs/impact/continual_replay_v1_lock.json')),
          'weights_sha256':{str(p):file_hash(p) for p in paths}}
    (logs/'ALL_MODELS_LOCKED.json').write_text(json.dumps(lock,indent=2)+'\n')
    with (logs/'evaluation_driver.log').open('a') as stream:
        subprocess.run([sys.executable,'scripts/impact/evaluate_continual_confirmation.py'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('CONTINUAL_CONFIRMATION_COMPLETE',flush=True)


if __name__=='__main__':main()
