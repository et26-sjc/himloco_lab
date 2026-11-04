"""Wait for collection/HPO completion, lock all arms, then evaluate once."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

logs=Path('logs/impact_force_study_v1')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wait-for-rep1',action='store_true',help='Use when rep1 is already training in another process')
args=parser.parse_args()
def run(command,name):
    path=logs/f'{name}.console.log'
    print('RUN',name,flush=True)
    with path.open('a') as stream:
        subprocess.run([sys.executable,*command],stdout=stream,stderr=subprocess.STDOUT,check=True)

while not (Path('datasets/go1_impact/force_study_v1/raw/rep5/test/report.json').exists()
           and (logs/'hpo/selected.json').exists()
           and (not args.wait_for_rep1 or (logs/'main/rep1/I0/forecast/report.json').exists())):
    time.sleep(10)

if not Path('datasets/go1_impact/force_study_v1/paired_branches/report.json').exists():
    run(['scripts/impact/paired_contact_branches.py','--headless','--device','cuda:0'],'paired_branches')
for rep in range(1,6):
    run(['scripts/impact/train_force_study.py','--rep',str(rep)],f'main_rep{rep}')
for kind in ('I0clean','I1','I2'):
    for rep in range(1,6):
        run(['scripts/impact/train_force_study.py','--rep',str(rep),'--input',kind,
             '--tasks','detection'],f'inputs_{kind}_rep{rep}')
# Secondary methods are declared before looking at the locked final test.
for rep in range(1,6):
    run(['scripts/impact/train_force_study.py','--rep',str(rep),'--group','alternatives',
         '--methods','huber','logmse','cdf','--targets','force20','loading20','impact20'],f'alternatives_rep{rep}')
(logs/'ALL_MODELS_LOCKED.json').write_text(json.dumps({'status':'all_requested_models_trained_before_final_test','replications':5},indent=2)+'\n')
run(['scripts/impact/evaluate_force_parallel.py'],'evaluation_parallel')
print('FORCE_STUDY_COMPLETE',flush=True)
