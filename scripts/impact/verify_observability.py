"""Re-score the primary result from preserved events/scores, without full frames."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from himloco_lab.impact.dataset import file_hash
from himloco_lab.impact.observability import matched_auc


def main():
    root=Path('artifacts/impact/observability_v1')
    summary=json.loads((root/'summary.json').read_text())
    lock=json.loads((root/'model_lock.json').read_text())
    for path,digest in lock['models'].items():assert file_hash(path)==digest
    for path,digest in lock['source_sha256'].items():assert file_hash(path)==digest,path
    differences=[]
    for rep in range(1,6):
        cache=Path('datasets/go1_impact/observability_v1')/f'rep{rep}/test/cache'
        events=dict(np.load(cache/'events.npz'));meta=json.loads((cache/'metadata.json').read_text())
        per_input={}
        for kind in ('phase','full'):
            pred=dict(np.load(root/f'rep{rep}/test_{kind}_predictions.npz'))
            report=json.loads((Path('logs/impact_observability_v1')/f'rep{rep}'/kind/'report.json').read_text())
            ids=pred['event_id']
            family=np.asarray(meta['terrain']['family_id'])[events['env'][ids]]
            use=(family==6)&events['valid_cutoff'][ids,3]
            values=[]
            for index,(seed,method,h) in enumerate(report['entries']):
                if method=='bce' and h==3:
                    row=matched_auc(events['severity'][ids[use]]>2000,pred['score'][index,use],pred['strata'][use])
                    expected=json.loads((root/f'rep{rep}/results.json').read_text())['results'][f'test/heldout/{kind}/bce/{seed}/3']['matched']['auc']
                    assert abs(row['auc']-expected)<1e-12
                    values.append(row['auc'])
            assert len(values)==2
            per_input[kind]=np.mean(values)
        differences.append(float(per_input['full']-per_input['phase']))
    np.testing.assert_allclose(differences,summary['primary']['replications'],rtol=0,atol=1e-12)
    mean=float(np.mean(differences));delta=float(t.ppf(.975,4)*np.std(differences,ddof=1)/np.sqrt(5))
    np.testing.assert_allclose([mean-delta,mean+delta],summary['primary']['CI95'],rtol=0,atol=1e-12)
    audits=[]
    for path in sorted(root.glob('rep*/*/data/native_audit.json')):
        report=json.loads(path.read_text());assert report['status']=='passed';audits.append(report)
    assert len(audits)==20
    dependencies=[Path('scripts/impact/train_force_study.py')]+list(Path('source/himloco_lab/himloco_lab/impact').glob('*.py'))
    test_evidence=json.loads((root/'contract_tests.json').read_text())
    assert test_evidence['successful']
    for path,digest in test_evidence['tested_source_sha256'].items():assert file_hash(path)==digest,path
    result={'status':'passed','primary_recomputed_without_full_proprio_frames':True,
            'paired_differences':differences,'native_roles_passed':len(audits),
            'native_events_recomputed':sum(a['events_recomputed'] for a in audits),
            'max_native_error_n_per_s':max(a['max_abs_label_error_n_per_s'] for a in audits),
            'formal_models_locked':240,'scientific_contract_tests_passed':test_evidence['count'],
            'dependency_sha256':{str(p):file_hash(p) for p in dependencies}}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('OBS_VERIFIED',json.dumps(result),flush=True)


if __name__=='__main__':main()
