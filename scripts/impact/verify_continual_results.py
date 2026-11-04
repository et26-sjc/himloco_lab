"""End-to-end artifact checks, memory accounting and executed-source provenance."""
import json
from pathlib import Path
import re

import numpy as np

from himloco_lab.impact.dataset import file_hash


def main():
    root=Path('artifacts/impact/continual_replay_v1')
    conf=root/'confirmation';logs=Path('logs/impact_continual_replay_v1/confirmation')
    lock=json.loads((logs/'ALL_MODELS_LOCKED.json').read_text())
    protocol=json.loads(Path('configs/impact/continual_replay_v1_lock.json').read_text())
    assert lock['protocol_sha256']==file_hash(Path('configs/impact/continual_replay_v1_lock.json'))
    for p,sha in protocol['source_sha256'].items():assert file_hash(Path(p))==sha,p
    assert protocol['analysis_locked_before_confirmation_training']['evaluator']==file_hash(Path('scripts/impact/evaluate_continual_confirmation.py'))
    assert json.loads((conf/'summary.json').read_text())['status']=='completed'
    previous=json.loads(Path('artifacts/impact/force_study_v1/final_verification.json').read_text())
    for p,sha in previous['unchanged_baseline_sha256'].items():assert file_hash(Path(p))==sha,p
    tests=Path('logs/impact_continual_validation_tests.log')
    test_text=tests.read_text();test_count=int(re.search(r'Ran (\d+) tests',test_text).group(1))
    assert test_text.rstrip().endswith('OK') and test_count>=37
    for p,sha in lock['weights_sha256'].items():assert file_hash(Path(p))==sha,p
    roles=[];count=0;max_memory=0;max_cpu_gpu_arrays=0
    for rep in range(1,6):
        report=json.loads((conf/f'rep{rep}'/'training.json').read_text())
        assert report['status']=='completed' and len(report['names'])==16 and len(report['history'])==11
        assert report['lock_sha256']==lock['protocol_sha256']
        assert report['data_metadata_sha256']['train']==file_hash(Path(f'datasets/go1_impact/continual_replay_v1/confirmation/rep{rep}/cache/train/metadata.json'))
        audits={r:json.loads((conf/f'rep{rep}'/'data'/r/'audit.json').read_text()) for r in ('train','monitor','calibration','test')}
        expected=sum(audits['train']['domains'][d]['robot_intervals'] for d in ('A','B','C'))
        for memory in report['history'][-1]['memory'].values():
            assert memory['seen']==expected
            assert sum(memory['counts'])==expected
            assert memory['payload_bytes']<=memory['serialized_bytes']<=8388608
            max_memory=max(max_memory,memory['serialized_bytes'])
            # The implementation caches the SAME allowed records on GPU.
            # Count this physical duplicate separately from unique replay bytes.
            gpu_cache=memory['rows']*(270+32+1)*4
            max_cpu_gpu_arrays=max(max_cpu_gpu_arrays,memory['payload_bytes']+gpu_cache)
        test=json.loads((conf/f'rep{rep}'/'test.json').read_text())
        assert len(test['records'])==18 and test['model_lock_sha256']==file_hash(logs/'ALL_MODELS_LOCKED.json')
        for r in test['records']:
            assert all(np.isfinite(r['domains'][d]['AP']) for d in ('A','B','C','U'))
            assert r['calibration']['selection']['false_alarms_per_robot_minute']<=.5
        for role,audit in audits.items():
            assert audit['status']=='passed' and audit['native_target_max_error']==0.
            raw=json.loads((conf/f'rep{rep}'/'data'/role/'report.json').read_text())
            baseline=previous['unchanged_baseline_sha256']
            assert raw['checkpoint_sha256']==baseline['logs/himloco_rsl_rl/go1_rough_pd/2026-10-22_23-51-36_go1_pd_him_seed1/model_10000.pt']
            assert raw['policy_sha256']==baseline['go1_lab_deploy/models/policy.pt']
            assert raw['observation_noise'] and raw['num_envs']==4096
            path=Path(f'datasets/go1_impact/continual_replay_v1/confirmation/rep{rep}/cache/{role}')
            meta=json.loads((path/'metadata.json').read_text())
            for key,sha in meta['files_sha256'].items():assert file_hash(path/f'{key}.npy')==sha
            assert file_hash(path/'native_audit.npz')==audit['native_audit_sha256']
            # Retain the corresponding original target slice for later audits
            # even when the bulk compact arrays are removed.
            with np.load(path/'native_audit.npz') as native:env=native['env_id']
            y=np.load(path/'targets.npy',mmap_mode='r')[:,env]
            target_path=path/'native_audit_targets.npz';np.savez_compressed(target_path,targets=y)
            proprio_path=path/'native_audit_proprio.npz'
            np.savez_compressed(proprio_path,
                pre=np.load(path/'pre.npy',mmap_mode='r')[:,env],post=np.load(path/'post.npy',mmap_mode='r')[:,env],
                done=np.load(path/'done.npy',mmap_mode='r')[:,env],terminated=np.load(path/'terminated.npy',mmap_mode='r')[:,env],
                family_id=np.asarray(meta['terrain']['family_id'])[env],tile_id=np.asarray(meta['terrain']['tile_id'])[env])
            roles.append({'rep':rep,'role':role,'seed':audit['seed'],'native_target_slice_sha256':file_hash(target_path),
                          'native_proprio_slice_sha256':file_hash(proprio_path)})
            count+=1
    result={'status':'passed','data_roles':count,'independent_collections':5,'initializations':2,
            'fixed_endpoint_checkpoint_files':len(lock['weights_sha256']),'method_seed_sequences':90,
            'max_serialized_replay_bytes':max_memory,
            'replay_budget_bytes':8388608,'all_unique_training_rows_inserted_once':True,
            'memory_budget_scope':'Serialized unique replay state, not total CPU+GPU process memory. GPU cache duplicates the same records; no additional past rows.',
            'max_replay_CPU_plus_GPU_persistent_array_bytes':max_cpu_gpu_arrays,
            'all_native_audit_target_errors':0.,'native_audit_targets':roles,
            'test_scoring_after_all_models_lock':True,'primary_results':str(conf/'summary.json')}
    result['unchanged_baseline_sha256']=previous['unchanged_baseline_sha256']
    result['contract_tests']={'passed':test_count,'log_sha256':file_hash(tests)}
    (root/'final_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='native_audit_targets'}))


if __name__=='__main__':main()
