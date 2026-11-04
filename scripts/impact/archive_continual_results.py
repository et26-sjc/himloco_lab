"""Preserve methods/weights/native audits, optionally remove verified bulk arrays.

Only this experiment's generated .npy arrays and redundant combined resume
copies are eligible for removal. Final model/Adam/RNG/bounded replay, fixed-node weights,
baseline, other studies, metadata and native audits remain preserved.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import tarfile

import torch

from himloco_lab.impact.dataset import file_hash
from archive_force_stage import digest

ROOT=Path('.').resolve()
ART=Path('artifacts/impact/continual_replay_v1')
DATA=Path('datasets/go1_impact/continual_replay_v1')
LOG=Path('logs/impact_continual_replay_v1')
DEST=Path('artifacts/impact/archives/continual_replay_v1_20261104')


def verify(manifest):
    path=DEST/manifest['bundle']
    assert file_hash(path)==manifest['bundle_sha256']
    rows={r['path']:r for r in manifest['bundle_files']}
    with tarfile.open(path,'r:gz') as tar:
        assert len(tar.getmembers())==len(rows)
        for member in tar.getmembers():
            assert member.isfile() and member.size==rows[member.name]['bytes']
            with tar.extractfile(member) as stream:assert digest(stream)==rows[member.name]['sha256']


def main(remove):
    assert json.loads((ART/'final_verification.json').read_text())['status']=='passed'
    assert json.loads((ART/'confirmation/summary.json').read_text())['status']=='completed'
    DEST.mkdir(parents=True,exist_ok=True)
    manifest_path=DEST/'manifest.json'
    if manifest_path.exists():manifest=json.loads(manifest_path.read_text());verify(manifest)
    else:
        # Keep useful final state and bounded memories, drop obsolete indices
        # into the unlimited all-past reference dataset.
        for resume in LOG.glob('confirmation/rep*/resume.pt'):
            dest=resume.with_name('final_training_state.pt')
            if not dest.exists():
                saved=torch.load(resume,map_location='cpu',weights_only=False)
                torch.save({k:saved[k] for k in ('model','optimizer','generator','completed','memories')},dest)
        prefixes=[ART,Path('scripts/impact'),Path('source/himloco_lab/himloco_lab/impact'),Path('configs/impact')]
        paths={p for prefix in prefixes for p in prefix.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        paths.update(Path('tests').glob('test_continual*.py'))
        paths.update(Path('docs').glob('GO1_IMPACT_CONTINUAL*.md'))
        paths.add(Path('README.md'))
        paths.add(Path('README_CN.md'))
        paths.update(p for p in [Path('logs/impact_continual_validation_tests.log'),
                                Path('logs/impact_continual_replay_v1_confirmation.log'),
                                Path('logs/impact_continual_replay_v1_pilot.log')] if p.exists())
        paths.update(LOG.glob('confirmation/*driver.log'))
        paths.update(LOG.rglob('seed*_B5.pt'));paths.update(LOG.rglob('seed*_C5.pt'))
        paths.update(LOG.rglob('seed*_freeze_A0.pt'));paths.update(LOG.rglob('stage0.pt'))
        paths.update(LOG.rglob('final_training_state.pt'))
        paths.update(LOG.glob('pilot_*.pt'))
        paths.update(p for p in DATA.rglob('*') if p.is_file() and p.suffix in ('.json','.npz'))
        bundle=DEST/'methods_results_weights.tar.gz';rows=[]
        with tarfile.open(bundle,'x:gz',compresslevel=1) as tar:
            for p in sorted(paths):
                sha=file_hash(p);tar.add(p,arcname=str(p),recursive=False)
                rows.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha})
        arrays=[]
        for p in sorted(DATA.rglob('*.npy')):
            assert 'cache' in p.parts and p.name in ('pre.npy','post.npy','targets.npy','age.npy','done.npy','terminated.npy')
            arrays.append({'path':str(p),'bytes':p.stat().st_size,'sha256':file_hash(p)})
        manifest={'created_UTC':datetime.now(timezone.utc).isoformat(),'stage':'continual_replay_v1',
                  'bundle':bundle.name,'bundle_sha256':file_hash(bundle),'bundle_files':rows,
                  'bulk_data_files':arrays,'bulk_data_bytes':sum(r['bytes'] for r in arrays),
                  'policy':'Methods/results/selected weights and small predetermined native audits backed up. Bulk arrays excluded.'}
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');verify(manifest)
    if remove and not (DEST/'DATA_REMOVAL.json').exists():
        freed=0;missing=[]
        for row in manifest['bulk_data_files']:
            p=ROOT/row['path']
            assert p.is_relative_to(ROOT/DATA) and p.suffix=='.npy'
            if not p.exists():missing.append(row['path']);continue
            assert p.stat().st_size==row['bytes'] and file_hash(p)==row['sha256']
            freed+=p.stat().st_size;p.unlink()
        resume_rows=[]
        for p in LOG.glob('confirmation/rep*/resume.pt'):
            replacement=p.with_name('final_training_state.pt')
            expected=next(r for r in manifest['bundle_files'] if r['path']==str(replacement))
            assert file_hash(replacement)==expected['sha256']
            resume_rows.append({'path':str(p),'bytes':p.stat().st_size,'sha256':file_hash(p),'model_optimizer_rng_bounded_replay_preserved':str(replacement)})
            freed+=p.stat().st_size;p.unlink()
        removal={'reason':'User requested methods, valid weights and reports rather than retaining bulky datasets.',
                 'removed_bulk_array_files':len(manifest['bulk_data_files'])-len(missing),
                 'removed_files':len(manifest['bulk_data_files'])-len(missing)+len(resume_rows),
                 'already_missing_bulk_files':missing,'freed_bytes':freed,
                 'freed_GiB':freed/2**30,'UTC':datetime.now(timezone.utc).isoformat(),
                 'removed_redundant_resume_files':resume_rows,
                 'preserved':'All metadata, historical hashes, reports, weights/Adam/RNG, bounded replay and native audit force/target/proprio subsets. Full frame-wise recomputation requires recollection.'}
        (DEST/'DATA_REMOVAL.json').write_text(json.dumps(removal,indent=2)+'\n')
        print('BULK_DATA_REMOVED_GiB',freed/2**30)
    print('CONTINUAL_ARCHIVE_VERIFIED',len(manifest['bundle_files']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--remove-compact-data',action='store_true')
    main(parser.parse_args().remove_compact_data)
