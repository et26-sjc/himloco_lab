"""Preserve force-study methods/weights and hash all retained data in place.

No raw dataset is duplicated, moved, removed, or modified. The small bundle is
an independent copy of methods/results/checkpoints; the data manifest identifies
the retained local dataset and is not an off-machine backup.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
DATA = Path('datasets/go1_impact/force_study_v1')
ART = Path('artifacts/impact/force_study_v1')
LOG = Path('logs/impact_force_study_v1')


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def file_digest(path):
    with path.open('rb') as stream:
        return digest(stream)


def files_under(relative):
    path = ROOT / relative
    return [path] if path.is_file() else sorted(
        p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts)


def create(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'INCOMPLETE').touch()
    required = json.loads((ROOT / ART / 'final_verification.json').read_text())
    assert required['status'] == 'passed'
    assert required['completed_model_instances'] == 780
    prefixes = [Path('scripts/impact'), Path('source/himloco_lab/himloco_lab/impact'),
                Path('configs/impact'), ART, LOG]
    paths = {p for prefix in prefixes for p in files_under(prefix)}
    paths.update((ROOT / 'tests').glob('test_impact*.py'))
    paths.add(ROOT / 'tests/test_force_study.py')
    paths.update((ROOT / 'docs').glob('GO1_*IMPACT*.md'))
    paths.update((ROOT / 'docs').glob('GO1_FORCE_LOADING_RATE*.md'))
    paths.update(ROOT / p for p in required['unchanged_baseline_sha256'])
    paths.update(ROOT / p for p in ('README.md', 'README_CN.md'))
    method_rows = []
    bundle = output / 'methods_results_weights.tar.gz'
    with tarfile.open(bundle, 'x:gz', compresslevel=1) as archive:
        for path in sorted(paths):
            relative = path.relative_to(ROOT).as_posix()
            sha = file_digest(path)
            archive.add(path, arcname=relative, recursive=False)
            method_rows.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha})
    data_rows = []
    for index, path in enumerate(files_under(DATA), 1):
        before = path.stat()
        sha = file_digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Data changed during hashing: {path}')
        data_rows.append({'path': path.relative_to(ROOT).as_posix(),
                          'bytes': after.st_size, 'sha256': sha})
        if index % 100 == 0:
            print('DATA_HASHED', index, flush=True)
    record = {
        'status': 'preserved', 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'stage': 'force_study_v1',
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'baseline_git_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'bundle': bundle.name, 'bundle_sha256': file_digest(bundle),
        'bundle_files': method_rows, 'retained_data_files': data_rows,
        'retained_data_bytes': sum(r['bytes'] for r in data_rows),
        'data_policy': 'All original files retained in place, including failed diagnostics. '
                       'Full SHA-256 inventory; raw data not duplicated or deleted. '
                       'This is a local preservation snapshot, not an external backup.',
        'result_interpretation': 'Includes incomplete/failed diagnostic directories for provenance; '
                                 'only previously audited completed runs support reported conclusions.'}
    (output / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    verify(output, include_data=False)
    (output / 'INCOMPLETE').unlink()
    print('FORCE_STAGE_PRESERVED', len(method_rows), len(data_rows), record['retained_data_bytes'], flush=True)


def verify(output, include_data):
    record = json.loads((output / 'manifest.json').read_text())
    if include_data and (output / 'DATA_REMOVAL.json').exists():
        raise SystemExit('Dataset binaries were removed by user request. '
                         'See DATA_REMOVAL.json; methods/results/weights remain verifiable without --include-data.')
    bundle = output / record['bundle']
    assert file_digest(bundle) == record['bundle_sha256']
    expected = {r['path']: r for r in record['bundle_files']}
    with tarfile.open(bundle, 'r:gz') as archive:
        members = archive.getmembers()
        assert len(members) == len(expected)
        for member in members:
            row = expected[member.name]
            assert member.isfile() and member.size == row['bytes']
            with archive.extractfile(member) as stream:
                assert digest(stream) == row['sha256'], member.name
    if include_data:
        for index, row in enumerate(record['retained_data_files'], 1):
            path = ROOT / row['path']
            assert path.stat().st_size == row['bytes']
            assert file_digest(path) == row['sha256'], row['path']
            if index % 100 == 0:
                print('DATA_VERIFIED', index, flush=True)
    print('ARCHIVE_CONTENT_VERIFIED', len(expected), 'data_rehashed', include_data, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/impact/archives/force_study_v1_20261104')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--include-data', action='store_true', help='Rehash the complete ~220 GiB retained dataset')
    args = parser.parse_args()
    if args.verify:
        verify(args.output, args.include_data)
    else:
        create(args.output)
