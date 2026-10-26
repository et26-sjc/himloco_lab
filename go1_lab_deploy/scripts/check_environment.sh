#!/usr/bin/env bash
# Offline model/configuration preflight inside the existing Jetson image.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "${1:-}" == --describe && "$#" -eq 1 ]]; then
    echo "Export root: $him_root"
    echo "Docker image: $him_image"
    echo 'Offline --check only, --network none; no motor commands or LCM.'
    exit 0
fi
if [[ "$#" -ne 0 ]]; then
    echo "Usage: bash $0 [--describe]" >&2
    exit 2
fi
him_verify_sources
docker image inspect "$him_image" --format 'Docker image: {{.Id}}; architecture: {{.Architecture}}'
docker run --rm --network none --runtime nvidia --entrypoint python3 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONIOENCODING=utf-8 \
    -v "$him_root:/opt/him:ro" "$him_image" \
    -u -B /opt/him/go1_lab_deploy/scripts/deploy_policy.py \
    --config /opt/him/himloco.json --check
