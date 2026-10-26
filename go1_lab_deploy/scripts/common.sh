#!/usr/bin/env bash
# Paths for the exported release. Sourcing this file never opens robot I/O.
set -euo pipefail
him_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
him_python=${HIM_PYTHON:-python3}
him_image=${HIM_IMAGE:-jetson-model-deployment}
him_container=${HIM_CONTAINER:-him_go1_lab_policy}
him_binary="$him_root/go1_lab_deploy/unitree_legged_sdk_bin/build/lcm_position_him"
him_log_dir="$him_root/logs"

him_verify_sources() {
    "$him_python" -B "$him_root/go1_lab_deploy/scripts/build_deployment_bundle.py" \
        --verify "$him_root"
}

him_verify_bridge() {
    "$him_python" -B "$him_root/go1_lab_deploy/scripts/verify_bridge_build.py" \
        --binary "$him_binary"
}

him_check_lcm_url() {
    him_configured_url=$("$him_python" -B - "$him_root/himloco.json" "${HIM_LCM_URL:-}" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    configured = json.load(stream)['runtime']['lcm_url']
if sys.argv[2] and sys.argv[2] != configured:
    raise SystemExit('HIM_LCM_URL must equal runtime.lcm_url in himloco.json')
print(configured)
PY
    )
    # Pass the same configured URL to the unchanged C++ bridge as to Python.
    export HIM_LCM_URL="$him_configured_url"
}
