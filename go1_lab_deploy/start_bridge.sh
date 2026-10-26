#!/usr/bin/env bash
# Terminal A: foreground HIM SDK bridge, retaining its manual Enter prompt.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
if [[ "${1:-}" == --describe && "$#" -eq 1 ]]; then
    echo "Export root: $him_root"
    echo "Bridge: $him_binary"
    echo 'Terminal A: manual Enter; SDK 3.8, UDP 8090, 2 ms control period.'
    echo 'Offline description only; no bridge is started.'
    exit 0
fi
if [[ "$#" -ne 0 ]]; then
    echo "Usage: sudo bash $0 [--describe]" >&2
    exit 2
fi
if [[ "$(uname -m)" != aarch64 && "$(uname -m)" != arm64 ]]; then
    echo 'This exported baseline is for the ARM64 Go1 Jetson host.' >&2
    exit 2
fi
him_verify_sources
him_check_lcm_url
if pgrep -af '[l]cm_position|[d]eploy_policy.py|[w]tw_supported_trial.py|[g]o1_fr.*sdk38'; then
    echo 'A bridge or controller is already present. Inspect and stop it first.' >&2
    exit 2
fi
if [[ -n "$(ss -H -uan 'sport = :8090')" ]]; then
    echo 'UDP 8090 is occupied. Do not start a second bridge.' >&2
    exit 2
fi
echo 'HIM terminal A. Keep body supported and legs clear before calibration.'
echo 'Press Enter at the bridge prompt, then start terminal B.'
echo 'After Python requests stop, stop this bridge with Ctrl+C; never Ctrl+Z.'
exec bash "$him_root/go1_lab_deploy/autostart/start_unitree_sdk.sh" "$him_binary"
