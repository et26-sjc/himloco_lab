#!/usr/bin/env bash
# Foreground bridge, or explicit operator-confirmed nohup/setsid startup.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
if [[ "${1:-}" == --describe && "$#" -eq 1 ]]; then
    echo "Export root: $him_root"
    echo "Bridge: $him_binary"
    echo 'Terminal A: manual Enter; SDK 3.8, UDP 8090, 2 ms control period.'
    echo 'Use --background for an operator-confirmed detached bridge.'
    echo 'Offline description only; no bridge is started.'
    exit 0
fi
him_background=false
if [[ "${1:-}" == --background && "$#" -eq 1 ]]; then
    him_background=true
elif [[ "$#" -ne 0 ]]; then
    echo "Usage: sudo bash $0 [--describe | --background]" >&2
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
if "$him_background"; then
    command -v nohup >/dev/null
    command -v setsid >/dev/null
    him_verify_bridge
    if [[ ! -t 0 ]]; then
        echo 'Run --background interactively to confirm bridge startup with Enter.' >&2
        exit 2
    fi
    echo 'Support the body; keep all legs clear. This starts communication, not calibration.'
    read -r -p 'Press Enter to start the detached bridge; Ctrl+C cancels: ' him_confirmation
    mkdir -p -- "$him_log_dir"
    him_console_name="bridge_$(date +%Y%m%d_%H%M%S)_$$.console.log"
    # The operator just confirmed. A private newline satisfies the unchanged
    # C++ prompt, with no stdin/stdout/stderr tied to the SSH terminal.
    nohup setsid "$him_binary" --lcm-url "$HIM_LCM_URL" \
        >"$him_log_dir/$him_console_name" 2>&1 <<< '' &
    him_background_pid=$!
    printf '%s\n' "$him_background_pid" >"$him_log_dir/background_bridge.pid"
    ln -sfn -- "$him_console_name" "$him_log_dir/latest_bridge.console.log"
    echo "Bridge launch PID: $him_background_pid"
    echo "Bridge output: $him_log_dir/$him_console_name"
    exit 0
fi
echo 'HIM terminal A. Keep body supported and legs clear before calibration.'
echo 'Press Enter at the bridge prompt, then start terminal B.'
echo 'After Python requests stop, stop this bridge with Ctrl+C; never Ctrl+Z.'
exec bash "$him_root/go1_lab_deploy/autostart/start_unitree_sdk.sh" "$him_binary"
