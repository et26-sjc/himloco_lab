#!/usr/bin/env bash
# Terminal B: native Lab PD policy; inherited Gym hardware transport/state machine.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
if [[ "${1:-}" == --describe && "$#" -eq 1 ]]; then
    echo "Export root: $him_root"
    echo "Configuration: $him_root/himloco.json"
    echo "Model: $him_root/models/policy.pt"
    echo 'Policy interface: lab_go1_pd_v1; Lab joint order; unscaled velocity commands.'
    echo "Docker image/container: $him_image / $him_container"
    echo "Persistent logs: $him_log_dir"
    echo 'Kp=40, Kd=1; 50 Hz. Configured vx/vy/yaw command maxima:'
    "$him_python" -B - "$him_root/himloco.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    print(json.load(stream)['runtime']['max_abs_command'])
PY
    echo 'Use --background for nohup/setsid without Docker -it; actual limits come from himloco.json.'
    echo 'Target-step stop disabled. RAMP/RUNNING tracking uses the configured persistence and instant threshold.'
    echo 'Calibration tracking, mechanical bounds, communication watchdogs and stop buttons retained.'
    echo 'First R2 calibrates; second R2 in READY starts policy. R2 stops RAMP/RUNNING; L2 stops every phase.'
    echo 'Offline description only; no container or publisher started.'
    exit 0
fi
him_background=false
if [[ "${1:-}" == --background && "$#" -eq 1 ]]; then
    him_background=true
elif [[ "$#" -ne 0 ]]; then
    echo "Usage: sudo bash $0 [--describe | --background]" >&2
    exit 2
fi
him_verify_sources
him_check_lcm_url
him_pid=''
while IFS= read -r candidate; do
    if [[ -r "/proc/$candidate/cmdline" && -e "/proc/$candidate/exe" ]]; then
        him_exe=$(readlink -f -- "/proc/$candidate/exe" || true)
        him_cmd=$(tr '\0' ' ' < "/proc/$candidate/cmdline")
        if [[ "$him_exe" == "$him_binary" ]]; then
            if [[ "$him_cmd" == *--observe-only* ]]; then
                echo 'The bridge is observe-only and cannot run the policy.' >&2
                exit 2
            fi
            him_pid=$candidate
            break
        fi
    fi
done < <(pgrep -f '[l]cm_position_him' || true)
if [[ -z "$him_pid" ]]; then
    echo "Start terminal A with this export's start_bridge.sh and press Enter first." >&2
    exit 2
fi
if [[ -z "$(ss -H -uan 'sport = :8090')" ]]; then
    echo 'Bridge UDP 8090 is not bound. Complete the terminal A Enter prompt.' >&2
    exit 2
fi
if pgrep -af '[d]eploy_policy.py|[w]tw_supported_trial.py'; then
    echo 'A Python controller is already running. Inspect it before starting another.' >&2
    exit 2
fi
if [[ -n "$(docker ps -aq --filter "name=^/${him_container}$")" ]]; then
    echo "Container $him_container already exists. Inspect it before restarting." >&2
    exit 2
fi
docker image inspect "$him_image" >/dev/null
mkdir -p -- "$him_log_dir"
him_log_name="him_lab_$(date +%Y%m%d_%H%M%S)_$$.jsonl"
echo "HIM log: $him_log_dir/$him_log_name"
echo 'Center sticks. First R2 calibrates; release and press R2 again only in READY.'
echo 'L2 stops every phase. R2 stops RAMP/RUNNING. No automatic duration limit.'
echo 'Stopping requests zero gains; body support must be ready.'
him_docker_args=(run --rm --init --name "$him_container" --network host --runtime nvidia
    --stop-signal SIGINT --stop-timeout 5 --label "him.deploy.root=$him_root"
    --entrypoint python3 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONIOENCODING=utf-8
    -v "$him_root:/opt/him:ro" -v "$him_log_dir:/logs")
if "$him_background"; then
    command -v nohup >/dev/null
    command -v setsid >/dev/null
else
    him_docker_args+=(-it)
fi
him_docker_args+=("$him_image" -u -B /opt/him/go1_lab_deploy/scripts/deploy_policy.py
    --config /opt/him/himloco.json --run --log "/logs/$him_log_name")
if "$him_background"; then
    him_console_name="${him_log_name%.jsonl}.console.log"
    nohup setsid docker "${him_docker_args[@]}" \
        >"$him_log_dir/$him_console_name" 2>&1 </dev/null &
    him_background_pid=$!
    printf '%s\n' "$him_background_pid" >"$him_log_dir/background_policy_client.pid"
    ln -sfn -- "$him_console_name" "$him_log_dir/latest_policy.console.log"
    echo "Policy Docker client launch PID: $him_background_pid"
    echo "Policy output: $him_log_dir/$him_console_name"
    echo 'Wait for HIM phase: WAIT_R2 in the output before using the remote.'
    echo 'The process is detached; Ctrl+C in this SSH terminal does NOT stop the policy.'
    echo 'Use remote L2/R2 or stop_background.sh to stop; support the body first.'
    exit 0
fi
echo 'Foreground mode: Ctrl+C here, then Ctrl+C terminal A.'
exec docker "${him_docker_args[@]}"
