#!/usr/bin/env bash
# Explicit startup only; no boot service and no automatic restart after a fault.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
if [[ "$#" -ne 0 ]]; then
    echo "Usage: sudo bash $0" >&2
    exit 2
fi
docker image inspect "$him_image" >/dev/null
if [[ -n "$(docker ps -aq --filter "name=^/${him_container}$")" ]]; then
    echo "Container $him_container already exists; inspect it before starting." >&2
    exit 2
fi
bash "$him_root/scripts/prepare_background_network.sh"
bash "$him_root/start_bridge.sh" --background
him_bridge_pid=$(cat "$him_log_dir/background_bridge.pid")
him_port_ready=false
for ((him_attempt=0; him_attempt<50; him_attempt++)); do
    if ! kill -0 "$him_bridge_pid" 2>/dev/null; then
        echo "Bridge launch exited; see $him_log_dir/latest_bridge.console.log" >&2
        exit 2
    fi
    if [[ -n "$(ss -H -uan 'sport = :8090')" ]]; then
        him_port_ready=true
        break
    fi
    sleep 0.1
done
if ! "$him_port_ready"; then
    echo 'Bridge UDP port not ready. Inspect status/output and stop this attempt before retrying.' >&2
    exit 2
fi
bash "$him_root/start_policy.sh" --background
echo "Follow startup: tail -f $him_log_dir/latest_policy.console.log"
echo 'After WAIT_R2, center sticks: first R2 calibrates; second R2 in READY starts policy.'
echo 'Disconnect only the computer-side cable after startup. Keep robot internal networking connected.'
echo 'Remote L2 stops every phase; R2 stops RAMP/RUNNING. There is no automatic restart.'
