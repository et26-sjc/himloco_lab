#!/usr/bin/env bash
# Operator stop, restricted to this release; no broad kill or restart.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
if [[ "$#" -ne 0 ]]; then
    echo "Usage: sudo bash $0" >&2
    exit 2
fi
echo 'Stopping requests zero gains. The body must be supported.'
him_cid=$(docker ps -aq --filter "name=^/${him_container}$")
if [[ -n "$him_cid" ]]; then
    him_owner=$(docker inspect -f '{{index .Config.Labels "him.deploy.root"}}' "$him_cid")
    if [[ "$him_owner" != "$him_root" ]]; then
        echo 'Container name belongs to another release; stop it through its own operator interface.' >&2
        exit 2
    fi
    # Created with SIGINT and --init: Python can publish stop and flush its log.
    if ! docker stop --time 5 "$him_cid"; then
        if docker inspect "$him_cid" >/dev/null 2>&1; then
            echo 'Container stop failed. Use remote L2 and inspect before proceeding.' >&2
            exit 2
        fi
    fi
fi
# Stop the bridge after the policy has finished. Its SIGTERM handler sends
# inactive frames before exit. Match /proc/exe, never a broad pgrep kill.
while IFS= read -r him_pid; do
    if [[ "$(readlink -f -- "/proc/$him_pid/exe" 2>/dev/null || true)" == "$him_binary" ]]; then
        kill -TERM "$him_pid"
        echo "Bridge SIGTERM requested: $him_pid"
    fi
done < <(pgrep -f '[l]cm_position_him' || true)
echo 'Stop requested; physical power-off is not verified. Restart explicitly to rearm.'
