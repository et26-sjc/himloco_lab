#!/usr/bin/env bash
# Read-only status; never opens robot communication or starts a controller.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
if [[ "$#" -ne 0 ]]; then
    echo "Usage: sudo bash $0" >&2
    exit 2
fi
him_top_pids=()
echo 'LCM/SDK route (LCM should use the internal interface after background startup):'
ip -4 route get 239.255.76.67
ip -4 route get 192.168.123.10
echo 'Bridge processes belonging to this release:'
while IFS= read -r him_pid; do
    if [[ "$(readlink -f -- "/proc/$him_pid/exe" 2>/dev/null || true)" == "$him_binary" ]]; then
        ps -p "$him_pid" -o pid,stat,tty,pcpu,pmem,etime,args
        him_top_pids+=("$him_pid")
    fi
done < <(pgrep -f '[l]cm_position_him' || true)
him_cid=$(docker ps -aq --filter "name=^/${him_container}$")
if [[ -n "$him_cid" ]]; then
    him_owner=$(docker inspect -f '{{index .Config.Labels "him.deploy.root"}}' "$him_cid")
    if [[ "$him_owner" != "$him_root" ]]; then
        echo 'Same container name belongs to another release; not inspecting it further.' >&2
        exit 2
    fi
    docker inspect -f 'Policy container: {{.Name}}; status={{.State.Status}}; host PID={{.State.Pid}}' "$him_cid"
    if [[ "$(docker inspect -f '{{.State.Running}}' "$him_cid")" == true ]]; then
        docker stats --no-stream --format 'Container CPU={{.CPUPerc}} memory={{.MemUsage}}' "$him_cid" || true
        docker top "$him_cid" -eo pid,ppid,stat,pcpu,pmem,comm || true
        # --init makes the first process docker-init; top includes its Python child.
        while IFS= read -r him_pid; do
            [[ "$him_pid" =~ ^[0-9]+$ ]] && him_top_pids+=("$him_pid")
        done < <(docker top "$him_cid" -eo pid 2>/dev/null | tail -n +2 || true)
    fi
else
    echo 'Policy container: not running.'
fi
for him_kind in bridge policy; do
    him_console="$him_log_dir/latest_${him_kind}.console.log"
    if [[ -f "$him_console" ]]; then
        echo "$him_kind output: $him_console"
        tail -n 8 -- "$him_console"
    fi
done
if [[ "${#him_top_pids[@]}" -gt 0 ]]; then
    him_pid_list=$(IFS=,; echo "${him_top_pids[*]}")
    echo "Live CPU/memory: sudo top -p $him_pid_list"
fi
echo 'CPU percentages are per logical CPU; 100% is one core, not the whole Jetson.'
