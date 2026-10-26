#!/usr/bin/env bash
# Pin this host-local LCM group to the internal SDK interface before sockets open.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "$#" -ne 0 || "$EUID" -ne 0 ]]; then
    echo "Usage: sudo bash $0 (normally called by start_background.sh)" >&2
    exit 2
fi
if pgrep -af '[l]cm_position|[d]eploy_policy.py|[w]tw_supported_trial.py|[g]o1_fr.*sdk38'; then
    echo 'Stop the previous bridge/controller before changing the LCM route.' >&2
    exit 2
fi
if [[ -n "$(ss -H -uan 'sport = :8090')" ]]; then
    echo 'UDP 8090 is occupied; network preparation stopped.' >&2
    exit 2
fi
him_verify_sources
him_group=$("$him_python" -B - "$him_root/himloco.json" <<'PY'
import ipaddress
import json
import sys
from urllib.parse import urlsplit
with open(sys.argv[1]) as stream:
    url = json.load(stream)['runtime']['lcm_url']
group = ipaddress.IPv4Address(urlsplit(url).hostname)
if not group.is_multicast:
    raise SystemExit('Expected an IPv4 multicast LCM group')
print(group)
PY
)
# This is the unchanged C++ bridge's SDK destination. Use its actual route,
# not the SSH/default interface, and leave both routes unchanged.
him_sdk_interface=$(ip -4 route get 192.168.123.10 | awk '{for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}')
if [[ -z "$him_sdk_interface" || "$him_sdk_interface" == lo ]]; then
    echo 'Cannot identify the internal Go1 SDK interface.' >&2
    exit 2
fi
mkdir -p -- "$him_log_dir"
if [[ ! -e "$him_log_dir/lcm_route_before_background.txt" ]]; then
    ip -4 route get "$him_group" >"$him_log_dir/lcm_route_before_background.txt"
fi
ip -4 route replace "$him_group/32" dev "$him_sdk_interface" scope link
echo 'LCM group pinned to internal SDK interface for computer-cable disconnection:'
ip -4 route get "$him_group"
