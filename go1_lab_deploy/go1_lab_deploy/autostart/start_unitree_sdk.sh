#!/usr/bin/env bash
# Explicit foreground bridge startup. Never select the old bundled binary.
set -euo pipefail
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [PATH_TO_NEW_lcm_position_him]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
binary=${1:-"$script_dir/../unitree_legged_sdk_bin/build/lcm_position_him"}
if [[ "${binary##*/}" != "lcm_position_him" ]]; then
  echo "Expected the newly built lcm_position_him binary; old lcm_position is unsupported." >&2
  exit 2
fi
if [[ ! -x "$binary" ]]; then
  echo "New HIM bridge is missing or not executable: $binary" >&2
  echo "Build unitree_legged_sdk_bin/CMakeLists.txt against the target SDK and LCM first." >&2
  exit 2
fi
"${PYTHON:-python3}" "$script_dir/../scripts/verify_bridge_build.py" --binary "$binary"
# The bridge keeps its operator prompt. No automatic Enter, background process,
# broad process killing, or automatic restart after a latched fault.
if [[ -n "${HIM_LCM_URL:-}" ]]; then
  exec "$binary" --lcm-url "$HIM_LCM_URL"
fi
exec "$binary"
