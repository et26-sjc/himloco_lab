#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    printf 'Usage: %s /absolute/path/to/unitree_legged_sdk\n' "$0" >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sdk_root="$(cd -- "$1" && pwd)"
mkdir -p -- "$script_dir/build"
(
    cd -- "$script_dir/build"
    cmake "$script_dir" -DUNITREE_SDK_ROOT="$sdk_root" -DHIM_GUARD_TEST_ONLY=OFF
    cmake --build . -- -j2
    ctest --output-on-failure
    ./lcm_position_him --version
    sha256sum lcm_position_him > lcm_position_him.sha256
)
printf 'Built HIM bridge; no robot communication was started.\n'
