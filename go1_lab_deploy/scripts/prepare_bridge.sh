#!/usr/bin/env bash
# Prepare a matching bridge on the Jetson; never launch robot communication.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "${1:-}" == --describe && "$#" -eq 1 ]]; then
    echo "Export root: $him_root"
    echo "Bridge destination: $him_binary"
    echo 'Use --reuse /absolute/path/to/lcm_position_him or --build /absolute/path/to/SDK.'
    echo 'Preparation checks --version only; no UDP or LCM is opened.'
    exit 0
fi
if [[ "$#" -ne 2 ]]; then
    echo "Usage: bash $0 --reuse BRIDGE_BINARY | --build SDK_ROOT | --describe" >&2
    exit 2
fi
him_verify_sources
case "$1" in
    --reuse)
        him_existing=$(readlink -f -- "$2")
        if [[ "$him_existing" == "$him_binary" ]]; then
            him_verify_bridge
            exit 0
        fi
        # Validate bytes, protocol and source digest BEFORE copying/executing
        # anything as the new bridge. The verifier only executes --version.
        "$him_python" -B "$him_root/go1_lab_deploy/scripts/verify_bridge_build.py" \
            --binary "$him_existing"
        if [[ -e "$him_binary" || -e "$him_binary.sha256" ]]; then
            echo 'Bridge destination already exists. Inspect it before replacing it.' >&2
            exit 2
        fi
        mkdir -p -- "$(dirname -- "$him_binary")"
        cp -- "$him_existing" "$him_binary"
        cp -- "$him_existing.sha256" "$him_binary.sha256"
        chmod +x -- "$him_binary"
        ;;
    --build)
        him_sdk=$(cd -- "$2" && pwd)
        him_build_dir=$(dirname -- "$him_binary")
        mkdir -p -- "$him_build_dir"
        (
            cd -- "$him_build_dir"
            cmake .. -DUNITREE_SDK_ROOT="$him_sdk" -DHIM_GUARD_TEST_ONLY=OFF
            cmake --build . --target lcm_position_him -- -j2
            sha256sum lcm_position_him > lcm_position_him.sha256
        )
        ;;
    *)
        echo 'Expected --reuse or --build.' >&2
        exit 2
        ;;
esac
him_verify_bridge
echo 'Matching bridge prepared. No robot communication was started.'
