#!/usr/bin/env python3
"""Reject stale HIM bridge builds before startup; this is not a signature check."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "unitree_legged_sdk_bin/lcm_position.cpp",
    "unitree_legged_sdk_bin/bridge_guard.hpp",
    "lcm_types/state_estimator_lcmt.lcm",
    "lcm_types/leg_control_data_lcmt.lcm",
    "lcm_types/pd_tau_targets_lcmt.lcm",
    "lcm_types/rc_command_lcmt.lcm",
)


def source_digest(deploy_root):
    # Match CMakeLists.txt: concatenate each file SHA-256 hex string, then hash.
    digests = [hashlib.sha256((Path(deploy_root) / name).read_bytes()).hexdigest()
               for name in SOURCE_FILES]
    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()


def verify_bridge(binary, deploy_root=DEPLOY_ROOT, run=None):
    binary = Path(binary).resolve()
    if binary.name != "lcm_position_him":
        raise ValueError("expected newly built lcm_position_him; old lcm_position is unsupported")
    receipt = binary.with_name(binary.name + ".sha256")
    fields = receipt.read_text(encoding="ascii").split()
    if len(fields) != 2 or fields[1] != binary.name or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
        raise ValueError("invalid bridge checksum receipt; rebuild with build_him_bridge.sh")
    payload = binary.read_bytes()
    if hashlib.sha256(payload).hexdigest() != fields[0]:
        raise ValueError("bridge binary checksum mismatch")
    expected = source_digest(deploy_root)
    # The newly implemented bridge embeds both. Reject an old executable before
    # asking it for --version: the original WTW main ignored command-line flags.
    if b"HIM_BRIDGE protocol=" not in payload or expected.encode("ascii") not in payload:
        raise ValueError("binary does not embed current HIM bridge sources; rebuild it")
    if run is None:
        run = subprocess.run
    result = run([str(binary), "--version"], stdin=subprocess.DEVNULL,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                 universal_newlines=True, check=True, timeout=5)
    wanted = "HIM_BRIDGE protocol=20261023 source_sha256=" + expected
    if result.stdout.strip() != wanted:
        raise ValueError("bridge version/source digest mismatch")
    if hashlib.sha256(binary.read_bytes()).hexdigest() != fields[0]:
        raise ValueError("bridge changed during verification")
    return wanted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--deploy-root", type=Path, default=DEPLOY_ROOT)
    args = parser.parse_args(argv)
    try:
        print(verify_bridge(args.binary, args.deploy_root))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print("Bridge build verification failed: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
