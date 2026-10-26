#!/usr/bin/env python3
"""Capture passive SDK states, or analyze saved physical alignment motions.

This script only subscribes to state. It never publishes motor targets, loads
a policy, or marks a deployment JSON as verified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POSE_STEPS = [
    ("neutral", "机身保持静止。若要确认 IMU 安装，请先用水平仪确认机身刚性参考面水平。"),
    ("roll_left_up", "从接近水平开始，缓慢抬高机器人左侧约 10～15 度，保持右侧较低。"),
    ("pitch_nose_up", "先恢复接近水平，再缓慢抬高机头约 10～15 度，保持机尾较低。"),
    ("yaw_left", "先恢复接近水平，从上往下看，缓慢把机头向左转约 10～15 度（逆时针）。"),
]


def guided(args):
    from go1_lab_deploy.himloco.contract import SDK_JOINT_NAMES
    if not 6 <= args.duration <= 30:
        raise ValueError("Guided capture duration must be 6 to 30 seconds")
    if (not 1 <= args.still_seconds <= 6 or
            args.duration < 2 * args.still_seconds + 2):
        raise ValueError("Still intervals must be 1 to 6 seconds with at least 2 seconds left for motion")
    steps = list(POSE_STEPS) if args.guided in ("imu", "all") else []
    if args.guided in ("joints", "all"):
        legs = {"FR": "右前", "FL": "左前", "RR": "右后", "RL": "左后"}
        motions = {"hip": "髋外展关节：让该腿向机器人左侧小幅摆动；不是前后摆动。",
                   "thigh": "大腿关节：让大腿相对机身向后小幅摆动。",
                   "calf": "小腿关节：轻轻伸展膝盖，让大小腿夹角变大。"}
        for name in SDK_JOINT_NAMES:
            leg, joint, _ = name.split("_")
            steps.append(("joint_" + name,
                legs[leg] + "腿，" + motions[joint] + " 只轻动指定关节约 5～15 度，不强扳，其他关节尽量不动。"))
    if args.only:
        selected = set(args.only)
        if len(selected) != len(args.only) or not selected.issubset({label for label, _ in steps}):
            raise ValueError("--only must name distinct labels from the selected guided group")
        steps = [(label, instruction) for label, instruction in steps if label in selected]
    print("只读采集：不加载策略、不发布电机目标。观测桥接必须使用 --observe-only。", flush=True)
    print("左右以前方为机头、站在机器人后方朝机头看为准。四腿离地、机身持续承托。", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, (label, instruction) in enumerate(steps, 1):
        print("\n步骤 {}/{}: {}\n{}".format(index, len(steps), label, instruction), flush=True)
        print("现在先摆好起始姿态，不要提前做该动作。Enter 开始；输入 q 退出。", flush=True)
        if input().strip().lower() == "q":
            return
        child = argparse.Namespace(config=args.config, capture=label, duration=args.duration,
                                   output=args.output / (label + ".jsonl"), cues=True,
                                   still_seconds=args.still_seconds)
        capture(child)
        print("本项结束。下一项采集前可恢复起始姿态。", flush=True)
    analysis = argparse.Namespace(analyze=args.output, level_confirmed=args.level_confirmed,
                                 output=args.output / ("alignment_report_" + args.guided + ".json"))
    analyze(analysis)


def capture(args):
    import lcm
    from go1_lab_deploy.himloco.config import load_config
    from go1_lab_deploy.himloco.state import StateReceiver
    from go1_lab_deploy.himloco.runner import _json_default, _wait_state
    cfg = load_config(args.config, require_ready=False)
    if not 3 <= args.duration <= 60:
        raise ValueError("Capture duration must be 3 to 60 seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receiver = StateReceiver(lcm.LCM(cfg["runtime"]["lcm_url"]))
    try:
        receiver.start()
        _wait_state(receiver, cfg["runtime"])
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(kind="header", label=args.capture,
                config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                motor_commands_published=0)) + "\n")
            print("CAPTURE_READY:", args.capture, "duration_s=", args.duration, flush=True)
            print("Keep still for first/last 1 second; perform only the labelled positive motion in between.", flush=True)
            end, previous, count = time.monotonic() + args.duration, None, 0
            started, stage = time.monotonic(), "still"
            if getattr(args, "cues", False):
                print("开始记录：先保持起始姿态不动 {} 秒。".format(args.still_seconds), flush=True)
            while time.monotonic() < end:
                if getattr(args, "cues", False):
                    elapsed = time.monotonic() - started
                    if stage == "still" and elapsed >= args.still_seconds:
                        stage = "move"
                        print("现在缓慢做本项动作；到达约 10～15 度后就保持，不必一直移动。neutral 项继续不动。", flush=True)
                    if stage == "move" and elapsed >= args.duration - args.still_seconds:
                        stage = "hold"
                        print("保持当前终点姿态不动，直到本项结束，不要回位。", flush=True)
                s = receiver.snapshot(cfg["runtime"]["state_timeout_s"], cfg["runtime"]["rc_timeout_s"])
                if s["source_tick"] != previous:
                    s.update(kind="sample", capture_monotonic_s=time.monotonic())
                    stream.write(json.dumps(s, default=_json_default, allow_nan=False) + "\n")
                    previous = s["source_tick"]
                    count += 1
                time.sleep(.02)
            stream.write(json.dumps(dict(kind="complete", fresh_samples=count,
                                         motor_commands_published=0)) + "\n")
        print("CAPTURE_SAVED:", str(args.output), "fresh_samples=", count, flush=True)
    finally:
        receiver.close()


def analyze(args):
    from go1_lab_deploy.himloco.alignment import analyze as analyze_data
    captures, hashes = {}, {}
    for path in sorted(args.analyze.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or rows[0].get("kind") != "header":
            continue
        if rows[-1].get("kind") != "complete":
            raise ValueError("Incomplete capture: " + path.name)
        label = rows[0]["label"]
        if label in captures:
            raise ValueError("Duplicate label: " + label)
        captures[label] = [r for r in rows[1:] if r.get("kind") == "sample"]
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = analyze_data(captures, level_confirmed=args.level_confirmed)
    report["capture_sha256"] = hashes
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    print(payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--capture", help="neutral, roll_left_up, pitch_nose_up, yaw_left, joint_NAME")
    modes.add_argument("--analyze", type=Path, help="Directory containing operator-labelled JSONL captures")
    modes.add_argument("--guided", choices=("imu", "joints", "all"), help="Interactive passive capture with operator cues")
    p.add_argument("--config", type=Path)
    p.add_argument("--duration", type=float, default=8)
    p.add_argument("--still-seconds", type=float, default=2,
                   help="Static start/end cue duration, only for --guided")
    p.add_argument("--only", nargs="+", help="Capture only these labels from --guided, preserving other files")
    p.add_argument("--output", type=Path)
    p.add_argument("--level-confirmed", action="store_true", help="Operator actually checked a rigid body reference with a level")
    args = p.parse_args()
    if (args.capture or args.guided) and (args.config is None or args.output is None):
        p.error("capture/guided requires --config and --output")
    if args.only and not args.guided:
        p.error("--only requires --guided")
    try:
        if args.guided:
            guided(args)
        elif args.capture:
            capture(args)
        else:
            analyze(args)
        return 0
    except (KeyboardInterrupt, EOFError):
        print("Passive capture stopped; no motor commands were published.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print("ALIGNMENT ERROR:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
