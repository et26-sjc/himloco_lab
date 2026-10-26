"""Exercise the real deployment Controller with MuJoCo-generated SDK feedback.

No LCM object, socket, bridge process or robot connection is created. This is a
simulation integration check of calibration/history/ramp and unchanged guards,
not a real-robot validation. MuJoCo supplies physical feedback rather than
copying commanded targets into a fictitious measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY/'go1_lab_deploy'))

import mujoco
import numpy as np

from go1_lab_deploy.himloco.config import load_config
from go1_lab_deploy.himloco.contract import SDK_JOINT_NAMES
from go1_lab_deploy.himloco.policy import JitPolicy
from go1_lab_deploy.himloco.runner import Controller, SafetyFault
from sim2sim.go1.config import SimConfig
from sim2sim.go1.controllers import PDController
from sim2sim.go1.mujoco_adapter import MujocoAdapter


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def fake_sdk_state(adapter, step, edges, pressed, command_sticks):
    q, dq = adapter.joints()
    state = adapter.state()
    sdk_indices = [adapter.names.index(name) for name in SDK_JOINT_NAMES]
    rotation = adapter.data.xmat[adapter.body_id].reshape(3, 3)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return dict(
        q_sdk=q[sdk_indices], dq_sdk=dq[sdk_indices],
        tau_sdk=adapter.data.actuator_force[adapter.actuator_ids][sdk_indices].copy(),
        quaternion_wxyz=adapter.data.xquat[adapter.body_id].copy(),
        gyro_sensor=state['omega'].copy(),
        rpy=np.array([state['roll'], state['pitch'], yaw]),
        l2=False, r2=pressed, r2_edges=edges,
        left_stick=np.array([command_sticks[1], command_sticks[0]]),
        right_stick=np.array([command_sticks[2], 0.]),
        state_age_s=0., rc_age_s=0., source_tick=step*adapter.cfg.decimation+1,
        source_timestamp_us=1_000_000+int(round(adapter.data.time*1e6)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(REPOSITORY/'go1_lab_deploy/himloco.json'))
    parser.add_argument('--duration', type=float, default=15.)
    parser.add_argument('--velocity', type=float, nargs=3, default=(.3, 0., 0.))
    parser.add_argument('--support-until-ready', action='store_true',
                        help='Idealized root clamp through WAIT_R2/CALIBRATE/SETTLE; release at READY. '
                             'Joint feedback remains simulated, not replaced by targets.')
    parser.add_argument('--output', default='artifacts/go1_lab_deploy/handover_simulation.json')
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be finite and positive')
    command = np.asarray(args.velocity, dtype=float)
    if not np.isfinite(command).all():
        parser.error('--velocity must contain finite physical-unit values')
    config = load_config(args.config, require_ready=True)
    limits = np.asarray(config['runtime']['max_abs_command'], dtype=float)
    if np.any(np.abs(command) > limits):
        parser.error('--velocity exceeds the unchanged deployment command limits')
    signs = np.asarray(config['runtime']['command_axis_signs'], dtype=float)
    sticks = command / (limits*signs)
    policy = JitPolicy(config['model']['path'], config['model']['sha256'])
    controller = Controller(config, policy)
    cfg = SimConfig(passive_damping=0.)
    if not np.isclose(cfg.policy_dt, config['runtime']['policy_dt']):
        raise AssertionError('Simulation/deployment policy periods differ')
    adapter = MujocoAdapter(cfg)
    pd = PDController(cfg.controller, adapter.names, adapter.effort_limits)
    kp, kd = pd.kp.copy(), pd.kd.copy()
    target_indices = [SDK_JOINT_NAMES.index(name) for name in adapter.names]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Select a new output file; previous evaluation is preserved')

    transitions = []
    running_velocity = []
    target_min = np.full(12, np.inf)
    target_max = np.full(12, -np.inf)
    max_tracking = 0.
    max_tilt = 0.
    min_height = float('inf')
    max_torque = np.zeros(12)
    physics_steps = 0
    policy_steps = 0
    saturation_count = 0
    edges = 0
    calibration_requested = False
    policy_requested = False
    running_time = 0.
    failure = None
    state_finite = True
    geometric_fall = False
    fall_dwell = 0.
    support_phases = ('WAIT_R2', 'CALIBRATE', 'SETTLE')
    supported_physics_steps = 0
    support_release_time = None
    root_pose = adapter.data.qpos[adapter.root_qpos:adapter.root_qpos+7].copy()
    phase = None
    for step in range(int(np.ceil(args.duration/cfg.policy_dt))):
        now = float(adapter.data.time)
        pressed = False
        if controller.phase == 'WAIT_R2' and step >= 1 and not calibration_requested:
            edges += 1
            pressed = True
            calibration_requested = True
        elif controller.phase == 'READY' and not policy_requested:
            edges += 1
            pressed = True
            policy_requested = True
        selected_sticks = sticks if controller.phase == 'RUNNING' else np.zeros(3)
        state = fake_sdk_state(adapter, step, edges, pressed, selected_sticks)
        try:
            result = controller.step(state, now)
        except SafetyFault as exc:
            failure = dict(kind='controller_guard', error=str(exc), time_s=now,
                phase_before_stop=phase, latched_phase=controller.phase,
                state=state, details=exc.details,
                previous_target_policy=controller.last_target,
                history=controller.history.get())
            break
        policy_steps += 1
        if result['phase'] != phase:
            phase = result['phase']
            transitions.append(dict(time_s=now, phase=phase))
        if args.support_until_ready and phase == 'READY' and support_release_time is None:
            support_release_time = now
        target_sdk = result['target_sdk']
        target = None if target_sdk is None else np.asarray(target_sdk)[target_indices]
        gain_scale = float(result.get('gain_scale', 0.))
        pd.kp, pd.kd = kp*gain_scale, kd*gain_scale
        if target is not None:
            target_min = np.minimum(target_min, target)
            target_max = np.maximum(target_max, target)
            max_tracking = max(max_tracking, float(np.max(np.abs(adapter.joints()[0]-target))))
        for _ in range(cfg.decimation):
            q, dq = adapter.joints()
            torque = np.zeros(12) if target is None else pd.step(target, q, dq)
            max_torque = np.maximum(max_torque, np.abs(torque))
            if target is not None:
                saturation_count += int((np.abs(pd.computed_torque) > adapter.effort_limits).sum())
            try:
                adapter.step(torque)
            except FloatingPointError as exc:
                state_finite = False
                failure = dict(kind='numerical_state', error=str(exc), time_s=float(adapter.data.time),
                    phase=phase, state=state)
                break
            physics_steps += 1
            if args.support_until_ready and phase in support_phases:
                # Idealized rigid fixture for the manually supported calibration
                # procedure. Only floating-root coordinates are constrained;
                # all twelve q/dq remain genuine integrated joint states.
                # This is deliberately reported as a constrained test trajectory,
                # not as an external-wrench model or unsupported standing test.
                adapter.data.qpos[adapter.root_qpos:adapter.root_qpos+7] = root_pose
                adapter.data.qvel[adapter.root_dof:adapter.root_dof+6] = 0.
                mujoco.mj_forward(adapter.model, adapter.data)
                supported_physics_steps += 1
            physical = adapter.state()
            height = float(physical['position'][2])
            tilt = float(np.arccos(np.clip(-physical['gravity'][2], -1., 1.)))
            min_height = min(min_height, height)
            max_tilt = max(max_tilt, tilt)
            _, _, trunk_contact = adapter.contacts()
            invalid = height < cfg.fall_height or tilt > cfg.fall_tilt or trunk_contact
            fall_dwell = fall_dwell+cfg.physics_dt if invalid else 0.
            if fall_dwell >= cfg.fall_duration:
                geometric_fall = True
                failure = dict(kind='geometric_fall', time_s=float(adapter.data.time),
                    phase=phase, height_m=height, tilt_rad=tilt, trunk_contact=trunk_contact)
                break
            if phase == 'RUNNING':
                running_time += cfg.physics_dt
                running_velocity.append(physical['velocity'].copy())
        if failure is not None:
            break

    completed = failure is None and physics_steps >= int(np.ceil(args.duration/cfg.policy_dt))*cfg.decimation
    passed = completed and controller.phase == 'RUNNING' and state_finite and not geometric_fall
    targeted = np.isfinite(target_min).all()
    report = dict(
        status='passed' if passed else 'guard_triggered' if failure and failure['kind'] == 'controller_guard'
            else 'failed' if failure else 'incomplete',
        scope='Offline MuJoCo-backed deployment handover; no robot hardware validation',
        test_fixture=dict(
            type='idealized_rigid_floating_root_clamp' if args.support_until_ready else 'none',
            constrained_phases=list(support_phases) if args.support_until_ready else [],
            release_time_s=support_release_time, supported_physics_steps=supported_physics_steps,
            root_pose=root_pose.tolist() if args.support_until_ready else None,
            note='Root pose/velocity constrained during calibration only; joint q/dq never replaced. '
                 'Constrained fixture test, not an unconstrained physical trajectory.'
                 if args.support_until_ready else 'Free floating body throughout; no external support.'),
        motor_commands_published=False, robot_connections_opened=False,
        config=str(Path(args.config).resolve()), config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        policy_sha256=policy.sha256, requested_duration_s=args.duration,
        elapsed_simulation_s=float(adapter.data.time), requested_velocity=command.tolist(),
        policy_dt=cfg.policy_dt, physics_dt=cfg.physics_dt, decimation=cfg.decimation,
        passive_damping=0., controller_kp=kp.tolist(), controller_kd=kd.tolist(),
        stages=transitions, final_phase=controller.phase, physics_steps=physics_steps,
        policy_steps=policy_steps, running_time_s=running_time,
        mean_running_body_velocity=(np.mean(running_velocity, axis=0).tolist() if running_velocity else None),
        state_finite=state_finite, geometric_fall=geometric_fall,
        min_base_height_m=min_height if np.isfinite(min_height) else None,
        max_tilt_rad=max_tilt, max_tracking_error_rad=max_tracking,
        target_joint_names=list(adapter.names),
        target_min_rad=target_min.tolist() if targeted else None,
        target_max_rad=target_max.tolist() if targeted else None,
        applied_torque_abs_peak_nm=max_torque.tolist(),
        torque_saturation_ratio=saturation_count/(max(1, physics_steps)*12),
        guards='Original Controller limits/tilt/tracking/freshness/cycle/R2/target gates preserved',
        failure=failure, mujoco_version=mujoco.__version__, model=adapter.model_metadata(),
    )
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, default=json_value, allow_nan=False)
        stream.write('\n')
    print('DEPLOY_HANDOVER_RESULT', json.dumps(report, default=json_value, allow_nan=False), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    sys.exit(main())
