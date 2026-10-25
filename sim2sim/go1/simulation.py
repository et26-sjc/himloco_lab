"""Shared closed loop and state-based metrics for GUI and batch evaluation."""
from contextlib import nullcontext
import time

import mujoco
import numpy as np

from .controllers import PDController
from .mujoco_adapter import MujocoAdapter


def run_episode(cfg, policy, command, duration, gui=False, real_time=False, zero_action=False):
    if duration <= 0 or not np.isfinite(duration):
        raise ValueError('Duration must be positive and finite')
    command = np.asarray(command, dtype=float)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError('Expected finite velocity [vx,vy,wz]')
    adapter = MujocoAdapter(cfg)
    controller = PDController(cfg.controller, adapter.names, adapter.effort_limits)
    policy.reset()
    initial_hash = adapter.initial_state_sha256()
    initial_position = adapter.state()['position'].copy()
    previous_position = initial_position.copy()
    previous_action = np.zeros(12)
    action = np.zeros(12)
    target = adapter.default_pos.copy()
    records, policy_inputs, policy_actions = [], [], []
    bad_time = 0.
    fall = False
    reason = None
    stopped_by_viewer = False
    policy_calls = 0
    action_updates = 0
    steps = int(round(duration/cfg.physics_dt))
    if gui:
        from mujoco import viewer as mj_viewer
        viewer_context = mj_viewer.launch_passive(adapter.model, adapter.data)
    else:
        viewer_context = nullcontext(None)
    started = time.monotonic()
    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = adapter.body_id
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -20
        for step in range(steps):
            if viewer is not None and not viewer.is_running():
                stopped_by_viewer = True
                break
            if step % cfg.decimation == 0:
                frame = adapter.observation(command, previous_action)
                action = np.zeros(12) if zero_action else policy.act(frame)
                target = adapter.targets(action)
                previous_action = action.copy()
                policy_calls += not zero_action
                action_updates += 1
                policy_inputs.append(frame)
                policy_actions.append(action.copy())
            q, dq = adapter.joints()
            torque = controller.step(target, q, dq)
            computed = controller.computed_torque.copy()
            adapter.step(torque)
            state = adapter.state()
            contacts, foot_forces, trunk_contact = adapter.contacts()
            delta = state['position'] - previous_position
            previous_position = state['position'].copy()
            invalid_pose = (state['position'][2] < cfg.fall_height or abs(state['roll']) > cfg.fall_tilt
                            or abs(state['pitch']) > cfg.fall_tilt or trunk_contact)
            bad_time = bad_time + cfg.physics_dt if invalid_pose else 0.
            saturation = np.abs(computed) > adapter.effort_limits
            stable_standing = (state['position'][2] > .20 and abs(state['roll']) < .35
                               and abs(state['pitch']) < .35 and np.linalg.norm(state['velocity'][:2]) < .15
                               and contacts.sum() >= 2)
            records.append(dict(time=float(adapter.data.time), position=state['position'], velocity=state['velocity'],
                                omega=state['omega'], roll=state['roll'], pitch=state['pitch'],
                                full_qpos=adapter.data.qpos.copy(), full_qvel=adapter.data.qvel.copy(),
                                q=q, dq=dq, q_target=target.copy(), computed_torque=computed,
                                applied_torque=torque.copy(), saturated=saturation, foot_contacts=contacts,
                                foot_normal_forces=foot_forces, path_delta=np.linalg.norm(delta[:2]),
                                standing_stable=stable_standing))
            if viewer is not None and step % cfg.decimation == 0:
                viewer.sync()
            if real_time:
                time.sleep(max(0., (step+1)*cfg.physics_dt-(time.monotonic()-started)))
            if bad_time >= cfg.fall_duration:
                fall = True
                reason = ('base_height' if state['position'][2] < cfg.fall_height else
                          'trunk_ground_contact' if trunk_contact else 'tilt')
                break
    if not records:
        raise RuntimeError('No simulated samples; duration must cover a physics step')
    trace = {key: np.asarray([r[key] for r in records]) for key in records[0]}
    trace['policy_observations'] = np.asarray(policy_inputs)
    trace['policy_actions'] = np.asarray(policy_actions)
    linear_error = trace['velocity'][:, :2] - command[:2]
    angular_error = trace['omega'][:, 2] - command[2]
    final_position = trace['position'][-1]
    steady = trace['time'] >= max(0., trace['time'][-1]-2.)
    metrics = dict(command=command.tolist(), mode='zero_action' if zero_action else 'policy',
                   duration_requested_s=duration, survival_s=float(trace['time'][-1]),
                   geometric_fall=fall, geometric_fall_rate=float(fall), fall_reason=reason,
                   completed_horizon=not fall and not stopped_by_viewer, viewer_closed=stopped_by_viewer,
                   linear_tracking_rmse=float(np.sqrt(np.mean(np.sum(linear_error**2, axis=1)))),
                   yaw_tracking_rmse=float(np.sqrt(np.mean(angular_error**2))),
                   mean_body_velocity=trace['velocity'].mean(0).tolist(),
                   mean_body_angular_velocity=trace['omega'].mean(0).tolist(),
                   mean_vx=float(trace['velocity'][:, 0].mean()), mean_vy=float(trace['velocity'][:, 1].mean()),
                   mean_wz=float(trace['omega'][:, 2].mean()),
                   base_height_mean=float(trace['position'][:, 2].mean()),
                   base_height_range=[float(trace['position'][:, 2].min()), float(trace['position'][:, 2].max())],
                   roll_range=[float(trace['roll'].min()), float(trace['roll'].max())],
                   pitch_range=[float(trace['pitch'].min()), float(trace['pitch'].max())],
                   joint_tracking_rmse=np.sqrt(np.mean((trace['q_target']-trace['q'])**2, axis=0)).tolist(),
                   applied_torque_rms=np.sqrt(np.mean(trace['applied_torque']**2, axis=0)).tolist(),
                   applied_torque_peak=np.max(np.abs(trace['applied_torque']), axis=0).tolist(),
                   joint_velocity_peak=np.max(np.abs(trace['dq']), axis=0).tolist(),
                   rated_velocity_exceedance_ratio=float((np.abs(trace['dq']) > np.array([
                       20.06 if '_calf_' in n else 30.1 for n in adapter.names])).mean()),
                   torque_saturation_ratio=float(trace['saturated'].mean()),
                   foot_contact_fraction=trace['foot_contacts'].mean(0).tolist(),
                   foot_normal_force_mean=trace['foot_normal_forces'].mean(0).tolist(),
                   world_displacement=(final_position-initial_position).tolist(),
                   horizontal_path_length=float(trace['path_delta'].sum()),
                   body_forward_distance=float(trace['velocity'][:, 0].sum()*cfg.physics_dt),
                   steady_standing_fraction=float(trace['standing_stable'][steady].mean()) if np.all(command == 0) else None,
                   physics_steps=len(records), policy_calls=policy_calls, action_updates=action_updates,
                   initial_state_sha256=initial_hash,
                   controller=dict(type=cfg.controller.type, kp=controller.kp.tolist(), kd=controller.kd.tolist()),
                   actual_gain_label=(f'kp{controller.kp[0]:g}_kd{controller.kd[0]:g}'
                        if np.all(controller.kp == controller.kp[0]) and np.all(controller.kd == controller.kd[0])
                        else 'per_joint_gains'),
                   config=cfg.to_dict(), policy=policy.metadata, model=adapter.model_metadata(),
                   mujoco_version=mujoco.__version__, wall_time_s=time.monotonic()-started)
    return metrics, trace
