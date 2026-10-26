"""HIM deployment state machine and 50 Hz LCM runner.

The calibration, measured-history warmup, action ramp and fault gates below are
deployment engineering, not operations inside the exported HIM policy.
"""
import json
import math
from pathlib import Path
import time

import numpy as np

from go1_lab_deploy.himloco.config import validate_config
from go1_lab_deploy.himloco.contract import (
    ACTION_SCALE, KD, KP, POLICY_DT, JointMap, ObservationHistory,
    action_to_target, build_observation, default_positions, joint_limits,
    projected_gravity,
)
from go1_lab_deploy.himloco.state import PROTOCOL_ID, StateError, StateReceiver
from go1_lab_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt


class SafetyFault(RuntimeError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


class Controller:
    """One step per policy period; getters never advance the history.

    Targets outside mechanical limits or an enabled step envelope cause a stop,
    not a silent change of the trained policy. During ramp only, previous_action
    is the normalized action recovered from the blended absolute target.
    Normal operation stores the raw policy action as in Lab's ActionManager.
    """

    def __init__(self, config, policy):
        validate_config(config, require_ready=True)
        self.config, self.policy = config, policy
        self.rt = config["runtime"]
        self.mapping = JointMap(config["model"]["training_dof_names"])
        self.q0 = default_positions(config["model"]["training_dof_names"])
        self.lower, self.upper = joint_limits(config["model"]["training_dof_names"])
        self.R_BS = np.asarray(config["alignment"]["sensor_to_body_rotation"], dtype=float)
        self.history = ObservationHistory()
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.phase = "WAIT_R2"
        self.phase_start = None
        self.last_time = None
        self.last_tick = None
        self.last_target = None
        self.tracking_error_since = np.full(12, np.nan)
        self.seen_edges = None
        self.calibration_start = None

    def _transition(self, phase, now):
        self.phase, self.phase_start = phase, now

    def _command(self, state):
        # Raw ly/lx/rx feed vx/vy/yaw; physical stick directions are configured
        # only through runtime.command_axis_signs, never joint/IMU sign changes.
        sticks = np.array([state["left_stick"][1], state["left_stick"][0],
                           state["right_stick"][0]], dtype=float)
        sticks[np.abs(sticks) <= self.rt["command_deadzone"]] = 0.0
        return (sticks * np.asarray(self.rt["command_axis_signs"])
                * np.asarray(self.rt["max_abs_command"]))

    def step(self, state, now):
        try:
            return self._step(state, now)
        except Exception:
            self.phase = "STOPPED"
            raise

    def _step(self, state, now):
        if self.phase == "STOPPED":
            raise SafetyFault("Controller is latched stopped")
        if self.last_time is not None:
            period = now - self.last_time
            if period <= 0 or period > POLICY_DT + self.rt["max_cycle_lateness_s"]:
                raise SafetyFault("Policy cycle gap {:.4f}s".format(period))
        else:
            period = None
        self.last_time = now
        if state["l2"]:
            raise SafetyFault("L2 emergency stop")
        if state["state_age_s"] > self.rt["state_timeout_s"] or state["rc_age_s"] > self.rt["rc_timeout_s"]:
            raise SafetyFault("Stale state or remote control")
        q = self.mapping.to_policy(state["q_sdk"])
        dq = self.mapping.to_policy(state["dq_sdk"])
        alignment = self.config["alignment"]
        attitude = state["rpy"] if alignment["attitude_source"] == "rpy" else state["quaternion_wxyz"]
        gravity = projected_gravity(attitude, alignment["attitude_source"], self.R_BS)
        omega = self.R_BS.dot(state["gyro_sensor"])
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        if tilt > self.rt["max_tilt_rad"]:
            raise SafetyFault("Body tilt {:.3f}rad exceeds reviewed limit".format(tilt))
        # Folded encoder readings can lie outside the training URDF. This
        # bounded allowance applies only before/during calibration, never to
        # policy targets or normal operation. It decays to the running allowance.
        margin = 0.02
        if self.phase == "WAIT_R2":
            margin = self.rt["startup_joint_margin_rad"]
        elif self.phase == "CALIBRATE":
            progress = min(1.0, (now - self.phase_start) / self.rt["calibration_duration_s"])
            margin = max(0.02, self.rt["startup_joint_margin_rad"] * (1.0 - progress))
        if np.any(q < self.lower - margin) or np.any(q > self.upper + margin):
            raise SafetyFault("Measured joint position outside {} limits".format(
                "calibration startup" if self.phase in ("WAIT_R2", "CALIBRATE") else "Go1 URDF"))
        tracking_status = {}
        if self.last_target is not None:
            if state["source_tick"] == self.last_tick:
                raise SafetyFault("No new SDK sample this policy cycle")
            errors = np.abs(q - self.last_target)
            limit = self.rt["max_tracking_error_rad"]
            moving = self.phase in ("RAMP", "RUNNING")
            timeout = self.rt.get("tracking_error_timeout_s", 0.0) if moving else 0.0
            instant_limit = (self.rt.get("max_instant_tracking_error_rad", limit)
                             if moving else limit)
            over = errors > limit
            # A different joint cannot inherit another joint's elapsed time.
            # Returning within the limit resets only that joint's timer.
            self.tracking_error_since[~over] = np.nan
            newly_over = over & np.isnan(self.tracking_error_since)
            self.tracking_error_since[newly_over] = now
            durations = np.zeros(12)
            durations[over] = now - self.tracking_error_since[over]
            largest = int(np.argmax(errors))
            longest = int(np.argmax(np.where(over, durations, -1.0)))
            tracking_status = dict(
                tracking_error_max_rad=float(errors[largest]),
                tracking_error_max_joint=self.mapping.policy_names[largest],
                tracking_error_over_limit_s=float(durations[longest]),
                tracking_error_over_limit_joint=(self.mapping.policy_names[longest]
                                                if np.any(over) else None),
                tracking_error_limit_rad=limit, tracking_error_timeout_s=timeout,
                tracking_error_instant_limit_rad=instant_limit)
            instant_fault = errors[largest] > instant_limit
            sustained = over & (durations >= timeout)
            if instant_fault or np.any(sustained):
                joint = (largest if instant_fault else
                         int(np.argmax(np.where(sustained, errors, -1.0))))
                raise SafetyFault(
                    "Joint tracking error: {} {:.4f} rad; {:.3f}s above {:.4f} rad "
                    "(delay {:.3f}s, instant limit {:.4f} rad)".format(
                        self.mapping.policy_names[joint], errors[joint], durations[joint],
                        limit, timeout, instant_limit),
                    details=dict(kind="tracking_error", phase=self.phase,
                                 trigger="instant" if instant_fault else "sustained",
                                 joint=self.mapping.policy_names[joint],
                                 error_rad=float(errors[joint]),
                                 duration_s=float(durations[joint]),
                                 limit_rad=limit, timeout_s=timeout,
                                 instant_limit_rad=instant_limit,
                                 previous_target_policy=self.last_target.copy(),
                                 q_policy=q.copy(), dq_policy=dq.copy(),
                                 error_policy_rad=errors.copy(),
                                 exceedance_policy_s=durations.copy(),
                                 candidate_sent=False))
        self.last_tick = state["source_tick"]
        edges = state["r2_edges"]
        if self.seen_edges is None:
            self.seen_edges = edges
        if edges - self.seen_edges > 1 or edges < self.seen_edges:
            raise SafetyFault("Ambiguous R2 edge sequence; restart the controller")
        pressed = edges > self.seen_edges
        self.seen_edges = edges
        if pressed:
            if self.phase == "WAIT_R2":
                if np.any(self._command(state) != 0):
                    raise SafetyFault("Center the command sticks before calibration")
                # Never command the out-of-URDF folded measurement itself.
                # The first legal target has zero gains; gains then ramp up.
                inset = self.rt["calibration_target_inset_rad"]
                self.calibration_start = np.clip(q, self.lower + inset, self.upper - inset)
                self._transition("CALIBRATE", now)
            elif self.phase == "READY":
                if np.any(self._command(state) != 0):
                    raise SafetyFault("Center the command sticks before policy handover")
                self._transition("RAMP", now)
            else:
                raise SafetyFault("R2 stop during {}".format(self.phase))
        result = dict(phase=self.phase, source_tick=state["source_tick"],
                      source_timestamp_us=state["source_timestamp_us"],
                      state_age_s=state["state_age_s"], rc_age_s=state["rc_age_s"],
                      period_s=period, q_policy=q, dq_policy=dq,
                      omega_body=omega, gravity_body=gravity, tilt_rad=tilt)
        result.update(tracking_status)
        if "tau_sdk" in state:
            # MCU torque estimate for tracking diagnosis, never a feedforward command.
            result["tau_est_policy_nm"] = self.mapping.to_policy(state["tau_sdk"])
        result.update(left_stick=np.asarray(state["left_stick"]).copy(),
                      right_stick=np.asarray(state["right_stick"]).copy())
        if self.phase == "WAIT_R2":
            result["target_sdk"] = None
            return result

        target = self.q0.copy()
        obs = None
        commands = np.zeros(3)
        raw = np.zeros(12, dtype=np.float32)
        clipped = raw.copy()
        applied = raw.copy()
        alpha = 0.0
        gain_scale = 1.0
        policy_inference_s = 0.0
        feedback_kind = "nominal_hold"
        policy_target = None
        if self.phase == "CALIBRATE":
            elapsed = now - self.phase_start
            fraction = min(1.0, elapsed / self.rt["calibration_duration_s"])
            gain_scale = min(1.0, elapsed / self.rt["calibration_gain_ramp_s"])
            target = self.calibration_start + fraction * (self.q0 - self.calibration_start)
            if fraction >= 1.0:
                self.history.reset()
                self.previous_action.fill(0)
                self._transition("SETTLE", now)
        elif self.phase == "SETTLE":
            errors = np.abs(q - self.q0)
            largest = int(np.argmax(errors))
            tolerance = self.rt["calibration_tolerance_rad"]
            remaining = self.rt["calibration_settle_timeout_s"] - (now - self.phase_start)
            joint_name = self.config["model"]["training_dof_names"][largest]
            result.update(settle_remaining_s=max(0.0, remaining),
                          max_nominal_error_rad=float(errors[largest]),
                          max_nominal_error_joint=joint_name,
                          calibration_tolerance_rad=tolerance)
            if remaining < 0:
                raise SafetyFault(
                    "Nominal pose did not settle before timeout: {} error {:.4f} rad, "
                    "tolerance {:.4f} rad, q {:.4f}, target {:.4f}".format(
                        joint_name, errors[largest], tolerance, q[largest], self.q0[largest]))
            if errors[largest] <= tolerance:
                obs = build_observation(commands, omega, gravity, q, dq, self.previous_action, self.q0)
                self.history.push(obs)
                if self.history.count >= self.rt["history_warmup_frames"]:
                    self._transition("READY", now)
            else:
                self.history.reset()
        else:
            if self.phase == "READY" and now - self.phase_start > self.rt["ready_timeout_s"]:
                raise SafetyFault("Policy handover wait timed out")
            if self.phase == "RUNNING":
                commands = self._command(state)
            obs = build_observation(commands, omega, gravity, q, dq, self.previous_action, self.q0)
            self.history.push(obs)  # the only history advancement in this cycle
            if self.phase in ("RAMP", "RUNNING"):
                inference_start = time.monotonic()
                raw = np.asarray(self.policy(self.history.get()), dtype=np.float32)
                policy_inference_s = time.monotonic() - inference_start
                # Lab clips q0 + scale * raw, not the raw network output.
                policy_target = action_to_target(raw, self.q0)
                clipped = (policy_target - self.q0) / ACTION_SCALE
                alpha = (min(1.0, (now - self.phase_start) / self.rt["action_ramp_s"])
                         if self.phase == "RAMP" else 1.0)
                target = (policy_target if alpha >= 1.0 else
                          self.q0 + alpha * (policy_target - self.q0))
                applied = (target - self.q0) / ACTION_SCALE
                if self.phase == "RAMP" and alpha >= 1.0:
                    self._transition("RUNNING", now)
                feedback_kind = "policy_raw" if alpha >= 1.0 else "ramp_applied"
        if not np.isfinite(target).all() or np.any(target < self.lower) or np.any(target > self.upper):
            raise SafetyFault("Policy target outside Go1 URDF limits", details=dict(
                kind="mechanical_target_limit", candidate_sent=False, phase=self.phase,
                joint_names=list(self.mapping.policy_names), rejected_target_policy=target.copy(),
                lower_policy=self.lower.copy(), upper_policy=self.upper.copy(),
                raw_action=raw.copy(), policy_target_before_ramp=policy_target,
                applied_action=applied.copy(), ramp_alpha=alpha, commands=commands.copy(),
                observation=None if obs is None else obs.copy(), history=self.history.get()))
        step_limit = self.rt["max_joint_target_step_rad"]
        result.update(target_step_guard_enabled=step_limit is not None,
                      target_step_limit_rad=step_limit,
                      target_step_max_rad=None, target_step_joint=None)
        if self.last_target is not None:
            target_delta = target - self.last_target
            largest_step = int(np.argmax(np.abs(target_delta)))
            result.update(target_step_max_rad=float(abs(target_delta[largest_step])),
                          target_step_joint=self.config["model"]["training_dof_names"][largest_step])
            if step_limit is not None and abs(target_delta[largest_step]) > step_limit:
                # Do not replace last_target/previous_action or publish the
                # rejected candidate. The caller sends stop before logging.
                details = dict(
                    kind="target_step", candidate_sent=False, phase=self.phase,
                    monotonic_s=now, period_s=period,
                    source_tick=state["source_tick"],
                    source_timestamp_us=state["source_timestamp_us"],
                    state_age_s=state["state_age_s"], rc_age_s=state["rc_age_s"],
                    joint=self.config["model"]["training_dof_names"][largest_step],
                    signed_step_rad=float(target_delta[largest_step]),
                    max_abs_step_rad=float(abs(target_delta[largest_step])),
                    limit_rad=step_limit,
                    previous_target_policy=self.last_target.copy(),
                    rejected_target_policy=target.copy(),
                    commands=commands.copy(), q_policy=q.copy(), dq_policy=dq.copy(),
                    tau_est_policy_nm=result.get("tau_est_policy_nm"),
                    omega_body=omega.copy(), gravity_body=gravity.copy(),
                    raw_action=raw.copy(), clipped_action=clipped.copy(),
                    applied_action=applied.copy(), ramp_alpha=alpha,
                    observation=None if obs is None else obs.copy(),
                    history=self.history.get(),
                    left_stick=result["left_stick"].copy(),
                    right_stick=result["right_stick"].copy())
                raise SafetyFault(
                    "Target step exceeds reviewed limit: {} {:.4f} rad > {:.4f} rad".format(
                        details["joint"], details["max_abs_step_rad"], step_limit),
                    details=details)
        self.last_target = target.copy()
        # Separate training's raw action history from startup target blending.
        self.previous_action = (raw if feedback_kind == "policy_raw" else applied).astype(np.float32, copy=True)
        result.update(phase=self.phase, commands=commands, observation=obs,
                      history=self.history.get(), history_count=self.history.count,
                      raw_action=raw, clipped_action=clipped, applied_action=applied,
                      action_feedback_kind=feedback_kind,
                      policy_target_before_ramp=policy_target,
                      ramp_alpha=alpha, target_policy=target,
                      policy_inference_s=policy_inference_s,
                      gain_scale=gain_scale,
                      target_sdk=self.mapping.to_sdk(target))
        return result


class CommandPublisher:
    def __init__(self, lc):
        self.lc, self.sequence, self.has_published = lc, 0, False

    def publish(self, result):
        msg = pd_tau_targets_lcmt()
        self.sequence += 1
        msg.id = self.sequence
        msg.robot_id = PROTOCOL_ID
        # Echo the timestamp made by this bridge: no host clock synchronization.
        msg.timestamp_us = int(result["source_timestamp_us"])
        msg.q_des = np.asarray(result["target_sdk"]).tolist()
        gain_scale = float(result["gain_scale"])
        if not math.isfinite(gain_scale) or not 0 <= gain_scale <= 1:
            raise ValueError("Invalid calibration gain scale")
        msg.kp = [KP * gain_scale] * 12
        msg.kd = [KD * gain_scale] * 12
        self.has_published = True
        self.lc.publish("pd_plustau_targets", msg.encode())

    def stop(self):
        msg = pd_tau_targets_lcmt()
        msg.robot_id, msg.id = PROTOCOL_ID, -1
        # Bridge interprets -1 as a latched stop, regardless of position fields.
        for _ in range(3):
            self.lc.publish("pd_plustau_targets", msg.encode())


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _wait_state(receiver, runtime):
    deadline = time.monotonic() + runtime["startup_timeout_s"]
    while True:
        try:
            return receiver.snapshot(runtime["state_timeout_s"], runtime["rc_timeout_s"])
        except StateError:
            if receiver.fault or time.monotonic() >= deadline:
                raise
            time.sleep(POLICY_DT)


def run_deployment(config, policy, log_path=None):
    """Called only by explicit --run after JIT preflight; imports LCM lazily."""
    validate_config(config, require_ready=True)
    import lcm
    controller = Controller(config, policy)
    rt = config["runtime"]
    if log_path is None:
        log_path = Path("logs") / ("him_deploy_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Open log before receiving or publishing commands; never overwrite a run.
    with log_path.open("x", encoding="utf-8") as log:
        def record(data):
            log.write(json.dumps(data, default=_json_default, allow_nan=False,
                                 separators=(",", ":")) + "\n")
        record(dict(event="start", config=config, protocol=PROTOCOL_ID, time=time.time(),
                    log_schema_version=3,
                    history_storage="Per-cycle observations and history_count; full history retained in fault snapshot"))
        log.flush()
        lc = lcm.LCM(rt["lcm_url"])
        receiver, publisher = StateReceiver(lc), CommandPublisher(lc)
        phase = None
        last_settle_report = None
        last_command_report = None
        previous_cycle_timing = None
        current_cycle_timing = None
        last_published_result = None
        state = None
        try:
            receiver.start()
            _wait_state(receiver, rt)
            print("Fresh SDK state. Center sticks; release then press R2 to calibrate. L2 stops.")
            print("Policy tracking guard: >{:.3f} rad for {:.3f}s on the SAME joint; "
                  ">{:.3f} rad stops immediately. Calibration tracking is unchanged.".format(
                      rt["max_tracking_error_rad"], rt.get("tracking_error_timeout_s", 0.0),
                      rt.get("max_instant_tracking_error_rad", rt["max_tracking_error_rad"])))
            deadline = time.monotonic()
            while True:
                now = time.monotonic()
                current_cycle_timing = dict(start_monotonic_s=now, stage="scheduler",
                                            scheduler_lateness_s=max(0.0, now - deadline))
                if now - deadline > rt["max_cycle_lateness_s"]:
                    raise SafetyFault("Policy scheduler missed deadline")
                current_cycle_timing["stage"] = "state_snapshot"
                snapshot_start = time.monotonic()
                state = receiver.snapshot(rt["state_timeout_s"], rt["rc_timeout_s"])
                snapshot_end = time.monotonic()
                current_cycle_timing["state_snapshot_s"] = snapshot_end - snapshot_start
                current_cycle_timing["stage"] = "controller"
                result = controller.step(state, now)
                # Inference might hang or overrun; do not send its late result.
                finished = time.monotonic()
                current_cycle_timing.update(controller_step_s=finished - snapshot_end,
                                            control_work_s=finished - now)
                if finished - now > POLICY_DT + rt["max_cycle_lateness_s"]:
                    raise SafetyFault("Inference/control work exceeded deadline")
                if finished - state["received_monotonic"] > rt["state_timeout_s"]:
                    raise SafetyFault("State expired during inference")
                # Recheck receiver fault/L2 after inference and before actuation.
                current_cycle_timing["stage"] = "stop_recheck"
                recheck_start = time.monotonic()
                latest = receiver.snapshot(rt["state_timeout_s"], rt["rc_timeout_s"])
                current_cycle_timing["stop_recheck_s"] = time.monotonic() - recheck_start
                if latest["l2"] or latest["r2_edges"] > state["r2_edges"]:
                    raise SafetyFault("Stop button arrived during policy computation")
                current_cycle_timing["stage"] = "publish"
                publish_start = time.monotonic()
                if result["target_sdk"] is not None:
                    publisher.publish(result)
                    last_published_result = result
                current_cycle_timing["publish_s"] = time.monotonic() - publish_start
                result.update(event="cycle", monotonic_s=now, compute_s=finished - now,
                              previous_cycle_timing=previous_cycle_timing)
                current_cycle_timing["stage"] = "logging"
                log_start = time.monotonic()
                # Each 270-value history repeats the last six 45-value observations.
                # Keep all observations/actions at 50 Hz, and avoid serializing
                # those same history values again on every control cycle.
                record({key: value for key, value in result.items() if key != "history"})
                current_cycle_timing["log_s"] = time.monotonic() - log_start
                current_cycle_timing["stage"] = "console"
                console_start = time.monotonic()
                if phase != result["phase"]:
                    phase = result["phase"]
                    print("HIM phase:", phase, "(READY: release then press R2 for policy)" if phase == "READY" else "")
                    log.flush()
                if (phase == "SETTLE" and "settle_remaining_s" in result and
                        (last_settle_report is None or now - last_settle_report >= 1.0)):
                    print("HIM settle: {:.1f}s left; {} error {:.4f} rad, tolerance {:.4f} rad; "
                          "history {}/{}".format(
                              result["settle_remaining_s"], result["max_nominal_error_joint"],
                              result["max_nominal_error_rad"], result["calibration_tolerance_rad"],
                              result["history_count"], rt["history_warmup_frames"]))
                    last_settle_report = now
                if (phase == "RUNNING" and
                        (last_command_report is None or now - last_command_report >= 1.0)):
                    print("HIM command: vx={:.3f} m/s, vy={:.3f} m/s, yaw={:.3f} rad/s "
                          "(requested, not measured speed)".format(*result["commands"]))
                    last_command_report = now
                current_cycle_timing["console_and_flush_s"] = time.monotonic() - console_start
                deadline += POLICY_DT
                sleep_start = time.monotonic()
                delay = deadline - sleep_start
                current_cycle_timing.update(stage="sleep", pre_sleep_work_s=sleep_start - now,
                                            requested_sleep_s=max(0.0, delay))
                if delay > 0:
                    time.sleep(delay)
                else:
                    # No catch-up cycles or multiple history shifts on one state.
                    deadline = time.monotonic()
                current_cycle_timing.update(stage="complete", actual_sleep_s=time.monotonic() - sleep_start)
                previous_cycle_timing = current_cycle_timing.copy()
        except KeyboardInterrupt:
            if publisher.has_published:
                publisher.stop()
            record(dict(event="stop", reason="KeyboardInterrupt"))
            print("HIM stopped by operator; bridge stop is latched.")
        except Exception as exc:
            # Send the stop before fault formatting or disk I/O can delay it.
            if publisher.has_published:
                publisher.stop()
            record(dict(event="fault", reason=str(exc), fault_monotonic_s=time.monotonic(),
                        fault_details=getattr(exc, "details", None),
                        last_state_snapshot=state,
                        current_cycle_timing=current_cycle_timing,
                        previous_cycle_timing=previous_cycle_timing,
                        last_published_cycle=last_published_result))
            raise
        finally:
            controller.phase = "STOPPED"
            try:
                if publisher.has_published:
                    publisher.stop()
            finally:
                receiver.close()
                log.flush()
    return str(log_path)


def observe_states(config, duration=10.0):
    """Raw SDK axes plus configured gravity, with NO command publications."""
    validate_config(config, require_ready=False)
    if duration <= 0 or not math.isfinite(duration):
        raise ValueError("Observation duration must be finite and positive")
    import lcm
    rt, alignment = config["runtime"], config["alignment"]
    receiver = StateReceiver(lcm.LCM(rt["lcm_url"]))
    try:
        receiver.start()
        _wait_state(receiver, rt)
        end = time.monotonic() + duration
        while time.monotonic() < end:
            s = receiver.snapshot(rt["state_timeout_s"], rt["rc_timeout_s"])
            attitude = s["rpy"] if alignment["attitude_source"] == "rpy" else s["quaternion_wxyz"]
            s["gravity_body"] = projected_gravity(attitude, alignment["attitude_source"],
                                                    alignment["sensor_to_body_rotation"])
            print(json.dumps(s, default=_json_default, allow_nan=False))
            time.sleep(0.2)
    finally:
        receiver.close()
