"""Receive coherent SDK samples; no policy ordering or Euler differentiation here."""
import collections
import threading
import time

import numpy as np

from go1_lab_deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from go1_lab_deploy.lcm_types.rc_command_lcmt import rc_command_lcmt
from go1_lab_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt


PROTOCOL_ID = 20261023


class StateError(RuntimeError):
    pass


def finite_vector(values, length, name):
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (length,) or not np.isfinite(a).all():
        raise StateError("Invalid {}".format(name))
    return a.copy()


class StateReceiver:
    """Join IMU and joints by SDK tick, and age *source progress*, not getters.

    The HIM bridge publishes only when SDK tick advances. A repeated LCM frame
    cannot refresh the freshness deadline. Receipts and checks use one local
    monotonic clock; bridge timestamps are only echoed back to that bridge.
    """

    def __init__(self, lc, clock=time.monotonic):
        self.lc, self.clock = lc, clock
        self._lock = threading.Lock()
        self._pending = collections.OrderedDict()
        self._sample = None
        self._rc = None
        self._last_r2 = None
        self._r2_edges = 0
        self._fault = None
        self._stopping = threading.Event()
        self._thread = None
        self._subscriptions = [
            lc.subscribe("state_estimator_data", self._imu_cb),
            lc.subscribe("leg_control_data", self._leg_cb),
            lc.subscribe("rc_command", self._rc_cb),
        ]

    def _fail(self, error):
        with self._lock:
            if self._fault is None:
                self._fault = str(error)

    def _receive(self, kind, msg, values):
        now = self.clock()
        if msg.robot_id != PROTOCOL_ID:
            raise StateError("Bridge protocol mismatch or latched bridge fault: {}".format(msg.robot_id))
        if not 0 <= msg.id <= 0xffffffff or msg.timestamp_us <= 0:
            raise StateError("Invalid SDK tick/timestamp")
        with self._lock:
            # Ignore duplicate/older packets (including reordered delivery).
            if self._sample is not None:
                delta = (msg.id - self._sample["source_tick"]) & 0xffffffff
                if delta == 0 or delta >= 0x80000000:
                    return
            entry = self._pending.setdefault(msg.id, {})
            if kind not in entry:
                entry[kind] = (msg.timestamp_us, now, values)
            if "imu" in entry and "leg" in entry:
                imu, leg = entry["imu"], entry["leg"]
                if imu[0] != leg[0]:
                    raise StateError("IMU/joint source timestamp mismatch")
                sample = dict(imu[2])
                sample.update(leg[2])
                sample.update(source_tick=msg.id, source_timestamp_us=msg.timestamp_us,
                              received_monotonic=min(imu[1], leg[1]))
                self._sample = sample
                del self._pending[msg.id]
            while len(self._pending) > 16:
                self._pending.popitem(last=False)

    def _imu_cb(self, channel, data):
        try:
            m = state_estimator_lcmt.decode(data)
            self._receive("imu", m, {
                "rpy": finite_vector(m.rpy, 3, "RPY"),
                "quaternion_wxyz": finite_vector(m.quat, 4, "quaternion"),
                "gyro_sensor": finite_vector(m.omegaBody, 3, "gyro"),
                "accel_sensor": finite_vector(m.aBody, 3, "accelerometer"),
            })
        except Exception as exc:
            self._fail(exc)

    def _leg_cb(self, channel, data):
        try:
            m = leg_control_data_lcmt.decode(data)
            self._receive("leg", m, {
                "q_sdk": finite_vector(m.q, 12, "joint position"),
                "dq_sdk": finite_vector(m.qd, 12, "joint velocity"),
                "tau_sdk": finite_vector(m.tau_est, 12, "joint torque"),
            })
        except Exception as exc:
            self._fail(exc)

    def _rc_cb(self, channel, data):
        try:
            m = rc_command_lcmt.decode(data)
            left = finite_vector(m.left_stick, 2, "left stick")
            right = finite_vector(m.right_stick, 2, "right stick")
            if np.max(np.abs(np.concatenate((left, right)))) > 1.01:
                raise StateError("RC stick outside [-1,1]")
            r2 = bool(m.right_lower_right_switch)
            with self._lock:
                # A held button at connection is not a start request.
                if self._last_r2 is False and r2:
                    self._r2_edges += 1
                self._last_r2 = r2
                self._rc = dict(left_stick=left, right_stick=right,
                                r2=r2, r2_edges=self._r2_edges,
                                l2=bool(m.left_lower_left_switch),
                                rc_received_monotonic=self.clock())
        except Exception as exc:
            self._fail(exc)

    def snapshot(self, state_timeout_s=0.05, rc_timeout_s=0.1):
        with self._lock:
            if self._fault is not None:
                raise StateError(self._fault)
            if self._sample is None or self._rc is None:
                raise StateError("Waiting for paired SDK state and remote control")
            result = {k: v.copy() if isinstance(v, np.ndarray) else v
                      for k, v in self._sample.items()}
            result.update({k: v.copy() if isinstance(v, np.ndarray) else v
                           for k, v in self._rc.items()})
        now = self.clock()
        result["state_age_s"] = now - result["received_monotonic"]
        result["rc_age_s"] = now - result["rc_received_monotonic"]
        if result["state_age_s"] > state_timeout_s:
            raise StateError("Stale SDK state ({:.3f}s)".format(result["state_age_s"]))
        if result["rc_age_s"] > rc_timeout_s:
            raise StateError("Stale RC state ({:.3f}s)".format(result["rc_age_s"]))
        return result

    @property
    def fault(self):
        with self._lock:
            return self._fault

    def start(self):
        if self._thread is not None:
            raise RuntimeError("State receiver already started")
        self._thread = threading.Thread(target=self._poll, name="him-lcm", daemon=True)
        self._thread.start()

    def _poll(self):
        try:
            while not self._stopping.is_set():
                if self.lc.handle_timeout(10) < 0:
                    raise StateError("LCM receiver failed")
        except Exception as exc:
            self._fail(exc)

    def close(self):
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for sub in self._subscriptions:
            self.lc.unsubscribe(sub)
