"""Explicit policy contract and small YAML/CLI configuration."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

LAB_JOINT_NAMES = tuple(f'{leg}_{joint}_joint' for joint in ('hip', 'thigh', 'calf')
                        for leg in ('FL', 'FR', 'RL', 'RR'))
LEGACY_JOINT_NAMES = tuple(f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'RL', 'RR')
                           for joint in ('hip', 'thigh', 'calf'))
MODEL_PATH = Path(__file__).parent / 'models/unitree_go1/scene_torque.xml'


def default_position(name):
    leg, joint, _ = name.split('_')
    return (0.1 if leg.endswith('L') else -0.1) if joint == 'hip' else (
        (0.8 if leg.startswith('F') else 1.0) if joint == 'thigh' else -1.5)


def gain_vector(value, names):
    """Resolve scalar, 12-vector, or named group/full-joint overrides."""
    if isinstance(value, dict):
        valid = {'default', 'hip', 'thigh', 'calf', *names}
        if set(value) - valid:
            raise ValueError(f'Unknown gain keys: {set(value) - valid}')
        values = [value.get(n, value.get(n.split('_')[1], value.get('default'))) for n in names]
        if any(v is None for v in values):
            raise ValueError('Gain mapping must cover all joints or supply default')
    else:
        values = value
    result = np.broadcast_to(np.asarray(values, dtype=float), (len(names),)).copy()
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Gains must be finite and nonnegative')
    return result


@dataclass
class ControllerConfig:
    type: str = 'pd'
    kp: object = 40.0
    kd: object = 1.0
    action_scale: float = 0.25
    hip_reduction: float = 1.0


@dataclass
class SimConfig:
    model: str = str(MODEL_PATH)
    profile: str = 'lab'
    physics_dt: float = 0.005
    decimation: int = 4
    initial_height: float = 0.4
    initial_yaw: float = 0.0
    fall_height: float = 0.16
    fall_tilt: float = 1.0
    fall_duration: float = 0.1
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    @property
    def joint_names(self):
        return LAB_JOINT_NAMES if self.profile == 'lab' else LEGACY_JOINT_NAMES

    @property
    def policy_dt(self):
        return self.physics_dt * self.decimation

    def to_dict(self):
        return asdict(self)


def load_config(path=None, **overrides):
    values = yaml.safe_load(Path(path).read_text()) if path else {}
    values = values or {}
    controller = ControllerConfig(**values.pop('controller', {}))
    cfg = SimConfig(controller=controller, **values)
    if path and not Path(cfg.model).is_absolute():
        cfg.model = str((Path(path).resolve().parent/cfg.model).resolve())
    for key, value in overrides.items():
        if value is not None:
            setattr(controller if key in ('kp', 'kd') else cfg, key, value)
    if cfg.profile not in ('lab', 'legacy') or cfg.controller.type != 'pd':
        raise ValueError('Expected profile lab/legacy and controller type pd')
    if (cfg.physics_dt <= 0 or not np.isfinite(cfg.physics_dt)
            or not isinstance(cfg.decimation, int) or cfg.decimation < 1):
        raise ValueError('Invalid simulation frequency')
    if (not np.isfinite([cfg.initial_height, cfg.initial_yaw, cfg.fall_height, cfg.fall_tilt, cfg.fall_duration]).all()
            or min(cfg.initial_height, cfg.fall_height, cfg.fall_tilt, cfg.fall_duration) <= 0):
        raise ValueError('Invalid initial pose or geometric fall thresholds')
    if not np.isfinite([controller.action_scale, controller.hip_reduction]).all() or (
            controller.action_scale <= 0 or controller.hip_reduction <= 0):
        raise ValueError('Invalid action scale or hip reduction')
    gain_vector(controller.kp, cfg.joint_names)
    gain_vector(controller.kd, cfg.joint_names)
    return cfg
