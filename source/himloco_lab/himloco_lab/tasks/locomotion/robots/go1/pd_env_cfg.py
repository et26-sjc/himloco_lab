"""Deterministic explicit PD comparison; all scene/MDP settings come from Go1."""
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils import configclass

from .velocity_env_cfg import Go1EnvCfg, Go1PlayEnvCfg


GO1_PD_ACTUATOR_CFG = IdealPDActuatorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    stiffness=40.0,
    damping=1.0,
    actuator_effort_limit={".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 35.55},
    actuator_velocity_limit={".*_hip_joint": 30.1, ".*_thigh_joint": 30.1, ".*_calf_joint": 20.06},
)
# Keep the same USD solver limits, friction and armature defaults as Actuator Net.
# Rated velocity above is telemetry, not a new solver clamp or torque-speed curve.
# Lab disables internal drive gains for explicit actuators and evaluates PD at
# every scene.write_data_to_sim(), i.e. all four 0.005 s PhysX substeps.


@configclass
class Go1PDEnvCfg(Go1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators = {"base_legs": GO1_PD_ACTUATOR_CFG.copy()}


@configclass
class Go1PDPlayEnvCfg(Go1PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators = {"base_legs": GO1_PD_ACTUATOR_CFG.copy()}
