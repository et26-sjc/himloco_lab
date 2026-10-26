"""Deployment-oriented reward/command profile, preserving the proven PD task."""
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from himloco_lab.tasks.locomotion import mdp
from himloco_lab.tasks.locomotion.velocity_env_cfg import RewardsCfg
from .pd_env_cfg import Go1PDEnvCfg, Go1PDPlayEnvCfg


@configclass
class Go1DeployV1RewardsCfg(RewardsCfg):
    # Feet have separate rigid bodies. Penalize eight thigh/calf bodies only;
    # keep normal foot contact and trunk termination unchanged.
    undesired_leg_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"])},
    )


def _apply_deploy_v1(cfg):
    cfg.rewards.base_height_l2.params["target_height"] = 0.32
    cfg.rewards.base_height_l2.weight = -2.0
    cfg.commands.base_velocity.min_command_norm = 0.10
    # Training retains heading following for half the environments. The other
    # half keep sampled yaw rates; the actor still observes velocity, not heading.
    cfg.commands.base_velocity.rel_heading_envs = 0.5


@configclass
class Go1PDDeployV1EnvCfg(Go1PDEnvCfg):
    rewards: Go1DeployV1RewardsCfg = Go1DeployV1RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_deploy_v1(self)


@configclass
class Go1PDDeployV1PlayEnvCfg(Go1PDPlayEnvCfg):
    rewards: Go1DeployV1RewardsCfg = Go1DeployV1RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_deploy_v1(self)
        # Play uses explicit velocity commands, as does deployment. Do not erase
        # fixed low-speed test commands with a training sampling deadzone.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.min_command_norm = 0.0
