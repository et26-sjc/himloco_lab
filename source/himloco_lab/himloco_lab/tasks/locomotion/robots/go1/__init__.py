"""Official Go1 USD with the official actuator network."""
import gymnasium as gym

for suffix, config in (("", "Go1EnvCfg"), ("-Play", "Go1PlayEnvCfg")):
    gym.register(
        id=f"Unitree-Go1-Velocity{suffix}",
        entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:{config}",
            "himloco_rsl_rl_cfg": "himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:Go1PPORunnerCfg",
        },
    )

for suffix, config in (("", "Go1PDEnvCfg"), ("-Play", "Go1PDPlayEnvCfg")):
    gym.register(
        id=f"Unitree-Go1-Velocity-PD{suffix}",
        entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.pd_env_cfg:{config}",
            "himloco_rsl_rl_cfg": "himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:Go1PDPPORunnerCfg",
        },
    )
