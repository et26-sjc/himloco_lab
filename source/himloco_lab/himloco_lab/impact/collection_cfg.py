"""Isolated collection overrides; standard Go1 tasks are never re-registered."""
from isaaclab.sensors import ContactSensorCfg

from himloco_lab.tasks.locomotion.robots.go1.pd_env_cfg import Go1PDEnvCfg

from .labels import ContactWindow


SCENARIOS = ("plane_walk", "rough_walk", "suite_walk", "stand", "drop", "airborne")


def make_collection_cfg(num_envs=4096, seed=1, scenario="rough_walk", physics_dt=0.005,
                        velocity=(0.7, 0.0, 0.0), randomization=False, observation_noise=False,
                        instrumented=True):
    if scenario not in SCENARIOS or num_envs < 1:
        raise ValueError("Invalid scenario or environment count")
    window = ContactWindow(physics_dt=physics_dt)
    cfg = Go1PDEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.dt = physics_dt
    cfg.decimation = window.decimation
    cfg.sim.render_interval = window.decimation
    cfg.scene.contact_forces.update_period = physics_dt
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.seed = seed
    if scenario == "suite_walk":
        from .terrains import research_terrain
        cfg.scene.terrain.terrain_generator = research_terrain(seed)
        cfg.scene.terrain.max_init_terrain_level = None
        # 4.2 m nominal travel stays inside each 12 m tile.
        cfg.episode_length_s = 6.0
    elif scenario != "rough_walk":
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    if not randomization:
        for name in ("physics_material", "add_base_mass", "randomize_rigid_body_com", "external_force", "push_robot"):
            setattr(cfg.events, name, None)
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        cfg.events.reset_base.params["pose_range"] = {k: (0.0, 0.0) for k in ("x", "y", "yaw")}
        cfg.events.reset_base.params["velocity_range"] = {
            k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    if scenario == "suite_walk":
        # Controlled diversity, independent of mass/friction/push randomization.
        cfg.events.reset_base.params["pose_range"] = {
            "x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-0.1, 0.1)}
        cfg.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        cfg.events.reset_robot_joints.params["velocity_range"] = (-0.1, 0.1)
    cfg.observations.policy.enable_corruption = observation_noise
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.min_command_norm = 0.0
    command.resampling_time_range = (1e9, 1e9)
    command.ranges.lin_vel_x = (velocity[0], velocity[0])
    command.ranges.lin_vel_y = (velocity[1], velocity[1])
    command.ranges.ang_vel_z = (velocity[2], velocity[2])
    command.low_vel_env_lin_x_ranges = (velocity[0], velocity[0])
    if scenario in ("stand", "drop", "airborne"):
        command.ranges.lin_vel_x = command.low_vel_env_lin_x_ranges = (0.0, 0.0)
        command.ranges.lin_vel_y = command.ranges.ang_vel_z = (0.0, 0.0)
    if scenario in ("drop", "airborne"):
        height = 0.6 if scenario == "drop" else 2.0
        cfg.scene.robot.init_state.pos = (0.0, 0.0, height)
    if instrumented:
        cfg.scene.impact_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_foot", update_period=physics_dt,
            history_length=window.history_length, track_air_time=False, debug_vis=False)
    return cfg
