"""Go1 rough terrain HIM baseline, independent of Go2 URDF and deployment settings."""
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG
from himloco_lab.tasks.locomotion import mdp
from himloco_lab.tasks.locomotion.velocity_env_cfg import RobotEnvCfg

# Isaac Lab's native USD order. Actions and joint observations explicitly share it.
GO1_JOINT_NAMES = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf")
                   for leg in ("FL", "FR", "RL", "RR")]

@configclass
class Go1EnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.joint_ordering = GO1_JOINT_NAMES
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.base_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.contact_forces.update_period = self.sim.dt
        for name in ("add_base_mass", "randomize_rigid_body_com", "external_force", "push_robot"):
            getattr(self.events, name).params["asset_cfg"].body_names = "trunk"
        self.observations.critic.base_external_force.params["asset_cfg"].body_names = "trunk"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"
        action = self.actions.JointPositionAction
        action.joint_names = GO1_JOINT_NAMES
        action.preserve_order = True
        # Keep current policy's 45D structure and clean critic supervision.
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.debug_vis = False
        self.scene.terrain.max_init_terrain_level = 3
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        for name in ("pyramid_stairs", "pyramid_stairs_inv"):
            terrains[name].step_height_range = (0.04, 0.18)
        terrains["hf_slope_with_noise"].noise_amplitude_range = (0.01, 0.05)
        terrains["discrete_obstacles"].max_height_range = (0.03, 0.12)
        self.rewards.base_height_l2.params["target_height"] = 0.30
        self.events.randomize_rigid_body_com.params["com_range"] = {
            "x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}

@configclass
class Go1PlayEnvCfg(Go1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 2
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.7, 0.7), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0))
        self.commands.base_velocity.low_vel_env_lin_x_ranges = (0.7, 0.7)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.terrain_levels = None
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.push_robot = None
        self.events.external_force = None
