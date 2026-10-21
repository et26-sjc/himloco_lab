"""Go2 configuration; keep its custom URDF, actuator and deployment metadata."""
from isaaclab.utils import configclass
from himloco_lab.assets.unitree import UNITREE_GO2_CFG
from himloco_lab.tasks.locomotion import mdp
from himloco_lab.tasks.locomotion.velocity_env_cfg import RobotEnvCfg as BaseRobotEnvCfg

@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    def __post_init__(self):
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        super().__post_init__()

@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.curriculum = True
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(1, 1), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0, 0), 
        )
        self.commands.base_velocity.low_vel_env_lin_x_ranges=(1,1)
        
        # Disable randomization events for play mode
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.push_robot = None
