"""Run with the OLD Isaac Gym Python to query its actual imported DOF order."""
import argparse
import json
from pathlib import Path
from isaacgym import gymapi

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', default='/home/sjc/HIMLoco-Go1')
parser.add_argument('--output', required=True)
args = parser.parse_args()
gym = gymapi.acquire_gym()
params = gymapi.SimParams()
params.up_axis = gymapi.UP_AXIS_Z
params.dt = .005
params.use_gpu_pipeline = False
params.physx.use_gpu = False
sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
assert sim is not None
options = gymapi.AssetOptions()
# LeggedRobot._create_envs and the inherited Go1RoughCfg.asset values.
settings = dict(default_dof_drive_mode=3, collapse_fixed_joints=True,
                replace_cylinder_with_capsule=True, flip_visual_attachments=False,
                fix_base_link=False, density=.001, angular_damping=0., linear_damping=0.,
                max_angular_velocity=1000., max_linear_velocity=1000., armature=0.,
                thickness=.01, disable_gravity=False)
for name, value in settings.items():
    setattr(options, name, value)
asset_dir = Path(args.baseline) / 'legged_gym/resources/robots/go1/urdf'
asset = gym.load_asset(sim, str(asset_dir), 'go1.urdf', options)
assert asset is not None
result = dict(joint_names=gym.get_asset_dof_names(asset),
              body_names=gym.get_asset_rigid_body_names(asset),
              asset=str(asset_dir / 'go1.urdf'), asset_options=settings,
              evidence='Isaac Gym gym.get_asset_dof_names on the training URDF with training AssetOptions')
Path(args.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
gym.destroy_sim(sim)
