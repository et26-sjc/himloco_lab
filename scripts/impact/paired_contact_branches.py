"""Controlled blind Go1 drops: hidden patch heights, verified common prefixes.

This concerns FUTURE force/rate uncertainty only. It does not establish limits
for post-contact estimation or for the walking policy's natural distribution.
"""
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(p);p.add_argument('--headless',action='store_true')
args=p.parse_args();launcher=AppLauncher(args);sys.argv=[sys.argv[0]]

import numpy as np
import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab import sim as sim_utils
from himloco_lab.impact.collection_cfg import make_collection_cfg
from himloco_lab.impact.recording_env import ImpactRecordingEnv
from himloco_lab.rsl_rl import HimlocoVecEnvWrapper
from himloco_lab.impact.dataset import file_hash


def main():
    torch.set_num_threads(4)
    output=Path('datasets/go1_impact/force_study_v1/paired_branches')
    if (output/'report.json').exists():
        print('PAIRED_BRANCHES_ALREADY_COMPLETE',flush=True)
        return
    output.mkdir(parents=True,exist_ok=False);(output/'INCOMPLETE').touch()
    n,groups,branches,steps=4096,128,32,60
    cfg=make_collection_cfg(n,73,'drop',.0025,observation_noise=False)
    cfg.sim.device=args.device
    cfg.scene.ground_patch=RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/HiddenPatch',
        spawn=sim_utils.CuboidCfg(size=(1.8,1.8,.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,-.05)))
    raw=ImpactRecordingEnv(cfg)
    env=HimlocoVecEnvWrapper(raw,history_length=5,privileged_history_length=0)
    try:
        robot,patch=raw.scene['robot'],raw.scene['ground_patch']
        root=robot.data.root_state_w.torch.clone();root[:,2]=.68
        root[:,7:]=0
        robot.write_root_state_to_sim(root)
        gen=torch.Generator(device=env.device).manual_seed(738)
        q=robot.data.default_joint_pos.torch.clone()
        delta=(torch.rand(groups,12,device=env.device,generator=gen)-.5)*.04
        q+=delta.repeat_interleave(branches,0)
        robot.write_joint_state_to_sim(q,torch.zeros_like(q))
        pose=patch.data.root_state_w.torch[:,:7].clone()
        height=torch.linspace(.01,.20,branches,device=env.device).repeat(groups)
        pose[:,2]=height-.05
        patch.write_root_pose_to_sim(pose)
        raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(dt=cfg.sim.dt)
        # Reset the policy-history wrapper after the prescribed initial state.
        env.obs_history_buf.zero_()
        env.obs_history_buf[:,:45]=raw.observation_manager.compute()['policy']
        histories=[];forces=[];dones=[]
        with torch.inference_mode():
            for step in range(steps):
                obs=env.get_observations()
                histories.append(obs.cpu().numpy().copy())
                _,_,_,done,_,_,_=env.step(torch.zeros(n,12,device=env.device))
                forces.append(raw.impact_normal_history_w[:,-8:].cpu().numpy().copy())
                dones.append(done.cpu().numpy().copy())
        np.savez(output/'branches.npz',history=np.stack(histories),forces=np.stack(forces),
                 done=np.stack(dones),height=height.cpu().numpy(),group=np.arange(n)//branches)
        report={'status':'completed','groups':groups,'branches':branches,'steps':steps,
                'physics_dt':.0025,'policy_dt':.02,'input':'270D raw noiseless actor history',
                'action':'zero policy action; nominal-pose PD remains active (Kp=40, Kd=1); not walking policy',
                'hidden':'per-environment kinematic contact patch top height .01..20m',
                'scope':'future prediction only; exact/approximate prefix equality must be measured',
                'sha256':file_hash(output/'branches.npz')}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(output/'INCOMPLETE').unlink()
        print('PAIRED_BRANCHES_COMPLETE',flush=True)
    finally:env.close()


try:main()
except BaseException:
    import traceback
    traceback.print_exc();launcher.app.close(exit_code=1)
else:launcher.app.close()
