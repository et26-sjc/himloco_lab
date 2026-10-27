"""Paired PD40/1 flat MuJoCo comparison, with explicit geometry diagnostics."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
import torch
from sim2sim.go1.config import SimConfig
from sim2sim.go1.mujoco_adapter import MujocoAdapter
from sim2sim.go1.policy_runner import PolicyRunner
from sim2sim.go1.simulation import run_episode

COMMANDS = [[.3,0.,0.],[-.3,0.,0.],[0.,0.,.3],[.1,0.,0.]]


def geometry(cfg,trace):
    a=MujocoAdapter(cfg)
    knees=[mujoco.mj_name2id(a.model,mujoco.mjtObj.mjOBJ_BODY,f'{leg}_calf') for leg in ('RL','RR')]
    bodies=[mujoco.mj_name2id(a.model,mujoco.mjtObj.mjOBJ_BODY,f'{leg}_{part}')
            for leg in ('FL','FR','RL','RR') for part in ('thigh','calf')]
    geoms=set(np.flatnonzero(np.isin(a.model.geom_bodyid,bodies) &
        ((a.model.geom_contype!=0)|(a.model.geom_conaffinity!=0))).tolist())-set(a.foot_geom_ids)
    rear=[i for i in geoms if a.model.geom_bodyid[i] in knees]
    if any(a.model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_CAPSULE for i in rear):
        raise ValueError('Expected the pinned Menagerie non-foot calf capsules')
    knee_z,clearances,contacts=[],[],[]
    for i in np.flatnonzero(trace['time']>2)[::4]:
        a.data.qpos[:]=trace['full_qpos'][i]
        a.data.qvel[:]=trace['full_qvel'][i]
        a.data.ctrl[a.actuator_ids]=trace['applied_torque'][i]
        mujoco.mj_forward(a.model,a.data)
        knee_z.append(a.data.xpos[knees,2].copy())
        # Capsule support function along world -Z, measured from the flat plane.
        clearances.append([a.data.geom_xpos[g,2]-a.model.geom_size[g,0]
            -a.model.geom_size[g,1]*abs(a.data.geom_xmat[g].reshape(3,3)[2,2]) for g in rear])
        contacts.append(any(c.dist<=0 and ((c.geom1==a.floor_id and c.geom2 in geoms) or
            (c.geom2==a.floor_id and c.geom1 in geoms)) for c in a.data.contact))
    return dict(rear_knee_origin_z_p5=float(np.percentile(knee_z,5)) if knee_z else None,
        rear_calf_collision_clearance_p5=float(np.percentile(clearances,5)) if clearances else None,
        geometric_nonfoot_contact_fraction=float(np.mean(contacts)) if contacts else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--duration',type=float,default=20.)
    p.add_argument('--output',default='artifacts/go1_deploy_v1/evaluation_20261027/mujoco')
    args=p.parse_args()
    if args.duration<=2: p.error('Use duration > 2 seconds')
    out=Path(args.output)
    if (out/'comparison.json').exists(): raise FileExistsError('Choose a new output path')
    torch.set_num_threads(1)
    cfg=SimConfig(passive_damping=0.)
    results=[];initial=None
    for label,path in (('baseline',args.baseline),('candidate',args.candidate)):
        policy=PolicyRunner(path)
        cases=[]
        for command in COMMANDS:
            metrics,trace=run_episode(cfg,policy,command,args.duration)
            if initial is None: initial=metrics['initial_state_sha256']
            assert metrics['initial_state_sha256']==initial
            mask=trace['time']>2
            cases.append(dict(command=command,survival_s=metrics['survival_s'],fall=metrics['geometric_fall'],
                mean_vx=float(trace['velocity'][mask,0].mean()) if mask.any() else None,
                mean_wz=float(trace['omega'][mask,2].mean()) if mask.any() else None,
                linear_rmse=float(np.sqrt(np.mean(np.sum((trace['velocity'][mask,:2]-command[:2])**2,axis=1)))) if mask.any() else None,
                yaw_rmse=float(np.sqrt(np.mean((trace['omega'][mask,2]-command[2])**2))) if mask.any() else None,
                base_height_mean=float(trace['position'][mask,2].mean()) if mask.any() else None,
                **geometry(cfg,trace)))
        results.append(dict(label=label,policy=policy.metadata,cases=cases))
        print('MUJOCO_PAIR_COMPLETE',label,flush=True)
    report=dict(status='passed',duration_s=args.duration,warmup_excluded_s=2.,kp=40.,kd=1.,passive_damping=0.,
        paired_initial_states=True,initial_state_sha256=initial,results=results,
        limitations='One nominal episode per command, no robustness confidence interval. Knee joint-origin '
        'height is not surface clearance. Calf capsule clearance/contact applies to the pinned Menagerie '
        'collision approximation, excludes all four foot spheres, and is not a hardware measurement.')
    out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('MUJOCO_PAIR_OK',out/'comparison.json')


if __name__=='__main__': main()
