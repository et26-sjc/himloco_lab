"""Controlled passive-parameter experiments; never overwrite the shipped model.

Eight combinations isolate Menagerie's extra damping/armature/frictionloss while
holding policy, Kp40/Kd1, initial state, command and frequency fixed.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from sim2sim.go1.cli import save_result
from sim2sim.go1.config import SimConfig
from sim2sim.go1.mujoco_adapter import MujocoAdapter
from sim2sim.go1.policy_runner import PolicyRunner, sha256
from sim2sim.go1.simulation import run_episode


def write_variant(destination, overrides):
    base = Path(SimConfig().model).parent
    destination.mkdir(parents=True, exist_ok=True)
    root = ET.parse(base/'go1_torque.xml').getroot()
    root.find('compiler').set('meshdir', os.path.relpath(base/'assets', destination))
    for joint in root.findall('.//default//joint'):
        for key, value in overrides.items():
            joint.set(key, str(value))
    ET.indent(root)
    ET.ElementTree(root).write(destination/'go1_torque.xml', encoding='unicode')
    scene = ET.parse(base/'scene_torque.xml').getroot()
    ET.ElementTree(scene).write(destination/'scene.xml', encoding='unicode')
    files = {n:sha256(destination/n) for n in ('go1_torque.xml', 'scene.xml')}
    files.update({os.path.relpath(p, destination):sha256(p) for p in (base/'assets').iterdir() if p.is_file()})
    record = dict(repository='https://github.com/google-deepmind/mujoco_menagerie',
                  commit=json.loads((base/'provenance.json').read_text())['commit'],
                  diagnostic_only=True, passive_parameter_overrides=overrides,
                  parent_model_sha256=sha256(base/'go1_torque.xml'), files=files)
    (destination/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    return str(destination/'scene.xml')


def summarize(q, targets, knee_z):
    return dict(thigh_mean=q[:, 4:8].mean(0).tolist(),
                thigh_range=np.column_stack((q[:, 4:8].min(0), q[:, 4:8].max(0))).tolist(),
                thigh_target_mean=targets[:, 4:8].mean(0).tolist(),
                calf_peak_to_peak=np.ptp(q[:, 8:12], axis=0).tolist(),
                knee_height_body_mean=knee_z.mean(0).tolist(),
                knee_above_trunk_fraction=(knee_z > 0).mean(0).tolist())


def mujoco_gait(cfg, trace):
    a = MujocoAdapter(cfg)
    ids = [mujoco.mj_name2id(a.model, mujoco.mjtObj.mjOBJ_BODY, f'{leg}_calf')
           for leg in ('FL', 'FR', 'RL', 'RR')]
    indices = np.flatnonzero(trace['time'] > 2)[::4]
    if not len(indices):
        return None
    knees = []
    for i in indices:
        a.data.qpos[:] = trace['full_qpos'][i]
        mujoco.mj_forward(a.model, a.data)
        rotation = a.data.xmat[a.body_id].reshape(3, 3)
        knees.append(((a.data.xpos[ids] - a.data.xpos[a.body_id]) @ rotation)[:, 2])
    return summarize(trace['q'][indices], trace['q_target'][indices], np.array(knees))


def plot_comparison(record, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reference = next(r for r in record['results'] if r['name'] == 'menagerie')
    corrected = next(r for r in record['results'] if r['name'] == 'zero_damping')
    series = [('Isaac Lab', record['isaaclab_gait']), ('MuJoCo original', reference['gait']),
              ('MuJoCo damping=0', corrected['gait'])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    fields = [('thigh_mean', 'Mean thigh angle (rad)'), ('calf_peak_to_peak', 'Calf peak-to-peak (rad)'),
              ('knee_height_body_mean', 'Mean knee height relative to trunk (m)'),
              ('knee_above_trunk_fraction', 'Fraction knee above trunk')]
    x = np.arange(4)
    for axis, (field, label) in zip(axes.flat, fields):
        for index, (name, data) in enumerate(series):
            values = data[field] if data is not None else np.full(4, np.nan)
            axis.bar(x+(index-1)*.25, values, width=.24, label=name)
        axis.set_xticks(x, ['FL', 'FR', 'RL', 'RR'])
        axis.set_ylabel(label)
        axis.grid(axis='y', alpha=.25)
    axes[0, 0].axhline(np.pi/2, linestyle='--', color='black', alpha=.5)
    axes[0, 0].legend(fontsize=9)
    axes[1, 0].axhline(0, color='black', linewidth=.8)
    fig.suptitle('Fixed Go1 HIM policy / Kp40 Kd1 / command vx=0.7\nSteady samples after 2 s; only passive damping changed in the corrected MuJoCo case')
    fig.savefig(output/'gait_comparison.png', dpi=180)
    fig.savefig(output/'gait_comparison.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--lab-gait', default='artifacts/sim2sim/gait_diagnosis/isaaclab/gait.npz')
    parser.add_argument('--output', default='artifacts/sim2sim/gait_diagnosis/passives')
    args = parser.parse_args()
    if args.duration <= 2:
        parser.error('Use duration > 2 seconds to collect steady gait samples')
    import torch
    torch.set_num_threads(1)
    policy = PolicyRunner(args.policy)
    output = Path(args.output)
    results = []
    initial = None
    for bits in itertools.product((False, True), repeat=3):
        overrides = {key:0. for key, enabled in zip(('damping', 'armature', 'frictionloss'), bits) if enabled}
        name = 'menagerie' if not overrides else 'zero_'+'_'.join(overrides)
        cfg = SimConfig()
        if overrides:
            cfg.model = write_variant(output/name/'model', overrides)
        metrics, trace = run_episode(cfg, policy, [.7, 0., 0.], args.duration)
        if initial is None:
            initial = metrics['initial_state_sha256']
        assert metrics['initial_state_sha256'] == initial
        save_result(output/name, metrics, trace)
        summary = dict(name=name, overrides=overrides, linear_rmse=metrics['linear_tracking_rmse'],
            mean_vx=metrics['mean_vx'], height=metrics['base_height_mean'], survival_s=metrics['survival_s'],
            fall=metrics['geometric_fall'], gait=mujoco_gait(cfg, trace),
            effective_damping=metrics['model']['passive_damping'], effective_armature=metrics['model']['armature'],
            effective_frictionloss=metrics['model']['frictionloss'])
        results.append(summary)
        print('PASSIVE_CASE', json.dumps(summary), flush=True)
    lab = np.load(args.lab_gait)
    valid = (lab['time'] > 2)[:, None] & lab['active']
    reference = summarize(lab['q'][valid], lab['q_target'][valid], lab['knee_body'][valid][:, :, 2])
    record = dict(policy=policy.metadata, command=[.7, 0, 0], kp=40., kd=1.,
                  same_initial_state=True, isaaclab_gait=reference, results=results,
                  note='Controlled diagnostic variants only; original Menagerie model and defaults unchanged.')
    (output/'comparison.json').write_text(json.dumps(record, indent=2)+'\n')
    plot_comparison(record, output)
    print('PASSIVE_DIAGNOSIS_OK', output/'comparison.json')


if __name__ == '__main__':
    main()
