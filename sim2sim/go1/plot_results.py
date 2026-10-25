"""Standalone scientific figures from saved gain-sweep metrics, no simulation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(summary, output):
    rows = json.loads(Path(summary).read_text())['episodes']
    groups = sorted({r['suite_group'] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    variables = [('linear_tracking_rmse', 'Linear tracking RMSE (m/s)'),
                 ('yaw_tracking_rmse', 'Yaw tracking RMSE (rad/s)'),
                 ('survival_s', 'Survival time (s)'), ('torque_saturation_ratio', 'Torque saturation fraction')]
    commands = [r['command'] for r in rows if r['suite_group'] == groups[0]]
    x = np.arange(len(commands))
    for ax, (key, label) in zip(axes.flat, variables):
        for j, group in enumerate(groups):
            values = [r for r in rows if r['suite_group'] == group]
            kp, kd = values[0]['controller']['kp'], values[0]['controller']['kd']
            label_gain = f'Kp={kp[0]:g}, Kd={kd[0]:g}' if len(set(kp)) == len(set(kd)) == 1 else f'group {group}'
            ax.bar(x+(j-(len(groups)-1)/2)*.25, [r[key] for r in values], width=.24, label=label_gain)
        ax.set_xticks(x, [str(c) for c in commands], rotation=12, fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.25)
    axes[0, 0].legend()
    fig.suptitle('Fixed HIM policy / Go1 Menagerie model: PD gain sensitivity\nOne deterministic episode per command; early falls censor tracking statistics')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)
