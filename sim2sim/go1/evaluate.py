"""Headless command suite and fixed-policy PD gain sensitivity experiments."""
import copy
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim2sim.go1.cli import parser, setup, save_result
from sim2sim.go1.simulation import run_episode

COMMANDS = ([0., 0., 0.], [.5, 0., 0.], [.7, 0., 0.], [0., 0., .3], [.7, 0., .3])


def main():
    cli = parser(__doc__)
    cli.add_argument('--pd-sweep', action='store_true', help='Sequential 40/1, 20/.5, 30/.75 with identical resets')
    args = cli.parse_args()
    cfg, policy = setup(args)
    if args.pd_sweep and (args.kp is not None or args.kd is not None):
        cli.error('--pd-sweep uses its three declared pairs; omit --kp/--kd')
    pairs = [(40., 1.), (20., .5), (30., .75)] if args.pd_sweep else [(cfg.controller.kp, cfg.controller.kd)]
    commands = [args.velocity] if args.velocity is not None else COMMANDS
    root = Path(args.output)
    rows = []
    references = {}
    for group, (kp, kd) in enumerate(pairs):
        current = copy.deepcopy(cfg)
        current.controller.kp, current.controller.kd = kp, kd
        for index, command in enumerate(commands):
            metrics, trace = run_episode(current, policy, command, args.duration)
            signature = metrics['initial_state_sha256']
            if index in references and references[index] != signature:
                raise AssertionError('PD sweep initial states differ')
            references[index] = signature
            metrics['suite_group'] = group
            save_result(root/f'group_{group}_command_{index}', metrics, trace)
            rows.append(metrics)
            print('EVAL_CASE', json.dumps({k:metrics[k] for k in ('command', 'controller', 'survival_s',
                'geometric_fall', 'linear_tracking_rmse', 'yaw_tracking_rmse')}), flush=True)
    report = dict(policy=policy.metadata, paired_initial_states=True, pd_sweep=args.pd_sweep, episodes=rows,
                  note='One deterministic episode per gain/command; fall-rate values are not population estimates.')
    report['groups'] = [dict(controller=rows[group*len(commands)]['controller'],
        case_count=len(commands), fall_case_count=sum(r['geometric_fall'] for r in rows if r['suite_group'] == group),
        completed_case_count=sum(r['completed_horizon'] for r in rows if r['suite_group'] == group))
        for group in range(len(pairs))]
    (root/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    fields = ['suite_group', 'actual_gain_label', 'command', 'survival_s', 'geometric_fall_rate',
              'linear_tracking_rmse', 'yaw_tracking_rmse', 'mean_vx', 'mean_vy', 'mean_wz',
              'base_height_mean', 'torque_saturation_ratio', 'rated_velocity_exceedance_ratio']
    with (root/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:row[k] for k in fields})
    print('MUJOCO_EVALUATION_OK', root/'summary.json')


if __name__ == '__main__':
    main()
