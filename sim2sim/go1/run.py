"""GUI or headless Go1 policy-controlled MuJoCo simulation."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim2sim.go1.cli import parser, setup, save_result
from sim2sim.go1.simulation import run_episode


def main():
    cli = parser(__doc__)
    cli.add_argument('--headless', action='store_true')
    cli.add_argument('--real-time', action='store_true')
    cli.add_argument('--zero-action', action='store_true', help='Diagnostic nominal posture, not policy control')
    args = cli.parse_args()
    cfg, policy = setup(args)
    metrics, trace = run_episode(cfg, policy, args.velocity or [.7, 0., .3], args.duration,
                                 gui=not args.headless, real_time=args.real_time, zero_action=args.zero_action)
    save_result(args.output, metrics, trace)
    print('MUJOCO_RUN_OK', metrics['survival_s'], 'fall=', metrics['geometric_fall'])


if __name__ == '__main__':
    main()
