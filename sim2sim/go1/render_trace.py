"""Render a recorded real rollout state; this does not run or replace GUI control."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from sim2sim.go1.config import SimConfig
from sim2sim.go1.mujoco_adapter import MujocoAdapter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    trace = np.load(args.trace)
    adapter = MujocoAdapter(SimConfig())
    adapter.data.qpos[:] = trace['full_qpos'][-1]
    adapter.data.qvel[:] = trace['full_qvel'][-1]
    mujoco.mj_forward(adapter.model, adapter.data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = adapter.body_id
    camera.distance = 1.8
    camera.azimuth = 130
    camera.elevation = -18
    with mujoco.Renderer(adapter.model, height=480, width=640) as renderer:
        renderer.update_scene(adapter.data, camera=camera)
        pixels = renderer.render()
    from matplotlib.image import imsave
    imsave(args.output, pixels)
    print('ROLLOUT_FRAME_RENDERED', args.output)
