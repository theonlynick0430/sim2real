"""Bimanual YAM relay with OSC on both arms: left picks the cube and sets it down at the
handoff spot, right picks it up there and places it at the goal. Runs on MJWarp (CUDA if available,
else Warp-CPU); the viewer shows world 0.

  mjpython examples/bimanual_relay.py                    # viewer (macOS needs `mjpython`)
  python examples/bimanual_relay.py --headless
  python examples/bimanual_relay.py --nworld 1024 --headless --device cuda:0
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without `pip install -e .`

import numpy as np

from sim2real.control import OSC
from sim2real.envs.scripted import RELAY_HANDOFF, BimanualRelay
from sim2real.sim.mjwarp_backend import MjWarpBackend
from examples._scenes import Viewer, make_scene


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--nworld", type=int, default=1)
  parser.add_argument("--device", default=None, help="warp device, e.g. cuda:0 or cpu (default: cuda if available)")
  parser.add_argument("--separation", type=float, default=0.4, help="distance between the arm bases [m]")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--perturb", action="store_true", help="ctrl-drag in the viewer pushes the robot (world 0)")
  args = parser.parse_args()

  scene = make_scene(bimanual=True, separation=args.separation)
  backend = MjWarpBackend(scene, nworld=args.nworld, device=args.device)

  oscs = [OSC(v, backend.nworld, device=backend.device) for v in backend.views]
  for osc, robot in zip(oscs, scene.robots):
    osc.check_compatible(robot.actuator_mode)
  backend.step1()
  park = [backend.robot_state(arm=a).tcp_pos.numpy()[0].copy() for a in range(backend.narm)]
  relay = BimanualRelay(backend.nworld, handoff=np.array(RELAY_HANDOFF), park=park)
  cube0, goal = np.array(scene.cfg.cube_pos), np.array(scene.cfg.goal_pos)

  viewer = Viewer(backend, args.headless, perturb=args.perturb)
  decimation = 10
  dt = scene.mjm.opt.timestep * decimation
  t, start = 0.0, time.time()
  while t < relay.duration:
    cmds = relay(t, cube0, backend.body_pos(scene.cube_body), goal, backend.body_quat(scene.cube_body)[0])
    for arm, (pos, closed, quat) in enumerate(cmds):
      oscs[arm].set_target(pos, quat_wxyz=quat)
      backend.set_gripper(scene.robots[arm].gripper_ctrl(closed), arm=arm)
    backend.control_substeps(oscs, decimation)  # both arms, one physics step at a time
    t += dt
    viewer.sync(dt)
  elapsed = time.time() - start
  viewer.close()

  err = np.linalg.norm(backend.body_pos(scene.cube_body) - goal, axis=-1)
  steps = int(round(relay.duration / scene.mjm.opt.timestep))
  print(f"[{backend.device}] nworld={backend.nworld} success={(err < 0.03).mean():.3f} mean_err={err.mean():.4f} m "
        f"| {steps * backend.nworld / elapsed:,.0f} env-steps/s ({elapsed:.1f}s wall)")


if __name__ == "__main__":
  main()
