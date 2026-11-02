"""Bimanual YAM relay with OSC on both arms: left picks the cube and sets it down at the
handoff spot, right picks it up there and places it at the goal. Identical code on both backends.

  mjpython examples/bimanual_relay.py                      # CPU (MuJoCo C) + viewer (macOS: `mjpython`)
  python examples/bimanual_relay.py --headless
  python examples/bimanual_relay.py --backend warp --nworld 1024 --headless   # MJWarp (GPU, or Warp-CPU)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without `pip install -e .`

import numpy as np

from sim2real.control import OSC
from sim2real.envs.scripted import RELAY_CUBE_POS, RELAY_GOAL_POS, RELAY_HANDOFF, BimanualRelay
from sim2real.robots import yam_bimanual
from sim2real.sim.backend import make_backend
from sim2real.sim.scene import SceneConfig, build_scene


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--backend", default="cpu", choices=["cpu", "warp"])
  ap.add_argument("--nworld", type=int, default=1)
  ap.add_argument("--device", default=None, help="warp device, e.g. cuda:0 or cpu")
  ap.add_argument("--separation", type=float, default=0.4, help="distance between the arm bases [m]")
  ap.add_argument("--headless", action="store_true")
  args = ap.parse_args()

  scene = build_scene(yam_bimanual(args.separation), SceneConfig(cube_pos=RELAY_CUBE_POS, goal_pos=RELAY_GOAL_POS))
  kw = {"device": args.device} if args.backend == "warp" else {}
  backend = make_backend(args.backend, scene, nworld=args.nworld, **kw)

  oscs = [OSC(v, backend.nworld, device=backend.device) for v in backend.views]
  for osc, robot in zip(oscs, scene.robots):
    osc.check_compatible(robot.actuator_mode)
  backend.step1()
  park = [backend.robot_state(arm=a).tcp_pos.numpy()[0].copy() for a in range(backend.narm)]
  relay = BimanualRelay(backend.nworld, handoff=np.array(RELAY_HANDOFF), park=park)
  cube0, goal = np.array(scene.cfg.cube_pos), np.array(scene.cfg.goal_pos)

  viewer = None
  if not args.headless:
    import mujoco.viewer

    vis = backend.to_mjdata(0)
    viewer = mujoco.viewer.launch_passive(scene.mjm, vis)

  decimation = 10
  dt = scene.mjm.opt.timestep * decimation
  t, wall = 0.0, time.time()
  while t < relay.duration:
    cmds = relay(t, cube0, backend.body_pos(scene.cube_body), goal, backend.body_quat(scene.cube_body)[0])
    for arm, (pos, closed, quat) in enumerate(cmds):
      oscs[arm].set_target(pos, quat_wxyz=quat)
      r = scene.robots[arm]
      backend.set_gripper(r.gripper_closed if closed else r.gripper_open, arm=arm)
    backend.control_substeps(oscs, decimation)  # both arms, one physics step at a time
    t += dt
    if viewer is not None:
      backend.to_mjdata(0, out=vis)
      viewer.sync()
      time.sleep(max(0.0, dt - (time.time() - wall) % dt))
  elapsed = time.time() - wall

  err = np.linalg.norm(backend.body_pos(scene.cube_body) - goal, axis=-1)
  steps = int(round(relay.duration / scene.mjm.opt.timestep))
  print(f"[{args.backend}] nworld={backend.nworld} success={(err < 0.03).mean():.3f} mean_err={err.mean():.4f} m "
        f"| {steps * backend.nworld / elapsed:,.0f} env-steps/s ({elapsed:.1f}s wall)")
  if viewer is not None:
    viewer.close()


if __name__ == "__main__":
  main()
