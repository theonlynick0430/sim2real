"""Scripted OSC pick-and-place. Identical code on both backends.

  python examples/scripted_pick.py                       # CPU (MuJoCo C) + viewer (macOS: use `mjpython`)
  python examples/scripted_pick.py --headless
  python examples/scripted_pick.py --backend warp --nworld 1024 --headless   # MJWarp (GPU, or Warp-CPU)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without `pip install -e .`

import numpy as np

from s2r.control import OSC
from s2r.envs.scripted import TOP_DOWN_QUAT, ScriptedPickPlace
from s2r.robots import ActuatorMode, get_robot
from s2r.sim.backend import make_backend
from s2r.sim.scene import build_scene


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--backend", default="cpu", choices=["cpu", "warp"])
  ap.add_argument("--nworld", type=int, default=1)
  ap.add_argument("--device", default=None, help="warp device, e.g. cuda:0 or cpu")
  ap.add_argument("--headless", action="store_true")
  ap.add_argument("--decimation", type=int, default=10, help="physics steps per policy step")
  args = ap.parse_args()

  robot = get_robot("yam", actuator_mode=ActuatorMode.TORQUE)
  scene = build_scene(robot)
  kw = {"device": args.device} if args.backend == "warp" else {}
  backend = make_backend(args.backend, scene, nworld=args.nworld, **kw)

  osc = OSC(backend.view, backend.nworld, device=backend.device)
  osc.check_compatible(robot.actuator_mode)
  script = ScriptedPickPlace(backend.nworld)

  # Per-world randomized cube start, fixed goal (host-side; fine for a script).
  rng = np.random.default_rng(0)
  cube0 = np.tile(np.array(scene.cfg.cube_pos), (backend.nworld, 1))
  if backend.nworld > 1:
    cube0[:, :2] += rng.uniform(-0.05, 0.05, size=(backend.nworld, 2))
  goal = np.array(scene.cfg.goal_pos)
  if args.backend == "cpu":
    backend.set_body_free_pose("cube_free", cube0)
  else:
    backend.set_body_free_pose("cube_free", cube0[0])  # broadcast; per-world randomization lives in PickPlaceEnv
    cube0[:] = cube0[0]
  backend.set_mocap_pos(scene.goal_body, goal)

  viewer = None
  if not args.headless:
    import mujoco.viewer

    vis = backend.to_mjdata(0)
    viewer = mujoco.viewer.launch_passive(scene.mjm, vis)

  dt = scene.mjm.opt.timestep * args.decimation
  t, wall = 0.0, time.time()
  while t < script.duration:
    pos, closed = script(t, cube0, goal)
    osc.set_target(pos, quat_wxyz=TOP_DOWN_QUAT)
    backend.set_gripper(robot.gripper_closed if closed else robot.gripper_open)
    backend.control_substeps(osc, args.decimation)
    t += dt
    if viewer is not None:
      backend.to_mjdata(0, out=vis)
      viewer.sync()
      time.sleep(max(0.0, dt - (time.time() - wall) % dt))
  elapsed = time.time() - wall

  cube = backend.body_pos(scene.cube_body)
  err = np.linalg.norm(cube - goal, axis=-1)
  success = err < 0.03
  steps = int(round(script.duration / scene.mjm.opt.timestep))
  print(f"[{args.backend}] nworld={backend.nworld} success={success.mean():.3f} mean_err={err.mean():.4f} m "
        f"| {steps * backend.nworld / elapsed:,.0f} env-steps/s ({elapsed:.1f}s wall)")
  if viewer is not None:
    viewer.close()


if __name__ == "__main__":
  main()
