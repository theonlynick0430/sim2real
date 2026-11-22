"""Scripted OSC pick-and-place on MJWarp (CUDA if available, else Warp-CPU). The viewer shows world 0.

  mjpython examples/scripted_pick.py                     # viewer (macOS needs `mjpython`)
  python examples/scripted_pick.py --headless
  python examples/scripted_pick.py --nworld 1024 --headless --device cuda:0
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without `pip install -e .`

import numpy as np

from sim2real.control import OSC
from sim2real.envs.scripted import TOP_DOWN_QUAT, ScriptedPickPlace
from sim2real.robots import ActuatorMode, get_robot
from sim2real.sim.mjwarp_backend import MjWarpBackend
from sim2real.sim.scene import build_scene
from examples._scenes import close_viewer


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--nworld", type=int, default=1)
  ap.add_argument("--device", default=None, help="warp device, e.g. cuda:0 or cpu (default: cuda if available)")
  ap.add_argument("--headless", action="store_true")
  ap.add_argument("--decimation", type=int, default=10, help="physics steps per policy step")
  args = ap.parse_args()

  robot = get_robot("yam", actuator_mode=ActuatorMode.TORQUE)
  scene = build_scene(robot)
  backend = MjWarpBackend(scene, nworld=args.nworld, device=args.device)

  osc = OSC(backend.view, backend.nworld, device=backend.device)
  osc.check_compatible(robot.actuator_mode)
  script = ScriptedPickPlace(backend.nworld)

  # Same cube start in every world (per-world randomization lives in PickPlaceEnv).
  cube0 = np.tile(np.array(scene.cfg.cube_pos), (backend.nworld, 1))
  goal = np.array(scene.cfg.goal_pos)
  backend.set_body_free_pose("cube_free", cube0[0])
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
  print(f"[{backend.device}] nworld={backend.nworld} success={success.mean():.3f} mean_err={err.mean():.4f} m "
        f"| {steps * backend.nworld / elapsed:,.0f} env-steps/s ({elapsed:.1f}s wall)")
  if viewer is not None:
    close_viewer(viewer)


if __name__ == "__main__":
  main()
