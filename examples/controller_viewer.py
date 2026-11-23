"""Scripted motion under a selectable controller, on MJWarp (CUDA if available, else Warp-CPU).

  mjpython examples/controller_viewer.py --controller osc        # TCP traces a circle (macOS needs `mjpython`)
  mjpython examples/controller_viewer.py --controller impedance  # joint-space sinusoid, torque mode
  mjpython examples/controller_viewer.py --controller pd         # joint-space sinusoid, MuJoCo position actuators
  python   examples/controller_viewer.py --controller osc --headless --seconds 5
  python   examples/controller_viewer.py --controller osc --perturb   # ctrl-drag a link to push the arm
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sim2real.control import OSC, JointImpedance, JointPD
from sim2real.envs.scripted import TOP_DOWN_QUAT
from sim2real.robots import ActuatorMode, get_robot
from sim2real.sim.mjwarp_backend import MjWarpBackend
from sim2real.sim.scene import build_scene
from examples._scenes import Viewer


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--controller", default="osc", choices=["osc", "impedance", "pd"])
  parser.add_argument("--device", default=None, help="warp device, e.g. cuda:0 or cpu (default: cuda if available)")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--perturb", action="store_true", help="ctrl-drag in the viewer pushes the robot (world 0)")
  parser.add_argument("--seconds", type=float, default=30.0)
  args = parser.parse_args()

  mode = ActuatorMode.POSITION if args.controller == "pd" else ActuatorMode.TORQUE
  robot = get_robot("yam", actuator_mode=mode)
  scene = build_scene(robot)
  backend = MjWarpBackend(scene, nworld=1, device=args.device)
  view, device = backend.view, backend.device
  if args.controller == "osc":
    ctrl = OSC(view, 1, device=device)
  elif args.controller == "impedance":
    ctrl = JointImpedance(view, 1, stiffness=[60, 60, 60, 20, 10, 10], inertia_shaped=False, device=device)
  else:
    ctrl = JointPD(view, 1, mode=ActuatorMode.POSITION, device=device)
  ctrl.check_compatible(mode)

  viewer = Viewer(backend, args.headless, perturb=args.perturb)
  decimation = 10
  dt = decimation * scene.mjm.opt.timestep
  home = np.asarray(robot.home_qpos)
  t, errors = 0.0, []
  while t < args.seconds and viewer.running():
    if args.controller == "osc":
      target = np.array([0.32 + 0.06 * np.cos(t), 0.08 * np.sin(t), 0.15])
      ctrl.set_target(target, quat_wxyz=TOP_DOWN_QUAT)
    else:
      amp = 0.3 * np.array([1, 0.5, 0.5, 1, 1, 1])
      target = home + amp * np.sin(t)
      ctrl.set_target(target, dq_des=amp * np.cos(t))
    backend.control_substeps(ctrl, decimation)
    t += dt
    state = backend.robot_state()
    actual = state.tcp_pos.numpy()[0] if args.controller == "osc" else state.q.numpy()[0]
    errors.append(np.linalg.norm(actual - target))
    viewer.sync(dt)
  viewer.close()
  print(f"[{args.controller}] mean tracking error over last half: {np.mean(errors[len(errors) // 2:]):.4f}")


if __name__ == "__main__":
  main()
