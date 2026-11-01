"""CPU (MuJoCo C) viewer with scripted motion under a selectable controller.

  mjpython examples/cpu_viewer.py --controller osc          # TCP traces a circle (macOS needs `mjpython`)
  mjpython examples/cpu_viewer.py --controller impedance    # joint-space sinusoid, torque mode
  mjpython examples/cpu_viewer.py --controller pd           # joint-space sinusoid, MuJoCo position actuators
  python   examples/cpu_viewer.py --controller osc --headless --seconds 5
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from s2r.control import OSC, JointImpedance, JointPD
from s2r.envs.scripted import TOP_DOWN_QUAT
from s2r.robots import ActuatorMode, get_robot
from s2r.sim.backend import make_backend
from s2r.sim.scene import build_scene


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--controller", default="osc", choices=["osc", "impedance", "pd"])
  ap.add_argument("--headless", action="store_true")
  ap.add_argument("--seconds", type=float, default=30.0)
  args = ap.parse_args()

  mode = ActuatorMode.POSITION if args.controller == "pd" else ActuatorMode.TORQUE
  robot = get_robot("yam", actuator_mode=mode)
  scene = build_scene(robot)
  backend = make_backend("cpu", scene)
  view = backend.view
  if args.controller == "osc":
    ctrl = OSC(view, 1, device="cpu")
  elif args.controller == "impedance":
    ctrl = JointImpedance(view, 1, stiffness=[60, 60, 60, 20, 10, 10], inertia_shaped=False, device="cpu")
  else:
    ctrl = JointPD(view, 1, mode=ActuatorMode.POSITION, device="cpu")
  ctrl.check_compatible(mode)

  viewer = None
  if not args.headless:
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(scene.mjm, backend.mjd)

  dt = scene.mjm.opt.timestep
  decim = 10
  home = np.asarray(robot.home_qpos)
  t, err_log = 0.0, []
  while t < args.seconds and (viewer is None or viewer.is_running()):
    tic = time.time()
    if args.controller == "osc":
      target = np.array([0.32 + 0.06 * np.cos(t), 0.08 * np.sin(t), 0.15])
      ctrl.set_target(target, quat_wxyz=TOP_DOWN_QUAT)
    else:
      amp = 0.3 * np.array([1, 0.5, 0.5, 1, 1, 1])
      target = home + amp * np.sin(t)
      ctrl.set_target(target, dq_des=amp * np.cos(t))
    backend.control_substeps(ctrl, decim)
    t += decim * dt
    st = backend.robot_state()
    cur = st.tcp_pos.numpy()[0] if args.controller == "osc" else st.q.numpy()[0]
    err_log.append(np.linalg.norm(cur - target))
    if viewer is not None:
      viewer.sync()
      time.sleep(max(0.0, decim * dt - (time.time() - tic)))
  print(f"[{args.controller}] mean tracking error over last half: {np.mean(err_log[len(err_log) // 2:]):.4f}")


if __name__ == "__main__":
  main()
