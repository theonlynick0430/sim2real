"""GPU IK + motion planning demo on YAM with cuRobo v2 (collision checking off). Linux + CUDA only.

  python examples/plan_demo.py --headless              # IK benchmark + a few plans, prints stats
  python examples/plan_demo.py --headless --batch 32   # also batched planning of 32 problems
  python examples/plan_demo.py                         # + replay plans kinematically in the MuJoCo viewer

Standalone: it is not wired into the envs or controllers. Targets are FK of random joint samples (so they
are reachable); every result is re-checked with MuJoCo FK on the exported URDF.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sim2real.planning import MujocoFK, rotation_error
from sim2real.planning.adapters import planner_config_from_robot
from sim2real.robots import get_robot


def ik_benchmark(cfg, fk, n, rng):
  from sim2real.planning import GpuIK

  ik = GpuIK(cfg.replace(max_batch_size=n))
  q = fk.sample_q(n, rng, margin=0.1)
  pos, quat = fk(q)
  ik.solve(pos, quat)  # first call captures CUDA graphs
  tic = time.perf_counter()
  res = ik.solve(pos, quat).numpy()
  dt = time.perf_counter() - tic
  p, r = fk(res.q[res.success])
  print(f"[ik] {n} targets in {dt * 1e3:.1f} ms | success {res.success.mean():.3f} | "
        f"pos err max {np.linalg.norm(p - pos[res.success], axis=-1).max() * 1e3:.2f} mm | "
        f"rot err max {np.degrees(rotation_error(r, quat[res.success]).max()):.2f} deg")


def report(name, traj, fk, pos=None, quat=None):
  t = traj.numpy()
  q_end = t.q[np.arange(len(t.q)), t.length - 1]
  msg = (f"[{name}] {len(t.q)} problems in {t.solve_time:.3f} s | success {t.success.mean():.3f} | "
         f"duration mean {np.mean((t.length - 1) * t.dt):.2f} s")
  if pos is not None:
    p, r = fk(q_end)
    msg += (f" | end err max {np.linalg.norm(p - pos, axis=-1).max() * 1e3:.2f} mm, "
            f"{np.degrees(rotation_error(r, quat).max()):.2f} deg")
  print(msg)
  return t


def replay(robot, trajs):
  """Kinematic playback (set qpos + mj_forward) of each trajectory in the passive viewer."""
  import mujoco
  import mujoco.viewer

  from examples._scenes import close_viewer
  from sim2real.sim.scene import build_scene

  scene = build_scene(robot)
  m = scene.mjm
  d = mujoco.MjData(m)
  qadr = [m.joint(j).qposadr[0] for j in robot.arm_joints]
  viewer = mujoco.viewer.launch_passive(m, d)
  for q, dt in trajs:
    for qt in q:
      if not viewer.is_running():
        break
      d.qpos[qadr] = qt
      mujoco.mj_forward(m, d)
      viewer.sync()
      time.sleep(dt)
    time.sleep(0.5)
  close_viewer(viewer)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--n-ik", type=int, default=1024, help="batch size for the IK benchmark")
  ap.add_argument("--n-plan", type=int, default=4, help="sequential plan_pose / plan_joint problems")
  ap.add_argument("--batch", type=int, default=0, help="also run BatchMotionPlanner on this many problems")
  ap.add_argument("--headless", action="store_true")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  import torch

  if not torch.cuda.is_available():
    sys.exit("cuRobo needs a CUDA GPU (Linux). Install with: uv pip install -e '.[planning]'")
  from sim2real.planning import GpuMotionPlanner

  rng = np.random.default_rng(args.seed)
  robot = get_robot("yam")
  workdir = Path(tempfile.mkdtemp(prefix="plan_demo_"))
  cfg = planner_config_from_robot(robot, workdir)
  fk = MujocoFK(cfg.urdf_path, cfg.joint_names, cfg.tool_frame)
  print(f"URDF: {cfg.urdf_path} | base {cfg.base_link} -> tool {cfg.tool_frame} | dof {cfg.dof}")

  ik_benchmark(cfg, fk, args.n_ik, rng)

  tic = time.perf_counter()
  mp = GpuMotionPlanner(cfg)
  mp.warmup()
  print(f"[plan] planner ready in {time.perf_counter() - tic:.1f} s")
  q0 = np.repeat(np.asarray(robot.home_qpos)[None], args.n_plan, 0)
  pos, quat = fk(fk.sample_q(args.n_plan, rng, margin=0.1))
  t_pose = report("plan_pose", mp.plan_pose(q0, pos, quat), fk, pos, quat)
  q_goal = fk.sample_q(args.n_plan, rng, margin=0.1)
  t_joint = report("plan_joint", mp.plan_joint(q0, q_goal), fk)
  print(f"[plan_joint] end joint err max {np.abs(t_joint.q[np.arange(args.n_plan), t_joint.length - 1] - q_goal).max():.4f} rad")

  if args.batch:
    bmp = GpuMotionPlanner(cfg.replace(max_batch_size=args.batch), batch=True)
    bmp.warmup()
    q0b = fk.sample_q(args.batch, rng, margin=0.1)
    posb, quatb = fk(fk.sample_q(args.batch, rng, margin=0.1))
    report("batch plan_pose", bmp.plan_pose(q0b, posb, quatb), fk, posb, quatb)

  if not args.headless:
    trajs = [(t.q[b, :t.length[b]], t.dt) for t in (t_pose, t_joint) for b in range(len(t.q)) if t.success[b]]
    replay(robot, trajs)


if __name__ == "__main__":
  main()
