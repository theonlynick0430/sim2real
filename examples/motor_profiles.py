"""Reference motor profiles for dyno experiments, from simulated YAM trajectories. Linux + CUDA only (the
plans come from the cuRobo-based motion planner).

  python examples/motor_profiles.py --out data/profiles --num-trajs 200            # generate a dataset
  python examples/motor_profiles.py --out data/profiles --num-trajs 200 --resume   # continue a stopped run
  python examples/motor_profiles.py --visualize --no-record --num-trajs 5          # just watch

The simulated motors are ideal: reflected rotor inertia (armature), no joint friction.

Each trajectory starts at a sampled configuration and chains a few skills with short pauses between them:
  joint_move  large move to a sampled joint goal, tracked by computed torque (inverse dynamics).
  ee_move     small move (position + orientation) to a nearby tool pose, tracked by the OSC.
The tool carries a payload whose mass and center of mass (in the tool frame) are sampled per trajectory and may
change before a skill. `--joint-weight` / `--ee-weight` set how often each skill is drawn; when no ee move is
possible from the current configuration, a joint move is used instead (marked `fallback_from` in the metadata).

Output (see sim2real/profiles/session.py): `<out>/traj_XXXXX.npz` with, per physics step and arm joint,
`q` [rad], `dq` [rad/s], `tau_motor` [Nm] (motor torque at the joint output) and `tau_ext` [Nm] (torque the
rest of the arm and the payload put on the motor's output; rotor inertia excluded), plus
`t`, `skill` (index into the skill list, -1 while pausing) and `meta` (JSON: the skill composition, payloads,
motor per joint). `<out>/session.json` lists every trajectory and is written whether the run finishes, fails
or is interrupted.

`--visualize` shows the rollouts in the MuJoCo viewer, in real time. The orange sphere on the tool is centered
on the payload's center of mass and its opacity is mass / `--payload-mass-max`.
"""

import argparse
import dataclasses
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
  from sim2real.profiles.sampling import SamplingConfig

  d = SamplingConfig()
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--num-trajs", type=int, default=100, help="trajectories in the dataset (indices 0..N-1)")
  p.add_argument("--joint-weight", type=float, default=d.joint_weight, help="relative weight of joint_move")
  p.add_argument("--ee-weight", type=float, default=d.ee_weight, help="relative weight of ee_move")
  p.add_argument("--out", type=Path, default=None, help="output directory (required unless --no-record)")
  p.add_argument("--resume", action="store_true", help="continue the dataset in --out, skipping finished files")
  p.add_argument("--visualize", action="store_true", help="show the rollouts in the MuJoCo viewer (real time)")
  p.add_argument("--no-record", action="store_true", help="generate without writing any file")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--min-skills", type=int, default=d.min_skills)
  p.add_argument("--max-skills", type=int, default=d.max_skills)
  p.add_argument("--payload-mass-max", type=float, default=d.payload_mass_max, help="kg")
  p.add_argument("--payload-com-range", type=float, nargs=3, default=d.payload_com_range, metavar=("X", "Y", "Z"),
                 help="payload center of mass is sampled within +- this box [m] in the tool frame")
  p.add_argument("--empty-prob", type=float, default=d.empty_prob, help="probability of carrying nothing")
  p.add_argument("--payload-change-prob", type=float, default=d.payload_change_prob,
                 help="probability that the payload changes before a skill")
  p.add_argument("--limit-margin", type=float, default=d.limit_margin,
                 help="fraction of each joint range kept free at both limits")
  p.add_argument("--max-vel", type=float, default=2.0, help="planner joint velocity limit [rad/s]")
  p.add_argument("--max-attempts", type=int, default=5, help="attempts per trajectory before it is skipped")
  p.add_argument("--device", default=None, help="warp device for the simulation (default: cuda:0)")
  args = p.parse_args()
  if args.out is None and not args.no_record:
    p.error("--out is required unless --no-record is given")
  if args.resume and args.no_record:
    p.error("--resume needs recording")
  return args


def main():
  args = parse_args()
  try:
    import torch

    import curobo  # noqa: F401
  except ImportError:
    sys.exit("motor_profiles needs the planner: uv pip install -e '.[planning]' (Linux + NVIDIA GPU)")
  if not torch.cuda.is_available():
    sys.exit("motor_profiles needs a CUDA GPU (the motion planner runs on cuRobo)")

  from examples._scenes import Viewer
  from sim2real.planning import GpuMotionPlanner, MujocoFK
  from sim2real.planning.adapters import planner_config_from_robot
  from sim2real.profiles.sampling import SamplingConfig
  from sim2real.profiles.scene import build_profile_scene
  from sim2real.profiles.session import Session
  from sim2real.profiles.skills import ProfileGenerator, Rejected, RolloutConfig
  from sim2real.robots import get_robot
  from sim2real.sim.mjwarp_backend import MjWarpBackend

  robot = get_robot("yam")
  sampling = SamplingConfig(
    joint_weight=args.joint_weight, ee_weight=args.ee_weight, min_skills=args.min_skills,
    max_skills=args.max_skills, limit_margin=args.limit_margin, payload_mass_max=args.payload_mass_max,
    payload_com_range=tuple(args.payload_com_range), empty_prob=args.empty_prob,
    payload_change_prob=args.payload_change_prob)
  rollout = RolloutConfig()
  config = {"robot": robot.name, "seed": args.seed, "max_vel": args.max_vel,
            "sampling": dataclasses.asdict(sampling), "rollout": dataclasses.asdict(rollout)}
  try:
    session = None if args.no_record else Session(args.out, config, resume=args.resume)
  except (FileExistsError, FileNotFoundError, ValueError) as problem:
    sys.exit(str(problem))

  status, error, viewer = "failed", None, None
  try:
    scene = build_profile_scene(robot)
    # One world, stepped from the host so every step can be recorded; nothing to gain from graph capture.
    backend = MjWarpBackend(scene, nworld=1, device=args.device or "cuda:0", use_graph=False)
    workdir = Path(tempfile.mkdtemp(prefix="motor_profiles_"))
    planner_cfg = planner_config_from_robot(robot, workdir, velocity=args.max_vel)
    fk = MujocoFK(planner_cfg.urdf_path, planner_cfg.joint_names, planner_cfg.tool_frame)
    planner = GpuMotionPlanner(planner_cfg)
    planner.warmup()
    viewer = Viewer(backend, headless=not args.visualize)
    generator = ProfileGenerator(backend, planner, fk, sampling, rollout, seed=args.seed,
                                 viewer=viewer if args.visualize else None)
    static_meta = {
      "robot": robot.name, "joint_names": list(robot.arm_joints), "motor_models": list(robot.motor_models),
      "torque_limits": list(robot.torque_limits), "seed": args.seed,
      "armature": backend.mjm.dof_armature[backend.view.dof_adr].tolist(),
      "units": {"t": "s", "q": "rad", "dq": "rad/s", "tau_motor": "Nm", "tau_ext": "Nm"},
    }

    tic, made = time.perf_counter(), 0
    for index in range(args.num_trajs):
      if session is not None and session.done(index):
        continue
      rejects = []
      for attempt in range(args.max_attempts):
        try:
          arrays, meta = generator.generate(index, attempt)
        except Rejected as reason:
          rejects.append(f"attempt {attempt}: {reason}")
          continue
        made += 1
        if session is not None:
          session.add(index, arrays, {**static_meta, **meta}, rejects)
        kinds = " ".join(s["type"].split("_")[0] for s in meta["skills"])
        print(f"[{index + 1}/{args.num_trajs}] {meta['num_samples'] * meta['dt']:5.1f} s | {kinds} | "
              f"attempt {attempt} | {time.perf_counter() - tic:6.0f} s elapsed", flush=True)
        break
      else:
        print(f"[{index + 1}/{args.num_trajs}] skipped after {args.max_attempts} attempts: {rejects[-1]}", flush=True)
        if session is not None:
          session.fail(index, rejects)
    status = "complete"
    print(f"generated {made} trajectories" + (f" in {args.out}" if session is not None else " (not recorded)"))
  except KeyboardInterrupt:
    status = "interrupted"
    print("interrupted" + ("; continue with --resume" if session is not None else ""))
  except BaseException as exc:
    error = exc
    raise
  finally:
    if session is not None:
      session.close(status, error)
      print(f"session: {session.path} ({status})")
    if viewer is not None:
      viewer.close()


if __name__ == "__main__":
  main()
