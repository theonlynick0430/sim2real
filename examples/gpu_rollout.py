"""Batched pick-and-place on MJWarp: expert or random actions, reports success + throughput.

  python examples/gpu_rollout.py --nworld 4096                 # CUDA
  python examples/gpu_rollout.py --nworld 8 --device cpu       # Warp-CPU (Mac debugging, slow)
  python examples/gpu_rollout.py --nworld 4096 --policy random
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import warp as wp

from examples._expert import expert_kernel
from s2r.envs.pick_place import PickPlaceConfig, PickPlaceEnv
from s2r.robots import get_robot
from s2r.sim.mjwarp_backend import MjWarpBackend
from s2r.sim.scene import build_scene


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--nworld", type=int, default=4096)
  ap.add_argument("--device", default=None)
  ap.add_argument("--policy", default="expert", choices=["expert", "random"])
  ap.add_argument("--steps", type=int, default=250, help="policy steps")
  args = ap.parse_args()

  scene = build_scene(get_robot("yam"))
  backend = MjWarpBackend(scene, nworld=args.nworld, device=args.device)
  env = PickPlaceEnv(backend, PickPlaceConfig())
  dev = backend.device
  print(f"device={dev} nworld={args.nworld} graph={backend.use_graph}")

  with wp.ScopedDevice(dev):
    action = wp.zeros((args.nworld, 4), dtype=float)
    succ_any = wp.zeros(args.nworld, dtype=int)
    rng = np.random.default_rng(0)
    obs = env.reset()

    # Warm-up (kernel compilation + graph capture) excluded from timing.
    env.step(action)
    env.reset()
    wp.synchronize_device(dev)

    total_success = np.zeros(args.nworld, bool)
    t0 = time.time()
    for _ in range(args.steps):
      if args.policy == "expert":
        wp.launch(expert_kernel, dim=args.nworld, inputs=[env.obs, env.step_count, env.cfg.action_scale, 0.10],
                  outputs=[action])
      else:
        action.assign(rng.uniform(-1, 1, (args.nworld, 4)).astype(np.float32))
      obs, rew, done = env.step(action)
      if args.policy == "expert":
        total_success |= env.success.numpy().astype(bool)  # host sync; fine for an example
    wp.synchronize_device(dev)
    el = time.time() - t0

  phys = args.steps * env.cfg.decimation * args.nworld
  print(f"success={total_success.mean():.3f}  policy-steps/s={args.steps * args.nworld / el:,.0f}  "
        f"physics-steps/s={phys / el:,.0f}  ({el:.2f}s)")
  print("buffers:", backend.overflow())


if __name__ == "__main__":
  main()
