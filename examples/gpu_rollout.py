"""Batched pick-and-place on MJWarp: expert or random actions, reports success + throughput.

  python examples/gpu_rollout.py --nworld 4096                 # CUDA
  python examples/gpu_rollout.py --nworld 8 --device cpu       # Warp-CPU (Mac debugging, slow)
  python examples/gpu_rollout.py --nworld 4096 --policy random
  python examples/gpu_rollout.py --nworld 4096 --bimanual      # two YAMs, relay expert
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import warp as wp

from examples._expert import RELAY_STEPS, launch_expert
from examples._scenes import make_scene
from sim2real.envs.pick_place import PickPlaceConfig, PickPlaceEnv
from sim2real.sim.mjwarp_backend import MjWarpBackend


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--nworld", type=int, default=4096)
  parser.add_argument("--device", default=None)
  parser.add_argument("--policy", default="expert", choices=["expert", "random"])
  parser.add_argument("--steps", type=int, default=None, help="policy steps (default: one expert episode)")
  parser.add_argument("--bimanual", action="store_true", help="two YAMs (relay task) instead of one")
  args = parser.parse_args()
  episode = RELAY_STEPS + 40 if args.bimanual else 250
  steps = args.steps or episode

  backend = MjWarpBackend(make_scene(args.bimanual), nworld=args.nworld, device=args.device)
  env = PickPlaceEnv(backend, PickPlaceConfig(episode_steps=episode))
  device = backend.device
  print(f"device={device} nworld={args.nworld} arms={env.narm} obs={env.obs_dim} act={env.action_dim} "
        f"graph={backend.use_graph}")

  with wp.ScopedDevice(device):
    action = wp.zeros((args.nworld, env.action_dim), dtype=float)
    rng = np.random.default_rng(0)

    # Warm-up (kernel compilation + graph capture) excluded from timing.
    env.reset()
    env.step(action)
    env.reset()
    wp.synchronize_device(device)

    succeeded = np.zeros(args.nworld, bool)
    start = time.time()
    for _ in range(steps):
      if args.policy == "expert":
        launch_expert(env, action)
      else:
        action.assign(rng.uniform(-1, 1, (args.nworld, env.action_dim)).astype(np.float32))
      env.step(action)
      if args.policy == "expert":
        succeeded |= env.success.numpy().astype(bool)  # host sync; fine for an example
    wp.synchronize_device(device)
    elapsed = time.time() - start

  physics_steps = steps * env.cfg.decimation * args.nworld
  print(f"success={succeeded.mean():.3f}  policy-steps/s={steps * args.nworld / elapsed:,.0f}  "
        f"physics-steps/s={physics_steps / elapsed:,.0f}  ({elapsed:.2f}s)")
  print("buffers:", backend.overflow())


if __name__ == "__main__":
  main()
