"""Throughput / memory sweep over nworld for the pick-place env (MJWarp, CUDA graph).

  python examples/gpu_bench.py --nworld 1024 8192 65536 262144 1048576
  python examples/gpu_bench.py --bimanual                    # two YAMs per world
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warp as wp

from examples._scenes import make_scene
from sim2real.envs.pick_place import PickPlaceEnv
from sim2real.sim.mjwarp_backend import MjWarpBackend


def bench(scene, nworld, device, steps):
  backend = MjWarpBackend(scene, nworld=nworld, device=device)
  env = PickPlaceEnv(backend)
  with wp.ScopedDevice(backend.device):
    action = wp.zeros((nworld, env.action_dim), dtype=float)
    env.reset()
    env.step(action)  # compile + capture
    wp.synchronize_device()
    t0 = time.time()
    for _ in range(steps):
      env.step(action)
    wp.synchronize_device()
    el = time.time() - t0
  mem = wp.get_mempool_used_mem_current(backend.device) if backend.device.is_cuda else 0
  return steps * nworld * env.cfg.decimation / el, mem


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--nworld", type=int, nargs="+", default=[1024, 8192, 65536])
  ap.add_argument("--device", default=None)
  ap.add_argument("--steps", type=int, default=50)
  ap.add_argument("--bimanual", action="store_true", help="two YAMs per world")
  args = ap.parse_args()
  scene = make_scene(args.bimanual)
  print(f"{'nworld':>10} {'physics steps/s':>18} {'mem [GB]':>10} {'KB/world':>10}")
  for n in args.nworld:
    try:
      sps, mem = bench(scene, n, args.device, args.steps)
      print(f"{n:>10} {sps:>18,.0f} {mem / 1e9:>10.2f} {mem / n / 1e3:>10.1f}")
    except Exception as e:  # OOM etc.
      print(f"{n:>10} failed: {type(e).__name__}: {e}")
      break


if __name__ == "__main__":
  main()
