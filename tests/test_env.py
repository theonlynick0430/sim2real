"""Batched env on MJWarp (CUDA, or Warp-CPU on Mac): masked reset + experts reach the goal."""

import numpy as np
import warp as wp

from examples._expert import RELAY_STEPS, launch_expert
from sim2real.envs.pick_place import PickPlaceConfig, PickPlaceEnv
from sim2real.sim.mjwarp_backend import MjWarpBackend


def _run_expert(scene, device, nworld, episode_steps):
  b = MjWarpBackend(scene, nworld=nworld, device=device)
  env = PickPlaceEnv(b, PickPlaceConfig(episode_steps=episode_steps))
  with wp.ScopedDevice(b.device):
    action = wp.zeros((nworld, env.action_dim), dtype=float)
    obs = env.reset()
    assert obs.shape == (nworld, env.obs_dim)
    cubes = obs.numpy()[:, env.layout.cube : env.layout.cube + 3]
    assert np.unique(cubes[:, 0]).size == nworld  # per-world randomization
    ok = np.zeros(nworld, bool)
    for _ in range(episode_steps):
      launch_expert(env, action)
      env.step(action)
      ok |= env.success.numpy().astype(bool)
  assert np.isfinite(env.obs.numpy()).all()
  return env, ok


def test_expert_succeeds(yam_torque_scene, warp_device):
  env, ok = _run_expert(yam_torque_scene, warp_device, nworld=8, episode_steps=250)
  assert env.obs_dim == 32 and env.action_dim == 4
  assert ok.mean() >= 0.75, ok


def test_bimanual_relay_expert_succeeds(yam_bimanual_scene, warp_device):
  env, ok = _run_expert(yam_bimanual_scene, warp_device, nworld=4, episode_steps=RELAY_STEPS + 40)
  assert env.narm == 2 and env.action_dim == 8 and env.obs_dim == 2 * 19 + 13
  assert ok.mean() >= 0.75, ok
