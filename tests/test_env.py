"""Batched env on MJWarp (CUDA, or Warp-CPU on Mac): masked reset + expert reaches the goal."""

import numpy as np
import warp as wp

from examples._expert import expert_kernel
from s2r.envs.pick_place import OBS_DIM, PickPlaceEnv
from s2r.sim.mjwarp_backend import MjWarpBackend


def test_expert_succeeds(yam_torque_scene, warp_device):
  nworld = 8
  b = MjWarpBackend(yam_torque_scene, nworld=nworld, device=warp_device)
  env = PickPlaceEnv(b)
  with wp.ScopedDevice(b.device):
    action = wp.zeros((nworld, 4), dtype=float)
    obs = env.reset()
    assert obs.shape == (nworld, OBS_DIM)
    cubes = obs.numpy()[:, 16:19]
    assert np.unique(cubes[:, 0]).size == nworld  # per-world randomization
    ok = np.zeros(nworld, bool)
    for _ in range(env.cfg.episode_steps):
      wp.launch(expert_kernel, dim=nworld, inputs=[env.obs, env.step_count, env.cfg.action_scale, 0.10],
                outputs=[action])
      env.step(action)
      ok |= env.success.numpy().astype(bool)
  assert ok.mean() >= 0.75, ok
  assert np.isfinite(env.obs.numpy()).all()
