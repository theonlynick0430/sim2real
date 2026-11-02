"""MJWarp-derived RobotState (CSR->dense M, mjw.jac gather, bias) == MuJoCo C (mj_fullM, mj_jacSite).

Run for the single-arm and the bimanual scene, every arm.
"""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest

from sim2real.sim.mjwarp_backend import MjWarpBackend
from sim2real.sim.mujoco_backend import MujocoBackend


@pytest.mark.parametrize("scene_name", ["yam_torque_scene", "yam_bimanual_scene"])
def test_state_parity(scene_name, warp_device, request):
  scene = request.getfixturevalue(scene_name)
  nworld = 4
  rng = np.random.default_rng(0)
  gpu = MjWarpBackend(scene, nworld=nworld, device=warp_device)
  cpu = MujocoBackend(scene)
  qpos = np.tile(cpu.mjd.qpos, (nworld, 1))
  qvel = np.zeros((nworld, scene.mjm.nv))
  for v in cpu.views:
    lo, hi = scene.mjm.jnt_range[[scene.mjm.joint(j).id for j in v.cfg.arm_joints]].T
    qpos[:, v.qpos_adr] = rng.uniform(lo + 0.1, hi - 0.1, (nworld, v.ndof))
    qvel[:, v.dof_adr] = rng.normal(0, 1.0, (nworld, v.ndof))
  gpu.d.qpos.assign(qpos.astype(np.float32))
  gpu.d.qvel.assign(qvel.astype(np.float32))
  mjw.forward(gpu.m, gpu.d)

  for arm in range(cpu.narm):
    sg = gpu.robot_state(arm=arm)
    M, J, bias, pos = sg.M.numpy(), sg.J.numpy(), sg.bias.numpy(), sg.tcp_pos.numpy()
    for w in range(nworld):
      cpu.mjd.qpos[:] = qpos[w]
      cpu.mjd.qvel[:] = qvel[w]
      mujoco.mj_forward(cpu.mjm, cpu.mjd)
      sc = cpu.robot_state(arm=arm)
      np.testing.assert_allclose(M[w], sc.M.numpy()[0], rtol=1e-3, atol=1e-5)
      np.testing.assert_allclose(J[w], sc.J.numpy()[0], rtol=1e-3, atol=1e-5)
      np.testing.assert_allclose(bias[w], sc.bias.numpy()[0], rtol=1e-3, atol=1e-4)
      np.testing.assert_allclose(pos[w], sc.tcp_pos.numpy()[0], atol=1e-5)
      assert np.all(np.linalg.eigvalsh(M[w]) > 0)
