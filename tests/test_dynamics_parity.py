"""MJWarp-derived RobotState (CSR->dense M, mjw.jac gather, bias) == MuJoCo C (mj_fullM, mj_jacSite)."""

import mujoco
import mujoco_warp as mjw
import numpy as np

from s2r.sim.mjwarp_backend import MjWarpBackend
from s2r.sim.mujoco_backend import MujocoBackend


def test_state_parity(yam_torque_scene, warp_device):
  scene = yam_torque_scene
  nworld = 4
  rng = np.random.default_rng(0)
  gpu = MjWarpBackend(scene, nworld=nworld, device=warp_device)
  cpu = MujocoBackend(scene)
  v = cpu.view
  qpos = np.tile(cpu.mjd.qpos, (nworld, 1))
  qvel = np.zeros((nworld, scene.mjm.nv))
  lo, hi = scene.mjm.jnt_range[:6].T
  qpos[:, v.qpos_adr] = rng.uniform(lo + 0.1, hi - 0.1, (nworld, 6))
  qvel[:, v.dof_adr] = rng.normal(0, 1.0, (nworld, 6))
  gpu.d.qpos.assign(qpos.astype(np.float32))
  gpu.d.qvel.assign(qvel.astype(np.float32))
  mjw.forward(gpu.m, gpu.d)
  sg = gpu.robot_state()
  M, J, bias, pos = sg.M.numpy(), sg.J.numpy(), sg.bias.numpy(), sg.tcp_pos.numpy()

  for w in range(nworld):
    cpu.mjd.qpos[:] = qpos[w]
    cpu.mjd.qvel[:] = qvel[w]
    mujoco.mj_forward(cpu.mjm, cpu.mjd)
    sc = cpu.robot_state()
    np.testing.assert_allclose(M[w], sc.M.numpy()[0], rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(J[w], sc.J.numpy()[0], rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(bias[w], sc.bias.numpy()[0], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(pos[w], sc.tcp_pos.numpy()[0], atol=1e-5)
    assert np.all(np.linalg.eigvalsh(M[w]) > 0)
