"""MJWarp-derived RobotState (CSR->dense M, mjw.jac gather, bias) == MuJoCo C reference (mj_fullM, mj_jacSite).

MuJoCo C is used here only as an independent oracle for the gather kernels; it is not a simulation path.
Run for the single-arm and the bimanual scene, every arm.
"""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest

from sim2real.sim.mjwarp_backend import MjWarpBackend


def _reference(mjm, mjd, view):
  """Arm quantities straight from MuJoCo C, float64, after mj_forward."""
  full = np.zeros((mjm.nv, mjm.nv))
  mujoco.mj_fullM(mjm, mjd, full)
  jacp, jacr = np.zeros((3, mjm.nv)), np.zeros((3, mjm.nv))
  mujoco.mj_jacSite(mjm, mjd, jacp, jacr, view.tcp_site_id)
  dof = view.dof_adr
  return {
    "M": full[np.ix_(dof, dof)],
    "J": np.concatenate([jacp[:, dof], jacr[:, dof]]),
    "bias": mjd.qfrc_bias[dof],
    "tcp_pos": mjd.site_xpos[view.tcp_site_id],
    "tcp_rot": mjd.site_xmat[view.tcp_site_id].reshape(3, 3),
  }


@pytest.mark.parametrize("scene_name", ["yam_torque_scene", "yam_bimanual_scene"])
def test_state_parity(scene_name, warp_device, request):
  scene = request.getfixturevalue(scene_name)
  mjm = scene.mjm
  nworld = 4
  rng = np.random.default_rng(0)
  b = MjWarpBackend(scene, nworld=nworld, device=warp_device)
  qpos = b.d.qpos.numpy().astype(np.float64)
  qvel = np.zeros((nworld, mjm.nv))
  for v in b.views:
    lo, hi = mjm.jnt_range[[mjm.joint(j).id for j in v.cfg.arm_joints]].T
    qpos[:, v.qpos_adr] = rng.uniform(lo + 0.1, hi - 0.1, (nworld, v.ndof))
    qvel[:, v.dof_adr] = rng.normal(0, 1.0, (nworld, v.ndof))
  b.d.qpos.assign(qpos.astype(np.float32))
  b.d.qvel.assign(qvel.astype(np.float32))
  mjw.forward(b.m, b.d)

  mjd = mujoco.MjData(mjm)
  tol = {"M": (1e-3, 1e-5), "J": (1e-3, 1e-5), "bias": (1e-3, 1e-4), "tcp_pos": (0, 1e-5), "tcp_rot": (0, 1e-5)}
  for arm, v in enumerate(b.views):
    s = b.robot_state(arm=arm)
    got = {k: getattr(s, k).numpy() for k in tol}
    for w in range(nworld):
      mjd.qpos[:], mjd.qvel[:] = qpos[w], qvel[w]
      mujoco.mj_forward(mjm, mjd)
      ref = _reference(mjm, mjd, v)
      for k, (rtol, atol) in tol.items():
        np.testing.assert_allclose(got[k][w], ref[k], rtol=rtol, atol=atol, err_msg=f"arm {arm} world {w} {k}")
      assert np.all(np.linalg.eigvalsh(got["M"][w]) > 0)
