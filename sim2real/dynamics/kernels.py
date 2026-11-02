"""Warp kernels that extract robot-level dynamics from MJWarp `Data`.

MJWarp provides (validated against mujoco_warp 3.17):
  * d.M         CSR lower-triangular mass matrix, (nworld, nC)  -> expanded here (no mj_fullM in MJWarp)
  * d.qfrc_bias C(q,dq)dq + g(q), (nworld, nv)                  -> gathered here
  * mjw.jac     host-level batched Jacobian launch, (nworld, 3, nv) -> called by backend, gathered here
  * d.site_xpos / d.site_xmat                                     -> gathered here
Everything else a controller needs (Lambda, pseudo-inverses, null-space) is
computed inside the controller kernels.
"""

import warp as wp


@wp.kernel
def gather_state_kernel(
  # model-level index maps
  qpos_adr: wp.array(dtype=int),
  dof_adr: wp.array(dtype=int),
  site_id: int,
  # mjw.Data
  qpos: wp.array2d(dtype=float),
  qvel: wp.array2d(dtype=float),
  qfrc_bias: wp.array2d(dtype=float),
  site_xpos: wp.array2d(dtype=wp.vec3),
  site_xmat: wp.array2d(dtype=wp.mat33),
  # outputs
  q_out: wp.array2d(dtype=float),
  dq_out: wp.array2d(dtype=float),
  bias_out: wp.array2d(dtype=float),
  tcp_pos_out: wp.array(dtype=wp.vec3),
  tcp_rot_out: wp.array(dtype=wp.mat33),
):
  w, i = wp.tid()
  q_out[w, i] = qpos[w, qpos_adr[i]]
  dq_out[w, i] = qvel[w, dof_adr[i]]
  bias_out[w, i] = qfrc_bias[w, dof_adr[i]]
  if i == 0:
    tcp_pos_out[w] = site_xpos[w, site_id]
    tcp_rot_out[w] = site_xmat[w, site_id]


@wp.kernel
def gather_mass_kernel(
  m_csr_index: wp.array2d(dtype=int),
  M_csr: wp.array2d(dtype=float),
  M_out: wp.array3d(dtype=float),
):
  w, i, j = wp.tid()
  idx = m_csr_index[i, j]
  val = float(0.0)
  if idx >= 0:
    val = M_csr[w, idx]
  M_out[w, i, j] = val


@wp.kernel
def gather_jac_kernel(
  dof_adr: wp.array(dtype=int),
  jacp: wp.array3d(dtype=float),
  jacr: wp.array3d(dtype=float),
  J_out: wp.array3d(dtype=float),
):
  w, i = wp.tid()
  d = dof_adr[i]
  for r in range(3):
    J_out[w, r, i] = jacp[w, r, d]
    J_out[w, r + 3, i] = jacr[w, r, d]


@wp.kernel
def tcp_point_kernel(site_id: int, site_xpos: wp.array2d(dtype=wp.vec3), point_out: wp.array(dtype=wp.vec3)):
  w = wp.tid()
  point_out[w] = site_xpos[w, site_id]


@wp.kernel
def scatter_ctrl_kernel(act_ids: wp.array(dtype=int), cmd: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float)):
  w, i = wp.tid()
  ctrl[w, act_ids[i]] = cmd[w, i]


@wp.kernel
def set_ctrl_column_kernel(act_id: int, value: wp.array(dtype=float), ctrl: wp.array2d(dtype=float)):
  w = wp.tid()
  ctrl[w, act_id] = value[w]
