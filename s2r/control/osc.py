"""Operational-space controller (Khatib), one fused Warp kernel per world.

  Lambda   = (J M^-1 J^T + eps I)^-1                       task-space inertia (damped)
  F        = Lambda [kp e - kd v],  v = J dq                 e = [p_des - p ; log(R_des R^T)]
  tau      = J^T F + N^T tau_0 + bias                        bias = C dq + g
  N^T      = I - J^T Jbar^T,  Jbar = M^-1 J^T Lambda          dynamically-consistent null space
  tau_0    = kp_null (q_null - q) - kd_null dq               posture task

J_dot dq is neglected (standard; negligible at manipulation speeds).
"""


import functools

import numpy as np
import warp as wp

from s2r.control.base import Controller
from s2r.control.linalg import linalg
from s2r.robots.config import ActuatorMode
from s2r.sim.state import ALL_NEEDS, RobotState


@wp.func
def orientation_error(R_des: wp.mat33, R: wp.mat33) -> wp.vec3:
  """World-frame rotation vector of R_des R^T (log map)."""
  q = wp.quat_from_matrix(R_des @ wp.transpose(R))
  v = wp.vec3(q[0], q[1], q[2])
  w = q[3]
  if w < 0.0:
    v = -v
    w = -w
  s = wp.length(v)
  if s < 1.0e-8:
    return 2.0 * v
  return v * (2.0 * wp.atan2(s, w) / s)


@functools.cache
def _osc_kernel(n: int):
  la = linalg(n)
  l6 = linalg(6)
  mat6n = wp.types.matrix(shape=(6, n), dtype=wp.float32)
  matn6 = wp.types.matrix(shape=(n, 6), dtype=wp.float32)

  @wp.kernel(module="unique")
  def kernel(
    q: wp.array2d(dtype=float),
    dq: wp.array2d(dtype=float),
    tcp_pos: wp.array(dtype=wp.vec3),
    tcp_rot: wp.array(dtype=wp.mat33),
    J_in: wp.array3d(dtype=float),
    M_in: wp.array3d(dtype=float),
    bias: wp.array2d(dtype=float),
    target_pos: wp.array(dtype=wp.vec3),
    target_rot: wp.array(dtype=wp.mat33),
    kp: wp.array(dtype=float),  # (6,) task stiffness [pos x3, rot x3]
    kd: wp.array(dtype=float),  # (6,)
    q_null: wp.array(dtype=float),
    kp_null: float,
    kd_null: float,
    lambda_damping: float,
    tau_limit: wp.array(dtype=float),
    cmd: wp.array2d(dtype=float),
  ):
    w = wp.tid()
    J = mat6n()
    for r in range(6):
      for c in range(n):
        J[r, c] = J_in[w, r, c]
    M = la.load_mat(M_in, w)
    qd = la.load_vec(dq, w)

    # M^-1 J^T, column by column, via one Cholesky of M.
    L = la.cholesky(M)
    MinvJt = matn6()
    for r in range(6):
      col = la.vec()
      for c in range(n):
        col[c] = J[r, c]
      x = la.cho_solve(L, col)
      for c in range(n):
        MinvJt[c, r] = x[c]

    Lambda_inv = J @ MinvJt
    for r in range(6):
      Lambda_inv[r, r] = Lambda_inv[r, r] + lambda_damping
    Lambda = l6.spd_inverse(Lambda_inv)

    # Task-space error and velocity.
    e_p = target_pos[w] - tcp_pos[w]
    e_r = orientation_error(target_rot[w], tcp_rot[w])
    v = J @ qd
    a = l6.vec()
    for i in range(3):
      a[i] = kp[i] * e_p[i] - kd[i] * v[i]
      a[i + 3] = kp[i + 3] * e_r[i] - kd[i + 3] * v[i + 3]
    F = Lambda @ a
    tau = wp.transpose(J) @ F

    # Null-space posture: tau_null = (I - J^T Jbar^T) tau_0
    tau0 = la.vec()
    for i in range(n):
      tau0[i] = kp_null * (q_null[i] - q[w, i]) - kd_null * dq[w, i]
    Jbar = MinvJt @ Lambda  # (n x 6)
    tau += tau0 - wp.transpose(J) @ (wp.transpose(Jbar) @ tau0)

    for i in range(n):
      cmd[w, i] = wp.clamp(tau[i] + bias[w, i], -tau_limit[i], tau_limit[i])

  return kernel


def quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
  import mujoco

  quat = np.asarray(quat, np.float64).reshape(-1, 4)
  out = np.zeros((quat.shape[0], 9))
  for i, qq in enumerate(quat):
    mujoco.mju_quat2Mat(out[i], qq)
  return out.reshape(-1, 3, 3)


class OSC(Controller):
  requires = ALL_NEEDS
  output = ActuatorMode.TORQUE

  def __init__(
    self,
    view,
    nworld: int,
    kp_pos: float = 400.0,
    kp_rot: float = 150.0,
    damping_ratio: float = 1.0,
    kp_null: float = 5.0,
    kd_null: float | None = None,
    lambda_damping: float = 1e-3,
    device=None,
  ):
    super().__init__(view, nworld, device)
    kp = np.array([kp_pos] * 3 + [kp_rot] * 3, np.float32)
    kd = 2.0 * damping_ratio * np.sqrt(kp)  # Lambda-weighted -> unit task-space mass
    self.kp_null = float(kp_null)
    self.kd_null = float(2.0 * np.sqrt(kp_null) if kd_null is None else kd_null)
    self.lambda_damping = float(lambda_damping)
    with wp.ScopedDevice(self.device):
      self.kp = wp.array(kp, dtype=float)
      self.kd = wp.array(kd.astype(np.float32), dtype=float)
      self.q_null = wp.array(view.home_qpos, dtype=float)
      self.target_pos = wp.zeros(nworld, dtype=wp.vec3)
      self.target_rot = wp.array(np.tile(np.eye(3, dtype=np.float32), (nworld, 1, 1)), dtype=wp.mat33)
    self._kernel = _osc_kernel(self.n)

  def set_target(self, pos, quat_wxyz=None, rot=None) -> None:
    """Host helper. pos (3,) or (nworld,3); orientation as MuJoCo wxyz quat or 3x3 matrix."""
    self.target_pos.assign(self._arr(pos, (self.nworld, 3)))
    if quat_wxyz is not None:
      rot = quat_wxyz_to_mat(quat_wxyz)
    if rot is not None:
      self.target_rot.assign(self._arr(rot, (self.nworld, 3, 3)))

  def hold_current(self, state: RobotState) -> None:
    wp.copy(self.target_pos, state.tcp_pos)
    wp.copy(self.target_rot, state.tcp_rot)

  def compute(self, state: RobotState) -> wp.array:
    wp.launch(
      self._kernel,
      dim=self.nworld,
      inputs=[
        state.q, state.dq, state.tcp_pos, state.tcp_rot, state.J, state.M, state.bias, self.target_pos,
        self.target_rot, self.kp, self.kd, self.q_null, self.kp_null, self.kd_null, self.lambda_damping,
        self.tau_limit,
      ],
      outputs=[self.cmd],
      device=self.device,
    )
    return self.cmd
