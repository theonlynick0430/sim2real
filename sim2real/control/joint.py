"""Joint-space controllers: PD and impedance.

JointPD
  POSITION actuators: cmd = q_des (MuJoCo's implicit position actuator / motor-side PD does the PD).
  TORQUE actuators:   tau = kp (q_des - q) + kd (dq_des - dq) [+ g(q), C(q,dq)dq if compensate_bias].
  Velocity control is JointPD with kp = 0 and a dq_des target.

JointImpedance (TORQUE)
  tau = K (q_des - q) + D (dq_des - dq) + bias            (inertia_shaped=False)
  tau = M [ddq_des + kp (q_des - q) + kd (dq_des - dq)] + bias   (inertia_shaped=True, computed-torque;
                                                                   ddq_des = 0 by default)
"""


import functools

import numpy as np
import warp as wp

from sim2real.control.base import Controller
from sim2real.control.linalg import linalg
from sim2real.robots.config import ActuatorMode
from sim2real.sim.state import NEED_BIAS, NEED_M, RobotState
from sim2real.utils import broadcast


@wp.kernel
def _passthrough_kernel(q_des: wp.array2d(dtype=float), cmd: wp.array2d(dtype=float)):
  w, i = wp.tid()
  cmd[w, i] = q_des[w, i]


@wp.kernel
def _pd_torque_kernel(
  q: wp.array2d(dtype=float),
  dq: wp.array2d(dtype=float),
  bias: wp.array2d(dtype=float),
  q_des: wp.array2d(dtype=float),
  dq_des: wp.array2d(dtype=float),
  kp: wp.array(dtype=float),
  kd: wp.array(dtype=float),
  use_bias: int,
  tau_limit: wp.array(dtype=float),
  cmd: wp.array2d(dtype=float),
):
  w, i = wp.tid()
  tau = kp[i] * (q_des[w, i] - q[w, i]) + kd[i] * (dq_des[w, i] - dq[w, i])
  if use_bias != 0:
    tau += bias[w, i]
  cmd[w, i] = wp.clamp(tau, -tau_limit[i], tau_limit[i])


@functools.cache
def _computed_torque_kernel(n: int):
  ops = linalg(n)

  @wp.kernel(module="unique")
  def kernel(
    q: wp.array2d(dtype=float),
    dq: wp.array2d(dtype=float),
    M: wp.array3d(dtype=float),
    bias: wp.array2d(dtype=float),
    q_des: wp.array2d(dtype=float),
    dq_des: wp.array2d(dtype=float),
    ddq_des: wp.array2d(dtype=float),
    kp: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    tau_limit: wp.array(dtype=float),
    cmd: wp.array2d(dtype=float),
  ):
    w = wp.tid()
    acc = ops.vec()
    for i in range(n):
      acc[i] = ddq_des[w, i] + kp[i] * (q_des[w, i] - q[w, i]) + kd[i] * (dq_des[w, i] - dq[w, i])
    tau = ops.load_mat(M, w) @ acc
    for i in range(n):
      cmd[w, i] = wp.clamp(tau[i] + bias[w, i], -tau_limit[i], tau_limit[i])

  return kernel


class _JointTargets(Controller):
  def __init__(self, view, nworld, kp, kd, device=None):
    super().__init__(view, nworld, device)
    with wp.ScopedDevice(self.device):
      self.q_des = wp.array(broadcast(view.home_qpos, (nworld, self.n)), dtype=float)
      self.dq_des = wp.zeros((nworld, self.n), dtype=float)
      self.kp = wp.array(broadcast(kp, self.n), dtype=float)
      self.kd = wp.array(broadcast(kd, self.n), dtype=float)

  def set_target(self, q_des, dq_des=None) -> None:
    self.q_des.assign(broadcast(q_des, (self.nworld, self.n)))
    self.dq_des.assign(broadcast(0.0 if dq_des is None else dq_des, (self.nworld, self.n)))


class JointPD(_JointTargets):
  def __init__(self, view, nworld, mode: ActuatorMode, kp=None, kd=None, compensate_bias=False, device=None):
    cfg = view.cfg
    super().__init__(view, nworld, cfg.kp if kp is None else kp, cfg.kd if kd is None else kd, device)
    self.output = mode
    self.compensate_bias = compensate_bias
    self.requires = frozenset({NEED_BIAS}) if (compensate_bias and mode == ActuatorMode.TORQUE) else frozenset()

  def compute(self, state: RobotState) -> wp.array:
    if self.output == ActuatorMode.POSITION:
      wp.launch(_passthrough_kernel, dim=(self.nworld, self.n), inputs=[self.q_des], outputs=[self.cmd],
                device=self.device)
    else:
      wp.launch(
        _pd_torque_kernel,
        dim=(self.nworld, self.n),
        inputs=[state.q, state.dq, state.bias, self.q_des, self.dq_des, self.kp, self.kd, int(self.compensate_bias),
                self.tau_limit],
        outputs=[self.cmd],
        device=self.device,
      )
    return self.cmd


class JointImpedance(_JointTargets):
  output = ActuatorMode.TORQUE

  def __init__(self, view, nworld, stiffness, damping=None, inertia_shaped=False, device=None):
    stiffness = broadcast(stiffness, view.ndof)
    if damping is None:
      if inertia_shaped:  # unit task inertia -> critical damping
        damping = 2.0 * np.sqrt(stiffness)
      else:  # keep the damping ratio implied by the robot's configured (kp, kd)
        cfg = view.cfg
        if not (cfg.kp and cfg.kd):
          raise ValueError("damping required: robot config has no default kp/kd")
        damping = np.asarray(cfg.kd) * np.sqrt(stiffness / np.asarray(cfg.kp))
    super().__init__(view, nworld, stiffness, damping, device)
    self.inertia_shaped = inertia_shaped
    self.requires = frozenset({NEED_BIAS, NEED_M}) if inertia_shaped else frozenset({NEED_BIAS})
    with wp.ScopedDevice(self.device):
      self.ddq_des = wp.zeros((nworld, self.n), dtype=float)

  def set_target(self, q_des, dq_des=None, ddq_des=None) -> None:
    """ddq_des: joint acceleration feedforward (computed-torque mode only); None -> zero."""
    if ddq_des is not None and not self.inertia_shaped:
      raise ValueError("ddq_des feedforward needs inertia_shaped=True (tau = M ddq_des + ...)")
    super().set_target(q_des, dq_des)
    self.ddq_des.assign(broadcast(0.0 if ddq_des is None else ddq_des, (self.nworld, self.n)))

  def compute(self, state: RobotState) -> wp.array:
    if self.inertia_shaped:
      wp.launch(
        _computed_torque_kernel(self.n),
        dim=self.nworld,
        inputs=[state.q, state.dq, state.M, state.bias, self.q_des, self.dq_des, self.ddq_des, self.kp, self.kd,
                self.tau_limit],
        outputs=[self.cmd],
        device=self.device,
      )
    else:
      wp.launch(
        _pd_torque_kernel,
        dim=(self.nworld, self.n),
        inputs=[state.q, state.dq, state.bias, self.q_des, self.dq_des, self.kp, self.kd, 1, self.tau_limit],
        outputs=[self.cmd],
        device=self.device,
      )
    return self.cmd
