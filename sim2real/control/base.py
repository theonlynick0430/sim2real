"""Controller base: pure (RobotState, targets, gains) -> actuator command.

Controllers own their target/gain buffers (warp arrays, nworld-batched) and an
output buffer. `compute()` is a single fused kernel launch, so it can sit inside
a CUDA graph on GPU and runs unchanged on Warp-CPU (Mac sim / real robot).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from sim2real.robots.config import ActuatorMode
from sim2real.sim.robot_view import RobotView
from sim2real.sim.state import RobotState


class Controller:
  requires: frozenset = frozenset()  # dynamics quantities needed from RobotState
  output: ActuatorMode = ActuatorMode.TORQUE  # actuator mode the command is meant for

  def __init__(self, view: RobotView, nworld: int, device=None):
    self.view = view
    self.n = view.ndof
    self.nworld = nworld
    self.device = wp.get_device(device)
    with wp.ScopedDevice(self.device):
      self.cmd = wp.zeros((nworld, self.n), dtype=float)
      self.tau_limit = wp.array(view.torque_limits, dtype=float)

  def check_compatible(self, mode: ActuatorMode) -> None:
    if mode != self.output:
      raise ValueError(f"{type(self).__name__} outputs {self.output.value} commands, but robot actuators are "
                       f"in {mode.value} mode")

  def compute(self, state: RobotState) -> wp.array:
    raise NotImplementedError

  # host helpers
  def _arr(self, value, shape) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, np.float32), shape).copy()
