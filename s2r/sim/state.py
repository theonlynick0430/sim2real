"""Backend-neutral robot state consumed by controllers.

All arrays are Warp arrays with a leading `nworld` dimension, living on the
backend's device (cuda for MJWarp, cpu for MuJoCo C / real robot). Conventions
follow MuJoCo: J rows 0-2 are translational, rows 3-5 rotational, all in the
world frame; `bias` = C(q, dq) dq + g(q) (MuJoCo `qfrc_bias`).
"""

from __future__ import annotations

import dataclasses

import warp as wp

# Which dynamics quantities a controller needs; backends skip the rest.
NEED_M = "M"
NEED_J = "J"
NEED_BIAS = "bias"
ALL_NEEDS = frozenset({NEED_M, NEED_J, NEED_BIAS})


@dataclasses.dataclass
class RobotState:
  q: wp.array  # (nworld, n) float32
  dq: wp.array  # (nworld, n)
  tcp_pos: wp.array  # (nworld,) vec3
  tcp_rot: wp.array  # (nworld,) mat33 (world-from-tcp)
  J: wp.array  # (nworld, 6, n)
  M: wp.array  # (nworld, n, n)
  bias: wp.array  # (nworld, n)

  @staticmethod
  def allocate(nworld: int, n: int, device) -> "RobotState":
    with wp.ScopedDevice(device):
      return RobotState(
        q=wp.zeros((nworld, n), dtype=float),
        dq=wp.zeros((nworld, n), dtype=float),
        tcp_pos=wp.zeros(nworld, dtype=wp.vec3),
        tcp_rot=wp.zeros(nworld, dtype=wp.mat33),
        J=wp.zeros((nworld, 6, n), dtype=float),
        M=wp.zeros((nworld, n, n), dtype=float),
        bias=wp.zeros((nworld, n), dtype=float),
      )
