"""Host-side forward kinematics of an exported URDF, through MuJoCo (no GPU, no cuRobo).

Used to sample reachable targets and to check IK / planner outputs independently of cuRobo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


class MujocoFK:
  """FK of `tool_link` in the URDF's base (root link) frame."""

  def __init__(self, urdf_path: str | Path, joint_names: Sequence[str], tool_link: str):
    self.model = mujoco.MjModel.from_xml_path(str(urdf_path))
    self.data = mujoco.MjData(self.model)
    self.qadr = np.array([self.model.joint(j).qposadr[0] for j in joint_names])
    self.tool = self.model.body(tool_link).id
    self.lower = self.model.jnt_range[[self.model.joint(j).id for j in joint_names], 0].copy()
    self.upper = self.model.jnt_range[[self.model.joint(j).id for j in joint_names], 1].copy()

  def __call__(self, q) -> tuple[np.ndarray, np.ndarray]:
    """q (B, dof) -> tool pos (B, 3), quat wxyz (B, 4), in the base frame."""
    q = np.atleast_2d(np.asarray(q, np.float64))
    pos, quat = np.empty((len(q), 3)), np.empty((len(q), 4))
    for i, qi in enumerate(q):
      self.data.qpos[self.qadr] = qi
      mujoco.mj_kinematics(self.model, self.data)
      pos[i] = self.data.xpos[self.tool]
      quat[i] = self.data.xquat[self.tool]
    return pos, quat

  def sample_q(self, n: int, rng: np.random.Generator, margin: float = 0.05) -> np.ndarray:
    """Uniform joint samples, `margin` (fraction of range) away from each limit."""
    span = self.upper - self.lower
    return rng.uniform(self.lower + margin * span, self.upper - margin * span, size=(n, len(span)))


def rotation_error(quat_a, quat_b) -> np.ndarray:
  """Angle [rad] between orientations (B, 4) wxyz, sign-invariant."""
  dot = np.abs(np.sum(np.asarray(quat_a) * np.asarray(quat_b), -1))
  return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
