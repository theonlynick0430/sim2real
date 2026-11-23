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
    joints = [self.model.joint(j).id for j in joint_names]
    self.qpos_adr = self.model.jnt_qposadr[joints]
    self.dof_adr = self.model.jnt_dofadr[joints]
    self.tool = self.model.body(tool_link).id
    self.lower, self.upper = self.model.jnt_range[joints].T.copy()

  def _set(self, q) -> None:
    """Set the arm joints and run kinematics (body poses in `self.data`)."""
    self.data.qpos[self.qpos_adr] = q
    mujoco.mj_kinematics(self.model, self.data)

  def __call__(self, q) -> tuple[np.ndarray, np.ndarray]:
    """q (B, dof) -> tool pos (B, 3), quat wxyz (B, 4), in the base frame."""
    q = np.atleast_2d(np.asarray(q, np.float64))
    pos, quat = np.empty((len(q), 3)), np.empty((len(q), 4))
    for i, qi in enumerate(q):
      self._set(qi)
      pos[i], quat[i] = self.data.xpos[self.tool], self.data.xquat[self.tool]
    return pos, quat

  def ee_trajectory(self, q, dq) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tool pose and twist along joint trajectories, in the base frame.

    q, dq (..., dof) -> pos (..., 3), quat wxyz (..., 4), vel (..., 6) = [linear, angular]: the velocity of
    the tool-frame origin and the tool angular velocity, both on base-frame axes.
    """
    q, dq = np.asarray(q, np.float64), np.asarray(dq, np.float64)
    lead, dof = q.shape[:-1], q.shape[-1]
    q, dq = q.reshape(-1, dof), dq.reshape(-1, dof)
    pos, quat, vel = np.empty((len(q), 3)), np.empty((len(q), 4)), np.empty((len(q), 6))
    jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
    for i, (qi, dqi) in enumerate(zip(q, dq)):
      self._set(qi)
      mujoco.mj_comPos(self.model, self.data)  # cdof / subtree_com, needed by mj_jacBody
      mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.tool)
      pos[i], quat[i] = self.data.xpos[self.tool], self.data.xquat[self.tool]
      vel[i, :3] = jacp[:, self.dof_adr] @ dqi
      vel[i, 3:] = jacr[:, self.dof_adr] @ dqi
    return pos.reshape(*lead, 3), quat.reshape(*lead, 4), vel.reshape(*lead, 6)

  def sample_q(self, n: int, rng: np.random.Generator, margin: float = 0.05) -> np.ndarray:
    """Uniform joint samples, `margin` (fraction of range) away from each limit."""
    span = self.upper - self.lower
    return rng.uniform(self.lower + margin * span, self.upper - margin * span, size=(n, len(span)))


def rotation_error(quat_a, quat_b) -> np.ndarray:
  """Angle [rad] between orientations (..., 4) wxyz, sign-invariant, via `mju_subQuat`.

  mju_subQuat's angle comes from atan2, so it stays accurate for tiny angles even with float32 inputs
  (2 * arccos(|a . b|) would turn ~1e-8 of rounding into ~5e-4 rad).
  """
  a, b = np.broadcast_arrays(np.asarray(quat_a, np.float64), np.asarray(quat_b, np.float64))
  diff = np.zeros(3)
  angles = np.empty(len(a.reshape(-1, 4)))
  for i, (qa, qb) in enumerate(zip(a.reshape(-1, 4), b.reshape(-1, 4))):
    mujoco.mju_subQuat(diff, qa, qb)
    angles[i] = np.linalg.norm(diff)
  return angles.reshape(a.shape[:-1])
