"""Small host-side helpers shared across sim2real (numpy; pose math goes through MuJoCo's `mju_*` functions)."""

from __future__ import annotations

import mujoco
import numpy as np


def broadcast(value, shape) -> np.ndarray:
  """`value` broadcast to `shape` as a new float32 array (e.g. one target for every world)."""
  return np.broadcast_to(np.asarray(value, np.float32), shape).copy()


def quat_to_mat(quat) -> np.ndarray:
  """wxyz quaternion(s) (..., 4) -> rotation matrices (..., 3, 3)."""
  quat = np.asarray(quat, np.float64)
  mats = np.zeros((*quat.shape[:-1], 9))
  for q, mat in zip(quat.reshape(-1, 4), mats.reshape(-1, 9)):
    mujoco.mju_quat2Mat(mat, q)
  return mats.reshape(*quat.shape[:-1], 3, 3)
