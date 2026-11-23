"""World <-> robot-base pose transforms (numpy, batched, quaternions wxyz).

Planner poses are expressed in the robot's base frame. For a robot mounted at (base_pos, base_quat) in
the world, convert world-frame targets with `world_to_base` and planner outputs with `base_to_world`.
"""

from __future__ import annotations

import numpy as np


def quat_mul(a, b) -> np.ndarray:
  """Hamilton product a * b, wxyz, broadcasting over leading dims."""
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                   aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_conj(q) -> np.ndarray:
  q = np.asarray(q, np.float64)
  return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q, v) -> np.ndarray:
  """Rotate vectors v (..., 3) by unit quaternions q (..., 4)."""
  v = np.asarray(v, np.float64)
  vq = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
  return quat_mul(quat_mul(q, vq), quat_conj(q))[..., 1:]


def world_to_base(pos_w, quat_w, base_pos, base_quat) -> tuple[np.ndarray, np.ndarray]:
  """Express world-frame poses (pos (...,3), quat (...,4)) in the base frame."""
  inv = quat_conj(base_quat)
  pos_b = quat_rotate(inv, np.asarray(pos_w, np.float64) - np.asarray(base_pos, np.float64))
  return pos_b, quat_mul(inv, quat_w)


def base_to_world(pos_b, quat_b, base_pos, base_quat) -> tuple[np.ndarray, np.ndarray]:
  """Express base-frame poses in the world frame (inverse of `world_to_base`)."""
  pos_w = quat_rotate(base_quat, pos_b) + np.asarray(base_pos, np.float64)
  return pos_w, quat_mul(base_quat, quat_b)
