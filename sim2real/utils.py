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


def resample_trajectory(q, dq, ddq, dt_in: float, dt_out: float, time_scale: float = 1.0):
  """Resample a joint trajectory (T, n) from `dt_in` to `dt_out`, optionally slowed down by `time_scale` (>= 1
  stretches time: velocities / time_scale, accelerations / time_scale^2).

  Position and velocity come from cubic Hermite interpolation of (q, dq) between consecutive input samples, so
  both are continuous and the input samples are reproduced exactly; `ddq` is interpolated linearly. The output
  starts at the first input sample and always ends on the last one (the final step may be shorter than `dt_out`
  in trajectory time). Returns (q, dq, ddq), each (T_out, n) float64.
  """
  q, dq, ddq = (np.asarray(x, np.float64) for x in (q, dq, ddq))
  if len(q) == 1:
    return q.copy(), dq / time_scale, ddq / time_scale**2
  h = dt_in * time_scale  # stretched spacing of the input samples
  dq, ddq = dq / time_scale, ddq / time_scale**2
  duration = (len(q) - 1) * h
  t = np.arange(0.0, duration, dt_out)
  t = np.append(t, duration) if duration - t[-1] > 1e-9 else t
  seg = np.minimum((t / h).astype(int), len(q) - 2)
  s = (t / h - seg)[:, None]
  q0, q1, v0, v1 = q[seg], q[seg + 1], h * dq[seg], h * dq[seg + 1]
  q_out = ((2 * s**3 - 3 * s**2 + 1) * q0 + (s**3 - 2 * s**2 + s) * v0 + (-2 * s**3 + 3 * s**2) * q1
           + (s**3 - s**2) * v1)
  dq_out = ((6 * s**2 - 6 * s) * (q0 - q1) + (3 * s**2 - 4 * s + 1) * v0 + (3 * s**2 - 2 * s) * v1) / h
  ddq_out = (1 - s) * ddq[seg] + s * ddq[seg + 1]
  return q_out, dq_out, ddq_out
