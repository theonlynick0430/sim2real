"""World <-> robot-base pose transforms (numpy, batched, quaternions wxyz), via MuJoCo's `mju_*Pose`.

Planner poses are expressed in the robot's base frame. For a robot mounted at (base_pos, base_quat) in
the world, convert world-frame targets with `world_to_base` and planner outputs with `base_to_world`.
"""

from __future__ import annotations

import mujoco
import numpy as np


def _compose(pos1, quat1, pos2, quat2) -> tuple[np.ndarray, np.ndarray]:
  """pose1 * pose2 for one pose1 and a batch of pose2 (pos (..., 3), quat (..., 4))."""
  pos1, quat1 = np.asarray(pos1, np.float64), np.asarray(quat1, np.float64)
  pos2, quat2 = np.asarray(pos2, np.float64), np.asarray(quat2, np.float64)
  lead = np.broadcast_shapes(pos2.shape[:-1], quat2.shape[:-1])
  pos2, quat2 = np.broadcast_to(pos2, (*lead, 3)).reshape(-1, 3), np.broadcast_to(quat2, (*lead, 4)).reshape(-1, 4)
  pos, quat = np.zeros((len(pos2), 3)), np.zeros((len(pos2), 4))
  for i in range(len(pos2)):
    mujoco.mju_mulPose(pos[i], quat[i], pos1, quat1, pos2[i], quat2[i])
  return pos.reshape(*lead, 3), quat.reshape(*lead, 4)


def world_to_base(pos_w, quat_w, base_pos, base_quat) -> tuple[np.ndarray, np.ndarray]:
  """Express world-frame poses (pos (...,3), quat (...,4)) in the base frame."""
  inv_pos, inv_quat = np.zeros(3), np.zeros(4)
  mujoco.mju_negPose(inv_pos, inv_quat, np.asarray(base_pos, np.float64), np.asarray(base_quat, np.float64))
  return _compose(inv_pos, inv_quat, pos_w, quat_w)


def base_to_world(pos_b, quat_b, base_pos, base_quat) -> tuple[np.ndarray, np.ndarray]:
  """Express base-frame poses in the world frame (inverse of `world_to_base`)."""
  return _compose(base_pos, base_quat, pos_b, quat_b)
