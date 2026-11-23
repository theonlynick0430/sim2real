"""Shared cuRobo / torch glue (imported lazily by ik.py and motion.py; needs a CUDA GPU)."""

from __future__ import annotations

import torch
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

from .config import PlannerConfig


def device_cfg(cfg: PlannerConfig) -> DeviceCfg:
  return DeviceCfg(device=torch.device(cfg.device), dtype=torch.float32)


def robot_cfg(cfg: PlannerConfig):
  """Pre-built cuRobo RobotCfg without collision spheres (so both collision costs are disabled)."""
  from curobo._src.types.robot import RobotCfg  # not re-exported publicly in cuRobo v2

  return RobotCfg.create(cfg.curobo_robot_dict(), device_cfg(cfg), load_collision_spheres=False)


def as_tensor(x, cfg: PlannerConfig, ndim: int) -> torch.Tensor:
  """numpy / list / torch -> float32 tensor on the planner device with `ndim` dims (adds a batch dim)."""
  t = torch.as_tensor(x, dtype=torch.float32, device=cfg.device)
  if t.ndim == ndim - 1:
    t = t.unsqueeze(0)
  if t.ndim != ndim:
    raise ValueError(f"expected {ndim - 1}D or {ndim}D input, got shape {tuple(t.shape)}")
  return t.contiguous()


def goal_pose(cfg: PlannerConfig, pos: torch.Tensor, quat_wxyz: torch.Tensor) -> GoalToolPose:
  quat = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True)
  return GoalToolPose.from_poses({cfg.tool_frame: Pose(position=pos, quaternion=quat)}, num_goalset=1)


def joint_state(q: torch.Tensor, joint_names) -> JointState:
  return JointState.from_position(q, joint_names=list(joint_names))


def pad_batch(t: torch.Tensor, size: int) -> torch.Tensor:
  """Repeat the last row so the batch dim equals `size` (batched planners solve a fixed batch)."""
  if t.shape[0] > size:
    raise ValueError(f"batch {t.shape[0]} exceeds max_batch_size {size}")
  if t.shape[0] == size:
    return t
  return torch.cat([t, t[-1:].expand(size - t.shape[0], *t.shape[1:])], 0)


def first_per_problem(x, batch: int) -> torch.Tensor:
  """Per-problem value from a cuRobo result field: [batch] -> as is, [batch, seeds, ...] -> best-ranked
  (seed 0). Rows past `batch` (CUDA-graph padding) are dropped."""
  x = torch.as_tensor(x)
  x = x.reshape(1) if x.ndim == 0 else x
  return x[:batch] if x.ndim == 1 else x[:batch, 0]
