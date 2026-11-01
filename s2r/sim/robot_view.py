"""RobotView: resolves a RobotConfig against a compiled MjModel into index maps.

Owns no physics and no backend state — it is the only place that knows how
robot-level names map onto model-level indices, so every backend (and the
real-robot dynamics model) can share it.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from s2r.robots.config import RobotConfig


@dataclasses.dataclass(frozen=True)
class RobotView:
  cfg: RobotConfig
  ndof: int
  qpos_adr: np.ndarray  # (n,) int32
  dof_adr: np.ndarray  # (n,) int32
  act_ids: np.ndarray  # (n,) int32
  tcp_site_id: int
  tcp_body_id: int
  gripper_act_id: int  # -1 if none
  # Index into the CSR (lower-triangular) mass matrix `M` (length nC) for each
  # (i, j) of the robot's dense n x n block; -1 means structural zero.
  m_csr_index: np.ndarray  # (n, n) int32
  torque_limits: np.ndarray  # (n,) float32
  home_qpos: np.ndarray  # (n,) float32

  @staticmethod
  def from_model(mjm: mujoco.MjModel, cfg: RobotConfig) -> "RobotView":
    qpos_adr, dof_adr = [], []
    for name in cfg.arm_joints:
      j = mjm.joint(name)
      if int(np.asarray(j.type).reshape(-1)[0]) not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        raise ValueError(f"arm joint '{name}' must be hinge or slide")
      qpos_adr.append(j.qposadr[0])
      dof_adr.append(j.dofadr[0])
    act_ids = [mjm.actuator(name).id for name in cfg.arm_actuators]
    site = mjm.site(cfg.tcp_site)
    grip = mjm.actuator(cfg.gripper_actuator).id if cfg.gripper_actuator else -1

    n = len(dof_adr)
    idx = np.full((n, n), -1, dtype=np.int32)
    for a, da in enumerate(dof_adr):
      for b, db in enumerate(dof_adr):
        row, col = max(da, db), min(da, db)
        start = mjm.M_rowadr[row]
        cols = mjm.M_colind[start : start + mjm.M_rownnz[row]]
        hit = np.nonzero(cols == col)[0]
        if hit.size:
          idx[a, b] = start + hit[0]

    return RobotView(
      cfg=cfg,
      ndof=n,
      qpos_adr=np.asarray(qpos_adr, np.int32),
      dof_adr=np.asarray(dof_adr, np.int32),
      act_ids=np.asarray(act_ids, np.int32),
      tcp_site_id=int(site.id),
      tcp_body_id=int(mjm.site_bodyid[site.id]),
      gripper_act_id=int(grip),
      m_csr_index=idx,
      torque_limits=np.asarray(cfg.torque_limits, np.float32),
      home_qpos=np.asarray(cfg.home_qpos, np.float32),
    )
