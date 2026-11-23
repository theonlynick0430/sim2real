"""RobotView: resolves a RobotConfig against a compiled MjModel into index maps.

Owns no physics and no backend state: it is the only place that knows how
robot-level names map onto model-level indices.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from sim2real.robots.config import RobotConfig


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
    joints = [mjm.joint(name).id for name in cfg.arm_joints]
    for name, j in zip(cfg.arm_joints, joints):
      if int(mjm.jnt_type[j]) not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        raise ValueError(f"arm joint '{name}' must be hinge or slide")
    dof_adr = mjm.jnt_dofadr[joints].astype(np.int32)
    site = mjm.site(cfg.tcp_site).id
    return RobotView(
      cfg=cfg,
      ndof=len(joints),
      qpos_adr=mjm.jnt_qposadr[joints].astype(np.int32),
      dof_adr=dof_adr,
      act_ids=np.array([mjm.actuator(name).id for name in cfg.arm_actuators], np.int32),
      tcp_site_id=int(site),
      tcp_body_id=int(mjm.site_bodyid[site]),
      gripper_act_id=mjm.actuator(cfg.gripper_actuator).id if cfg.gripper_actuator else -1,
      m_csr_index=csr_index(mjm, dof_adr),
      torque_limits=np.asarray(cfg.torque_limits, np.float32),
      home_qpos=np.asarray(cfg.home_qpos, np.float32),
    )


def csr_index(mjm: mujoco.MjModel, dofs: np.ndarray) -> np.ndarray:
  """(n, n) index into the CSR mass matrix `M` (length nC) for each entry of the dense block over `dofs`,
  -1 for structural zeros. MuJoCo's own `mj_fullM` does the CSR -> dense expansion: feed it M = 1..nC."""
  probe = mujoco.MjData(mjm)
  probe.M[:] = np.arange(1, mjm.nC + 1)
  full = np.zeros((mjm.nv, mjm.nv))
  mujoco.mj_fullM(mjm, probe, full)
  return full[np.ix_(dofs, dofs)].astype(np.int32) - 1
