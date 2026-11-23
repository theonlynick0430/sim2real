"""Batched GPU inverse kinematics (cuRobo v2 `InverseKinematics`), collision checking disabled."""

from __future__ import annotations

import dataclasses

import numpy as np

from .config import PlannerConfig


@dataclasses.dataclass
class IKResult:
  q: "torch.Tensor | np.ndarray"  # (B, dof) best solution per target  # noqa: F821
  success: "torch.Tensor | np.ndarray"  # (B,) bool
  pos_err: "torch.Tensor | np.ndarray"  # (B,) m
  rot_err: "torch.Tensor | np.ndarray"  # (B,) rad
  solve_time: float = 0.0  # s, as reported by cuRobo

  def numpy(self) -> "IKResult":
    f = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)  # noqa: E731
    return IKResult(f(self.q), f(self.success).astype(bool), f(self.pos_err), f(self.rot_err), self.solve_time)


class GpuIK:
  """Solve up to `cfg.max_batch_size` tool-pose targets (base frame, wxyz) in one GPU call."""

  def __init__(self, cfg: PlannerConfig):
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

    from . import _curobo

    self.cfg = cfg
    self._c = _curobo
    ik_cfg = InverseKinematicsCfg.create(
      robot=_curobo.robot_cfg(cfg),
      scene_model=None,
      self_collision_check=False,
      load_collision_spheres=False,
      device_cfg=_curobo.device_cfg(cfg),
      num_seeds=cfg.num_ik_seeds,
      position_tolerance=cfg.position_tolerance,
      orientation_tolerance=cfg.orientation_tolerance,
      max_batch_size=cfg.max_batch_size,
      use_cuda_graph=True,
    )
    self.solver = InverseKinematics(ik_cfg)
    if list(self.solver.joint_names) != list(cfg.joint_names):
      raise ValueError(f"cuRobo joint order {self.solver.joint_names} != {cfg.joint_names}")

  def solve(self, pos, quat_wxyz, seed_q=None, current_q=None) -> IKResult:
    """pos (B,3), quat_wxyz (B,4): targets in the base frame (numpy or torch; unbatched inputs allowed).

    seed_q (B,dof): optional initial guesses, used alongside cuRobo's random seeds.
    current_q (B,dof): optional; ranks solutions by closeness to this configuration (default: cfg.default_q).
    Returns torch tensors on the planner device (`.numpy()` for host arrays).
    """
    c, cfg = self._c, self.cfg
    pos, quat = c.as_tensor(pos, cfg, 2), c.as_tensor(quat_wxyz, cfg, 2)
    batch = pos.shape[0]
    seed = c.as_tensor(seed_q, cfg, 2).unsqueeze(1) if seed_q is not None else None
    # Always give cuRobo a current state. Its goal buffer (and the CUDA graph captured from it) is built on the
    # first call; a current state appearing later changes the buffer structure and needs a graph reset, which
    # cuRobo disables by default ("CUDA graph reset is not available"), and omitting it later silently reuses
    # the previous call's. Without `current_q`, fall back to cfg.default_q. (Batch size needs no care here:
    # cuRobo pads to max_batch_size itself.)
    if current_q is None:
      current_q = np.tile(np.asarray(cfg.default_q, dtype=np.float32), (batch, 1))
    current = c.joint_state(c.as_tensor(current_q, cfg, 2), cfg.joint_names)
    res = self.solver.solve_pose(c.goal_pose(cfg, pos, quat), current_state=current, seed_config=seed)
    q = res.js_solution.position if res.js_solution is not None else res.solution
    return IKResult(
      q=q[:batch] if q.ndim == 2 else q[:batch, 0],  # (B, dof) or ranked (B, seeds, dof)
      success=c.first_per_problem(res.success, batch).bool(),
      pos_err=c.first_per_problem(res.position_error, batch),
      rot_err=c.first_per_problem(res.rotation_error, batch),
      solve_time=float(res.solve_time),
    )
