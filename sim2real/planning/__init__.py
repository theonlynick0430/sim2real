"""GPU IK + motion planning on cuRobo v2 (collision checking disabled).

Self-contained: nothing here imports the rest of sim2real except `adapters.py`, so the folder can be
copied elsewhere as-is. Hard deps: numpy, mujoco (URDF export + host FK). The solvers (`GpuIK`,
`GpuMotionPlanner`) need torch + cuRobo + a CUDA GPU and are imported lazily, so the rest works anywhere.

  cfg = planner_config_from_robot(get_robot("yam"), workdir)      # or PlannerConfig(urdf_path=..., ...)
  ik = GpuIK(cfg.replace(max_batch_size=256)); res = ik.solve(pos, quat_wxyz)
  mp = GpuMotionPlanner(cfg); mp.warmup(); traj = mp.plan_pose(q_start, pos, quat_wxyz)
"""

from .config import PlannerConfig
from .fk import MujocoFK, rotation_error
from .frames import base_to_world, world_to_base
from .mjcf_to_urdf import UrdfInfo, export_urdf

_LAZY = {
  "GpuIK": ".ik", "IKResult": ".ik",
  "GpuMotionPlanner": ".motion", "Trajectory": ".motion",
  "planner_config_from_robot": ".adapters", "export_robot_urdf": ".adapters",
}


def __getattr__(name):
  if name in _LAZY:
    import importlib

    return getattr(importlib.import_module(_LAZY[name], __name__), name)
  raise AttributeError(name)


__all__ = ["PlannerConfig", "MujocoFK", "rotation_error", "base_to_world", "world_to_base", "UrdfInfo",
           "export_urdf", *_LAZY]
