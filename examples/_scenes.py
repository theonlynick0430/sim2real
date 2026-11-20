"""Scene presets shared by the examples."""

from sim2real.envs.scripted import RELAY_CUBE_POS, RELAY_GOAL_POS
from sim2real.robots import get_robot, yam_bimanual
from sim2real.sim.scene import SceneConfig, build_scene


def make_scene(bimanual: bool):
  """Single YAM pick-place, or bimanual YAM relay (cube in front of left arm, goal in front of right)."""
  if bimanual:
    return build_scene(yam_bimanual(), SceneConfig(cube_pos=RELAY_CUBE_POS, goal_pos=RELAY_GOAL_POS))
  return build_scene(get_robot("yam"))


def close_viewer(viewer, timeout: float = 5.0) -> None:
  """Close a `mujoco.viewer.launch_passive` handle and wait for its render thread to exit.

  `Handle.close()` only requests exit; the viewer keeps running on a daemon thread. Returning from main
  while it is still tearing down GL races the interpreter's atexit hooks (glfw.terminate, Warp's CUDA
  context release) and segfaults intermittently. Joining the thread first makes exit clean.
  """
  import threading

  viewer.close()
  main = threading.main_thread()
  for t in threading.enumerate():
    if t is not main and t.daemon and "_launch_internal" in repr(getattr(t, "_target", "")):
      t.join(timeout)
