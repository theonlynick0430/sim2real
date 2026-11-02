"""Scene presets shared by the examples."""

from s2r.envs.scripted import RELAY_CUBE_POS, RELAY_GOAL_POS
from s2r.robots import get_robot, yam_bimanual
from s2r.sim.scene import SceneConfig, build_scene


def make_scene(bimanual: bool):
  """Single YAM pick-place, or bimanual YAM relay (cube in front of left arm, goal in front of right)."""
  if bimanual:
    return build_scene(yam_bimanual(), SceneConfig(cube_pos=RELAY_CUBE_POS, goal_pos=RELAY_GOAL_POS))
  return build_scene(get_robot("yam"))
