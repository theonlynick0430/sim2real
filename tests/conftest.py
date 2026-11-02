import pytest
import warp as wp

wp.config.quiet = True


@pytest.fixture(scope="session")
def yam_torque_scene():
  from s2r.robots import ActuatorMode, get_robot
  from s2r.sim.scene import build_scene

  return build_scene(get_robot("yam", actuator_mode=ActuatorMode.TORQUE))


@pytest.fixture(scope="session")
def warp_device():
  """CUDA if available, else Warp-CPU (MJWarp's CPU path is for debugging, but correct)."""
  return "cuda:0" if wp.is_cuda_available() else "cpu"


@pytest.fixture(scope="session")
def yam_bimanual_scene():
  """Two torque-mode YAMs (bases at y = +-0.2) with the relay cube/goal layout."""
  from s2r.envs.scripted import RELAY_CUBE_POS, RELAY_GOAL_POS
  from s2r.robots import yam_bimanual
  from s2r.sim.scene import SceneConfig, build_scene

  return build_scene(yam_bimanual(), SceneConfig(cube_pos=RELAY_CUBE_POS, goal_pos=RELAY_GOAL_POS))
