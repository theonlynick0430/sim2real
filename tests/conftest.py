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
