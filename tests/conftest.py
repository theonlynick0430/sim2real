import numpy as np
import pytest
import warp as wp

wp.config.quiet = True


@pytest.fixture(scope="session")
def yam_torque_scene():
  from sim2real.robots import ActuatorMode, get_robot
  from sim2real.sim.scene import build_scene

  return build_scene(get_robot("yam", actuator_mode=ActuatorMode.TORQUE))


@pytest.fixture(scope="session")
def warp_device():
  """CUDA if available, else Warp-CPU (MJWarp's CPU path is for debugging, but correct)."""
  return "cuda:0" if wp.is_cuda_available() else "cpu"


@pytest.fixture(scope="session")
def yam_bimanual_scene():
  """Two torque-mode YAMs (bases at y = +-0.2) with the relay cube/goal layout."""
  from examples._scenes import make_scene

  return make_scene(bimanual=True)


def _randomize_arms(backend, rng, margin: float, vel_std: float):
  """Random arm joint positions (`margin` [rad] inside the limits) and velocities in every world, then forward.

  Returns the full (qpos, qvel) arrays that were set, float64, for building MuJoCo C references.
  """
  mjm, nworld = backend.mjm, backend.nworld
  qpos, qvel = backend.d.qpos.numpy().astype(np.float64), np.zeros((nworld, mjm.nv))
  for view in backend.views:
    lo, hi = mjm.jnt_range[[mjm.joint(j).id for j in view.cfg.arm_joints]].T
    qpos[:, view.qpos_adr] = rng.uniform(lo + margin, hi - margin, (nworld, view.ndof))
    qvel[:, view.dof_adr] = rng.normal(0, vel_std, (nworld, view.ndof))
  backend.d.qpos.assign(qpos.astype(np.float32))
  backend.d.qvel.assign(qvel.astype(np.float32))
  backend.forward()
  return qpos, qvel


@pytest.fixture(scope="session")
def randomize_arms():
  """`randomize_arms(backend, rng, margin, vel_std) -> (qpos, qvel)`, see `_randomize_arms`."""
  return _randomize_arms
