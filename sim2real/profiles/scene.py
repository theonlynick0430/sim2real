"""Robot-only scene for motor reference profiles: one arm, a floor, and a payload body on the tool.

The payload is a single body welded to the TCP's parent body. Its mass and position are edited in the compiled
model between skills (`set_payload`), and its sphere geom is the visualization: centered on the true center of
mass, opacity proportional to mass.
"""

from __future__ import annotations

import mujoco
import numpy as np

from sim2real.robots.config import ActuatorMode, RobotConfig
from sim2real.sim.scene import Scene, SceneConfig, as_mounts, compile_with_home_key, compose_robots

PAYLOAD_BODY = "payload"
PAYLOAD_GEOM = "payload_geom"
PAYLOAD_RADIUS = 0.025  # m: sphere drawn at the COM, and the solid sphere the payload's inertia is taken from


def build_profile_scene(robot: RobotConfig, cfg: SceneConfig = SceneConfig()) -> Scene:
  """One torque-mode arm on a floor plane with an (initially massless) payload body on the tool.

  The motors are whatever the robot's model says; for YAM that is ideal (armature = reflected rotor inertia,
  no joint friction).
  """
  if robot.actuator_mode != ActuatorMode.TORQUE:
    raise ValueError("motor profiles need torque-mode actuators")
  mounts = as_mounts(robot)
  spec = compose_robots(mounts)
  spec.modelname = f"{robot.name}_profiles"
  spec.option.timestep = cfg.timestep
  spec.option.integrator = cfg.integrator
  spec.option.iterations = cfg.solver_iterations
  spec.option.ls_iterations = cfg.ls_iterations

  world = spec.worldbody
  world.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
  # The table surface the base is bolted to. Trajectories are rejected if anything comes near it.
  world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], rgba=[0.25, 0.3, 0.35, 1])

  tcp_body = next(b for b in spec.bodies if any(s.name == robot.tcp_site for s in b.sites))
  payload = tcp_body.add_body(name=PAYLOAD_BODY, mass=0.0, inertia=[0.0, 0.0, 0.0], explicitinertial=True)
  payload.add_geom(name=PAYLOAD_GEOM, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[PAYLOAD_RADIUS, 0, 0],
                   rgba=[1.0, 0.45, 0.0, 0.0], contype=0, conaffinity=0, group=1)
  return Scene(mjm=compile_with_home_key(spec, mounts), mounts=mounts, cfg=cfg)


def set_payload(backend, mass: float, com_tool, max_mass: float) -> None:
  """Give the payload `mass` [kg] with its center of mass at `com_tool` [m] in the TCP site frame.

  Edits the host model in place and re-uploads it (`backend.reload_model`), so it takes effect on the next
  physics step. The sphere's opacity is mass / max_mass (invisible when empty).
  """
  mjm, robot = backend.mjm, backend.scene.robot
  site, body = mjm.site(robot.tcp_site).id, mjm.body(PAYLOAD_BODY).id
  offset = np.zeros(3)
  mujoco.mju_rotVecQuat(offset, np.asarray(com_tool, np.float64), mjm.site_quat[site])
  mjm.body_pos[body] = mjm.site_pos[site] + offset
  mjm.body_mass[body] = mass
  mjm.body_inertia[body] = 0.4 * mass * PAYLOAD_RADIUS**2  # solid sphere
  mjm.geom_rgba[mjm.geom(PAYLOAD_GEOM).id, 3] = np.clip(mass / max_mass, 0.0, 1.0)
  mujoco.mj_setConst(mjm, mujoco.MjData(mjm))
  backend.reload_model()
