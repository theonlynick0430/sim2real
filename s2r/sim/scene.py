"""Scene composition with MjSpec: robot + table + cube + goal marker.

The scene owns everything that is *not* the robot (objects, contacts, solver
options) and applies the robot's actuator mode. Output is a compiled MjModel
that both backends consume unchanged.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from s2r.robots.config import ActuatorMode, RobotConfig


@dataclasses.dataclass(frozen=True)
class SceneConfig:
  timestep: float = 0.002
  integrator: int = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # step1/step2 compatible (no RK4)
  solver_iterations: int = 10
  ls_iterations: int = 20
  cube_half_size: float = 0.02
  cube_mass: float = 0.05
  cube_pos: tuple[float, float, float] = (0.35, 0.0, 0.02)
  goal_pos: tuple[float, float, float] = (0.30, 0.15, 0.02)
  friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
  # MJWarp per-world buffer sizes. Measured peak for YAM grasping the cube on CPU:
  # 36 contacts / 153 constraint rows (every finger primitive touches the cube).
  # Overflow silently drops contacts/constraints, so keep headroom.
  nconmax: int = 48
  njmax: int = 192


@dataclasses.dataclass
class Scene:
  """Compiled scene + the handful of names env/backends need."""

  mjm: mujoco.MjModel
  robot: RobotConfig
  cfg: SceneConfig
  cube_body: str = "cube"
  goal_body: str = "goal"
  home_key: str = "home"


def _apply_actuator_mode(spec: mujoco.MjSpec, robot: RobotConfig) -> None:
  if robot.actuator_mode == ActuatorMode.TORQUE:
    for name, lim in zip(robot.arm_actuators, robot.torque_limits):
      act = spec.actuator(name)
      act.set_to_motor()
      act.inheritrange = 0.0
      act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
      act.ctrlrange = [-lim, lim]
      act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
      act.forcerange = [-lim, lim]
    # No "magic" gravity compensation in torque mode: the controller compensates
    # gravity explicitly via qfrc_bias, exactly as on the real robot.
    for body in spec.bodies:
      body.gravcomp = 0.0
  elif robot.actuator_mode == ActuatorMode.POSITION:
    if robot.kp:
      for name, kp, kd, lim in zip(robot.arm_actuators, robot.kp, robot.kd, robot.torque_limits):
        act = spec.actuator(name)
        act.set_to_position(kp=kp, kv=kd)
        act.inheritrange = 1.0  # ctrlrange = joint range
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-lim, lim]
  else:
    raise ValueError(robot.actuator_mode)


def build_scene(robot: RobotConfig, cfg: SceneConfig = SceneConfig()) -> Scene:
  spec = mujoco.MjSpec.from_file(str(robot.model_path))
  spec.modelname = f"{robot.name}_pick_place"

  # Options.
  spec.option.timestep = cfg.timestep
  spec.option.integrator = cfg.integrator
  spec.option.iterations = cfg.solver_iterations
  spec.option.ls_iterations = cfg.ls_iterations
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

  _apply_actuator_mode(spec, robot)
  for name, body, pos, quat in robot.extra_sites:
    spec.body(body).add_site(name=name, pos=list(pos), quat=list(quat), size=[0.005, 0, 0], rgba=[0, 0, 1, 1],
                             group=4)
  for b1, b2 in robot.contact_excludes:
    spec.add_exclude(bodyname1=b1, bodyname2=b2)

  # Visuals.
  spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], width=300, height=300)
  spec.add_material(name="grid", textures=["", "grid"], texrepeat=[5, 5], texuniform=True, reflectance=0.2)
  wb = spec.worldbody
  wb.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

  # Table surface (robot base sits on it at z=0).
  wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], material="grid",
              friction=list(cfg.friction))

  # Cube.
  h = cfg.cube_half_size
  cube = wb.add_body(name="cube", pos=list(cfg.cube_pos))
  cube.add_freejoint(name="cube_free")
  cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h, h, h], mass=cfg.cube_mass,
                rgba=[0.8, 0.2, 0.2, 1], friction=list(cfg.friction), condim=4)

  # Goal marker: mocap body, visual only (mocap_pos is per-world state in both backends).
  goal = wb.add_body(name="goal", mocap=True, pos=list(cfg.goal_pos))
  goal.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[h, 0, 0], rgba=[0.2, 0.8, 0.2, 0.4],
                contype=0, conaffinity=0, group=1)

  # Replace robot keyframes with one scene-level "home" (keyframes must cover full nq/nu).
  for key in list(spec.keys):
    spec.delete(key)
  mjm = spec.compile()

  qpos = mjm.qpos0.copy()
  ctrl = np.zeros(mjm.nu)
  for jname, q in zip(robot.arm_joints, robot.home_qpos):
    qpos[mjm.joint(jname).qposadr[0]] = q
  if robot.actuator_mode == ActuatorMode.POSITION:
    for aname, q in zip(robot.arm_actuators, robot.home_qpos):
      ctrl[mjm.actuator(aname).id] = q
  if robot.gripper_actuator:
    ctrl[mjm.actuator(robot.gripper_actuator).id] = robot.gripper_open
  spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist(),
               mpos=list(cfg.goal_pos), mquat=[1, 0, 0, 0])
  mjm = spec.compile()
  return Scene(mjm=mjm, robot=robot, cfg=cfg)
