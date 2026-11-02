"""Scene composition with MjSpec: mounted robot(s) + table + cube + goal marker.

Each robot is loaded from its own MJCF/URDF (`robot_spec`, which applies the
config: actuator mode, extra sites, contact excludes) and attached at its mount
pose with a name prefix (`compose_robots`), so one arm and several arms share
the same robot-level edits. The scene owns everything that is *not* a robot (objects, solver
options, contact budget). Output is a compiled MjModel both backends consume.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import mujoco
import numpy as np

from s2r.robots.config import ActuatorMode, RobotConfig, RobotMount


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
  mounts: tuple[RobotMount, ...]  # one per arm, in arm-index order
  cfg: SceneConfig
  cube_body: str = "cube"
  goal_body: str = "goal"
  home_key: str = "home"

  @property
  def robots(self) -> tuple[RobotConfig, ...]:
    """Per-arm configs with model-level (prefixed) names."""
    return tuple(m.resolved for m in self.mounts)

  @property
  def robot(self) -> RobotConfig:
    """Arm 0 (single-arm convenience)."""
    return self.robots[0]

  @property
  def narm(self) -> int:
    return len(self.mounts)


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


def robot_spec(robot: RobotConfig) -> mujoco.MjSpec:
  """The robot's own MJCF/URDF with every config-driven edit applied (un-prefixed names).

  Single place for robot-level spec edits, shared by the sim scene and the real-robot model.
  """
  spec = mujoco.MjSpec.from_file(str(robot.model_path))
  for key in list(spec.keys):  # scenes build their own full-state keyframe
    spec.delete(key)
  _apply_actuator_mode(spec, robot)
  for name, body, pos, quat in robot.extra_sites:
    spec.body(body).add_site(name=name, pos=list(pos), quat=list(quat), size=[0.005, 0, 0], rgba=[0, 0, 1, 1],
                             group=4)
  for b1, b2 in robot.contact_excludes:
    spec.add_exclude(bodyname1=b1, bodyname2=b2)
  return spec


def compose_robots(mounts: Sequence[RobotMount]) -> mujoco.MjSpec:
  """A spec holding every mounted robot at its base pose, names prefixed per mount.

  Several robots are attached with `MjSpec.attach(prefix=...)`. A single un-prefixed robot is
  used as the base spec directly (its top-level bodies moved to the mount pose), keeping its
  original names; MuJoCo's attach would warn about the empty prefix.
  """
  mounts = tuple(mounts)
  if len(mounts) == 1 and not mounts[0].prefix:
    m = mounts[0]
    spec = robot_spec(m.robot)
    for body in spec.worldbody.bodies:  # world <- mount <- body
      pos, quat = np.zeros(3), np.zeros(4)
      mujoco.mju_mulPose(pos, quat, np.asarray(m.pos, float), np.asarray(m.quat, float),
                         np.asarray(body.pos, float), np.asarray(body.quat, float))
      body.pos, body.quat = pos, quat
    return spec

  prefixes = [m.prefix for m in mounts]
  if not all(prefixes) or len(set(prefixes)) != len(prefixes):
    raise ValueError(f"multiple robots need unique, non-empty prefixes, got {prefixes}")
  spec = mujoco.MjSpec()
  for m in mounts:
    child = robot_spec(m.robot)
    # Options are scene-level (set by the caller after composing); match them here so attach
    # doesn't report robot-vs-parent option conflicts.
    child.option.integrator = spec.option.integrator
    child.option.timestep = spec.option.timestep
    frame = spec.worldbody.add_frame(pos=list(m.pos), quat=list(m.quat))
    spec.attach(child, prefix=m.prefix, frame=frame)
  return spec


def as_mounts(robots: RobotConfig | RobotMount | Sequence[RobotMount]) -> tuple[RobotMount, ...]:
  if isinstance(robots, RobotConfig):
    return (RobotMount(robots),)
  if isinstance(robots, RobotMount):
    return (robots,)
  return tuple(robots)


def build_scene(
  robots: RobotConfig | RobotMount | Sequence[RobotMount], cfg: SceneConfig = SceneConfig()
) -> Scene:
  """Table + cube + goal + one or more mounted robots (e.g. `yam_bimanual()`)."""
  mounts = as_mounts(robots)
  spec = compose_robots(mounts)
  spec.modelname = "_".join(m.prefix.strip("/") or m.robot.name for m in mounts) + "_pick_place"

  # Scene-level options (override whatever the robots' MJCF set).
  spec.option.timestep = cfg.timestep
  spec.option.integrator = cfg.integrator
  spec.option.iterations = cfg.solver_iterations
  spec.option.ls_iterations = cfg.ls_iterations
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

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

  # One scene-level "home" keyframe (keyframes must cover full nq/nu).
  mjm = spec.compile()
  qpos = mjm.qpos0.copy()
  ctrl = np.zeros(mjm.nu)
  for robot in (m.resolved for m in mounts):
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
  return Scene(mjm=mjm, mounts=mounts, cfg=cfg)
