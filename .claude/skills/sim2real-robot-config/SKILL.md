---
name: sim2real-robot-config
description: How to add or modify a robot (Franka, UR, YAM, ...) in sim2real via RobotConfig, RobotMount and the MjSpec scene builder, including actuator modes, TCP sites, contact excludes and multi-arm mounts. Use when adding a robot, changing robot assets/gains, or composing multi-robot scenes.
---

# Robot configs and scenes

## RobotConfig (`sim2real/robots/config.py`)

A frozen dataclass; everything a backend, controller, or real driver needs. Fields:

| field | meaning |
|---|---|
| `name` | registry key (`get_robot(name, **overrides)`) |
| `model_path` | MJCF **or** URDF; both go through `mujoco.MjSpec.from_file` |
| `arm_joints` / `arm_actuators` | controlled joints (hinge/slide only) and one actuator each, same order |
| `tcp_site` | end-effector frame used for J and OSC |
| `home_qpos`, `torque_limits` | length `ndof`; limits clamp every controller output |
| `kp`, `kd` | default joint gains (POSITION mode actuators, `JointPD` defaults, impedance damping ratio) |
| `gripper_actuator`, `gripper_open`, `gripper_closed` | single position actuator for the gripper (optional) |
| `actuator_mode` | `ActuatorMode.TORQUE` (motor, needed by OSC/impedance) or `POSITION` (implicit PD) |
| `contact_excludes` | body pairs with contacts disabled (saves GPU contact budget) |
| `extra_sites` | `(name, body, pos, quat_wxyz)` sites added at build time — define a TCP the asset lacks |

`__post_init__` validates lengths. Use `cfg.replace(...)` for variants, e.g.
`get_robot("yam", actuator_mode=ActuatorMode.POSITION)`.

## Adding a robot

1. Vendor the asset under `sim2real/assets/<robot>/` (keep its LICENSE; `package-data` already ships `assets/**`).
2. Create `sim2real/robots/<robot>.py`:
   ```python
   ASSETS = Path(__file__).resolve().parents[1] / "assets" / "<robot>"
   FRANKA = register_robot(RobotConfig(
     name="franka", model_path=ASSETS / "panda.xml",
     arm_joints=tuple(f"joint{i}" for i in range(1, 8)), arm_actuators=(...),
     tcp_site="...", home_qpos=(...), torque_limits=(...), kp=(...), kd=(...),
     gripper_actuator="...", gripper_open=..., gripper_closed=...,
   ))
   ```
3. Import it in `sim2real/robots/__init__.py` so it registers on `import sim2real.robots`.
4. Controllers are generated per DoF (`linalg(n)`), so a 7-DoF arm needs no controller changes.
5. Test: build the scene, run `tests/test_dynamics_parity.py`-style CPU vs MJWarp checks, and an OSC tracking
   test. Measure peak contacts (`MujocoBackend.mjd.ncon`, `nefc`) while grasping and size `SceneConfig.nconmax/njmax`.

Gotchas:
- Menagerie TCP sites are often the flange, not the pad center — add an `extra_sites` pinch site (see `yam.py`).
- Torque limits should come from the motor datasheet / MJCF `forcerange`; they become `ctrlrange` in TORQUE mode.
- Source keyframes are deleted; the scene creates one full-state `home` key from `home_qpos`.

## Scene building (`sim2real/sim/scene.py`)

- `robot_spec(robot)`: loads the asset and applies **all** robot-level edits (actuator mode, extra sites,
  excludes). It's shared with the real-robot dynamics model, so edits here apply to both.
- `_apply_actuator_mode`: TORQUE → `set_to_motor`, ctrl/force range = ±torque_limit, **gravcomp = 0**.
  POSITION → `set_to_position(kp, kv=kd)`.
- `compose_robots(mounts)`: one un-prefixed mount uses the spec directly (names unchanged); several mounts are
  attached with `MjSpec.attach(child, prefix=..., frame=...)` (prefixes every name, incl. excludes/equalities).
  Child options are synced to the parent first to avoid attach conflict warnings.
- `build_scene(robots, SceneConfig)`: adds table, cube (`cube_free` freejoint), mocap goal, solver options,
  and the `home` keyframe. Scene owns everything that isn't a robot.
- `SceneConfig`: `timestep=0.002`, `integrator=implicitfast` (never RK4), `nconmax=48`, `njmax=192` per world.

## Multi-arm (`RobotMount`)

`RobotMount(robot, prefix="left/", pos=..., quat=...)`; `mount.resolved` is the config with prefixed names
(what the compiled model sees). `yam_bimanual(separation)` returns `(left, right)` mounts, left at +y.
Mount poses should come from the calibrated real rig — the real driver places each arm's dynamics model at
the same pose so sim and real share one world frame.
