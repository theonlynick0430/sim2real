---
name: sim2real-planning
description: The sim2real GPU IK / motion planning library (sim2real/planning) on cuRoboV2 with collision checking disabled — its API, the MJCF→URDF export, cuRobo v2 API facts at the pinned commit, the self-containment rule, and how to test on a Mac vs a Linux GPU. Use when using or changing GpuIK / GpuMotionPlanner, adding a robot to the planner, bumping cuRobo, or wiring planning into envs/scripts.
---

# GPU IK + motion planning (`sim2real/planning/`)

Wraps NVIDIA cuRobo v2 ("cuRoboV2", github.com/NVlabs/curobo, Apache-2.0). **Linux + NVIDIA GPU only**: no
macOS/CPU support. World and self collision are **off**. The module is **standalone**: nothing in envs,
controllers or existing examples uses it yet.

## Layout and the self-containment rule

| file | role |
|---|---|
| `mjcf_to_urdf.py` | `export_urdf(mjm, base_body, tool_site, joint_names, out_path, effort, velocity) -> UrdfInfo` |
| `config.py` | `PlannerConfig` (frozen) + `curobo_robot_dict()` (format 2.0, **no `collision_spheres`**) |
| `ik.py` | `GpuIK(cfg).solve(pos, quat_wxyz, seed_q=None, current_q=None) -> IKResult` |
| `motion.py` | `GpuMotionPlanner(cfg, batch=False, **plan_kwargs)`: `warmup()`, `plan_pose`, `plan_joint` -> `Trajectory` |
| `fk.py` | `MujocoFK(urdf, joints, tool_link)`: host FK of the URDF via MuJoCo, `sample_q`; `rotation_error` (`mju_subQuat`) |
| `frames.py` | `world_to_base` / `base_to_world` (numpy, wxyz, via `mju_mulPose` / `mju_negPose`) |
| `_curobo.py` | torch/cuRobo glue: tensors, `RobotCfg` without spheres, batch padding, result slicing, `to_numpy` |
| `adapters.py` | `planner_config_from_robot(RobotConfig, workdir, velocity=2.0, **overrides)`, `export_robot_urdf` |

- **Only `adapters.py` may import other `sim2real.*` modules.** Everything else uses relative imports.
  `tests/test_planning_urdf.py::test_planning_is_self_contained` enforces this. To move the package out of
  the repo, copy the folder and drop `adapters.py`.
- `__init__.py` imports `GpuIK`, `GpuMotionPlanner` and `adapters` **lazily**. Importing `sim2real.planning`
  must keep working without torch/cuRobo (on the Mac). Keep torch/cuRobo imports inside functions or `_curobo.py`.
- Hard deps are only numpy and mujoco. torch and cuRobo come from the `planning` extra in `pyproject.toml`.
- Pose math uses MuJoCo's `mju_*` functions. The package can't import `sim2real.utils`, so it keeps its own
  small helpers (e.g. `mjcf_to_urdf.quat_to_mat`).

## Usage

```python
cfg = planner_config_from_robot(get_robot("yam"), workdir)       # exports <workdir>/yam.urdf
res = GpuIK(cfg.replace(max_batch_size=256)).solve(pos, quat)     # (B,3), (B,4) -> res.q (B,dof), success, pos_err, rot_err
mp = GpuMotionPlanner(cfg); mp.warmup()
traj = mp.plan_pose(q_start, pos, quat)     # IK happens inside cuRobo (several goal configs, best traj wins)
traj = mp.plan_joint(q_start, q_goal)       # joint-space goal, no IK
bmp = GpuMotionPlanner(cfg.replace(max_batch_size=32), batch=True)   # BatchMotionPlanner, B <= 32 together
```

- **Pose goals:** for a desired end-effector pose call `plan_pose` directly. Use `GpuIK` + `plan_joint` only
  to pick the goal configuration yourself.
- **Conventions:**
  - Poses are in the **robot base frame** (the robot's root body, i.e. the `RobotMount` frame), with
    **wxyz** quaternions.
  - Inputs can be numpy or torch, with or without a batch dim. Outputs are torch on `cfg.device`; call
    `.numpy()` for host arrays.
- **Multi-arm:** one planner per arm on the un-prefixed `RobotConfig`. Convert world targets with
  `world_to_base(pos, quat, mount.pos, mount.quat)`.
- **`Trajectory` layout:**
  - `q`, `dq` and `ddq` are `(B, T, dof)`, sampled at `dt = cfg.interpolation_dt`.
  - `T` is the max `length` over problems. Past `length[b]`, `q` holds the final configuration and
    `dq`/`ddq` are 0.
  - `success` is `(B,)`. A failed problem whose cuRobo result is `None` gets `q = q_start` with
    `length = 1`.
  - `plan_pose(..., return_ee=True)` / `plan_joint(..., return_ee=True)` also fill the planned tool-frame
    trajectory in the base frame: `ee_pos (B,T,3)`, `ee_quat (B,T,4)` wxyz, `ee_vel (B,T,6)` = [linear,
    angular] (zero past `length[b]`). It is computed by host FK (`MujocoFK.ee_trajectory`, MuJoCo on the
    exported URDF), not cuRobo, so it matches `MujocoFK` exactly; not included in `solve_time`. Off by
    default (fields are `None`); cost is ~B·T host FK + Jacobian calls.
- **Batch sizing:**
  - `GpuIK` handles at most `cfg.max_batch_size` targets per call; cuRobo's `solve_pose` pads smaller
    batches to `max_batch_size` itself, so the batch size may vary freely between calls.
  - `GpuIK.solve` always passes cuRobo a current state (`current_q`, else `cfg.default_q`). cuRobo builds
    its goal buffer and CUDA graph on the first call; a current state that first appears on a later call
    needs a graph reset, which is off by default (`curobo.runtime.cuda_graph_reset`) and raises
    "CUDA graph reset is not available", leaving the CUDA context unusable. Keep the inputs structurally
    identical across calls.
  - The batched planner always solves exactly `max_batch_size` problems. Smaller batches are padded with
    the last row and trimmed from the output.
- **Warmup and CUDA graphs:** call `warmup()` once. The first call captures CUDA graphs, which takes seconds;
  later calls are fast. Construct each solver once and reuse it. Each instance holds its own graphs and GPU
  memory, so a `GpuIK` plus a planner means two cuRobo solvers.

## MJCF → URDF export (cuRobo reads URDF only)

- **What's exported:** the serial chain from the root body to the body holding `tcp_site`. That is one link
  per body, one joint per link (or a `fixed` joint for bodies without joints), body inertials, and the TCP
  site as a fixed child link named after the site.
- **What's dropped:** side branches such as the gripper fingers, and all visual/collision geometry.
- **`extra_sites`:** `export_robot_urdf` builds the model with `sim2real.sim.scene.robot_spec`, so config
  `extra_sites` such as YAM's `pinch_site` are included.
- **Missing names:** an unknown `base_body` or `tool_site` raises `KeyError` (from MuJoCo's name lookup).
- **Refuses (raises `ValueError`) on:**
  - more than one joint on a body;
  - ball or free joints;
  - joints offset from their body origin (`jnt_pos != 0`);
  - chain joints that don't match `joint_names` in order.

  Supporting any of these means extending the exporter, e.g. adding an intermediate fixed link for
  offsets.
- **Limits:** `jnt_range` for position. Velocity comes from the `velocity` argument; the **default of
  2 rad/s for YAM is a guess**, since `RobotConfig` has no velocity limits. Effort comes from
  `torque_limits`. cuRobo reads the velocity limits from the URDF. `max_acc` / `max_jerk` live in
  `PlannerConfig` (cuRobo cspace).
- **MuJoCo reload:** the URDF embeds `<mujoco><compiler fusestatic="false" .../></mujoco>` so MuJoCo can
  reload it with the fixed tool link kept as a body. `MujocoFK` and the tests rely on this.
- **Exactness:** URDF rpy is fixed-axis XYZ, `R = Rz(y) Ry(p) Rx(r)`, handled in `mat_to_rpy` including
  gimbal lock. The YAM export matches the MJCF `pinch_site` to about 1e-15.

## cuRobo v2 facts (pinned `main` @ `163e335c`, 2026-11-21)

- **Why this commit:** v0.8.0 (Apr 2026) rewrote the API and v0.9.0 (Nov 2026) is the latest tag. We pin a
  later `main` commit because it honors `self_collision_check=False` during validation too and adds
  `enable_inverse_dynamics`, which we set to `False`. Move to the next tag once it ships.
- **Old API names don't exist.** 0.7.x `MotionGen`, `IKSolverConfig`, `plan_single`/`plan_batch` and
  `MotionGenPlanConfig` are gone.
- **IK:**
  - Build with `InverseKinematicsCfg.create(robot=..., scene_model=None, self_collision_check=False,
    load_collision_spheres=False, num_seeds, position_tolerance, orientation_tolerance, max_batch_size,
    use_cuda_graph=True)`.
  - Solve with `InverseKinematics.solve_pose(GoalToolPose.from_poses({tool: Pose(position, quaternion)},
    num_goalset=1), current_state=JointState, seed_config=[B, n, dof])`.
  - Result fields `success`, `js_solution.position`, `position_error` and `rotation_error` are ranked per
    seed, with shape `[B, seeds, ...]`. We take seed 0 and trim the padding rows.
- **Motion planning:**
  - Build with `MotionPlannerCfg.create(..., interpolation_dt, enable_inverse_dynamics=False)`.
  - `MotionPlanner.plan_pose(goal, current_state, max_attempts=20, ...)` and `plan_cspace(goal_state,
    current_state)` handle one problem. `BatchMotionPlanner` has the same methods over `[B, dof]`.
- **Planner result fields:**
  - `interpolated_trajectory` (JointState) is a fixed buffer, `[B, seeds, T, dof]` or `[B, T, dof]`.
  - **Result buffers are reused across calls**: the next plan call overwrites them in place. Anything kept must
    be copied (`_extract` clones `q`, `dq`, `ddq`, `success`); views made the non-batch planner return the last
    problem's trajectory for every row.
  - `interpolated_last_tstep` is the **count** of valid steps per problem, not the last index.
  - `get_interpolated_plan()` refuses batches, so we slice the buffer ourselves.
- **Collision is off** when there's no scene, `self_collision_check=False`, and the robot has zero spheres
  (cuRobo then disables both collision costs). Gotcha: `MotionPlannerCfg.create` has **no
  `load_collision_spheres` flag** and loads spheres from a robot dict. That's why `_curobo.robot_cfg`
  pre-builds `curobo._src.types.robot.RobotCfg.create(dict, device_cfg, load_collision_spheres=False)`.
  This is an internal import path, so check it first when bumping cuRobo.
- **Install:** the distribution is `nvidia-curobo` and the import is `curobo`. The CUDA extra is
  `cu12-torch`; use `cu13-torch` for CUDA 13 drivers. Kernels JIT-compile at runtime, with no build step.
  Its requirement `warp-lang>=0.10` is compatible with our `warp-lang==1.18.0` pin.
- **When bumping the pin:**
  1. Re-check the signatures above against the new source (`curobo/_src/solver/solver_ik*.py`,
     `curobo/_src/motion/motion_planner*.py`).
  2. Run the GPU suite.

## Testing

```bash
python -m pytest tests/test_planning_urdf.py          # anywhere (Mac OK): export FK == MJCF, metadata, frames, self-containment
# Linux + CUDA, after: uv pip install -e ".[dev,planning]"
python -m pytest tests/test_planning_curobo.py -v     # IK success >= 95% on 256 targets, plan_pose/plan_joint, batched
python examples/plan_demo.py --headless --batch 32    # IK benchmark + sequential/batched plans; drop --headless to replay
```

- **Skips:** the cuRobo suite skips itself without cuRobo or CUDA.
- **Targets:** always FK of random in-limit joint samples (`MujocoFK.sample_q(margin=0.1)`), so they are
  reachable.
- **Independent checks:** outputs are re-checked with `MujocoFK`, not cuRobo's own FK. Trajectory checks
  cover the start equal to `q_start`, staying within joint limits, `|dq| <= 1.05 * max_vel`, no
  per-step jumps beyond `max_vel * dt`, and the end reaching the goal.
- **On a Mac:** you can check `GpuIK` / `GpuMotionPlanner` plumbing on the CPU with torch installed and
  cuRobo modules stubbed in `sys.modules`; `device="cpu"` works for the wrappers themselves. Only real
  cuRobo calls need the GPU.

## Not done yet (natural next steps)

- **No collision:** add spheres with `curobo.robot_builder` / `fit_collision_spheres` plus a `scene_model`
  for the table/cube.
- **Not integrated:** for example, feed `traj.q`/`traj.dq` as `JointPD`/`JointImpedance` targets at
  `traj.dt`, or OSC from FK. Keep `step1 → controller → step2`, and keep planning out of the RL hot loop,
  since it's a host-side call.
- **Validated on a GPU** (RTX 4090, torch 2.11 + cu128): the full cuRobo suite passes. That first run found
  the result-buffer aliasing and the IK current-state / CUDA-graph issue described above.
