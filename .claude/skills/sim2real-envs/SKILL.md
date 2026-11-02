---
name: sim2real-envs
description: How the batched MJWarp RL environment (PickPlaceEnv) and scripted policies in sim2real are structured — obs/action layout, reward, masked resets as kernels, multi-arm support, and experts. Use when creating a new task/env, changing observations, rewards or resets, or writing scripted/expert policies.
---

# Environments (`sim2real/envs/`)

## PickPlaceEnv (`pick_place.py`) — MJWarp only

Per policy step, all Warp kernels, no host syncs:

```
_apply_action_kernel (per arm)  → OSC target_pos + gripper value
control_substeps(oscs, decimation)  → CUDA graph: decimation × [step1, OSC per arm, step2]
mjw.kinematics → _reward_done_kernel → _masked_reset → _observe (_arm_obs_kernel per arm + _shared_obs_kernel)
```

- **Rates**: physics/OSC 500 Hz (`dt=2 ms`), policy 50 Hz (`decimation=10`), episodes 250 steps.
- **Action** `(nworld, 4·narm)`: per arm `[dx, dy, dz]` ∈ [-1,1] × `action_scale` **relative to the current TCP**
  (accumulating into the target caused wind-up) + gripper (>0 closes). Targets clamped to a per-arm workspace
  translated by the mount position. Orientation held top-down.
- **Obs** (`ObsLayout`): per arm `q | dq | tcp_pos | finger | cube−tcp` (2n+7), then shared
  `cube_pos | cube_quat | goal | goal−cube` (13). Single 6-DoF arm = 32 dims. Use layout offsets, never literals.
- **Reward**: reach (nearest TCP of any arm) + lifted + lifted·place + 5 on success (`success_tol=0.03`).
- **Reset**: `reset_key` (nworld,) int, 0 = home, -1 = keep; `mjw.reset_data_keyframe` with that mask, then
  `_randomize_kernel` (cube/goal xy, seeded by `seed` + epoch) and `_reset_target_kernel` (OSC target = home TCP).
- Outputs are Warp arrays: `obs`, `reward`, `done`, `success`; `wp.to_torch(env.obs)` is zero-copy.

## Writing a new task

1. Scene: extend `build_scene` or write a sibling builder that reuses `compose_robots` (objects belong to the
   scene, not the robot). Keep a single full-state `home` keyframe.
2. Env: copy the PickPlaceEnv structure. Rules:
   - preallocate everything in `__init__`; `step` = kernel launches only
   - per-world randomness inside kernels (`wp.rand_init(seed, w + epoch * 7919)` in `_randomize_kernel`)
   - masked resets, not full resets
   - one controller per arm; pass the list to `control_substeps`
   - `check_compatible` each controller against the robot's actuator mode
3. Host-side reads (`.numpy()`) only in `__init__` (e.g. home TCP) or tests.
4. Tests: an expert/scripted policy reaching high success on Warp-CPU with few worlds (see `tests/test_env.py`).

## Scripted policies (`scripted.py`) — backend-agnostic

- `ScriptedPickPlace(nworld)`: time-phased waypoints (`PHASES`: above_cube → descend → grasp → lift → to_goal → place → release → retreat),
  `__call__(t, cube_pos0, goal) -> (pos, closed)`. `TOP_DOWN_QUAT` is the tool orientation.
- `BimanualRelay(nworld, handoff, park)`: left arm picks cube → handoff, parks; right arm latches cube pose,
  yaw-aligns its grasp (`box_grasp_yaw`), places at goal. Returns `[(pos, closed, quat)] × 2`.
  Yaw alignment was required: a ~6° cube yaw made the second grasp slip.
- Layout constants `RELAY_CUBE_POS`, `RELAY_GOAL_POS`, `RELAY_HANDOFF`.
- GPU experts for the env are kernels in `examples/_expert.py` (`launch_expert(env, action)`), time-phased.
