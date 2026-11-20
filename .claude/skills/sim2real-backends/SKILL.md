---
name: sim2real-backends
description: The sim2real Backend interface (MuJoCo C CPU and MJWarp GPU), how RobotState is filled on each, validated MuJoCo/MJWarp 3.17 API facts, CUDA graph capture, and GPU performance rules. Use when touching sim/, dynamics/, bumping mujoco/mujoco-warp, or optimizing throughput.
---

# Backends (`sim2real/sim/`, `sim2real/dynamics/`)

## Interface (`backend.py`)

`make_backend("cpu" | "warp", scene, nworld=1, device=None)`. Both implement:

- physics: `reset()`, `step1()`, `step2()`, `control_substeps(ctrls, n)` (n × [step1, per-arm controller →
  `set_arm_ctrl`, step2])
- robot I/O (per arm): `robot_state(needs, arm=0)`, `set_arm_ctrl(cmd, arm=0)`, `set_gripper(value, arm=0)`
- `views` / `states` tuples (one per arm, `scene.mounts` order); `view` / `state` / `narm` shortcuts
- host accessors — **sync points, scripts/tests only**: `body_pos`, `body_quat`, `set_mocap_pos`,
  `set_body_free_pose`, `to_mjdata(world, out)`

`RobotView.from_model(mjm, cfg)` resolves names → `qpos_adr`, `dof_adr`, `act_ids`, `tcp_site_id`,
`tcp_body_id`, `gripper_act_id`, and `m_csr_index` (dense n×n → CSR M index, -1 = structural zero).

## How each backend fills RobotState

| quantity | `MujocoBackend` (CPU, float64 physics) | `MjWarpBackend` (N worlds, float32) |
|---|---|---|
| M | `mj_fullM(m, d, dst)` then sub-block | `d.M` is CSR lower-tri `(nworld, nC)` → `gather_mass_kernel` via `m_csr_index` |
| J | `mj_jacSite` | `mjw.jac(m, d, jacp, jacr, point, body)` (host-level batched, all nv cols) → `gather_jac_kernel` |
| bias | `qfrc_bias` | `d.qfrc_bias` |
| TCP | `site_xpos/xmat` | `d.site_xpos/xmat` |

`fill_state_from_mjdata` (CPU) is shared with the real-robot dynamics model. CPU writes through zero-copy
`.numpy()` views of Warp-CPU arrays.

## Validated API facts (mujoco / mujoco-warp 3.17.0; guarded by `tests/test_api_contract.py`)

- `qM` → `M` (CSR, `nC`) rename in both; `mj_fullM(m, d, dst)`.
- `mjw.make_data(mjm, nworld, nconmax, njmax, nccdmax)`: `nconmax` per world from a shared pool
  (`naconmax = nconmax·nworld`); `njmax` hard per-world limit; `nccdmax` per-world share of the CCD pool
  (`naccdmax = nccdmax·nworld`, defaults to `nconmax`), used only by convex (GJK/EPA) pairs — box-box here.
  The convex narrowphase allocates ~3 KB scratch per CCD slot on every call, so a large `nccdmax` makes the
  transient peak ~4x the steady-state memory. **Overflow drops contacts** and ORs `OverflowType` bits into
  `d.overflow` (per world); `reset_data_keyframe` zeroes the bits of the worlds it resets.
- `mjw.step1/step2`, `kinematics`, `forward`, `reset_data_keyframe(m, d, key | per-world int array, -1 = keep)`,
  `get_data_into(mjd, mjm, d, world_id=)`.
- Unsupported in MJWarp: PGS solver, plugins, autodiff; RK4 incompatible with step1/step2.
- MJWarp runs on Warp-CPU (slow, ~3k steps/s at 16 worlds) — fine for parity tests on macOS.
- When bumping versions: update pins together, run `test_api_contract.py` and `test_dynamics_parity.py` first.
  Check the installed source (`python -c "import mujoco_warp, inspect; ..."`) rather than assuming.

## CUDA graphs (`MjWarpBackend.control_substeps`)

On CUDA, the first call per `(controller ids, n)` warms up (compiles kernels), then captures
`n × [step1, ctrl, step2]` with `wp.ScopedCapture` and replays it with `wp.capture_launch`. The cache keeps the
controller objects alive (id reuse + baked buffer addresses). Consequences: never reallocate buffers used inside
`compute`; never put host syncs or Python data-dependent branches in the captured path. `use_graph=False` disables
it for debugging.

## Performance rules (1M-world target)

- Contacts dominate. YAM fingers peak at 36 contacts / 153 constraint rows grasping a cube (bimanual 26/125).
  Data ≈ 46 KB/world at 48/192 → 1M worlds ≈ 46 GB. Trimming finger colliders is the next lever.
- Use `contact_excludes` / contype masks; primitives over meshes.
- `overflow()` (host sync) reports `nacon`/`nefc` usage plus the accumulated `OverflowType` flags:
  `ccd` / `contacts` / `nefc` mean a contact or constraint was silently dropped (the task can still succeed —
  the expert places the cube with the CCD pool at zero). The backend ORs `d.overflow` into an accumulator
  in `reset_data()` before MJWarp zeroes it, so **always reset through `backend.reset_data(mask)`**, never
  `mjw.reset_data_keyframe` directly. Check it in debug runs and tests, not in the loop.
- CCD demand measured with the experts: 2–4 slots/world (overflow bit set at `nccdmax=2`, clear at 4);
  `SceneConfig.nccdmax=16` keeps 4x margin and fits 131072 single-arm worlds on a 24 GB RTX 4090.
- Fuse work into one kernel per world; avoid temporaries; `wp.to_torch` is zero-copy for RL frameworks.
- Domain randomization: MJWarp per-world model fields (`body_mass`, `geom_friction`, `actuator_gainprm`, …).
