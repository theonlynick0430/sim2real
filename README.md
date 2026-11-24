# sim2real

Robot-agnostic sim-to-real infrastructure on **MuJoCo Warp**. There is one simulation path: it runs on CUDA for
training and on Warp-CPU for debugging and viewing on a Mac.

- The default robot is the I2RT YAM; its model is vendored from MuJoCo Menagerie under the MIT license.
- Tasks: single-arm pick-and-place, and a bimanual YAM relay (left arm hands the cube to the right arm via a handoff spot).

## How to run

**1. Install** with conda (Python 3.12 + uv, from `environment.yml`) and uv (project deps), from the repo root:
```bash
conda env create -f environment.yml
conda activate sim2real
uv pip install -e ".[dev]"         # pins mujoco / mujoco-warp 3.17.0, warp-lang 1.18.0
```
Run every command below inside the activated `sim2real` env.

**2. Test** — 24 tests, ~80 s on a Mac (the cuRobo planning suite skips without a CUDA GPU). Tests use CUDA if present, else Warp-CPU:
```bash
python -m pytest
```

**3. View on a Mac (MJWarp on Warp-CPU + passive viewer of world 0)** — the viewer needs `mjpython` instead of
`python`. Warp-CPU manages ~350 physics steps/s for one world, so playback is a little slower than real time:
```bash
mjpython examples/controller_viewer.py --controller osc   # hold/track with osc | impedance | pd
mjpython examples/scripted_pick.py                        # single-arm OSC pick-and-place
mjpython examples/bimanual_relay.py                       # two YAMs hand off the cube
python examples/scripted_pick.py --headless               # no viewer (remote / CI)
```

**4. Run on GPU (NVIDIA, many worlds)** — the same scripts; the device defaults to CUDA when present. Add
`--device cpu` to debug on a Mac:
```bash
python examples/scripted_pick.py --nworld 1024 --headless
python examples/bimanual_relay.py --nworld 1024 --headless
python examples/gpu_rollout.py --nworld 4096 [--bimanual]          # batched RL env + scripted expert
python examples/gpu_bench.py --nworld 1024 65536 262144 1048576 [--bimanual]   # throughput / memory sweep
```

**5. GPU IK / motion planning (cuRoboV2, Linux + NVIDIA GPU only)** — standalone, not wired into the envs or
controllers yet. Collision checking is disabled. The arm is exported from MJCF to a kinematic URDF (cuRobo only
reads URDF); `tests/test_planning_urdf.py` checks that export on any machine:
```bash
uv pip install -e ".[dev,planning]"           # cuRobo pinned in pyproject (cu12; use cu13-torch for CUDA 13)
python -m pytest tests/test_planning_curobo.py -v
python examples/plan_demo.py --headless --batch 32   # IK benchmark + sequential and batched planning
```
```python
from sim2real.planning import GpuIK, GpuMotionPlanner
from sim2real.planning.adapters import planner_config_from_robot
cfg = planner_config_from_robot(get_robot("yam"), "/tmp/planning")   # poses in the robot base frame, wxyz
res = GpuIK(cfg.replace(max_batch_size=256)).solve(pos, quat)        # res.q, res.success, res.pos_err, res.rot_err
mp = GpuMotionPlanner(cfg); mp.warmup()
traj = mp.plan_pose(q_start, pos, quat)       # or mp.plan_joint(q_start, q_goal); batch=True for many at once
traj.q, traj.dq, traj.ddq, traj.dt, traj.length, traj.success   # (B, T, dof) at cfg.interpolation_dt
```
`sim2real/planning/` imports nothing else from sim2real except `adapters.py`, so it can be moved out as a package.

**6. Motor reference profiles for dyno experiments (Linux + NVIDIA GPU: uses the planner)** — YAM trajectories
composed of planned skills (`joint_move` under computed torque, `ee_move` under OSC) with a payload on the tool
whose mass and center of mass vary, recorded per motor at 500 Hz:
```bash
python examples/motor_profiles.py --out data/profiles --num-trajs 200            # generate a dataset
python examples/motor_profiles.py --out data/profiles --num-trajs 200 --resume   # continue a stopped run
python examples/motor_profiles.py --visualize --no-record --num-trajs 5          # watch, write nothing
```
- `--joint-weight` / `--ee-weight` set the skill mix; `--help` lists the payload, margin and speed options.
- Each `traj_XXXXX.npz` holds `t`, `q`, `dq`, `tau_motor`, `tau_ext` (per arm joint; Nm at the joint output),
  `skill` (index into the skill list, -1 while pausing) and `meta` (JSON: skill composition, payload per
  skill, motor per joint). `session.json` lists every trajectory and is written on success, failure or Ctrl-C.
- The simulated motors are ideal: reflected rotor inertia (armature), no joint friction.
- `tau_ext` is the torque the rest of the arm and the payload put on the motor's output. Rotor inertia counts
  as part of the motor and is not in it (`tau_motor = armature * qacc - tau_ext`): on a dyno, drive the motor
  along `q`/`dq` while the load side applies `tau_ext`; an ideal motor then needs `tau_motor`.
- In the viewer the orange sphere on the tool is centered on the payload's center of mass; its opacity is the
  mass relative to `--payload-mass-max`.
- About 2.5 s of wall time per simulated second (one world, stepped from the host), i.e. ~15 s per trajectory.

> **macOS note:** Python 3.12 ignores `.pth` files that carry the `hidden` flag. On some setups the editable install ends up hidden, and then `import sim2real` fails. The examples add the repo to `sys.path` themselves, and pytest sets `pythonpath`.

## Agent skills

`.claude/skills/` holds project skills that give coding agents (e.g. Claude Code) the context to extend this
repo. They're checked in, so everyone who clones the repo gets them, and they contain no user- or machine-specific paths.

| skill | covers |
|---|---|
| `sim2real-architecture` | module map, invariants, how to verify changes — start here |
| `sim2real-robot-config` | adding robots (`RobotConfig`), scene building, actuator modes, multi-arm mounts |
| `sim2real-controllers` | writing controllers as Warp kernels, equations, testing |
| `sim2real-backends` | the MJWarp backend, `RobotState` filling, validated MJWarp API, CUDA graphs, perf |
| `sim2real-envs` | batched RL env structure, obs/action/reward/reset, scripted experts |
| `sim2real-planning` | GPU IK / motion planning on cuRoboV2: API, MJCF→URDF export, pinned cuRobo facts, GPU testing |

Keep them in sync when you change the corresponding code.

## Layout

```
sim2real/
  robots/    config.py   RobotConfig (frozen dataclass) + registry; RobotMount (config + prefix + base pose)
             yam.py      YAM config; yam_bimanual() -> two mounts
  sim/       scene.py    MjSpec scene: mounted robot(s) + table + cube + goal; actuator-mode rewrite; contact budget
             robot_view.py  RobotConfig -> model indices (qpos/dof/actuator ids, TCP site, CSR M index)
             state.py    RobotState: q, dq, tcp_pos, tcp_rot, J (6xn), M (nxn), bias  [warp arrays, (nworld, ...)]
             mjwarp_backend.py   MjWarpBackend: N worlds on CUDA or Warp-CPU, per-arm views/states,
                                 CUDA-graph-captured control substeps
  dynamics/  kernels.py  MJWarp -> RobotState extraction (CSR->dense M, jac gather, bias, TCP)
  control/   linalg.py   per-DoF fixed-size Cholesky / SPD inverse (warp funcs, generated per n)
             joint.py    JointPD (position or torque), JointImpedance (optionally inertia-shaped)
             osc.py      Operational-space control: one fused kernel per world
  envs/      pick_place.py  batched RL env (MJWarp), 1..N arms: obs / reward / masked reset as kernels
             scripted.py    host-side scripted pick-and-place + BimanualRelay
  planning/  (self-contained; only adapters.py imports sim2real)
             mjcf_to_urdf.py  MuJoCo chain (base body -> TCP site) -> kinematic + inertial URDF
             config.py   PlannerConfig -> cuRobo robot dict without collision spheres
             ik.py       GpuIK: batched cuRobo IK        motion.py  GpuMotionPlanner: plan_pose / plan_joint
             fk.py       host FK of the URDF via MuJoCo  frames.py  world <-> base pose transforms
             adapters.py RobotConfig -> URDF + PlannerConfig
  profiles/  scene.py    robot-only scene + payload body on the tool; set_payload (mass, COM)
             sampling.py start configs, joint / ee goals, payloads; Checker (limits, contacts, singularity)
             skills.py   ProfileGenerator: plan + roll out joint_move / ee_move, record q, dq, tau_motor, tau_ext
             session.py  traj_XXXXX.npz + session.json, resume
```

## Key design decisions

**1. A robot is data, not a class hierarchy.**
- `RobotConfig` holds everything about a robot:
  - model path (MJCF or URDF, both loaded through `MjSpec.from_file`)
  - arm joints and actuators
  - TCP site, plus `extra_sites` for frames the asset lacks
  - home pose, torque limits, gains
  - gripper settings and contact excludes
  - actuator mode
- Adding Franka or UR7e means writing one config and calling `register_robot`.
- `RobotView` resolves names to indices once and owns no physics.

**2. Dynamics are computed at the backend/state seam.**
- Controllers are pure `(RobotState, targets, gains) -> command`.
- The backend fills `RobotState` from MJWarp:

| quantity | MJWarp |
|---|---|
| M | `d.M` is **CSR lower-triangular `(nworld, nC)`**; we expand it with a precomputed index table (`RobotView.m_csr_index`, built with MuJoCo's `mj_fullM`) |
| J (TCP) | `mjw.jac(m, d, jacp, jacr, point, body)`, a host-level launch over all worlds that returns all `nv` columns; we gather the arm's columns |
| bias = C·q̇ + g | `d.qfrc_bias` |
| Λ, J⁺, null space | custom, inside the controller kernel |

- `tests/test_dynamics_parity.py` checks these against MuJoCo C (`mj_fullM`, `mj_jacSite`) as an independent reference. MuJoCo C is not used at runtime.

**3. Controllers are written once, as Warp kernels.**
- On GPU, a controller is one fused kernel per world, placed inside the CUDA graph between `step1` and `step2`. There are no host syncs.
- On a Mac, the same kernels and the same MJWarp physics run on Warp's CPU device (no graph capture), at about 3 ms per physics step for one world.
- Custom kernels are needed because MJWarp has no task-space quantities. Per-op torch/linalg calls would also cost far more than one register-resident 6×6 / n×n solve per thread.
- Matrix types are generated per DoF count (`linalg(n)`), so 6- and 7-DoF arms share code.

**4. `step1 → controller → step2`, not `step`.**
- M, J, and bias are then evaluated at the *current* state at no extra cost, with no one-step lag.
- This requires the Euler or implicit/implicitfast integrator, not RK4.

**5. The actuator model is a robot-config choice.**
- **TORQUE mode:**
  - `motor` actuators, with gravity compensation (`gravcomp`) disabled.
  - Gravity comes from `bias`, exactly as on hardware.
  - Required by OSC and impedance control.
- **POSITION mode:** MuJoCo's implicit position actuators, used by `JointPD` with the configured kp/kd.
- Each controller declares `output`, and `check_compatible()` catches mismatches.

**6. Multiple arms are mounts, controlled independently.**
- A `RobotMount` is a robot config plus a name prefix and a base pose. A bimanual rig is a tuple of mounts (`yam_bimanual(separation)`).
- `compose_robots` attaches each robot's spec with `MjSpec.attach(prefix=...)`. A single un-prefixed robot takes a direct path instead, so single-arm names are unchanged.
- The backend holds one `RobotView` and one `RobotState` per arm. `robot_state`, `set_arm_ctrl` and `set_gripper` take `arm=`. `control_substeps` takes one controller per arm and runs them all between the same `step1` and `step2`; on CUDA they share one graph.
- **Why separate per-arm controllers are exact:** fixed-base arms share no joints, so the mass matrix is block-diagonal (asserted in `tests/test_bimanual.py`). Two 6-DoF OSCs produce the same torques as one 12-DoF OSC, at a fraction of the cost. Arms with a shared torso or base would need a whole-body controller instead.
- **Coordination lives above the controllers:** `BimanualRelay` for scripts, the policy in the env.

**7. Control rates.**
- Physics and the low-level controller run at 500 Hz (`dt = 2 ms`).
- The policy runs at 50 Hz (`decimation = 10`) and only updates targets.
- The env's actions are TCP deltas relative to the *current* TCP. Accumulating deltas into the target caused wind-up.

## MuJoCo / MJWarp API notes (validated against 3.17.0; enforced by `tests/test_api_contract.py`)

- **The `qM` → `M` (CSR, `nC`) rename** happened in both MuJoCo C and MJWarp. `mj_fullM` (used for the CSR index table and by the parity test) is now `(m, d, dst)`.
- **`make_data(nconmax, njmax, nccdmax)`:**
  - `nconmax` is per world, drawn from a shared pool `naconmax = nconmax·nworld`.
  - `njmax` is a hard per-world limit.
  - `nccdmax` is the per-world share of the CCD contact pool (`naccdmax = nccdmax·nworld`), used only by geom pairs without a closed-form collider (box-box here). It defaults to `nconmax`, and the convex narrowphase allocates about 3 KB of scratch per slot on every call.
  - MJWarp prints overflow warnings, and overflowed contacts are dropped. It also sets `OverflowType` bits in `d.overflow` per world, but `reset_data_keyframe` zeroes them for the worlds it resets, so the backend accumulates them (`MjWarpBackend.reset_data` + `overflow()`).
- **Exposed per-step pieces:** `step1`/`step2`, `kinematics`, `reset_data_keyframe(m, d, key_array)` for a per-world masked reset, and `get_data_into` for viewing one world.
- **MJWarp runs on Warp-CPU on macOS.** It is slow (about 350 physics steps/s for one world, 3k at 16 worlds) but correct, and is how this repo is tested, debugged and viewed on a Mac.
- **Unsupported in MJWarp:** the PGS solver, plugins, and autodiff. RK4 does not work with `step1`/`step2`.
- **`MjSpec.attach(child, prefix, frame=...)`** prefixes every name, including contact excludes and equality constraints.
  - It warns on an empty prefix, which is why a single robot is not attached.
  - It also warns when child and parent options differ; the scene sets options after composing.

## Performance notes and bottlenecks

- **Contacts dominate the budget.**
  - Bimanual relay peaks at 26 contacts and 125 constraint rows (measured on MuJoCo C), within the same 48/192 budget: the arms work sequentially and touch one cube.
  - The YAM fingers are built from about 13 small collision primitives each, so grasping the cube peaks at **36 contacts and 153 constraint rows** (measured on MuJoCo C).
  - `SceneConfig` sets `nconmax=48`, `njmax=192`, `nccdmax=16`.
  - Data memory is about **46 KB per world**, so 1M worlds needs about 46 GB, which fits an 80 GB A100/H100.
  - **CCD scratch is the transient peak.** With `nccdmax` left at its default of `nconmax`, the narrowphase scratch made the high-water mark about 4x the steady state (190 KB/world), so a 24 GB RTX 4090 ran out of memory above 65536 worlds. The measured CCD demand is 2 to 4 slots per world during a grasp (the expert sets the CCD overflow bit at 2, not at 4); `nccdmax=16` keeps 4x margin and lets the 4090 run 131072 single-arm worlds at about 8M physics steps/s.
  - With 16/64 buffers it is about 24 KB per world. That requires trimming the finger colliders first (the next optimization).
- **Graph capture:** `control_substeps` is captured once per (controller, decimation) and replayed. The env's per-step kernels (action, reward, reset, obs, plus 2× `kinematics`) run outside the graph and could be folded in.
- **Host syncs:** `MjWarpBackend.body_pos`, `set_body_free_pose`, and `overflow()` read device memory. They are for scripts only and should not be called in the RL loop.
- **Check `overflow()` after any change to contacts or buffers.** A dropped contact does not fail the task: with the CCD pool at zero the expert still places the cube (the sphere and capsule finger contacts carry the grasp) while the box-box contacts silently vanish. `overflow()["ccd"]`, `["contacts"]` and `["nefc"]` are the bits that mean something was dropped; `tests/test_env.py` asserts all three are clear after an expert episode. `ITERATIONS` only means the solver hit `solver_iterations`, which the bimanual relay does routinely.
- **Precision:** physics is float32 (on CUDA and Warp-CPU alike). Λ uses a damped inverse (`lambda_damping`), and torques are clamped to actuator limits.
- **Friction:** YAM joints have no joint friction (`frictionloss=0`). Menagerie's model set an unidentified placeholder of 0.1 Nm on every joint; it was removed here until friction is measured on the real motors. OSC error is below 0.3 mm.

## Status

| | MJWarp on Warp-CPU (Mac) | MJWarp on CUDA |
|---|---|---|
| state parity (M, J, bias, TCP) vs MuJoCo C reference | ✅ | not yet run |
| OSC / impedance / PD tracking | ✅ (same errors as the former MuJoCo C path) | not yet run |
| scripted pick-and-place | ✅ 6 mm placement | not yet run |
| batched env + expert | ✅ 16/16 success | not yet run; **graph-capture path untested** |
| bimanual: independent OSC, scripted relay | ✅ relay places within 10 mm | not yet run |
| bimanual batched env + relay expert | ✅ 4/4 success | not yet run |

**Next steps:**
- Run `pytest` and `gpu_bench.py` on the GPU box.
- Trim finger colliders to shrink the contact budget.
- Domain randomization through MJWarp's per-world model fields (`body_mass`, `geom_friction`, `actuator_gainprm`, …).
