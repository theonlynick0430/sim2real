# sim2real

Robot-agnostic sim-to-real infrastructure on **MuJoCo C (CPU)** and **MuJoCo Warp (GPU)**.

- The default robot is the I2RT YAM; its model is vendored from MuJoCo Menagerie under the MIT license.
- Tasks: single-arm pick-and-place, and a bimanual YAM relay (left arm hands the cube to the right arm via a handoff spot).

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"     # pins mujoco / mujoco-warp 3.17.0, warp 1.18.0
.venv/bin/python -m pytest                                     # 22 tests; MJWarp tests use CUDA if present, else Warp-CPU

# CPU (macOS): viewer + scripted motion  (macOS passive viewer needs `mjpython`)
.venv/bin/mjpython examples/cpu_viewer.py --controller osc     # or impedance | pd
.venv/bin/mjpython examples/scripted_pick.py                   # OSC pick-and-place, MuJoCo C
.venv/bin/mjpython examples/bimanual_relay.py                  # two YAMs, OSC on both arms

# GPU: same scripted pick on MJWarp, batched RL env rollout, throughput sweep
python examples/scripted_pick.py --backend warp --nworld 1024 --headless
python examples/gpu_rollout.py --nworld 4096                   # add --device cpu to debug on a Mac
python examples/gpu_rollout.py --nworld 4096 --bimanual        # bimanual relay env + expert
python examples/bimanual_relay.py --backend warp --nworld 1024 --headless
python examples/gpu_bench.py --nworld 1024 65536 262144 1048576   # add --bimanual for two arms per world
```

> **macOS note:** Python 3.12 ignores `.pth` files that carry the `hidden` flag. On some setups the editable install ends up hidden, and then `import s2r` fails. The examples add the repo to `sys.path` themselves, and pytest sets `pythonpath`.

## Layout

```
s2r/
  robots/    config.py   RobotConfig (frozen dataclass) + registry; RobotMount (config + prefix + base pose)
             yam.py      YAM config; yam_bimanual() -> two mounts
  sim/       scene.py    MjSpec scene: mounted robot(s) + table + cube + goal; actuator-mode rewrite; contact budget
             robot_view.py  RobotConfig -> model indices (qpos/dof/actuator ids, TCP site, CSR M index)
             state.py    RobotState: q, dq, tcp_pos, tcp_rot, J (6xn), M (nxn), bias  [warp arrays, (nworld, ...)]
             backend.py  Backend interface (per-arm views/states) + make_backend("cpu" | "warp")
             mujoco_backend.py   MuJoCo C, 1 world (float64 physics)
             mjwarp_backend.py   MJWarp, N worlds, CUDA-graph-captured control substeps
  dynamics/  kernels.py  MJWarp -> RobotState extraction (CSR->dense M, jac gather, bias, TCP)
  control/   linalg.py   per-DoF fixed-size Cholesky / SPD inverse (warp funcs, generated per n)
             joint.py    JointPD (position or torque), JointImpedance (optionally inertia-shaped)
             osc.py      Operational-space control: one fused kernel per world
  envs/      pick_place.py  batched RL env (MJWarp), 1..N arms: obs / reward / masked reset as kernels
             scripted.py    backend-agnostic scripted pick-and-place + BimanualRelay
  real/      driver.py   RobotHardware (MIT-mode contract), RobotDynamicsModel, RealArm, RealControlLoop,
                         SimulatedRig / SimulatedHardware
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
- `RobotView` resolves names to indices once. It owns no physics, so the CPU backend, GPU backend, and real-robot model all share it.

**2. Dynamics are computed at the backend/state seam.**
- Controllers are pure `(RobotState, targets, gains) -> command`.
- Each backend fills `RobotState` from its own native API:

| quantity | MuJoCo C (CPU / real) | MJWarp (GPU) |
|---|---|---|
| M | `mj_fullM` | `d.M` is **CSR lower-triangular `(nworld, nC)`**; we expand it with a precomputed index table (`RobotView.m_csr_index`) |
| J (TCP) | `mj_jacSite` | `mjw.jac(m, d, jacp, jacr, point, body)`, a host-level launch over all worlds that returns all `nv` columns; we gather the arm's columns |
| bias = C·q̇ + g | `qfrc_bias` | `d.qfrc_bias` |
| Λ, J⁺, null space | — (custom) | — (custom, inside the controller kernel) |

- The real robot uses the same MJCF, robot only, through MuJoCo C (`RobotDynamicsModel`). So "sim dynamics" and "robot dynamics" come from one model.

**3. Controllers are written once, as Warp kernels.**
- On GPU, a controller is one fused kernel per world, placed inside the CUDA graph between `step1` and `step2`. There are no host syncs.
- On a Mac or the real robot, the same kernels run on Warp's CPU device, at about 50 µs per physics step, including MuJoCo.
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
- On hardware both modes map onto MIT-mode commands, `τ = kp(q*−q) + kd(q̇*−q̇) + τ_ff`.

**6. Multiple arms are mounts, controlled independently.**
- A `RobotMount` is a robot config plus a name prefix and a base pose. A bimanual rig is a tuple of mounts (`yam_bimanual(separation)`).
- `compose_robots` attaches each robot's spec with `MjSpec.attach(prefix=...)`. A single un-prefixed robot takes a direct path instead, so single-arm names are unchanged.
- Backends hold one `RobotView` and one `RobotState` per arm. `robot_state`, `set_arm_ctrl` and `set_gripper` take `arm=`. `control_substeps` takes one controller per arm and runs them all between the same `step1` and `step2`; on CUDA they share one graph.
- **Why separate per-arm controllers are exact:** fixed-base arms share no joints, so the mass matrix is block-diagonal (asserted in `tests/test_bimanual.py`). Two 6-DoF OSCs produce the same torques as one 12-DoF OSC, at a fraction of the cost. Arms with a shared torso or base would need a whole-body controller instead.
- **Coordination lives above the controllers:** `BimanualRelay` for scripts, the policy in the env.
- **Real rigs:** each arm's `RobotDynamicsModel` is placed at its mount pose, so the TCP pose and J are in the same world frame as the sim, and the same targets work in both.
  - `RealControlLoop` reads every arm, then computes, then commands.
  - `SimulatedRig` advances physics once per tick.

**7. Control rates.**
- Physics and the low-level controller run at 500 Hz (`dt = 2 ms`).
- The policy runs at 50 Hz (`decimation = 10`) and only updates targets.
- The env's actions are TCP deltas relative to the *current* TCP. Accumulating deltas into the target caused wind-up.

## MuJoCo / MJWarp API notes (validated against 3.17.0; enforced by `tests/test_api_contract.py`)

- **The `qM` → `M` (CSR, `nC`) rename** happened in both MuJoCo C and MJWarp. `mj_fullM` is now `(m, d, dst)`.
- **`make_data(nconmax, njmax)`:**
  - `nconmax` is per world, drawn from a shared pool `naconmax = nconmax·nworld`.
  - `njmax` is a hard per-world limit.
  - MJWarp prints overflow warnings, and overflowed contacts are dropped.
- **Exposed per-step pieces:** `step1`/`step2`, `kinematics`, `reset_data_keyframe(m, d, key_array)` for a per-world masked reset, and `get_data_into` for viewing one world.
- **MJWarp runs on Warp-CPU on macOS.** It is slow (about 3k physics steps/s at 16 worlds) but correct, and is used here for parity tests and debugging.
- **Unsupported in MJWarp:** the PGS solver, plugins, and autodiff. RK4 does not work with `step1`/`step2`.
- **`MjSpec.attach(child, prefix, frame=...)`** prefixes every name, including contact excludes and equality constraints.
  - It warns on an empty prefix, which is why a single robot is not attached.
  - It also warns when child and parent options differ; the scene sets options after composing.

## Performance notes and bottlenecks

- **Contacts dominate the budget.**
  - Bimanual relay peaks at 26 contacts and 125 constraint rows (CPU), within the same 48/192 budget: the arms work sequentially and touch one cube.
  - The YAM fingers are built from about 13 small collision primitives each, so grasping the cube peaks at **36 contacts and 153 constraint rows** (measured on CPU).
  - `SceneConfig` sets `nconmax=48`, `njmax=192`.
  - Data memory is about **46 KB per world**, so 1M worlds needs about 46 GB, which fits an 80 GB A100/H100.
  - With 16/64 buffers it is about 24 KB per world. That requires trimming the finger colliders first (the next optimization).
- **Graph capture:** `control_substeps` is captured once per (controller, decimation) and replayed. The env's per-step kernels (action, reward, reset, obs, plus 2× `kinematics`) run outside the graph and could be folded in.
- **Host syncs:** `Backend.body_pos`, `set_body_free_pose`, and `overflow()` read device memory. They are for scripts only and should not be called in the RL loop.
- **Precision:** physics is float32 on GPU and float64 on CPU. Λ uses a damped inverse (`lambda_damping`), and torques are clamped to actuator limits.
- **Friction:** YAM joints have `frictionloss=0.1`, which creates a steady-state deadband, so the OSC gains default high (`kp_pos=400`). With friction removed, OSC error is below 0.3 mm.

## Status

| | CPU (MuJoCo C, Mac) | MJWarp on Warp-CPU (Mac) | MJWarp on CUDA |
|---|---|---|---|
| state parity (M, J, bias, TCP) | reference | ✅ matches CPU | not yet run |
| OSC / impedance / PD tracking | ✅ | ✅ | not yet run |
| scripted pick-and-place | ✅ 6 mm placement | ✅ 4/4 worlds | not yet run |
| batched env + expert | — | ✅ 16/16 success | not yet run; **graph-capture path untested** |
| bimanual: parity, independent OSC, scripted relay | ✅ relay places within 9 mm | ✅ parity; relay 2/2 worlds | not yet run |
| bimanual batched env + relay expert | — | ✅ 4/4 success | not yet run |
| real robot | `RealControlLoop` verified on `SimulatedRig` (1 and 2 arms); dynamics model at mount == sim; `I2RTYamHardware` is a stub | | |

**Next steps:**
- Run `pytest` and `gpu_bench.py` on the GPU box.
- Trim finger colliders to shrink the contact budget.
- Domain randomization through MJWarp's per-world model fields (`body_mass`, `geom_friction`, `actuator_gainprm`, …).
- Wire the i2rt SDK into `RobotHardware`.
