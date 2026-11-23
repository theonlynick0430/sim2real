---
name: sim2real-architecture
description: Map of the sim2real repo (one MJWarp sim path on CUDA or Warp-CPU), its core invariants, and how to verify changes. Use before any non-trivial change to sim2real/, examples/ or tests/, or when deciding where new code belongs.
---

# sim2real architecture

Robot-agnostic sim-to-real stack with a single simulation path: MJWarp. It runs on CUDA (N worlds, RL) and on
Warp-CPU (macOS debugging, tests and the passive viewer) with the same code. There is no MuJoCo C runtime path
and no real-robot code here (that lives in a separate repo).

## Data flow (the one seam everything hangs on)

```
RobotConfig ──► Scene (MjSpec → MjModel) ──► MjWarpBackend (cuda | cpu)
                                                │  step1()
                                                ▼
                                   RobotState (q, dq, tcp_pos, tcp_rot, J, M, bias)
                                                │  controller.compute(state)  [Warp kernel]
                                                ▼
                                   cmd (nworld, n) ──► set_arm_ctrl ──► step2()
```

## Where things live

| Module | Owns | Skill for details |
|---|---|---|
| `sim2real/robots/` | `RobotConfig` (pure data), `RobotMount`, registry, `yam.py` | `sim2real-robot-config` |
| `sim2real/sim/scene.py` | MjSpec composition, actuator-mode rewrite, contact budget | `sim2real-robot-config` |
| `sim2real/sim/robot_view.py` | names → model indices (incl. CSR M index) | `sim2real-backends` |
| `sim2real/sim/state.py` | `RobotState`, `NEED_*` flags | `sim2real-backends` |
| `sim2real/sim/mjwarp_backend.py` | `MjWarpBackend`, the only backend | `sim2real-backends` |
| `sim2real/dynamics/kernels.py` | MJWarp Data → RobotState gather kernels | `sim2real-backends` |
| `sim2real/control/` | `Controller` base, `JointPD`, `JointImpedance`, `OSC`, `linalg(n)` | `sim2real-controllers` |
| `sim2real/envs/` | batched `PickPlaceEnv`, scripted policies | `sim2real-envs` |
| `sim2real/utils.py` | shared host helpers: `broadcast`, `quat_to_mat` (pose math goes through MuJoCo `mju_*`) | — |
| `sim2real/planning/` | GPU IK / motion planning on cuRoboV2 (no collision), MJCF→URDF export. Self-contained: only `adapters.py` may import other sim2real modules (enforced by `tests/test_planning_urdf.py`). Torch/cuRobo imported lazily; Linux + CUDA only | `sim2real-planning` |
| `examples/` | runnable scripts (below); `_scenes.py` / `_expert.py` are shared helpers, not entry points | — |

## Invariants (do not break)

1. **A robot is data.** No per-robot subclasses; new robots are a `RobotConfig` + `register_robot`.
2. **Controllers are pure** `(RobotState, own targets/gains) -> cmd`, one Warp kernel launch, no host sync.
   The backend fills `RobotState` from MJWarp; controllers never touch `mjw.Data` / `MjData`.
3. **`step1 → controller → step2`**, never `step`, so M/J/bias are at the current state. Integrator must be
   Euler/implicit/implicitfast (not RK4).
4. **Torque mode has no gravcomp**: gravity comes from `bias`, exactly as on hardware.
5. **All arrays are `(nworld, ...)` Warp arrays** on the backend device (CUDA, or Warp-CPU on a Mac).
   Don't add a separate MuJoCo C sim path; debug and view MJWarp on Warp-CPU instead.
6. **Multi-arm = tuple of `RobotMount`s**, one controller per arm (exact for fixed bases: block-diagonal M).
7. **No host syncs in the RL hot loop** (`.numpy()`, `body_pos`, `overflow()` are for scripts/tests only).
8. **Pinned versions** (`pyproject.toml`): mujoco / mujoco-warp move fast. Validate API assumptions against the
   installed version (`tests/test_api_contract.py`), don't assume MJWarp supports every MuJoCo C feature.

## Verify a change

```bash
python -m pytest                                   # all tests; CUDA if present, else Warp-CPU
python examples/scripted_pick.py --headless        # pick-and-place, expect success=1.000, ~6 mm
python examples/bimanual_relay.py --headless       # two-arm relay, expect success=1.000, ~5 or ~10 mm
python examples/controller_viewer.py --headless --seconds 5   # osc tracking error ~0.008
python examples/gpu_rollout.py --nworld 16 --device cpu   # batched env path on a Mac
```

Run from the repo root inside the conda env (`conda env create -f environment.yml && conda activate sim2real &&
uv pip install -e ".[dev]"`). Add dependencies to `pyproject.toml`, not to `environment.yml` (conda only provides
Python and uv).

## Running and viewing

| script | what | viewer |
|---|---|---|
| `controller_viewer.py --controller osc\|impedance\|pd` | one arm tracking a moving target | ✅ |
| `scripted_pick.py` | single-arm scripted pick-and-place | ✅ |
| `bimanual_relay.py` | two-arm cube handoff | ✅ |
| `gpu_rollout.py [--bimanual]` | batched `PickPlaceEnv` + expert, success and steps/s | — |
| `gpu_bench.py --nworld ...` | throughput / memory sweep | — |

All take `--device` (default: CUDA if present, else Warp-CPU). Viewer scripts take `--headless`; on macOS run
them with `mjpython`. Viewing is display-only and wrapped by `examples._scenes.Viewer(backend, headless)`: each
`sync(dt)` copies world 0 into a `mujoco.MjData` (`backend.to_mjdata(0, out=...)`) that
`mujoco.viewer.launch_passive` draws, paced to real time; `close()` joins the render thread (`close_viewer`;
avoids exit segfaults). Warp-CPU runs ~350 physics
steps/s for one world, so playback is ~0.7x real time. To view a new script or env, follow the same pattern.

## Style

Prefer official APIs for kinematics, dynamics and pose math (MuJoCo `mj_*` / `mju_*`, MJWarp `mjw.*`, Warp
builtins such as `wp.quat_to_axis_angle`) over hand-written versions; MJWarp's `mujoco_warp._src` is private.
Shared host helpers live in `sim2real/utils.py`; the planning package keeps its own (it must stay self-contained).

2-space indentation, 120-col lines, `from __future__ import annotations` in typed modules, frozen dataclasses for
config. Keep abstractions minimal: add an interface only when a second implementation exists.
