---
name: sim2real-architecture
description: Map of the sim2real repo (MuJoCo CPU + MJWarp GPU + real robot), its core invariants, and how to verify changes. Use before any non-trivial change to sim2real/, examples/ or tests/, or when deciding where new code belongs.
---

# sim2real architecture

Robot-agnostic sim-to-real stack. One set of controllers runs in three places:
MuJoCo C (CPU, macOS viewer, 1 world), MJWarp (GPU, N worlds, RL), and the real robot.

## Data flow (the one seam everything hangs on)

```
RobotConfig ──► Scene (MjSpec → MjModel) ──► Backend (cpu | warp)
                                                │  step1()
                                                ▼
                                   RobotState (q, dq, tcp_pos, tcp_rot, J, M, bias)
                                                │  controller.compute(state)  [Warp kernel]
                                                ▼
                                   cmd (nworld, n) ──► set_arm_ctrl ──► step2()
Real robot:  hardware.read() → RobotDynamicsModel (mj_forward) → RobotState → same controller → MIT command
```

## Where things live

| Module | Owns | Skill for details |
|---|---|---|
| `sim2real/robots/` | `RobotConfig` (pure data), `RobotMount`, registry, `yam.py` | `sim2real-robot-config` |
| `sim2real/sim/scene.py` | MjSpec composition, actuator-mode rewrite, contact budget | `sim2real-robot-config` |
| `sim2real/sim/robot_view.py` | names → model indices (incl. CSR M index) | `sim2real-backends` |
| `sim2real/sim/state.py` | `RobotState`, `NEED_*` flags | `sim2real-backends` |
| `sim2real/sim/{backend,mujoco_backend,mjwarp_backend}.py` | unified backend interface + 2 impls | `sim2real-backends` |
| `sim2real/dynamics/kernels.py` | MJWarp Data → RobotState gather kernels | `sim2real-backends` |
| `sim2real/control/` | `Controller` base, `JointPD`, `JointImpedance`, `OSC`, `linalg(n)` | `sim2real-controllers` |
| `sim2real/envs/` | batched `PickPlaceEnv` (MJWarp), scripted policies | `sim2real-envs` |
| `sim2real/real/driver.py` | `RobotHardware`, `RobotDynamicsModel`, `RealArm`, `RealControlLoop`, `SimulatedRig` | `sim2real-real-robot` |

## Invariants (do not break)

1. **A robot is data.** No per-robot subclasses; new robots are a `RobotConfig` + `register_robot`.
2. **Controllers are pure** `(RobotState, own targets/gains) -> cmd`, one Warp kernel launch, no host sync.
   They never touch MjData/mjw.Data directly.
3. **Backends fill `RobotState` from their native API**; controllers don't know which backend they're on.
4. **`step1 → controller → step2`**, never `step`, so M/J/bias are at the current state. Integrator must be
   Euler/implicit/implicitfast (not RK4).
5. **Torque mode has no gravcomp**: gravity comes from `bias`, exactly as on hardware.
6. **All arrays are `(nworld, ...)` Warp arrays** on the backend device; CPU backend is `nworld=1` on Warp-CPU.
7. **Multi-arm = tuple of `RobotMount`s**, one controller per arm (exact for fixed bases: block-diagonal M).
8. **No host syncs in the RL hot loop** (`.numpy()`, `body_pos`, `overflow()` are for scripts/tests only).
9. **Pinned versions** (`pyproject.toml`): mujoco / mujoco-warp move fast. Validate API assumptions against the
   installed version (`tests/test_api_contract.py`), don't assume CPU/GPU feature parity.

## Verify a change

```bash
python -m pytest                                   # all tests; MJWarp uses CUDA if present, else Warp-CPU
python examples/scripted_pick.py --headless        # CPU pick-and-place, expect success=1.000, ~6 mm
python examples/bimanual_relay.py --headless       # two-arm relay, expect success=1.000, ~9 mm
python examples/gpu_rollout.py --nworld 16 --device cpu   # batched env path on a Mac
```

Run from the repo root with the project venv active (`pip install -e ".[dev]"`). On macOS the passive viewer
needs `mjpython` instead of `python`.

## Style

2-space indentation, 120-col lines, `from __future__ import annotations` in typed modules, frozen dataclasses for
config. Keep abstractions minimal: add an interface only when a second implementation exists.
