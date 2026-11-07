# Wheel-Legged Biped Locomotion with RL

<p align="center"><img src="https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/hero-robot.jpg" alt="WBR wheel-legged biped robot" width="420"></p>

**WBR is a dual-mode wheel-legged biped robot**: two driven wheels with a two-link leg on each
side (hip + knee), plus a passive caster. Four wheel-legged capabilities — two-wheel walking,
jumping, four-wheel driving and on-command mode switching — are learned end-to-end with
reinforcement learning in Isaac Gym.

> **This repository is a personal learning record.** It documents my training runs and the
> checkpoints they produced; it is not a polished framework and comes with no support.
> The code keeps only the WBR robot. The demo robots inherited from upstream `legged_gym`
> (ANYmal, A1, Cassie, …) were removed; the task registry holds 13 WBR tasks.
> [English](README.md) | 中文

## Overview

A reinforcement-learning controller for a **wheel-legged biped**: a torso carried by two wheels,
each with a two-link leg (hip + knee, 3 DoF per side) plus a passive caster.

| Capability | Task | What it does |
|---|---|---|
| Two-wheel walking | `wbr_flat` | Walk and hold height under velocity / yaw / height commands (delivered with 0.1 push and ±5% friction randomization) |
| Two-wheel jumping | `wbr_jump` | Crouch, push off, tuck in the air, absorb; standing jumps of 54.9 / 98.9 / 92.2 mm |
| Rough terrain | `wbr_terrain` | Same policy on random undulation; `wbr_p8_L1~L3` (7 element types × 3 levels), `wbr_p8_combo_L1~L3`, `wbr_p8_demo` (corridor) are scripted evaluation scenes |
| Step-down | `wbr_drop` / `wbr_drop15` | Passive drops without height sensing: 15 cm vertical drops ≥28/32 at every speed (landing rule v2: wheel-lock + slow-speed leg extension in flight) |
| Four-wheel driving | `wbr_flat4` | Static stable stance, both casters down, drive and turn (vx±1 / yaw±1, 15 DR items) |
| Mode switching | `wbr_switch`, `wbr_uni` | Stand up from four-wheel to two-wheel and lie back down; five-slot phase machine, stand-up via the P7T2 platform script chain (throw + catch + dwell handover) |

The terrain acceptance tasks reuse the `wbr_flat` policy — they are scenes, not new training problems.

## Demos

Five behaviours of the same policy weights on both simulation engines
(Isaac Gym = training-native; MuJoCo = the `sim2sim_mujoco` deployment chain).
Images and videos will be attached via a GitHub release (`demo-media-v2`) once recorded.

<!-- To be recorded by the author. Suggested naming:
     fourw-<engine> / twow-<engine> / rise-<engine> / crouch-<engine> / jump-<engine>, engine ∈ {isaac, mujoco} -->
<table>
<tr><th></th><th>Isaac Gym</th><th>MuJoCo</th></tr>
<tr><th>Four-wheel driving</th><td></td><td></td></tr>
<tr><th>Two-wheel walking</th><td></td><td></td></tr>
<tr><th>Stand-up</th><td></td><td></td></tr>
<tr><th>Lie-down (crouch)</th><td></td><td></td></tr>
<tr><th>Jump</th><td></td><td></td></tr>
</table>

## Web console

`wbr_console` is a browser control surface over the simulation, started with `./run_bench.sh`
(→ `http://127.0.0.1:8770`). **Two engines**: Isaac Gym (training-native) and MuJoCo
(the `sim2sim_mujoco` deployment chain — no GPU init, sub-second deterministic stop). Five tabs:

- **Simulation run** — start / stop, live command sliders (Vx / Yaw / Height / Roll / Jump);
  in chain mode one-click stand-up / lie-down / load-jump / back-to-two-wheel / back-to-four-wheel /
  reset / slot-switch, with a command feedback bar and a settle gate (|vx|≤0.08 ∧ |pitch|≤5.7° for
  8 consecutive steps before any network switch), GPU status, and a dock for the native Isaac window; with MuJoCo the 3D viewport is embedded in the page.
- **Configure Isaac / Configure MuJoCo** — slot-based: four-wheel 164 / two-wheel walk 161 /
  two-wheel jump 161 / lie-down 173 / five-slot chain (2-wheel or 4-wheel start), each slot one
  contract + asset + checkpoint; the MuJoCo chain supports a complex-terrain (four-quadrant demo) map;
  stand-up via the P7T2 script chain (default) or the P7S learned policy; per-phase timing parameters.
- **Telemetry plots** — live curves of velocities, wheel/leg states, torques and phase, from the
  running simulation or a saved TSV log.
- **Reference** — parameter glossary, shortcuts and storage paths.

<!-- To be captured by the author: console_run_isaac / console_run_mj / console_cfg_isaac /
     console_cfg_mj / console_plot / console_help (.png) -->
<table>
<tr><th>Simulation run · Isaac Gym</th><th>Simulation run · MuJoCo</th></tr>
<tr><td></td><td></td></tr>
<tr><th>Configure Isaac</th><th>Configure MuJoCo</th></tr>
<tr><td></td><td></td></tr>
<tr><th>Telemetry plots</th><th>Reference</th></tr>
<tr><td></td><td></td></tr>
</table>

## Requirements

| Item | Version / note |
|---|---|
| GPU | NVIDIA (CUDA-capable) |
| OS | Ubuntu 20.04 (tested) |
| Python | 3.8 — Isaac Gym Preview 3 ships binaries for 3.8 only |
| Simulation | **Isaac Gym Preview 3**, from the NVIDIA developer package (not on PyPI) |
| PyTorch | The one bundled with your Isaac Gym Preview 3 environment |
| PPO | `rsl_rl` **1.0.2, already vendored** in `rsl_rl-1.0.2/` — do not `pip install rsl-rl` |

## Installation

```bash
# 1. Isaac Gym (download the package from NVIDIA's developer site first)
cd isaacgym/python && pip install -e . && cd ../..

# 2. this repository
pip install -e .

# 3. plotting / terrain generation
pip install matplotlib numpy scipy
```

If Python cannot find its own shared library:

```
ImportError: libpython3.8.so.1.0: cannot open shared object file
```

export the library path of your environment before running anything:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Check the install — this must print `8`:

```bash
python -c "
import isaacgym, torch
import legged_gym.envs
from legged_gym.utils import task_registry
print(len(task_registry.task_classes))
"
```

> Import `legged_gym.envs` before `legged_gym.utils`: the task registry is filled by the former,
> and `isaacgym` must be imported before `torch`.

## Quick start

```bash
# train (headless; writes to logs/<experiment_name>/<run_name>/)
python legged_gym/scripts/train.py --task wbr_flat --headless

# continue a run for 300 more iterations
python legged_gym/scripts/train.py --task wbr_terrain --resume --max_iterations 300

# MuJoCo deployment chain (no GPU init; the two-wheel-start four-slot chain runs 9.6 s with zero terminations)
python -m legged_gym.scripts.sim2sim_mujoco.deploy.uni_chain --start 2w

# interactive web console (two engines, five slots, P7T2 platform stand-up, plots) → http://127.0.0.1:8770
cd legged_gym/scripts/wbr_console && ./run_bench.sh
```

`play02.py` accepts the iteration suffix only, so copy the release's `model_*.pt` and `config.json`
into your `logs/<task>/` directory and pass the folder name as `--load_run`.

Telemetry and plots: `play02.py` writes TSV segments to `legged_gym/scripts/wbr_play_data/`,
`plot02.py` turns them into a self-contained HTML page.

## Tasks

| Task | Robot | Observations | Actions | Purpose |
|---|---|---|---|---|
| `wbr_flat` | two-wheel (`wbr_v1.urdf`) | 161 | 6 | walk / hold height (randomization delivery base) |
| `wbr_jump` | two-wheel | 161 | 6 | two-wheel task with the jump phase machine (delivered as P8J3d) |
| `wbr_terrain` | two-wheel | 161 | 6 | walking on random undulation (no height map in the observation) |
| `wbr_demo` / `wbr_combo` / `wbr_elements` | two-wheel | 161 | 6 | P8 acceptance scenes: corridor / combined terrain / single-element grid |
| `wbr_drop` / `wbr_drop15` / `wbr_drop15view` | two-wheel | 161 | 6 | step-down training/acceptance maps with the landing rule |
| `wbr_flat4` | four-wheel (`wbr_v2.urdf`) | 164 | 6 | static stable stance, drive and turn |
| `wbr_switch` | four-wheel | 173 | 6 | stand-up / lie-down with phase scripts |
| `wbr_uni` | multi-slot | 161/164/173 | 6 | unified phase machine (0–11), all modes in one env |

The terrain acceptance tasks reuse the `wbr_flat` policy — they are scenes, not new training
problems.

## Repository layout

```
legged_gym/
├── envs/
│   ├── base/          LeggedRobot base class, config defaults
│   └── wbr/           ★ this project
│       ├── flat/      wbr.py, wbr_flat_config.py — two-wheel walking + jumping
│       ├── flat4/     wbr_flat4.py, wbr_flat4_config.py — four-wheel driving
│       ├── switch/    stand-up / lie-down phase machine
│       ├── uni/       unified multi-mode env, FSM, observation builder
│       └── terrain/   offline height-map generators and probes
├── scripts/           train / play / telemetry, and the wbr_console web app
└── utils/             task registry, helpers, terrain, logger
rsl_rl-1.0.2/          bundled PPO implementation
resources/robots/wbr/  URDFs and meshes
releases/              demo weights only (F3e four-wheel, R81 two-wheel/jump, P7L lie-down); full training via train.py
```

## Open issues

Listed first, because they say more about the project than the successes do:

- **Jump height does not follow the command monotonically.** Three standing-jump levels are
  reachable (54.9 / 98.9 / 92.2 mm) but the top level still lands below the middle one — a policy
  capability boundary; the command-to-apex mapping is non-linear.
- **The two engines disagree during transients.** For the same weights, jump apex differs by
  ±25–42 mm between Isaac and MuJoCo (burst-phase transients sensitive to history); walking parity
  is ≤1.3%, and Isaac is the deployment reference.
- **The chain still trails standalone on some metrics.** The five-slot chain passes acceptance
  (C1×3 / C2 / C3), but chain-vs-standalone jump differences are not fully attributed yet.
- **Discrete obstacles are out of scope.** Step-down covers passive drops (15 cm vertical ≥28/32);
  rocks, gaps and step-ups are not attempted.
- **Only simulation.** No hardware numbers, no sim-to-real margin calibration (the Sim2Sim MuJoCo
  chain is in place as an intermediate validation layer; sensitivity study done — the two-wheel
  policy depends strongly on the base_lin_vel observation).
- **Training is fragile.** PPO collapses near the end of long runs; checkpoints are selected
  per-release rather than taken from the last iteration.

## Acknowledgements

- [**legged_gym**](https://github.com/leggedrobotics/legged_gym) by Nikita Rudin, Robotic Systems
  Lab, ETH Zurich — the training framework this repository derives from.
- [**rsl_rl**](https://github.com/leggedrobotics/rsl_rl) v1.0.2 — the PPO implementation used here.
- [**NVIDIA Isaac Gym**](https://developer.nvidia.com/isaac-gym) — the physics simulation backend.
- [**fgmn/Wheel_legged_gym**](https://github.com/fgmn/Wheel_legged_gym) — the wheel-legged robot
  base this work extends.
- [**OpenCode**](https://opencode.ai) — AI assistance for development, refactoring
  and documentation; all code reviewed and owned by the author.

All redistributed code retains its original license; see [LICENSE](LICENSE), [NOTICE](NOTICE) and
`licenses/`.

## License

BSD 3-Clause — see [LICENSE](LICENSE). This is research code; use at your own risk.

---

**Author**: SYH ([@Yihuan01](https://github.com/Yihuan01))
