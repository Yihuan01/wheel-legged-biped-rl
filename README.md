# Wheel-Legged Biped Locomotion with RL

![WBR wheel-legged biped robot](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/robot-stand.jpg)

**WBR is a dual-mode wheel-legged biped robot**: two driven wheels with a two-link leg on each
side (hip + knee), plus a passive caster. Four wheel-legged capabilities — two-wheel walking,
jumping, four-wheel driving and on-command mode switching — are learned end-to-end with
reinforcement learning in Isaac Gym.

[中文说明](README.zh-CN.md)

> **This repository is a personal learning record.** It documents my training runs and the
> checkpoints they produced; it is not a polished framework and comes with no support.
> The code keeps only the WBR robot. The demo robots inherited from upstream `legged_gym`
> (ANYmal, A1, Cassie, …) were removed, and the task registry holds 8 WBR tasks.

## Overview

A reinforcement-learning controller for a **wheel-legged biped**: a torso carried by two wheels,
each with a two-link leg (hip + knee, 3 DoF per side) plus a passive caster.

| Capability | Task | What it does |
|---|---|---|
| Two-wheel walking | `wbr_flat` | Walk and hold height on flat ground under velocity / yaw / height commands |
| Two-wheel jumping | `wbr_flat` | Crouch, push off, tuck in the air, absorb on landing |
| Rough terrain | `wbr_terrain` | Same policy on random undulation; `wbr_demo` / `wbr_combo` / `wbr_elements` are scripted evaluation scenes |
| Four-wheel driving | `wbr_flat4` | Static stable stance, both casters down, drive and turn |
| Mode switching | `wbr_switch`, `wbr_uni` | Stand up from four-wheel to two-wheel and lie back down, phase by phase |

## Demos

Each mode as an animated clip, recorded with the released demo weights in Isaac Gym. The GIFs are
attached to the [demo-media-v1 release](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/tag/demo-media-v1)
so cloning the repository does not carry them:

| | |
|---|---|
| **Four-wheel driving** (`P7F3e`) | **Two-wheel walking** (`P6R81`) |
| ![Four-wheel driving](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/fourw_drive.gif) | ![Two-wheel walking](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/twow_walk.gif) |
| **Jump** (`P6R81`) | **Stand-up / lie-down** (`P7T` script + `P7L`) |
| ![Jump](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/jump.gif) | ![Stand-up and lie-down](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/standup.gif) |

## Web console

`wbr_console` is a browser control surface over the simulation, started with `./run_bench.sh`
(→ `http://127.0.0.1:8770`). Four tabs:

- **Simulation run** — start / stop, live command sliders (Vx / Yaw / Height / Roll / Jump),
  one-click stand-up / lie-down / reset / load-jump, GPU status, and a dock for the native
  Isaac Gym window.
- **Model configuration** — the four-slot chain (drive → stand-up → walk → lie-down), each slot
  swappable with any checkpoint whose dimensions match; stand-up via learned policy (P7S) or the
  P7T script chain; per-phase timing parameters.
- **Telemetry plots** — live curves of velocities, wheel/leg states, torques and phase, from the
  running simulation or a saved TSV log.
- **Reference** — parameter glossary, shortcuts and storage paths.

| Simulation run | Model configuration |
|---|---|
| ![Simulation run](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-run.png) | ![Model configuration](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-model.png) |
| **Telemetry plots** | **Reference** |
| ![Telemetry plots](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-plot.png) | ![Reference](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-help.png) |

The console itself ships in the repository; the screenshots above are attached to the
[demo-media-v1 release](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/tag/demo-media-v1)
to keep clones light.

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

# replay a released checkpoint
python legged_gym/scripts/play02.py --task wbr_flat \
    --task wbr_uni --demo_chain

# interactive web console (four modes, phase scripts, plots) → http://127.0.0.1:8770
cd legged_gym/scripts/wbr_console && ./run_bench.sh
```

`play02.py` accepts the iteration suffix only, so copy the release's `model_*.pt` and `config.json`
into your `logs/<task>/` directory and pass the folder name as `--load_run`.

Telemetry and plots: `play02.py` writes TSV segments to `legged_gym/scripts/wbr_play_data/`,
`plot02.py` turns them into a self-contained HTML page.

## Tasks

| Task | Robot | Observations | Actions | Purpose |
|---|---|---|---|---|
| `wbr_flat` | two-wheel (`wbr_v1.urdf`) | 161 | 6 | walk / hold height / jump on flat ground |
| `wbr_terrain` | two-wheel | 161 | 6 | walking on random undulation (no height map in the observation) |
| `wbr_demo` | two-wheel | 161 | 6 | 47 m demo corridor with 9 terrain elements |
| `wbr_combo` | two-wheel | 161 | 6 | three difficulty levels × three element orders |
| `wbr_elements` | two-wheel | 161 | 6 | single-element acceptance grid (8 elements × 3 levels) |
| `wbr_flat4` | four-wheel (`wbr_v2.urdf`) | 164 | 6 | static stable stance, drive and turn |
| `wbr_switch` | four-wheel | 173 | 6 | stand-up / lie-down with phase scripts |
| `wbr_uni` | four-wheel | 173 | 6 | unified phase machine, all modes in one env (replay only) |

The three terrain acceptance tasks reuse the `wbr_flat` policy — they are scenes, not new training
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

Each directory carries its own `README.md` listing what is inside it.

## Versions

[VERSIONS.md](VERSIONS.md) records the versions uploaded to this public repository — one entry
per public upload, with the checkpoints included in that upload. See each `releases/<pkg>/README.md`
for the demo weights shipped here.

## Open issues

Listed first, because they say more about the project than the successes do:

- **Jump height does not increase monotonically with the command.** The top command level produces a
  lower jump than the middle one: the command shapes crouch depth, not take-off velocity.
- **The chain is not as good as standalone.** Standing height in the four-slot chain is ≈ 1.5 cm
  lower than a standalone policy, and jumps lose 1–2 cm.
- **Obstacle terrain is unsolved.** Discrete steps and gaps above a few centimetres still end in a
  fall.
- **Only simulation.** No hardware numbers, no sim-to-real margin calibration.
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
