# Attention-Guided Evolutionary Art

## Overview
This repository reconstructs target images with ordered alpha-blended geometric primitives. The active live reconstruction module is now [python/src/live_refiner.py](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\src\live_refiner.py).

The current implementation includes:
- A two-pass live optimizer with separate appearance and geometry rollback.
- Grid-seeded initialization for the coarse round.
- A fixed four-round live refiner schedule at 50, 100, and 200 resolution.
- Diverse high-error region sampling for polygon insertion.
- Structure-guided aspect ratios and thin-stroke support.
- Round-to-round palette refinement in LAB space.
- A live display path plus a headless execution path.
- Hard runtime limits and a `<30s remaining` switch to final-only optimization.

## Current Status
The code paths above are implemented and tested, but the 5-minute reconstruction quality is still limited by the cost of finite-difference geometry updates.

The latest measured 5-minute `grape.jpg` run at `200x200` produced:
- `rgb_mse = 0.07396`
- `ssim = 0.15606`
- `psnr = 11.31 dB`
- `gradient_mse = 0.01458`
- `gradient_corr = -0.02610`
- `polygon_count = 32`
- stage coverage: only round `A` completed under the 5-minute cap on this machine

Output artifacts from that run:
- Image: [python/outputs/refiner_eval/grape_refiner_5min.png](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\outputs\refiner_eval\grape_refiner_5min.png)
- Metrics: [python/outputs/refiner_eval/grape_refiner_5min_metrics.json](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\outputs\refiner_eval\grape_refiner_5min_metrics.json)

That means the pipeline is functional, but it is not yet producing a high-fidelity 5-minute reconstruction on the grape target.

## Python Setup
Requirements:
- Python 3.12+
- `uv`
- Node.js with `npx.cmd` available if you want to generate repomix snapshots

Setup:

```powershell
Set-Location python
uv sync
```

## Main Run Commands
Live window:

```powershell
Set-Location python
uv run python run.py .\targets\grape.jpg --minutes 5 --resolution 200
```

Headless run:

```powershell
Set-Location python
uv run python run.py .\targets\grape.jpg --no-display --minutes 5 --resolution 200
```

Final evaluation helper:

```powershell
Set-Location python
uv run python final_reconstruct_eval.py
```

## Current Live Refiner Behavior
The live refiner currently does the following:

1. Round A at `50x50`
   - starts from `24` grid-seeded ellipses
   - adds batches of `8`
   - uses `position_lr=0.8`, `size_lr=0.3`, `color_lr=0.05`
2. Round B at `100x100`
   - scales the current polygons
   - adds batches of `6`
   - uses `position_lr=0.6`, `size_lr=0.2`, `color_lr=0.04`
3. Round C at `200x200`
   - scales the current polygons
   - adds batches of `5`
   - uses `position_lr=0.4`, `size_lr=0.15`, `color_lr=0.03`
4. Round D at `200x200`
   - stays at full working resolution
   - adds batches of `3`
   - uses `position_lr=0.2`, `size_lr=0.08`, `color_lr=0.02`

Optimizer details:
- appearance updates and geometry updates are evaluated separately
- color gradients use `trans_after * effective_alpha`
- geometry updates use finite differences with rollback only for geometry
- palette refinement blends `70%` local LAB target color with `30%` current polygon color between rounds

## Display and Controls
The live UI still exposes the existing control surface from the refiner module:
- `P`: pause/resume
- `R`: save screenshot
- `Q`: quit
- `S`: segmentation overlay toggle
- `E`: residual view mode cycle
- `V`: focus view cycle
- `1`, `2`, `3`: direct focus view selection
- `+`, `-`: softness scaling
- `G`, `D`: force-growth / force-decomposition requests

## Testing
Compile:

```powershell
Set-Location python
uv run python -m compileall .\src .\tests .\scripts .\run.py .\final_reconstruct_eval.py
```

Focused tests used during the recent refiner update:

```powershell
Set-Location python
$env:PYTHONPATH = (Get-Location).Path
uv run pytest .\tests\test_live_optimizer.py .\tests\test_refiner_live.py -q
```

## Repomix Snapshot
Reusable repomix generator:

```powershell
Set-Location python
uv run python .\scripts\build_python_repomix.py
```

That script creates [python/python_repomix.xml](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\python_repomix.xml) for the `python/` folder while excluding:
- all cache folders
- `.venv`
- `targets`
- `outputs`
- `scripts`
- `tests`
- `uv.lock`
- `.python-version`

Included files currently come from:
- top-level Python files such as `run.py`, `final_reconstruct_eval.py`, `compare.py`, `demo.py`, and `pyproject.toml`
- everything under `python/src/`

The script also verifies that the generated XML paths exactly match that manifest and fails if any missing or unexpected file is present.

## Repository Layout
- [python/src/live_refiner.py](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\src\live_refiner.py): active live refiner module
- [python/src/live_optimizer.py](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\src\live_optimizer.py): joint optimizer and rollback logic
- [python/src/live_schedule.py](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\src\live_schedule.py): growth helpers, grid seeding, residual targeting
- [python/run.py](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\run.py): CLI entry point
- [docs/academic_paper.md](C:\Users\ahada\Documents\abdulahad\evolutionary-art\docs\academic_paper.md): implementation-focused paper draft
