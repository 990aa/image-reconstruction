# Attention-Guided Evolutionary Art

## Overview
This project reconstructs a target image with explicit geometric primitives instead of latent generative sampling. The current best code path is a sequential greedy painter implemented in [python/src/live_refiner.py](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\src\live_refiner.py) and [python/src/live_optimizer.py](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\src\live_optimizer.py).

The live system now works by adding one shape at a time:
- score many candidate shapes in the highest-error regions
- solve each candidate color analytically from the target pixels it covers
- hill-climb only the winning shape geometry
- commit that single shape permanently to the canvas

That is the best-performing implementation currently in the repository.

## Best Verified Result
The current kept 5-minute verification run is:
- target: `python/targets/grape.jpg`
- resolution: `200x200`
- runtime: `5 minutes`
- run id: `grape_20260401_164124`

Measured final metrics:
- `rgb_mse = 0.013905`
- `ssim = 0.55509`
- `psnr = 18.568 dB`
- `lab_mse = 71.802`
- `gradient_mse = 0.00588`
- `gradient_mae = 0.04979`
- `gradient_corr = 0.69528`
- `accepted_polygons = 596`

Current kept artifacts:
- Image: [stage_detail.png](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\outputs\stage_checkpoints\grape_20260401_164124\stage_detail.png)
- Metrics: [run_metrics.json](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\outputs\stage_checkpoints\grape_20260401_164124\run_metrics.json)

![Best reconstruction](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\outputs\stage_checkpoints\grape_20260401_164124\stage_detail.png)

Historical note:
The previous kept best was the restored March 31, 2026 sequential run `grape_20260331_093438`, which reached `rgb_mse = 0.020303` and `ssim = 0.45878`. The April 1, 2026 tuning pass improved that benchmark materially and is now the active best snapshot.

## Current Algorithm
The current best variant still uses three sequential stages, but with larger early resolutions, heavier candidate search, and stronger alpha coverage:

1. Foundation at `100x100`
   - `200` shapes max, or `budget // 6`
   - ellipses and quads
   - `candidate_count=80`
   - `mutation_steps=160`
   - `alpha=0.55 -> 0.85`
2. Structure at `150x150`
   - `400` shapes max, or `budget // 3`
   - ellipses, quads, and triangles
   - `candidate_count=64`
   - `mutation_steps=128`
   - `alpha=0.40 -> 0.72`
3. Detail at `200x200`
   - remaining shapes
   - ellipses, triangles, and thin strokes
   - `candidate_count=72`
   - `mutation_steps=156`
   - `alpha=0.28 -> 0.60`

Important implementation details:
- routing uses absolute residual, with detail routing `high_frequency + 0.40 * residual`
- shape colors are solved analytically, not by learning rates
- geometry is refined by mutation-based hill climbing, not finite-difference gradients
- mutation radii scale by stage, up to `6 px` shifts and `15 deg` rotations in detail
- the canvas starts from the target mean color, not a random or white field

## Setup
Requirements:
- Python `3.14+`
- `uv`
- Node.js if you want repomix snapshots

```powershell
Set-Location python
uv sync
```

## Main Commands
Run the live UI:

```powershell
Set-Location python
uv run python .\run.py .\targets\grape.jpg --minutes 5 --resolution 200
```

Run headless:

```powershell
Set-Location python
uv run python .\run.py .\targets\grape.jpg --no-display --minutes 5 --resolution 200
```

Repomix snapshot:

```powershell
Set-Location python
uv run python .\scripts\build_python_repomix.py
```

## Testing
Compile:

```powershell
Set-Location python
$env:PYTHONPATH = (Get-Location).Path
uv run python -m compileall .\src .\tests .\run.py .\final_reconstruct_eval.py .\scripts\build_python_repomix.py
```

Regression suite:

```powershell
Set-Location python
$env:PYTHONPATH = (Get-Location).Path
uv run pytest .\tests\test_live_optimizer.py .\tests\test_live_renderer.py .\tests\test_mse.py .\tests\test_preprocessing.py .\tests\test_refiner_live.py -q
```

## Key Files
- [python/src/live_refiner.py](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\src\live_refiner.py): multi-stage sequential reconstruction driver
- [python/src/live_optimizer.py](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\src\live_optimizer.py): candidate scoring, analytic color solve, and hill climbing
- [python/src/core_renderer.py](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\src\core_renderer.py): soft rasterizer and primitive compositing
- [python/run.py](C:\Users\ahada\Documents\abdulahad\evolutionary-art\python\run.py): CLI entry point
- [docs/academic_paper.md](C:\Users\ahada\Documents\abdulahad\evolutionary-art\docs\academic_paper.md): current implementation report
- [docs/approach_retrospective.md](C:\Users\ahada\Documents\abdulahad\evolutionary-art\docs\approach_retrospective.md): best approach and failed-approach analysis
