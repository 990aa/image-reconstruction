# Attention-Guided Evolutionary Reconstruction with a Live Refiner Pipeline

## Abstract
This document describes the current implementation state of the repository rather than an aspirational design. The system reconstructs target images with ordered alpha-blended primitives and a live refiner pipeline implemented in `python/src/live_refiner.py`. The current codebase includes grid-seeded initialization, a four-round multi-resolution schedule, independent rollback for appearance and geometry updates, transmittance-weighted color gradients, diverse error-region sampling, structure-guided primitive aspect ratios, and LAB-based round palette refinement. The implementation is functional and test-backed, but the finite-difference geometry stage remains computationally expensive. On the current machine, a capped 5-minute `200x200` run on `grape.jpg` completed only the first round and produced `rgb_mse=0.07396`, `ssim=0.15606`, and `gradient_mse=0.01458`, indicating that the present system is operational but not yet high-fidelity under that wall-clock budget.

## 1. Scope
The goal of the project is interpretable image reconstruction with explicit primitive geometry instead of latent generative synthesis. Every accepted update corresponds to a visible primitive change, so optimization traces can be inspected in terms of position, size, color, and shape.

The active live module is:
- `python/src/live_refiner.py`

Supporting modules used by the current live path are:
- `python/src/live_optimizer.py`
- `python/src/live_schedule.py`
- `python/src/core_renderer.py`

## 2. Current Pipeline

### 2.1 Primitive Representation
The reconstruction canvas is composed from an ordered list of transparent primitives. The current renderer supports:
- ellipses
- triangles
- thin strokes

Each primitive carries:
- center
- size
- rotation
- RGB color
- alpha
- shape-specific parameters

The ordering matters because compositing is done front-to-back in sequence.

### 2.2 Appearance and Geometry Optimization
The current `LiveJointOptimizer.step()` implementation is split into two independent passes:

1. Appearance pass
   - updates color every step
   - evaluates loss immediately after the appearance update
   - rolls back only appearance parameters if that update is worse
2. Geometry pass
   - runs on interval
   - computes finite-difference gradients for centers and sizes
   - evaluates loss after geometry alone
   - rolls back only geometry parameters if that update is worse

This replaced the older coupled rollback behavior where a good color update could be lost because a geometry update in the same step overshot.

### 2.3 Color Gradient
The current color gradient uses visibility-aware weights:

$$
w_i = \mathrm{trans\_after}_i \cdot \mathrm{effective\_alpha}_i
$$

and

$$
\nabla_{c_i}\mathcal{L} \propto \sum_{x,y} w_i(x,y)\,r(x,y)
$$

where `trans_after` is the remaining transmittance from primitive `i` to the output and `effective_alpha` is coverage multiplied by scalar alpha. This prevents deeply buried primitives from receiving a color update as if they were fully visible.

### 2.4 Grid-Seeded Initialization
The first coarse round no longer starts from random polygons. Instead:
- the `50x50` target is divided into a regular grid
- each seed polygon is placed at a grid-cell center
- the polygon color is the mean RGB color of that target cell
- alpha is initialized to `0.85`
- rotation is initialized to `0`
- the initial primitive type is ellipse

For the standard coarse run this uses `24` polygons over an effective `5x5` grid with one cell dropped.

### 2.5 Diverse Region Sampling
Polygon insertion no longer uses a single repeatedly chosen maximum-error region. The live schedule now:
- computes a region-summed error map
- extracts a top-`K` candidate pool
- samples distinct regions without replacement

The same idea is applied in `live_schedule.py` so batch growth is less likely to collapse into a single hotspot.

### 2.6 Structure-Guided Shape Initialization
New primitives use local structure measurements:
- low-structure regions receive near-circular aspect ratios
- stronger edges produce elongated aspect ratios
- major axes are oriented perpendicular to the gradient direction
- thin strokes derive length and width from local structure strength

This logic lives in `live_schedule.py` and is used by the current live refiner insertion path.

### 2.7 Palette Refinement
Between rounds, each polygon is recolored by:
1. sampling target pixels under that polygon’s coverage
2. computing the mean LAB color
3. blending `70%` sampled target color with `30%` current polygon color

This pass is currently implemented between resolution transitions in the live refiner helper.

## 3. Current Round Schedule
The current helper-backed schedule in `live_refiner.py` is a fixed four-round plan:

1. Round A
   - resolution: `50x50`
   - initial polygons: `24`
   - additions: `10` batches of `8`
   - learning rates: `position=0.8`, `size=0.3`, `color=0.05`
2. Round B
   - resolution: `100x100`
   - additions: `8` batches of `6`
   - learning rates: `position=0.6`, `size=0.2`, `color=0.04`
3. Round C
   - resolution: `200x200`
   - additions: `10` batches of `5`
   - learning rates: `position=0.4`, `size=0.15`, `color=0.03`
4. Round D
   - resolution: `200x200`
   - additions: `10` batches of `3`
   - learning rates: `position=0.2`, `size=0.08`, `color=0.02`

Other current schedule settings:
- `position_update_interval = 1`
- `size_update_interval = 3`
- `max_fd_polygons = None` for `50x50` and `100x100`
- `max_fd_polygons = 40` for `200x200`
- if remaining runtime drops below `30` seconds, remaining additions are skipped and the system switches to final-only optimization

## 4. Runtime Behavior
The system still exposes:
- a live display path
- a headless path
- pause, quit, screenshot, residual-mode, segmentation, focus-view, and softness controls

The module file has been renamed to `live_refiner.py`, but several internal symbol names still retain the older `Phase7...` naming for compatibility with the existing CLI and tests.

## 5. Verification Performed
Recent verification performed against the current implementation:

### 5.1 Code-Level Checks
- `uv run python -m compileall` on active Python source and support files
- focused pytest runs for:
  - `test_live_optimizer.py`
  - `test_refiner_live.py`
- a manifest-verified repomix snapshot builder at `python/scripts/build_python_repomix.py`
  - outputs `python/python_repomix.xml`
  - includes the Python root entrypoints and `python/src/`
  - excludes cache folders, `.venv`, `targets`, `outputs`, `scripts`, `tests`, `uv.lock`, and `.python-version`

### 5.2 Sanity Checks
Completed checks:
- geometry positions visibly changed during a 50-step single-ellipse test
- coarse grid seeding achieved `rgb_mse=0.076436` on `grape.jpg` at `50x50`
- `trans_after` behaved as expected on a stacked-ellipse visibility check

### 5.3 Five-Minute Reconstruction Check
The current verified 5-minute run used:
- target: `python/targets/grape.jpg`
- resolution: `200x200`
- runtime cap: `300` seconds

Measured results:
- `elapsed_seconds = 300.59`
- `iterations = 201`
- `polygon_count = 32`
- `rgb_mse = 0.07396`
- `lab_mse = 321.31`
- `psnr = 11.31 dB`
- `ssim = 0.15606`
- `gradient_mse = 0.01458`
- `gradient_mae = 0.07681`
- `gradient_corr = -0.02610`
- stage markers: only `A`
- shape counts: `32` ellipses, `0` triangles, `0` strokes

Saved outputs:
- `python/outputs/refiner_eval/grape_refiner_5min.png`
- `python/outputs/refiner_eval/grape_refiner_5min_metrics.json`

## 6. Discussion
The codebase is more correct than the earlier version in several important ways:
- color and geometry rollback are decoupled
- color gradients respect visibility
- growth targets are more diverse
- shape initialization is more content-aware
- file/module naming has been updated away from `phase7` in the workspace

However, correctness has not yet translated into strong 5-minute reconstruction quality on the grape target. The main bottleneck is still the finite-difference geometry cost, which limits how many optimization steps and rounds complete within a fixed wall-clock budget.

## 7. Limitations
Current limitations of the repository state are:
- the refiner does not yet finish all planned rounds inside 5 minutes on the current machine
- the measured 5-minute grape reconstruction is not yet visually strong enough to claim easy recognition
- many public-facing function and class names still use legacy `Phase7` naming even though the file/module names no longer do
- the documentation now matches the code, but the code itself still needs further optimization work for high-quality time-bounded results

## 8. Conclusion
The repository currently contains a working live refiner implementation with better optimizer correctness, cleaner growth logic, and up-to-date file naming. The documentation in this file reflects the current verified state of the codebase rather than an ideal future result. The next engineering priority is runtime reduction for geometry updates so the multi-round schedule can actually complete within the intended 5-minute budget.
