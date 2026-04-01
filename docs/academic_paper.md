# Sequential Greedy Evolutionary Reconstruction

## Abstract
This document describes the current best verified implementation in the repository as of April 1, 2026. The system reconstructs a target image with explicit alpha-blended geometric primitives using a sequential greedy search rather than a joint gradient optimizer. The active pipeline evaluates many candidate shapes in high-error regions, solves each candidate color analytically, hill-climbs only the best candidate geometry, and commits that single shape to the canvas. A 5-minute headless reconstruction of `python/targets/grape.jpg` at `200x200` currently yields `rgb_mse=0.013905`, `ssim=0.55509`, and `gradient_corr=0.69528`, which is the strongest verified result presently kept in the workspace.

## 1. Project Goal
The project aims to approximate arbitrary images with interpretable geometric primitives. Instead of generating a latent image and decoding it, the system constructs the result from a visible sequence of primitive additions. Each accepted step corresponds to a concrete geometric decision:
- where a shape is placed
- how large it is
- how it is rotated
- what color it contributes
- what primitive family it belongs to

This makes the optimization trace inspectable and suitable for studying where reconstruction quality improves or degrades.

## 2. Active Implementation
The current active modules are:
- `python/src/live_refiner.py`
- `python/src/live_optimizer.py`
- `python/src/core_renderer.py`
- `python/run.py`

The current codebase no longer uses the earlier finite-difference live optimizer as its main path. The restored best path is a sequential painter.

## 3. Method

### 3.1 Primitive Dictionary
The renderer currently supports:
- ellipses
- quads
- triangles
- thin strokes

Each primitive is alpha-blended onto the current canvas in order.

### 3.2 Sequential Addition
The central design choice is sequential greedy growth:
1. compute the current error map between target and canvas
2. sample candidate centers from the strongest error regions
3. generate candidate shapes around those centers
4. solve candidate colors analytically from covered target pixels
5. render and score every candidate independently
6. hill-climb the best candidate geometry
7. commit only that single winning shape

This keeps the optimization problem local and bounded instead of jointly updating hundreds of existing parameters.

### 3.3 Analytic Color
Color is not learned with a gradient step in the active best variant. For each candidate shape, the renderer computes its coverage mask and solves the color directly from the target and current canvas under that mask. This removes RGB from the mutation search and makes geometry search much cheaper.

### 3.4 Geometry Search
Geometry is refined with a mutation-based hill climber:
- initial candidate pool sampled from top residual regions
- local mutations over center, size, and rotation
- accept only mutations that reduce RGB MSE

This replaced the earlier slower finite-difference update path for the best current workflow.

## 4. Schedule
The current best schedule is a three-stage plan:

1. Foundation
   - resolution: `100x100`
   - shapes: `min(200, budget // 6)`
   - types: ellipses and quads
   - candidate search: `80`
   - mutation steps: `160`
   - alpha range: `0.55 -> 0.85`
   - mutation radius: `2.5 px`, `18%` size, `15 deg`
2. Structure
   - resolution: `150x150`
   - shapes: `min(400, budget // 3)`
   - types: ellipses, quads, triangles
   - candidate search: `64`
   - mutation steps: `128`
   - alpha range: `0.40 -> 0.72`
   - mutation radius: `4 px`, `18%` size, `15 deg`
3. Detail
   - resolution: `200x200`
   - remaining shapes
   - types: ellipses, triangles, thin strokes
   - candidate search: `72`
   - mutation steps: `156`
   - alpha range: `0.28 -> 0.60`
   - mutation radius: `6 px`, `18%` size, `15 deg`

The detail stage uses a high-frequency-biased residual map with an added `0.40 * residual` term so that unresolved low-frequency color masses still receive attention. The earlier stages use plain residual targeting.

## 5. Verification

### 5.1 Code Verification
Verified locally with:
- `uv run pytest .\tests\test_live_optimizer.py .\tests\test_live_renderer.py .\tests\test_mse.py .\tests\test_preprocessing.py .\tests\test_refiner_live.py -q`

At the time of this update, that suite passed with `13` tests.

### 5.2 Five-Minute Reconstruction
The currently kept best run in the workspace is:
- run id: `grape_20260401_164124`
- target: `python/targets/grape.jpg`
- resolution: `200x200`
- runtime budget: `5 minutes`
- accepted shapes: `596`

Measured final metrics:
- `rgb_mse = 0.013905`
- `rmse = 0.11792`
- `psnr = 18.568 dB`
- `ssim = 0.55509`
- `lab_mse = 71.802`
- `gradient_mse = 0.00588`
- `gradient_mae = 0.04979`
- `gradient_corr = 0.69528`

Saved artifacts:
- `python/outputs/stage_checkpoints/grape_20260401_164124/stage_detail.png`
- `python/outputs/stage_checkpoints/grape_20260401_164124/run_metrics.json`

### 5.3 Historical Note
The previous kept best run was `grape_20260331_093438`, which reached `rgb_mse = 0.020303`, `ssim = 0.45878`, and `gradient_corr = 0.59014`. The April 1 tuning pass improved those values to `0.013905`, `0.55509`, and `0.69528` respectively, so the earlier restored run is now best understood as the predecessor baseline rather than the active best result.

## 6. Discussion
The tuned sequential greedy method outperformed both the earlier gradient-style optimizer and the first restored sequential baseline. The main lesson from the experiments is that the system benefits from:
- stronger alpha coverage in the early passes
- higher early-stage resolution so promoted shapes remain structurally meaningful
- more aggressive mutation radii for geometric search
- heavier candidate search in the foundation stage
- residual routing that remains detail-aware without becoming blind to low-frequency color error

The system still has room to improve. The current best image is recognizable and substantially better than the earlier gradient-based live refiner, but it is still visibly approximate in the grape interiors and leaf boundaries.

## 7. Conclusion
The repository’s current best implementation is a tuned three-stage sequential greedy geometric reconstructor. It is now the code path reflected in the repository documentation, the kept output artifacts, and the retrospective notes. The next practical work should focus on careful tuning of this branch rather than replacing it with a different optimization family.
