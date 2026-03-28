# Attention-Guided Evolutionary Art

## Overview
This project reconstructs small RGB target images (100x100) using an evolutionary hill-climbing process that places semi-transparent geometric primitives (triangles, quadrilaterals, ellipses) one at a time.

The system is attention-guided: each proposal is sampled from a probability map derived from per-pixel error, so new shapes are preferentially proposed in poorly reconstructed regions.

## Core Python Pipeline
Main implementation files:
- [python/src/image_loader.py](python/src/image_loader.py): Loads and normalizes images to float32 [0,1].
- [python/src/canvas.py](python/src/canvas.py): White-canvas initialization and copying.
- [python/src/mse.py](python/src/mse.py): Scalar MSE, raw error map, Gaussian-smoothed probability map.
- [python/src/polygon.py](python/src/polygon.py): Shape dataclass/enum and candidate generation.
- [python/src/renderer.py](python/src/renderer.py): Rasterization and alpha blending.
- [python/src/preprocessing.py](python/src/preprocessing.py): Phase 1 preprocessing (4-level Gaussian pyramid, LAB k-means segmentation, Sobel structure map, complexity scoring, adaptive recommendations).
- [python/src/optimizer.py](python/src/optimizer.py): Main hill-climbing loop with phase scheduling, multi-scale perceptual LAB loss weighting, adaptive alpha selection, edge-aware shape strategy, splitting, palette refinement, and polygon death/replacement maintenance.
- [python/src/display.py](python/src/display.py): Four-panel live visualization with threaded optimizer/display separation.
- [python/demo.py](python/demo.py): Multi-target batch runner, frame/GIF/grid/formula/stat generation.
- [python/run.py](python/run.py): Custom-image entry point with preprocessing summary and live run launch.

## Part 2 Features Implemented
Phase 2 (Perceptual color matching):
- Acceptance and optimization guidance use perceptual LAB-space error instead of RGB-only MSE.
- Candidate color is segmentation-aware: cluster centroid in LAB plus local LAB patch blending.
- Adaptive alpha selector evaluates low/medium/high alpha (0.15, 0.40, 0.70) and chooses the best candidate.
- Palette refinement pass runs periodically to retune accepted polygon colors.

Phase 3 (Advanced polygon strategies):
- Content-aware shape selection uses structure map magnitude and phase context.
- Edge regions use oriented geometry guided by local gradient direction.
- Accepted polygons can be split into two smaller children when split candidates reduce loss.
- Periodic polygon death/replacement maintenance removes weak contributors and proposes new candidates.

## Targets
Targets are in [python/targets](python/targets):
- [python/targets/heart.png](python/targets/heart.png)
- [python/targets/logo.png](python/targets/logo.png)
- [python/targets/face.png](python/targets/face.png)

## Setup
Requirements:
- Python 3.12+
- Node.js 20+
- `uv` for Python package management

### Python setup
From repo root:

```powershell
Set-Location python
uv sync
```

### Slides setup
From repo root:

```powershell
Set-Location slides
npm install
```

## Testing
Run algorithm tests:

```powershell
Set-Location python
uv run python -m pytest tests/test_mse.py tests/test_renderer.py tests/test_optimizer.py -v
```

Run full test suite including Part 2 and Part 3 behavioral checks:

```powershell
Set-Location python
uv run python -m pytest tests -v
```

## Run the Demo
### Full run (default)
Generates three target runs (5000 iters each), frames, GIF replays, final images, formula image, and comparison grid.

```powershell
Set-Location python
uv run python demo.py
```

### Short verification run
Quick artifact smoke test (200 iters each):

```powershell
Set-Location python
uv run python demo.py --iterations 200
```

## Live Matplotlib Visualization (AI in Action)
Use the custom-image runner to open the 4-panel live Matplotlib window and watch the optimizer evolve the canvas in real time.

Shape cycle used continuously during optimization:
- triangle
- quadrilateral
- ellipse

Run on any image path:

```powershell
Set-Location python
uv run python run.py .\targets\face.png
```

You will see a startup analysis summary (complexity score, recommended polygon budget, detected color regions), then a live window with:
- target image
- attention/error map
- evolving canvas
- live stats + MSE decay

Run all provided sample targets one by one:

```powershell
Set-Location python
uv run python run.py .\targets\heart.png
uv run python run.py .\targets\logo.png
uv run python run.py .\targets\face.png
```

Useful overrides:

```powershell
# manually force polygon budget
uv run python run.py .\targets\face.png --polygons 300

# override coarse-phase max polygon size
uv run python run.py .\targets\face.png --max-size 26
```

Headless/check mode (no GUI window):

```powershell
uv run python run.py .\targets\face.png --no-display
```

## Generated Outputs
Output artifacts are written to [python/outputs](python/outputs), including:
- Per-target frame snapshots every 50 accepted polygons
- Final canvases (`*_final.jpg`)
- Replays (`*_replay.gif`)
- [python/outputs/comparison_grid.jpg](python/outputs/comparison_grid.jpg)
- [python/outputs/mse_formula.png](python/outputs/mse_formula.png)
- [python/outputs/run_stats.json](python/outputs/run_stats.json)

## Slide Decks
Two presentation formats are provided:

1. Reveal.js web deck (interactive HTML)
2. Native PowerPoint `.pptx` generated with `pptxgenjs`

### Build Reveal.js deck

```powershell
Set-Location slides
npm run build
```

Build output: [slides/dist/index.html](slides/dist/index.html)

### Generate PPTX deck

```powershell
Set-Location slides
npm run pptx
```

Output: [slides/dist/Attention_Guided_Evolutionary_Art_Abdul_Ahad.pptx](slides/dist/Attention_Guided_Evolutionary_Art_Abdul_Ahad.pptx)

### Verify PPTX is populated (non-blank)
This performs structural checks (slide count, text/image/shape presence, font-range sanity):

```powershell
Set-Location python
uv run python scripts/verify_pptx.py ..\slides\dist\Attention_Guided_Evolutionary_Art_Abdul_Ahad.pptx --expected-slides 10
```

## Academic Report
Detailed report (Python component only):
- [docs/academic_report.md](docs/academic_report.md)

Supporting diagrams:
- [docs/figures/architecture_diagram.png](docs/figures/architecture_diagram.png)
- [docs/figures/optimization_flow.png](docs/figures/optimization_flow.png)

## Submission Package
A zip file can be generated after building artifacts:

```powershell
# from repository root
Compress-Archive -Path README.md,docs,python\src,python\targets,python\tests,python\demo.py,python\scripts,python\outputs,slides\dist\index.html,slides\dist\assets,slides\dist\images,slides\dist\Attention_Guided_Evolutionary_Art_Abdul_Ahad.pptx -DestinationPath submission_evolutionary_art.zip -Force
```

## Notes
- The Python algorithm is the main project component.
- Slides are presentation artifacts generated from algorithm outputs.
- Re-running [python/demo.py](python/demo.py) updates outputs consumed by both Reveal and PPTX decks.