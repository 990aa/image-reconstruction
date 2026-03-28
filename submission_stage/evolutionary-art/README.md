# Attention-Guided Evolutionary Art

Name: Abdul Ahad  
Reg. No.: 245805010

## Overview
This project reconstructs small RGB target images (100x100) using an evolutionary hill-climbing process that places semi-transparent geometric primitives (triangles, quadrilaterals, ellipses) one at a time.

The system is attention-guided: each proposal is sampled from a probability map derived from per-pixel error, so new shapes are preferentially proposed in poorly reconstructed regions.

## Repository Structure
- [python](python): Core algorithm, tests, demo runner, generated outputs.
- [slides](slides): Reveal.js slide deck and PPTX generation scripts.
- [docs](docs): Academic report and supporting figures.

## Core Python Pipeline
Main implementation files:
- [python/src/image_loader.py](python/src/image_loader.py): Loads and normalizes images to float32 [0,1].
- [python/src/canvas.py](python/src/canvas.py): White-canvas initialization and copying.
- [python/src/mse.py](python/src/mse.py): Scalar MSE, raw error map, Gaussian-smoothed probability map.
- [python/src/polygon.py](python/src/polygon.py): Shape dataclass/enum and candidate generation.
- [python/src/renderer.py](python/src/renderer.py): Rasterization and alpha blending.
- [python/src/optimizer.py](python/src/optimizer.py): Main hill-climbing loop, phase scheduling, acceptance tracking.
- [python/src/display.py](python/src/display.py): Four-panel live visualization with threaded optimizer/display separation.
- [python/demo.py](python/demo.py): Multi-target batch runner, frame/GIF/grid/formula/stat generation.

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