# Attention-Guided Evolutionary Art

## Overview
This project reconstructs target images with transparent geometric primitives using an attention-guided evolutionary optimizer.

The implementation uses:
- Perceptual LAB loss from scikit-image color space conversions.
- MiniBatchKMeans segmentation in LAB space.
- Sobel structure and orientation maps from scikit-image filters.
- Sigmoid coarse-to-fine weighting: `w_fine = sigmoid(8 * (i / max_iter - 0.4))`.
- Population-assisted hill climbing with 6 parallel optimizer variants.
- Interactive live visualization and post-run analysis tooling.

Default resolution is 200x200. 300x300 is available with `--resolution 300`.

## Main Components
- [python/src/preprocessing.py](python/src/preprocessing.py): 4-level pyramid, complexity score, adaptive polygon recommendations, LAB segmentation, Sobel maps.
- [python/src/optimizer.py](python/src/optimizer.py): hill climbing loop, adaptive alpha search, phase-aware geometry behavior, splitting, palette refinement every 500 accepted polygons, death/replacement.
- [python/src/population.py](python/src/population.py): 6-thread variant population and barrier-synchronized recombination.
- [python/src/display.py](python/src/display.py): four-panel live UI with controls, overlays, error-mode cycling, and progress telemetry.
- [python/src/output_tools.py](python/src/output_tools.py): log-spaced evolution recorder and quality-vs-budget analysis.
- [python/run.py](python/run.py): custom-image entry point and artifact generation.
- [python/compare.py](python/compare.py): naive vs improved baseline comparison with split canvas and MSE curves.

## Setup
Requirements:
- Python 3.12+
- Node.js 20+
- `uv`

Python environment:

```powershell
Set-Location python
uv sync
```

Slides environment:

```powershell
Set-Location slides
npm install
```

## Running Live Evolution
Run on any image:

```powershell
Set-Location python
uv run python run.py .\targets\face.png
```

Useful options:

```powershell
# 300x300 opt-in
uv run python run.py .\targets\face.png --resolution 300

# explicit non-square fit behavior
uv run python run.py .\targets\face.png --fit-mode letterbox

# force polygon budget and size schedule start
uv run python run.py .\targets\face.png --polygons 280 --max-size 24

# controlled short live run (auto close)
uv run python run.py .\targets\face.png --iterations 1200 --close-after-seconds 8 --output-prefix face_short
```

Live controls:
- `P`: pause/resume workers
- `S`: segmentation overlay
- `E`: cycle error map mode (RGB, structure-weighted, LAB)
- `R`: save screenshot
- `Q`: clean shutdown
- `1`/`2`/`3`: switch shown variant stream

## Phase 6 Output and Comparison Tooling
The live runner now emits post-run analysis artifacts by default:
- log-spaced evolution frames and montage grid
- quality-vs-budget plot and CSV

Example:

```powershell
Set-Location python
uv run python run.py .\targets\internet_landscape.jpg --fit-mode crop --iterations 1200 --close-after-seconds 8 --output-prefix internet_landscape
```

Naive-vs-improved comparison:

```powershell
Set-Location python
uv run python compare.py .\targets\internet_landscape.jpg --iterations 800 --no-display --output .\outputs\internet_landscape_compare.png
```

## Internet Image Validation Results
Three custom internet images were tested with identical comparison settings (`--iterations 800`):

| Target | Naive Final MSE | Improved Final MSE | Improvement Gap | Figure |
| --- | ---: | ---: | ---: | --- |
| Portrait | 555.1732 | 114.7067 | 440.4665 | [docs/figures/internet_portrait_compare.png](docs/figures/internet_portrait_compare.png) |
| Landscape | 208.1470 | 1.9645 | 206.1825 | [docs/figures/internet_landscape_compare.png](docs/figures/internet_landscape_compare.png) |
| Graphic | 359.3228 | 14.0109 | 345.3118 | [docs/figures/internet_graphic_compare.png](docs/figures/internet_graphic_compare.png) |

Short live runs (with auto-close) also showed stable downward trends and artifact generation:

| Target Prefix | Final Iteration | Accepted Polygons | Final MSE |
| --- | ---: | ---: | ---: |
| internet_portrait | 123 | 109 | 139.3204 |
| internet_landscape | 134 | 120 | 6.0246 |
| internet_graphic | 70 | 70 | 66.7524 |

Quality-vs-budget artifacts:
- [docs/figures/internet_portrait_quality_vs_budget.png](docs/figures/internet_portrait_quality_vs_budget.png)
- [docs/figures/internet_landscape_quality_vs_budget.png](docs/figures/internet_landscape_quality_vs_budget.png)
- [docs/figures/internet_graphic_quality_vs_budget.png](docs/figures/internet_graphic_quality_vs_budget.png)

Log-spaced evolution grids:
- [docs/figures/internet_portrait_log_evolution_grid.png](docs/figures/internet_portrait_log_evolution_grid.png)
- [docs/figures/internet_landscape_log_evolution_grid.png](docs/figures/internet_landscape_log_evolution_grid.png)
- [docs/figures/internet_graphic_log_evolution_grid.png](docs/figures/internet_graphic_log_evolution_grid.png)

## Testing
Run the full test suite:

```powershell
Set-Location python
uv run python -m pytest tests -v
```

Current status in this workspace: 20 tests passed.

## Outputs
Generated artifacts are stored in [python/outputs](python/outputs), including:
- `*_final.jpg`, `*_replay.gif`, acceptance snapshots.
- `*_quality_vs_budget.png` and `*_quality_vs_budget.csv`.
- `*_log_evolution_grid.png` and `*_iter_XXXX.png` checkpoints.
- `*_compare.png` naive-vs-improved dashboards.
- [python/outputs/comparison_grid.jpg](python/outputs/comparison_grid.jpg), [python/outputs/mse_formula.png](python/outputs/mse_formula.png), [python/outputs/run_stats.json](python/outputs/run_stats.json).

## Academic Report and Diagrams
- Report: [docs/academic_report.md](docs/academic_report.md)
- Pipeline diagrams:
	- [docs/figures/architecture_diagram.png](docs/figures/architecture_diagram.png)
	- [docs/figures/optimization_flow.png](docs/figures/optimization_flow.png)
- Internet-image result figures:
	- [docs/figures/internet_portrait_compare.png](docs/figures/internet_portrait_compare.png)
	- [docs/figures/internet_landscape_compare.png](docs/figures/internet_landscape_compare.png)
	- [docs/figures/internet_graphic_compare.png](docs/figures/internet_graphic_compare.png)

## Slides
Build Reveal.js deck:

```powershell
Set-Location slides
npm run build
```

Generate PPTX:

```powershell
Set-Location slides
npm run pptx
```

PPTX output: [slides/dist/Attention_Guided_Evolutionary_Art_Abdul_Ahad.pptx](slides/dist/Attention_Guided_Evolutionary_Art_Abdul_Ahad.pptx)