# Attention-Guided Evolutionary Art

## Overview
This project reconstructs arbitrary target images with transparent geometric primitives using an attention-guided evolutionary optimizer.

Phase 7 introduces full multi-resolution progressive growth for arbitrary images:
- Automatic complexity analysis and polygon budget recommendation.
- Automatic pyramid schedule selection and cross-resolution polygon scaling.
- Five-panel live visualization with signed residual diagnostics and geometric outline view.
- Interactive growth/correction controls and live softness tuning.
- Hard timeout protection with graceful throttling near deadline.

Default resolution is 200x200 and can be overridden with the CLI.

## Setup
Requirements:
- Python 3.12+
- Node.js 20+
- uv

Python environment:

```powershell
Set-Location python
uv sync
```

Slides environment (optional):

```powershell
Set-Location slides
npm install
```

## Run Interface
Primary interface:

```powershell
Set-Location python
uv run python run.py path/to/image.jpg [--polygons 400] [--minutes 3] [--resolution 200]
```

Behavior:
- Automatically detects complexity.
- Automatically builds the multi-resolution schedule.
- Automatically runs progressive polygon growth through scheduled rounds.
- Automatically enforces a hard timeout budget so execution does not stall.

## CLI Options
- --polygons: override the automatic polygon budget.
- --minutes: optimization runtime budget in minutes.
- --timeout-seconds: hard safety timeout in seconds (defaults to minutes*60 + 45).
- --resolution: base square resolution.
- --fit-mode: auto, crop, or letterbox image fitting.
- --no-prompt: disables interactive fit-mode prompt when fit-mode is auto.
- --seed: deterministic random seed.
- --update-interval-ms: UI refresh interval (default 2000 ms).
- --close-after-seconds: auto-close UI after N seconds.
- --no-display: run headless optimization mode.
- --iterations: test/compatibility cap on total optimizer iteration points.

## Execution Modes
Live interactive mode:

```powershell
Set-Location python
uv run python run.py .\targets\internet_landscape.jpg --minutes 3 --resolution 200
```

Headless mode:

```powershell
Set-Location python
uv run python run.py .\targets\internet_landscape.jpg --no-display --minutes 3 --resolution 200
```

Hard-timeout-controlled run:

```powershell
Set-Location python
uv run python run.py .\targets\internet_landscape.jpg --minutes 3 --timeout-seconds 170
```

Fast automated smoke run:

```powershell
Set-Location python
uv run python run.py .\targets\internet_landscape.jpg --close-after-seconds 8 --iterations 120 --minutes 0.2
```

Comparison mode (naive vs improved):

```powershell
Set-Location python
uv run python compare.py .\targets\internet_landscape.jpg --iterations 800 --no-display --output .\outputs\internet_landscape_compare.png
```

External timeout wrapper mode:

```powershell
Set-Location python
uv run python scripts\interrupt_timeout.py --timeout-seconds 180 -- uv run python run.py .\targets\internet_landscape.jpg --minutes 3
```

## Phase 7 Five-Panel Display
Panel 1: Target
- Static reference image.

Panel 2: Current reconstruction
- Full-resolution evolving canvas.
- Updated on the UI interval (default 2 seconds) to avoid wasting optimization compute on redraw overhead.

Panel 3: Residual error (signed)
- Shows target minus reconstruction.
- Red indicates the reconstruction is too dark in that region.
- Blue indicates the reconstruction is too bright in that region.
- Directional residual is more actionable than scalar heatmaps.

Panel 4: Polygon visualization
- White background with outlines only.
- Large polygons: blue.
- Medium polygons: green.
- Small polygons: red.
- Reveals how geometry is progressively added and refined.

Panel 5: Loss curve
- Log-scale MSE versus iteration.
- Dashed vertical markers show resolution transitions.
- Orange vertical markers show polygon-batch additions.
- Typical sawtooth pattern appears as growth adds new polygons, followed by rapid loss recovery.

## Keyboard Controls
Original controls:
- P: pause/resume workers.
- S: segmentation overlay toggle.
- E: residual visualization mode cycle.
- R: save screenshot.
- Q: graceful quit.
- 1/2/3: switch focus display stream.

New controls:
- G: force immediate polygon growth step.
- D: force immediate residual decomposition and correction pass.
- V: cycle panel-2 focus between reconstruction, residual, and polygon outlines.
- + / -: adjust softness live (sharper or softer edges during optimization).

## Experimental Results
Rigorous paired experiments were run on three different internet images with identical comparison settings (--iterations 800).

| Target | Naive Final MSE | Improved Final MSE | Improvement Gap |
| --- | ---: | ---: | ---: |
| Portrait | 555.1732 | 114.7067 | 440.4665 |
| Landscape | 208.1470 | 1.9645 | 206.1825 |
| Graphic | 359.3228 | 14.0109 | 345.3118 |

Artifacts:
- [docs/figures/internet_portrait_compare.png](docs/figures/internet_portrait_compare.png)
- [docs/figures/internet_landscape_compare.png](docs/figures/internet_landscape_compare.png)
- [docs/figures/internet_graphic_compare.png](docs/figures/internet_graphic_compare.png)
- [docs/figures/internet_portrait_quality_vs_budget.png](docs/figures/internet_portrait_quality_vs_budget.png)
- [docs/figures/internet_landscape_quality_vs_budget.png](docs/figures/internet_landscape_quality_vs_budget.png)
- [docs/figures/internet_graphic_quality_vs_budget.png](docs/figures/internet_graphic_quality_vs_budget.png)
- [docs/figures/internet_portrait_log_evolution_grid.png](docs/figures/internet_portrait_log_evolution_grid.png)
- [docs/figures/internet_landscape_log_evolution_grid.png](docs/figures/internet_landscape_log_evolution_grid.png)
- [docs/figures/internet_graphic_log_evolution_grid.png](docs/figures/internet_graphic_log_evolution_grid.png)

## Documentation
- Research paper: [docs/academic_paper.md](docs/academic_paper.md)

## Testing and Quality Checks
Run tests:

```powershell
Set-Location python
uv run python -m pytest tests -v
```

Run linting/format/type checks:

```powershell
Set-Location python
uvx ruff check --fix
uvx ruff format
uvx ty check
```

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

PPTX output:
- [slides/dist/Attention_Guided_Evolutionary_Art_Abdul_Ahad.pptx](slides/dist/Attention_Guided_Evolutionary_Art_Abdul_Ahad.pptx)