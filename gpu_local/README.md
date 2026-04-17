# Local GPU Module

This module is a modularized local-GPU implementation extracted from `colab/evolArt.ipynb`.

## Setup

```powershell
Set-Location gpu_local
uv sync
```

## Run Reconstruction

```powershell
Set-Location gpu_local
uv run python .\run.py "C:\path\to\your\image.jpg" --minutes 5 --resolution 200 --polygons 1500 --seed 42 --fit-mode crop --save-stage-checkpoints --save-svg
```

Outputs are written to `gpu_local/outputs/gpu_local/` by default.

## Optional Notebook Features

- `evolutionary_art_gpu.visualization.plot_3d_exploded_view`: interactive 3D primitive stack with Plotly.
- `evolutionary_art_gpu.ablation.run_ablation_suite`: ablation/baseline suite ported from the notebook.
