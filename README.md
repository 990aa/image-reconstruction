# Evolutionary Art Reconstruction

This repository is split into three clean, platform-specific modules that share the same sequential primitive reconstruction logic:

- `cpu_optimized`: optimized CPU implementation (headless + live dashboard + tests)
- `colab`: canonical Colab notebook implementation (`evolArt.ipynb`)
- `gpu_local`: modular local-GPU Python package extracted from the notebook

## 1) CPU Optimized Module

### Setup

```powershell
Set-Location cpu_optimized
uv sync
```

### Run Headless (custom image path)

```powershell
Set-Location cpu_optimized
uv run python .\run.py "C:\path\to\your\image.jpg" --no-display --minutes 5 --resolution 200 --polygons 1500 --seed 42 --fit-mode crop --no-prompt
```

### Run Live Dashboard

```powershell
Set-Location cpu_optimized
uv run python .\run.py "C:\path\to\your\image.jpg" --minutes 5 --resolution 200 --polygons 1500 --seed 42 --fit-mode crop --no-prompt
```

### Optional Demo Recorder

```powershell
Set-Location cpu_optimized
uv run python .\scripts\render_internet_demo.py --minutes 1.0 --frame-stride 2 --frame-duration-ms 120
```

## 2) Colab Notebook Module

Notebook location:

- `colab/evolArt.ipynb`

### Run In Colab

1. Open `colab/evolArt.ipynb` in Google Colab.
2. Upload your image to `/content/`.
3. Set `IMAGE_PATH` in the first cell, for example: `IMAGE_PATH = "/content/my_image.jpg"`.
4. Run all cells in order.

## 3) Local GPU Modular Module

### Setup

```powershell
Set-Location gpu_local
uv sync
```

### Run Local GPU Reconstruction (custom image path)

```powershell
Set-Location gpu_local
uv run python .\run.py "C:\path\to\your\image.jpg" --minutes 5 --resolution 200 --polygons 1500 --seed 42 --fit-mode crop --save-stage-checkpoints --save-svg
```

### Disable UI

```powershell
Set-Location gpu_local
uv run python .\run.py "C:\path\to\your\image.jpg" --no-display --minutes 5 --resolution 200 --polygons 1500
```

Outputs are written to `gpu_local/outputs/gpu_local/`.

## Consistency Notes

- Stage schedule, candidate search, mutation policy, and analytic color solve are synchronized with the notebook logic.
- `gpu_local` also includes notebook-derived utility modules:
	- `evolutionary_art_gpu.visualization` (3D exploded view)
	- `evolutionary_art_gpu.ablation` (ablation + baseline suite)

## Validation

### CPU tests

```powershell
Set-Location cpu_optimized
$env:PYTHONPATH = (Get-Location).Path
uv run pytest .\tests -q
```

### GPU module compile smoke check

```powershell
Set-Location gpu_local
$env:PYTHONPATH = (Get-Location).Path
uv run python -m compileall .\src .\run.py
```
