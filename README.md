# Iterative Primitive Reconstruction

This repository is split into three platform-specific modules that share the same sequential primitive reconstruction pipeline:

- `cpu_optimized`: optimized CPU implementation (headless + live dashboard + tests)
- `colab`: canonical notebook implementation (`iterative_art.ipynb`)
- `gpu_local`: modular local-GPU package (`iterative_art_gpu`) extracted from the notebook

Run all commands from the repository root using `uv --directory ...`.

## 1) CPU Optimized Module

### Setup

```powershell
uv --directory cpu_optimized sync --group dev
```

### Run Headless (custom image path)

```powershell
uv --directory cpu_optimized run python run.py "C:\path\to\your\image.jpg" --no-display --minutes 5 --resolution 200 --polygons 1500 --seed 42 --fit-mode crop --no-prompt
```

### Run Live Dashboard

```powershell
uv --directory cpu_optimized run python run.py "C:\path\to\your\image.jpg" --minutes 5 --resolution 200 --polygons 1500 --seed 42 --fit-mode crop --no-prompt
```

### Optional Demo Recorder

```powershell
uv --directory cpu_optimized run python scripts\render_internet_demo.py --minutes 1.0 --frame-stride 2 --frame-duration-ms 120
```

## 2) Colab Notebook Module

Notebook location: `colab/iterative_art.ipynb`

### Run in Colab

1. Open `colab/iterative_art.ipynb` in Google Colab.
2. Upload your image to `/content/`.
3. Set `IMAGE_PATH` in the first cell, for example: `IMAGE_PATH = "/content/my_image.jpg"`.
4. Run all cells in order.

## 3) Local GPU Modular Module

### Setup

```powershell
uv --directory gpu_local sync --group dev
```

### Run Local GPU Reconstruction (custom image path)

```powershell
uv --directory gpu_local run python run.py "C:\path\to\your\image.jpg" --minutes 5 --resolution 200 --polygons 1500 --seed 42 --fit-mode crop --save-stage-checkpoints --save-svg
```

### Disable UI

```powershell
uv --directory gpu_local run python run.py "C:\path\to\your\image.jpg" --no-display --minutes 5 --resolution 200 --polygons 1500
```

Outputs are written to `gpu_local/outputs/iterative_art_gpu_local/` by default.

## Consistency Notes

- Stage schedule, candidate search, mutation policy, and analytic color solve are synchronized with the notebook logic.
- `gpu_local` includes notebook-derived utility modules:
  - `iterative_art_gpu.visualization` (3D exploded view)
  - `iterative_art_gpu.ablation` (ablation + baseline suite)

## Validation

### CPU tests

```powershell
uv --directory cpu_optimized run pytest tests -q
```

### GPU compile smoke check

```powershell
uv --directory gpu_local run python -m compileall src run.py
```

### Lint, format, and type checks

```powershell
uvx ruff check --fix
uvx ruff format
uvx ty check
```

