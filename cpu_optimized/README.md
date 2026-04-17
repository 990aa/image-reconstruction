# CPU Optimized Module

This module contains the optimized CPU implementation with headless and live-dashboard runners.
Run all commands from the repository root.

## Setup

```powershell
uv --directory cpu_optimized sync --group dev
```

## Run Headless (custom image path)

```powershell
uv --directory cpu_optimized run python run.py "C:\path\to\your\image.jpg" --no-display --minutes 5 --resolution 200 --polygons 1500 --seed 42 --fit-mode crop --no-prompt
```

## Run Live Dashboard

```powershell
uv --directory cpu_optimized run python run.py "C:\path\to\your\image.jpg" --minutes 5 --resolution 200 --polygons 1500 --seed 42 --fit-mode crop --no-prompt
```

## Run Tests

```powershell
uv --directory cpu_optimized run pytest tests -q
```
