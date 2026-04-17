# CPU Optimized Module

This module contains the optimized CPU implementation with headless and live-dashboard runners.

## Setup

```powershell
Set-Location cpu_optimized
uv sync
```

## Run Headless (custom image path)

```powershell
Set-Location cpu_optimized
uv run python .\run.py "C:\path\to\your\image.jpg" --no-display --minutes 5 --resolution 200 --polygons 1500 --seed 42 --fit-mode crop --no-prompt
```

## Run Live Dashboard

```powershell
Set-Location cpu_optimized
uv run python .\run.py "C:\path\to\your\image.jpg" --minutes 5 --resolution 200 --polygons 1500 --seed 42 --fit-mode crop --no-prompt
```

## Run Tests

```powershell
Set-Location cpu_optimized
$env:PYTHONPATH = (Get-Location).Path
uv run pytest .\tests -q
```
