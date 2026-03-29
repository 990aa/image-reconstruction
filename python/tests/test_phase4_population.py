from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from PIL import Image
from skimage import color

from src.live_phase7 import build_phase7_plan, run_phase7_headless
from src.preprocessing import preprocess_target_array


def _load_square_target(path: Path, resolution: int) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        square = rgb.crop((left, top, left + side, top + side))
        resized = square.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(
        np.float32, copy=False
    )


def _lab_rmse(a: np.ndarray, b: np.ndarray) -> float:
    la = color.rgb2lab(np.clip(a, 0.0, 1.0))
    lb = color.rgb2lab(np.clip(b, 0.0, 1.0))
    mse = np.mean((la - lb) ** 2, dtype=np.float32)
    return float(np.sqrt(max(float(mse), 0.0)))


def test_phase4_full_single_optimizer_pipeline_three_images() -> None:
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "targets" / "grape.jpg",
        root / "targets" / "internet_portrait.jpg",
        root / "targets" / "internet_landscape.jpg",
    ]

    for image_path in targets:
        target = _load_square_target(image_path, resolution=96)
        pre = preprocess_target_array(
            target,
            polygon_override=240,
            random_seed=11,
            base_resolution=96,
        )
        plan = build_phase7_plan(
            base_resolution=96,
            polygon_budget=240,
            complexity_score=float(pre.complexity_score),
        )

        t0 = time.monotonic()
        result = run_phase7_headless(
            target_image=pre.target_rgb,
            segmentation_map=pre.segmentation_map,
            plan=plan,
            random_seed=11,
            minutes=1.0,
            hard_timeout_seconds=90.0,
            max_total_steps=None,
        )
        elapsed = time.monotonic() - t0

        lab_loss = _lab_rmse(pre.target_rgb, result.final_canvas)

        assert elapsed < 300.0
        assert lab_loss < 20.0
