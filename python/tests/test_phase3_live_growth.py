from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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


def _lab_mse(a: np.ndarray, b: np.ndarray) -> float:
    la = color.rgb2lab(np.clip(a, 0.0, 1.0))
    lb = color.rgb2lab(np.clip(b, 0.0, 1.0))
    return float(np.mean((la - lb) ** 2, dtype=np.float32))


def test_phase3_stage_a_and_a_plus_b_on_grape() -> None:
    root = Path(__file__).resolve().parents[1]
    target = _load_square_target(root / "targets" / "grape.jpg", resolution=96)

    pre = preprocess_target_array(
        target,
        polygon_override=240,
        random_seed=7,
        base_resolution=96,
    )
    base_plan = build_phase7_plan(
        base_resolution=96,
        polygon_budget=240,
        complexity_score=float(pre.complexity_score),
    )

    stage_a_plan = replace(
        base_plan, stage_b_batches=0, stage_c_batches=0, stage_d_steps=0
    )
    stage_a_result = run_phase7_headless(
        target_image=pre.target_rgb,
        segmentation_map=pre.segmentation_map,
        plan=stage_a_plan,
        random_seed=7,
        minutes=0.0,
        hard_timeout_seconds=40.0,
        max_total_steps=None,
    )
    stage_a_lab = _lab_mse(pre.target_rgb, stage_a_result.final_canvas)
    assert stage_a_lab < 30.0

    stage_ab_plan = replace(base_plan, stage_c_batches=0, stage_d_steps=0)
    stage_ab_result = run_phase7_headless(
        target_image=pre.target_rgb,
        segmentation_map=pre.segmentation_map,
        plan=stage_ab_plan,
        random_seed=7,
        minutes=0.0,
        hard_timeout_seconds=60.0,
        max_total_steps=None,
    )
    stage_ab_lab = _lab_mse(pre.target_rgb, stage_ab_result.final_canvas)
    assert stage_ab_lab < 15.0
    assert stage_ab_lab < stage_a_lab
