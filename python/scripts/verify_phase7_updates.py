from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live_phase7 import Phase7ControlState, build_phase7_plan, execute_phase7_schedule
from src.preprocessing import preprocess_target_array


def prepare_image_square(image_path: Path, resolution: int) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        square = rgb.crop((left, top, left + side, top + side))
        out = square.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return (np.asarray(out, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    targets = [
        project_root / "targets" / "internet_portrait.jpg",
        project_root / "targets" / "internet_landscape.jpg",
        project_root / "targets" / "internet_graphic.jpg",
    ]

    summary: dict[str, dict[str, float | int]] = {}

    for target_path in targets:
        target_rgb = prepare_image_square(target_path, resolution=220)
        pre = preprocess_target_array(
            target_rgb,
            polygon_override=420,
            random_seed=42,
            base_resolution=220,
        )
        plan = build_phase7_plan(
            base_resolution=220,
            polygon_budget=420,
            complexity_score=float(pre.complexity_score),
        )

        update_count = 0
        changing_updates = 0
        prev_canvas: np.ndarray | None = None

        def on_update(
            canvas: np.ndarray,
            _polygons,
            _polygon_resolution: int,
            _losses: list[float],
            _resolution_markers: list[int],
            _batch_markers: list[int],
            _round_name: str,
            _iteration: int,
            _loss: float,
            _running: bool,
            _status: str,
        ) -> None:
            nonlocal update_count, changing_updates, prev_canvas
            update_count += 1
            if prev_canvas is not None:
                delta = float(np.mean(np.abs(canvas - prev_canvas), dtype=np.float32))
                if delta > 1e-6:
                    changing_updates += 1
            prev_canvas = np.array(canvas, copy=True)

        result = execute_phase7_schedule(
            target_image=pre.target_rgb,
            plan=plan,
            random_seed=42,
            minutes=0.6,
            hard_timeout_seconds=48.0,
            controls=Phase7ControlState(),
            shared_update_callback=on_update,
            max_total_steps=1100,
        )

        final_mse = float(
            np.mean((pre.target_rgb - result.final_canvas) ** 2, dtype=np.float32)
        )
        summary[target_path.name] = {
            "final_mse": final_mse,
            "iterations": int(result.iterations),
            "polygon_count": int(result.polygon_count),
            "update_count": int(update_count),
            "changing_updates": int(changing_updates),
        }

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
