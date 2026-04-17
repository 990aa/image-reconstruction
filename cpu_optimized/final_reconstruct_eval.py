from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from src.live_refiner import build_phase_plan, run_phase_headless
from src.preprocessing import preprocess_target_array


@dataclass
class ReconstructionMetric:
    image_name: str
    resolution: int
    polygons: int
    minutes: float
    iterations: int
    accepted_polygons: int
    mse: float
    rmse: float
    psnr_db: float
    ssim: float
    accuracy_percent: float
    pixel_match_5pct: float
    target_file: str
    reconstruction_file: str
    abs_error_file: str


def prepare_square_image(image_path: Path, resolution: int) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        cropped = rgb.crop((left, top, left + side, top + side))
        resized = cropped.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(
        np.float32, copy=False
    )


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    arr = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def absolute_error_map(target: np.ndarray, recon: np.ndarray) -> np.ndarray:
    abs_err = np.mean(np.abs(target - recon), axis=2, dtype=np.float32)
    denom = max(float(np.quantile(abs_err, 0.99)), 1e-6)
    norm = np.clip(abs_err / denom, 0.0, 1.0)
    return np.repeat(norm[:, :, None], 3, axis=2).astype(np.float32, copy=False)


def compute_metrics(
    target: np.ndarray, recon: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    mse = float(np.mean((target - recon) ** 2, dtype=np.float32))
    rmse = float(np.sqrt(max(mse, 0.0)))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    ssim = float(structural_similarity(target, recon, channel_axis=2, data_range=1.0))
    accuracy = float(max(0.0, min(1.0, 1.0 - mse)) * 100.0)
    pixel_match = float(
        np.mean(np.abs(target - recon) <= 0.05, dtype=np.float32) * 100.0
    )
    return mse, rmse, psnr, ssim, accuracy, pixel_match


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    targets_dir = project_root / "python" / "targets"
    docs_figures = project_root / "docs" / "figures"
    docs_figures.mkdir(parents=True, exist_ok=True)

    image_names = [
        "internet_portrait.jpg",
        "internet_landscape.jpg",
        "internet_graphic.jpg",
    ]

    resolution = 500
    polygons = 1000
    minutes = 5.0
    seed = 42

    metrics: list[ReconstructionMetric] = []

    for image_name in image_names:
        image_path = targets_dir / image_name
        target_rgb = prepare_square_image(image_path, resolution=resolution)

        preprocessed = preprocess_target_array(
            target_rgb,
            polygon_override=polygons,
            random_seed=seed,
            base_resolution=resolution,
        )
        plan = build_phase_plan(
            base_resolution=resolution,
            polygon_budget=polygons,
            complexity_score=float(preprocessed.complexity_score),
        )

        result = run_phase_headless(
            target_image=preprocessed.target_rgb,
            segmentation_map=preprocessed.segmentation_map,
            plan=plan,
            random_seed=seed,
            minutes=minutes,
            hard_timeout_seconds=(minutes * 60.0 + 45.0),
            max_total_steps=None,
        )

        recon = np.clip(result.final_canvas, 0.0, 1.0).astype(np.float32, copy=False)
        target = np.clip(preprocessed.target_rgb, 0.0, 1.0).astype(
            np.float32, copy=False
        )

        mse, rmse, psnr, ssim, accuracy, pixel_match = compute_metrics(target, recon)
        abs_err = absolute_error_map(target, recon)

        stem = image_name.rsplit(".", 1)[0]
        target_file = docs_figures / f"{stem}_refiner_target_500.png"
        recon_file = docs_figures / f"{stem}_refiner_reconstruction_500.png"
        err_file = docs_figures / f"{stem}_refiner_abs_error_500.png"

        save_rgb_image(target_file, target)
        save_rgb_image(recon_file, recon)
        save_rgb_image(err_file, abs_err)

        metrics.append(
            ReconstructionMetric(
                image_name=image_name,
                resolution=resolution,
                polygons=polygons,
                minutes=minutes,
                iterations=int(result.iterations),
                accepted_polygons=int(result.polygon_count),
                mse=mse,
                rmse=rmse,
                psnr_db=psnr,
                ssim=ssim,
                accuracy_percent=accuracy,
                pixel_match_5pct=pixel_match,
                target_file=str(target_file.relative_to(project_root)).replace(
                    "\\", "/"
                ),
                reconstruction_file=str(recon_file.relative_to(project_root)).replace(
                    "\\", "/"
                ),
                abs_error_file=str(err_file.relative_to(project_root)).replace(
                    "\\", "/"
                ),
            )
        )

    payload = {
        "settings": {
            "resolution": resolution,
            "polygons": polygons,
            "minutes": minutes,
            "seed": seed,
            "hard_timeout_seconds": minutes * 60.0 + 45.0,
        },
        "results": [asdict(item) for item in metrics],
    }

    out_json = docs_figures / "refiner_final_reconstruction_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved metrics: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
