from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.display import run_live_display
from src.preprocessing import preprocess_target_array


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run attention-guided evolutionary art on a custom image."
    )
    parser.add_argument("image_path", type=Path, help="Path to an input image.")
    parser.add_argument(
        "--polygons",
        type=int,
        default=None,
        help="Override adaptive polygon budget.",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=None,
        help="Override coarse-phase maximum polygon size in pixels.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8000,
        help="Maximum proposal iterations per variant.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[200, 300],
        default=200,
        help="Base preprocessing resolution (200 default, 300 opt-in).",
    )
    parser.add_argument(
        "--fit-mode",
        choices=["auto", "crop", "letterbox"],
        default="auto",
        help="How to square non-square images before resize.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive fit-mode prompt when --fit-mode=auto.",
    )
    parser.add_argument(
        "--target-mse",
        type=float,
        default=0.01,
        help="Target MSE used for convergence-time estimation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic runs.",
    )
    parser.add_argument(
        "--update-interval-ms",
        type=int,
        default=100,
        help="Display refresh interval in milliseconds.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run preprocessing and summary only (useful for tests/headless runs).",
    )
    return parser


def _resolve_fit_mode(
    requested: str,
    *,
    no_prompt: bool,
    interactive: bool,
) -> str:
    if requested in {"crop", "letterbox"}:
        return requested

    if no_prompt or not interactive:
        return "crop"

    try:
        choice = input(
            "Fit image to square: [C]enter-crop (default) or [L]etterbox? "
        ).strip().lower()
    except EOFError:
        return "crop"
    if choice.startswith("l"):
        return "letterbox"
    return "crop"


def _prepare_image_square(
    image_path: Path,
    *,
    resolution: int,
    fit_mode: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        orig_w, orig_h = rgb.size

        if fit_mode == "crop":
            side = min(orig_w, orig_h)
            left = (orig_w - side) // 2
            top = (orig_h - side) // 2
            square = rgb.crop((left, top, left + side, top + side))
            fitted = square.resize((resolution, resolution), Image.Resampling.LANCZOS)
        else:
            scale = min(resolution / orig_w, resolution / orig_h)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (resolution, resolution), color=(255, 255, 255))
            paste_x = (resolution - new_w) // 2
            paste_y = (resolution - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))
            fitted = canvas

        arr = np.asarray(fitted, dtype=np.float32) / 255.0

    return arr.astype(np.float32, copy=False), (orig_w, orig_h)


def _estimate_runtime_seconds(
    preprocessed,
    *,
    max_iterations: int,
    target_mse: float,
    seed: int,
) -> tuple[float | None, float | None]:
    del seed

    resolution = float(preprocessed.base_resolution)
    complexity = float(np.clip(preprocessed.complexity_score, 0.0, 1.0))

    base_rate_200 = 2300.0
    resolution_penalty = (200.0 / max(resolution, 1.0)) ** 2
    complexity_penalty = 1.0 - 0.22 * complexity
    iter_rate = base_rate_200 * resolution_penalty * complexity_penalty

    target_factor = max(0.5, min(2.0, (0.01 / max(target_mse, 1e-6)) ** 0.35))
    expected_iters = (1200.0 + 2600.0 * complexity) * target_factor * (
        resolution / 200.0
    ) ** 2
    usable_iters = min(float(max_iterations), expected_iters)
    eta_seconds = usable_iters / max(iter_rate, 1e-6)

    return float(iter_rate), float(eta_seconds)


def print_analysis(preprocessed, *, original_size: tuple[int, int], fit_mode: str, iterations: int, target_mse: float, iter_rate: float | None, eta_seconds: float | None) -> None:
    h, w, _ = preprocessed.target_rgb.shape
    unique_regions = int(np.unique(preprocessed.segmentation_map).size)

    print("=== Input Analysis ===")
    print(f"source_size: {original_size[0]}x{original_size[1]}")
    print(f"fit_mode: {fit_mode}")
    print(f"target_size: {w}x{h}")
    print(f"complexity_score: {preprocessed.complexity_score:.4f}")
    print(f"recommended_polygons: {preprocessed.recommended_polygons}")
    print(f"detected_color_regions: {unique_regions}")
    print(f"k_clusters: {preprocessed.recommended_k}")
    print(
        "pyramid_levels: "
        + ", ".join([f"{img.shape[1]}x{img.shape[0]}" for img in preprocessed.pyramid])
    )
    print(
        "size_schedule_px: "
        + ", ".join(
            [
                f"coarse_start={preprocessed.recommended_size_schedule['coarse_start']:.1f}",
                f"coarse_end={preprocessed.recommended_size_schedule['coarse_end']:.1f}",
                f"structural_end={preprocessed.recommended_size_schedule['structural_end']:.1f}",
                f"detail_end={preprocessed.recommended_size_schedule['detail_end']:.1f}",
            ]
        )
    )

    if iter_rate is None:
        print("estimated_iteration_rate: unknown")
    else:
        print(f"estimated_iteration_rate: {iter_rate:,.0f} iterations/s")

    if eta_seconds is None:
        print(f"estimated_time_to_mse_{target_mse:.4f}: unknown")
    else:
        print(f"estimated_time_to_mse_{target_mse:.4f}: ~{eta_seconds:,.1f} s")

    print(f"max_iterations: {iterations}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.image_path.exists() or not args.image_path.is_file():
        parser.error(f"Image path does not exist: {args.image_path}")

    interactive = sys.stdin.isatty()
    fit_mode = _resolve_fit_mode(
        args.fit_mode,
        no_prompt=args.no_prompt,
        interactive=interactive,
    )

    prepared_rgb, original_size = _prepare_image_square(
        args.image_path,
        resolution=args.resolution,
        fit_mode=fit_mode,
    )

    preprocessed = preprocess_target_array(
        prepared_rgb,
        polygon_override=args.polygons,
        max_size_override=args.max_size,
        random_seed=args.seed,
        base_resolution=args.resolution,
    )

    iter_rate, eta_seconds = _estimate_runtime_seconds(
        preprocessed,
        max_iterations=args.iterations,
        target_mse=args.target_mse,
        seed=args.seed,
    )

    print_analysis(
        preprocessed,
        original_size=original_size,
        fit_mode=fit_mode,
        iterations=args.iterations,
        target_mse=args.target_mse,
        iter_rate=iter_rate,
        eta_seconds=eta_seconds,
    )

    if args.no_display:
        return 0

    print("Launching population-assisted live visualization window...")
    print("Controls: P pause/resume, S segmentation overlay, E error mode, R screenshot, Q quit, 1/2/3 variant view")

    run_live_display(
        target_image=preprocessed.target_rgb,
        target_pyramid=preprocessed.pyramid,
        structure_map=preprocessed.structure_map,
        gradient_angle_map=preprocessed.gradient_angle_map,
        segmentation_map=preprocessed.segmentation_map,
        cluster_centroids_lab=preprocessed.cluster_centroids_lab,
        cluster_variances_lab=preprocessed.cluster_variances_lab,
        max_iterations=args.iterations,
        max_polygons=preprocessed.recommended_polygons,
        size_schedule=preprocessed.recommended_size_schedule,
        update_interval_ms=args.update_interval_ms,
        random_seed=args.seed,
        target_mse=args.target_mse,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
