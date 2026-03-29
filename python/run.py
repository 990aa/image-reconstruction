from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.live_phase7 import (
    build_phase7_plan,
    run_phase7_headless,
    run_phase7_live_display,
)
from src.preprocessing import preprocess_target_array


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 7 live evolutionary visualization for arbitrary images."
    )
    parser.add_argument("image_path", type=Path, help="Path to an input image.")
    parser.add_argument(
        "--polygons",
        type=int,
        default=None,
        help="Override automatic polygon budget.",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=3.0,
        help="Runtime budget in minutes.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Hard safety timeout in seconds. Defaults to minutes*60 + 45.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=200,
        help="Base square resolution.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic runs.",
    )
    parser.add_argument(
        "--update-interval-ms",
        type=int,
        default=2000,
        help="Display refresh interval in milliseconds.",
    )
    parser.add_argument(
        "--close-after-seconds",
        type=float,
        default=None,
        help="Auto-close live visualization after N seconds.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run optimization without opening the live UI.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Compatibility/testing override for maximum optimizer iteration points.",
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
        choice = (
            input("Fit image to square: [C]enter-crop (default) or [L]etterbox? ")
            .strip()
            .lower()
        )
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
    expected_iters = (
        (1200.0 + 2600.0 * complexity) * target_factor * (resolution / 200.0) ** 2
    )
    usable_iters = min(float(max_iterations), expected_iters)
    eta_seconds = usable_iters / max(iter_rate, 1e-6)

    return float(iter_rate), float(eta_seconds)


def print_analysis(
    preprocessed,
    *,
    original_size: tuple[int, int],
    fit_mode: str,
    minutes: float,
    hard_timeout_seconds: float | None,
    polygon_budget: int,
    plan,
    iter_rate: float | None,
    eta_seconds: float | None,
) -> None:
    h, w, _ = preprocessed.target_rgb.shape
    unique_regions = int(np.unique(preprocessed.segmentation_map).size)

    print("=== Input Analysis ===")
    print(f"source_size: {original_size[0]}x{original_size[1]}")
    print(f"fit_mode: {fit_mode}")
    print(f"target_size: {w}x{h}")
    print(f"complexity_score: {preprocessed.complexity_score:.4f}")
    print(f"recommended_polygons: {preprocessed.recommended_polygons}")
    print(f"polygon_budget: {polygon_budget}")
    print(f"detected_color_regions: {unique_regions}")
    print(f"k_clusters: {preprocessed.recommended_k}")
    print(
        "pyramid_levels: "
        + ", ".join([f"{img.shape[1]}x{img.shape[0]}" for img in preprocessed.pyramid])
    )
    print(f"runtime_budget_minutes: {minutes:.2f}")
    if hard_timeout_seconds is None:
        print("hard_timeout_seconds: none")
    else:
        print(f"hard_timeout_seconds: {hard_timeout_seconds:.1f}")

    print("auto_rounds:")
    for round_cfg in plan.rounds:
        print(
            "  "
            + ", ".join(
                [
                    f"{round_cfg.name}",
                    f"res={round_cfg.resolution}",
                    f"batches={round_cfg.batch_schedule}",
                    f"size=[{round_cfg.min_size:.2f},{round_cfg.max_size:.2f}]",
                    f"steps={round_cfg.max_steps_per_cycle}/{round_cfg.post_add_steps}",
                ]
            )
        )

    if iter_rate is None:
        print("estimated_iteration_rate: unknown")
    else:
        print(f"estimated_iteration_rate: {iter_rate:,.0f} iterations/s")

    if eta_seconds is None:
        print("estimated_time_to_steady_state: unknown")
    else:
        print(f"estimated_time_to_steady_state: ~{eta_seconds:,.1f} s")


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

    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    if args.minutes <= 0.0 and args.iterations is None:
        parser.error("Use --minutes > 0 or provide --iterations for a bounded run")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive")

    hard_timeout_seconds: float | None = (
        float(args.timeout_seconds)
        if args.timeout_seconds is not None
        else (float(args.minutes) * 60.0 + 45.0 if args.minutes > 0.0 else None)
    )

    preprocessed = preprocess_target_array(
        prepared_rgb,
        polygon_override=args.polygons,
        random_seed=args.seed,
        base_resolution=args.resolution,
    )

    polygon_budget = (
        int(args.polygons)
        if args.polygons is not None
        else int(preprocessed.recommended_polygons)
    )
    plan = build_phase7_plan(
        base_resolution=args.resolution,
        polygon_budget=polygon_budget,
        complexity_score=float(preprocessed.complexity_score),
    )

    iter_rate, eta_seconds = _estimate_runtime_seconds(
        preprocessed,
        max_iterations=max(500, polygon_budget * 40),
        target_mse=0.01,
        seed=args.seed,
    )

    print_analysis(
        preprocessed,
        original_size=original_size,
        fit_mode=fit_mode,
        minutes=float(args.minutes),
        hard_timeout_seconds=hard_timeout_seconds,
        polygon_budget=polygon_budget,
        plan=plan,
        iter_rate=iter_rate,
        eta_seconds=eta_seconds,
    )

    if args.no_display:
        result = run_phase7_headless(
            target_image=preprocessed.target_rgb,
            segmentation_map=preprocessed.segmentation_map,
            plan=plan,
            random_seed=args.seed,
            minutes=float(args.minutes),
            hard_timeout_seconds=hard_timeout_seconds,
            max_total_steps=args.iterations,
        )
    else:
        print("Launching Phase 7 five-panel live visualization...")
        print(
            "Controls: P pause, S segmentation, E residual mode, R screenshot, Q quit, 1/2/3 or V view, G force growth, D decomposition pass, +/- softness"
        )
        result = run_phase7_live_display(
            target_image=preprocessed.target_rgb,
            segmentation_map=preprocessed.segmentation_map,
            plan=plan,
            random_seed=args.seed,
            minutes=float(args.minutes),
            hard_timeout_seconds=hard_timeout_seconds,
            update_interval_ms=args.update_interval_ms,
            close_after_seconds=args.close_after_seconds,
            max_total_steps=args.iterations,
        )

    print("=== Run Summary ===")
    print(f"final_iteration: {result.iterations}")
    print(f"accepted_polygons: {result.polygon_count}")
    print(f"final_mse: {result.final_loss:.6f}")
    if result.loss_history:
        print(f"mse_improvement: {result.loss_history[0] - result.final_loss:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
