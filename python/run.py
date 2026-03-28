from __future__ import annotations

import argparse
from pathlib import Path

from src.display import run_live_display
from src.preprocessing import preprocess_target_image


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
        help="Maximum proposal iterations.",
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


def print_analysis(preprocessed) -> None:
    h, w, _ = preprocessed.target_rgb.shape
    unique_regions = len(set(preprocessed.segmentation_map.reshape(-1).tolist()))

    print("=== Input Analysis ===")
    print(f"target_size: {w}x{h}")
    print(f"complexity_score: {preprocessed.complexity_score:.4f}")
    print(f"recommended_polygons: {preprocessed.recommended_polygons}")
    print(f"detected_color_regions: {unique_regions}")
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.image_path.exists() or not args.image_path.is_file():
        parser.error(f"Image path does not exist: {args.image_path}")

    preprocessed = preprocess_target_image(
        args.image_path,
        polygon_override=args.polygons,
        max_size_override=args.max_size,
        random_seed=args.seed,
    )

    print_analysis(preprocessed)

    if args.no_display:
        return 0

    print("Launching live visualization window...")
    print("Shape cycle: triangle -> quadrilateral -> ellipse")

    run_live_display(
        target_image=preprocessed.target_rgb,
        target_pyramid=preprocessed.pyramid,
        structure_map=preprocessed.structure_map,
        max_iterations=args.iterations,
        max_polygons=preprocessed.recommended_polygons,
        size_schedule=preprocessed.recommended_size_schedule,
        update_interval_ms=args.update_interval_ms,
        random_seed=args.seed,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
