# Run with:
# uv run python .\scripts\render_internet_demo.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageSequence

from run import _accuracy_metrics, _prepare_image_square
from src.live_refiner import build_phase_plan, record_phase_demo_gif
from src.preprocessing import preprocess_target_array


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record live matplotlib reconstruction demos for the internet_* targets."
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=1.0,
        help="Runtime per target in minutes.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional hard timeout per target.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=200,
        help="Square demo resolution.",
    )
    parser.add_argument(
        "--polygons",
        type=int,
        default=1500,
        help="Primitive budget passed into the stage planner.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Capture every Nth dashboard update.",
    )
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=120,
        help="Animated GIF frame duration in milliseconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "demo_viz",
        help="Directory for saved demo GIFs and metadata.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    targets_dir = project_root / "targets"
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_names = [
        "internet_graphic.jpg",
        "internet_landscape.jpg",
        "internet_portrait.jpg",
    ]

    per_target_records: list[dict[str, object]] = []
    combined_frames: list[Image.Image] = []

    for target_name in target_names:
        image_path = targets_dir / target_name
        target_rgb, original_size = _prepare_image_square(
            image_path,
            resolution=int(args.resolution),
            fit_mode="crop",
        )
        preprocessed = preprocess_target_array(
            target_rgb,
            polygon_override=int(args.polygons),
            random_seed=int(args.seed),
            base_resolution=int(args.resolution),
        )
        plan = build_phase_plan(
            base_resolution=int(args.resolution),
            polygon_budget=int(args.polygons),
            complexity_score=float(preprocessed.complexity_score),
        )

        gif_path = output_dir / f"{image_path.stem}_live_demo.gif"
        result, metadata = record_phase_demo_gif(
            target_image=preprocessed.target_rgb,
            segmentation_map=preprocessed.segmentation_map,
            plan=plan,
            random_seed=int(args.seed),
            minutes=float(args.minutes),
            hard_timeout_seconds=(
                float(args.timeout_seconds)
                if args.timeout_seconds is not None
                else float(args.minutes) * 60.0 + 10.0
            ),
            output_path=gif_path,
            frame_stride=int(args.frame_stride),
            frame_duration_ms=int(args.frame_duration_ms),
        )

        if int(metadata["update_events"]) < int(result.polygon_count):
            raise RuntimeError(
                f"Visualization update stream failed for {target_name}: "
                f"updates={metadata['update_events']} polygons={result.polygon_count}"
            )
        if float(metadata["final_recorded_loss"]) >= float(
            metadata["initial_recorded_loss"]
        ):
            raise RuntimeError(
                f"Visualization did not show loss reduction for {target_name}: "
                f"{metadata['initial_recorded_loss']} -> {metadata['final_recorded_loss']}"
            )

        final_metrics = _accuracy_metrics(preprocessed.target_rgb, result.final_canvas)
        record = {
            "target": target_name,
            "original_size": list(original_size),
            "gif_path": str(gif_path.relative_to(project_root)).replace("\\", "/"),
            "accepted_polygons": int(result.polygon_count),
            "iterations": int(result.iterations),
            "metrics": final_metrics,
            "demo_metadata": metadata,
        }
        per_target_records.append(record)
        (output_dir / f"{image_path.stem}_live_demo.json").write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

        with Image.open(gif_path) as gif:
            for frame in ImageSequence.Iterator(gif):
                combined_frames.append(frame.convert("P"))

    if not combined_frames:
        raise RuntimeError("No frames were captured for the internet demo sequence.")

    sequence_path = output_dir / "internet_sequence_live_demo.gif"
    combined_frames[0].save(
        sequence_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=int(args.frame_duration_ms),
        loop=0,
        optimize=False,
    )

    summary = {
        "targets": per_target_records,
        "sequence_gif": str(sequence_path.relative_to(project_root)).replace("\\", "/"),
    }
    (output_dir / "internet_sequence_live_demo.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved combined demo: {sequence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
