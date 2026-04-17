from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

from iterative_art_gpu.exporters import export_svg, save_rgb_image
from iterative_art_gpu.pipeline import prepare_square_image, run_phase_local_gpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local GPU runner extracted from the Colab implementation."
    )
    parser.add_argument("image_path", type=Path, help="Path to input image")
    parser.add_argument("--polygons", type=int, default=1500, help="Shape budget")
    parser.add_argument("--minutes", type=float, default=10.0, help="Runtime budget")
    parser.add_argument("--resolution", type=int, default=200, help="Square resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--fit-mode",
        choices=["crop", "letterbox"],
        default="crop",
        help="How to square non-square inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "iterative_art_gpu_local",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live dashboard updates",
    )
    parser.add_argument(
        "--save-stage-checkpoints",
        action="store_true",
        help="Save stage end images",
    )
    parser.add_argument(
        "--save-svg",
        action="store_true",
        help="Export final primitive stack to SVG",
    )
    return parser


def _accuracy_metrics(target: np.ndarray, canvas: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((target - canvas) ** 2, dtype=np.float32))
    rmse = float(np.sqrt(max(mse, 0.0)))
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))
    ssim = float(
        structural_similarity(
            np.clip(target, 0.0, 1.0),
            np.clip(canvas, 0.0, 1.0),
            channel_axis=2,
            data_range=1.0,
        )
    )
    return {
        "rgb_mse": mse,
        "rmse": rmse,
        "psnr_db": psnr,
        "ssim": ssim,
    }


def _build_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    return fig, axes


def _render_dashboard(
    fig,
    axes,
    *,
    stage_name: str,
    iteration: int,
    polygon_count: int,
    target: np.ndarray,
    canvas: np.ndarray,
    losses: list[float],
) -> None:
    ax_target, ax_canvas, ax_error, ax_curve = axes.reshape(-1)

    ax_target.clear()
    ax_target.imshow(target)
    ax_target.set_title("Target")
    ax_target.axis("off")

    ax_canvas.clear()
    ax_canvas.imshow(canvas)
    ax_canvas.set_title(f"Reconstruction | {polygon_count} shapes")
    ax_canvas.axis("off")

    err = np.mean(np.abs(target - canvas), axis=2)
    err = np.clip(err / max(float(np.quantile(err, 0.99)), 1e-6), 0.0, 1.0)
    ax_error.clear()
    ax_error.imshow(err, cmap="magma", vmin=0.0, vmax=1.0)
    ax_error.set_title("Abs Error")
    ax_error.axis("off")

    ax_curve.clear()
    ax_curve.set_title("MSE Reduction")
    ax_curve.set_yscale("log")
    ax_curve.grid(True, alpha=0.25)
    if losses:
        ax_curve.plot(
            np.maximum(np.asarray(losses, dtype=np.float64), 1e-9), color="tab:blue"
        )

    fig.suptitle(f"Stage: {stage_name} | Iter: {iteration}", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.canvas.draw_idle()
    plt.pause(0.001)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.image_path.exists() or not args.image_path.is_file():
        parser.error(f"Image path does not exist: {args.image_path}")
    if args.minutes <= 0.0:
        parser.error("--minutes must be positive")
    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    if args.polygons <= 0:
        parser.error("--polygons must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_rgb, original_size = prepare_square_image(
        args.image_path,
        resolution=int(args.resolution),
        fit_mode=str(args.fit_mode),
    )

    fig = None
    axes = None
    if not args.no_display:
        fig, axes = _build_dashboard()

    stage_records: list[dict[str, object]] = []
    last_update = time.perf_counter()

    def _progress(stage_name, iteration, polygon_count, canvas, target, losses):
        nonlocal last_update
        if fig is None or axes is None:
            return
        now = time.perf_counter()
        if stage_name == "done" or (now - last_update) >= 1.0:
            _render_dashboard(
                fig,
                np.asarray(axes),
                stage_name=stage_name,
                iteration=int(iteration),
                polygon_count=int(polygon_count),
                target=target,
                canvas=canvas,
                losses=losses,
            )
            last_update = now

    def _stage_checkpoint(stage_name, canvas, metrics):
        payload = {
            "stage": stage_name,
            **{
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in metrics.items()
            },
        }
        if args.save_stage_checkpoints:
            stage_path = output_dir / f"stage_{stage_name}.png"
            save_rgb_image(stage_path, canvas)
            payload["image_path"] = stage_path.as_posix()
        stage_records.append(payload)

    result = run_phase_local_gpu(
        target_rgb=image_rgb,
        resolution=int(args.resolution),
        polygons=int(args.polygons),
        minutes=float(args.minutes),
        seed=int(args.seed),
        progress_callback=_progress,
        stage_checkpoint_callback=_stage_checkpoint,
        progress_interval_seconds=0.5,
    )

    final_png = output_dir / "final_reconstruction.png"
    save_rgb_image(final_png, result.final_canvas)

    svg_path: Path | None = None
    if args.save_svg:
        svg_path = export_svg(
            result.batch,
            width=int(args.resolution),
            height=int(args.resolution),
            background_color=result.background_color,
            filename=output_dir / "zoom.svg",
        )

    metrics = _accuracy_metrics(result.preprocessed.target_rgb, result.final_canvas)
    summary = {
        "image_path": str(args.image_path),
        "original_size": [int(original_size[0]), int(original_size[1])],
        "fit_mode": str(args.fit_mode),
        "resolution": int(args.resolution),
        "minutes": float(args.minutes),
        "seed": int(args.seed),
        "polygons": int(args.polygons),
        "accepted_polygons": int(result.batch.count),
        "iterations": int(result.iterations),
        "metrics": metrics,
        "final_image": final_png.as_posix(),
        "svg": None if svg_path is None else svg_path.as_posix(),
        "stage_markers": [[name, int(idx)] for name, idx in result.stage_markers],
        "stage_checkpoints": stage_records,
    }

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Local GPU Run Summary ===")
    print(f"accepted_polygons: {result.batch.count}")
    print(f"iterations: {result.iterations}")
    print(f"rgb_mse: {metrics['rgb_mse']:.6f}")
    print(f"ssim: {metrics['ssim']:.5f}")
    print(f"final_image: {final_png}")
    print(f"summary_json: {summary_path}")

    if fig is not None:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
