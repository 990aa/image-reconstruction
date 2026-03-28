from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.canvas import create_white_canvas
from src.mse import perceptual_mse_lab
from src.optimizer import HillClimbingOptimizer, get_current_size
from src.polygon import ShapeType, generate_shape
from src.preprocessing import preprocess_target_array
from src.renderer import render_polygon


def _prepare_image_square(
    image_path: Path,
    *,
    resolution: int,
    fit_mode: str,
) -> np.ndarray:
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
            canvas.paste(
                resized, ((resolution - new_w) // 2, (resolution - new_h) // 2)
            )
            fitted = canvas

        arr = np.asarray(fitted, dtype=np.float32) / 255.0
    return arr.astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare naive vs improved optimizer on the same image.",
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--resolution", type=int, choices=[200, 300], default=200)
    parser.add_argument("--fit-mode", choices=["crop", "letterbox"], default="crop")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (defaults to outputs/<stem>_compare.png)",
    )
    return parser


def run_naive(
    target: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    h, w, _ = target.shape

    canvas = create_white_canvas(width=w, height=h)
    mse = perceptual_mse_lab(canvas, target)
    history: list[float] = [mse]

    uniform_map = np.full((h, w), 1.0 / float(h * w), dtype=np.float32)
    cycle = (ShapeType.TRIANGLE, ShapeType.QUADRILATERAL, ShapeType.ELLIPSE)

    for i in range(iterations):
        shape_type = cycle[i % len(cycle)]
        size_px = get_current_size(i, max(iterations, 1))

        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        candidate = generate_shape(
            shape_type=shape_type,
            probability_map=uniform_map,
            target_image=target,
            size_px=size_px,
            center_xy=center,
            orientation=float(rng.uniform(0.0, 2.0 * np.pi)),
            alpha=0.40,
            rng=rng,
        )
        random_color = rng.uniform(0.0, 1.0, size=3).astype(np.float32)
        candidate = replace(
            candidate,
            color=(
                float(random_color[0]),
                float(random_color[1]),
                float(random_color[2]),
            ),
            alpha=0.40,
        )

        trial = render_polygon(canvas, candidate)
        trial_mse = perceptual_mse_lab(trial, target)
        if trial_mse < mse:
            canvas = trial
            mse = trial_mse

        history.append(mse)

    return canvas, history


def run_improved(
    preprocessed,
    *,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    optimizer = HillClimbingOptimizer(
        target_image=preprocessed.target_rgb,
        max_iterations=iterations,
        target_pyramid=preprocessed.pyramid,
        structure_map=preprocessed.structure_map,
        gradient_angle_map=preprocessed.gradient_angle_map,
        segmentation_map=preprocessed.segmentation_map,
        cluster_centroids_lab=preprocessed.cluster_centroids_lab,
        cluster_variances_lab=preprocessed.cluster_variances_lab,
        size_schedule=preprocessed.recommended_size_schedule,
        random_seed=seed,
    )
    optimizer.run(iterations=iterations)
    return np.array(optimizer.canvas, copy=True), list(optimizer.mse_history)


def main() -> int:
    args = build_parser().parse_args()
    if not args.image_path.exists() or not args.image_path.is_file():
        raise SystemExit(f"Image not found: {args.image_path}")

    prepared = _prepare_image_square(
        args.image_path,
        resolution=args.resolution,
        fit_mode=args.fit_mode,
    )
    preprocessed = preprocess_target_array(
        prepared,
        random_seed=args.seed,
        base_resolution=args.resolution,
    )

    naive_canvas, naive_hist = run_naive(
        preprocessed.target_rgb,
        iterations=args.iterations,
        seed=args.seed + 101,
    )
    improved_canvas, improved_hist = run_improved(
        preprocessed,
        iterations=args.iterations,
        seed=args.seed,
    )

    fig = plt.figure(figsize=(16, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.28, wspace=0.25)

    ax_target = fig.add_subplot(gs[0, 0])
    ax_naive = fig.add_subplot(gs[0, 1])
    ax_improved = fig.add_subplot(gs[0, 2])
    ax_curve = fig.add_subplot(gs[1, :])

    ax_target.imshow(preprocessed.target_rgb)
    ax_target.set_title("Target")
    ax_target.axis("off")

    ax_naive.imshow(naive_canvas)
    ax_naive.set_title(f"Naive Final | MSE {naive_hist[-1]:.4f}")
    ax_naive.axis("off")

    ax_improved.imshow(improved_canvas)
    ax_improved.set_title(f"Improved Final | MSE {improved_hist[-1]:.4f}")
    ax_improved.axis("off")

    x_naive = np.arange(len(naive_hist))
    x_improved = np.arange(len(improved_hist))
    ax_curve.plot(x_naive, naive_hist, color="tab:red", linewidth=2.0, label="Naive")
    ax_curve.plot(
        x_improved, improved_hist, color="tab:blue", linewidth=2.0, label="Improved"
    )
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("Iteration")
    ax_curve.set_ylabel("Perceptual MSE (LAB)")
    ax_curve.set_title("Naive vs Improved Optimization")
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend(loc="upper right")

    out_path = (
        args.output
        if args.output is not None
        else Path("outputs") / f"{args.image_path.stem.lower()}_compare.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")

    print("=== Comparison Summary ===")
    print(f"image: {args.image_path}")
    print(f"iterations: {args.iterations}")
    print(f"naive_final_mse: {naive_hist[-1]:.6f}")
    print(f"improved_final_mse: {improved_hist[-1]:.6f}")
    print(f"improvement_gap: {naive_hist[-1] - improved_hist[-1]:.6f}")
    print(f"output: {out_path}")

    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
