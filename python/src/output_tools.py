from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.canvas import create_white_canvas
from src.mse import perceptual_mse_lab
from src.optimizer import HillClimbingOptimizer
from src.renderer import render_polygons


LOG_SNAPSHOT_ITERATIONS: tuple[int, ...] = (1, 10, 50, 100, 250, 500, 1000, 2000, 5000)
BUDGET_COUNTS: tuple[int, ...] = (10, 20, 50, 100, 200, 300, 500)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(image), mode="RGB").save(path)


def save_log_evolution_frames(
    optimizer: HillClimbingOptimizer,
    *,
    output_dir: Path,
    prefix: str,
    iterations: tuple[int, ...] = LOG_SNAPSHOT_ITERATIONS,
) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    usable = [i for i in iterations if i <= optimizer.iteration]
    if not usable:
        return {}

    snapshots = optimizer.iteration_snapshots
    if not snapshots:
        return {}

    saved: dict[int, Path] = {}
    available_sorted = sorted(snapshots.keys())

    for target_iter in usable:
        if target_iter in snapshots:
            frame = snapshots[target_iter]
            selected_iter = target_iter
        else:
            candidates = [i for i in available_sorted if i <= target_iter]
            if not candidates:
                continue
            selected_iter = candidates[-1]
            frame = snapshots[selected_iter]

        out = output_dir / f"{prefix}_iter_{target_iter:04d}.png"
        _save_rgb(out, frame)
        saved[target_iter] = out

    if saved:
        cols = 3
        rows = int(np.ceil(len(saved) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.0 * rows))
        axes_arr = np.atleast_2d(axes)

        items = list(saved.items())
        for idx, (iteration, path) in enumerate(items):
            r = idx // cols
            c = idx % cols
            img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            axes_arr[r, c].imshow(img)
            mse_val = optimizer.mse_history[min(iteration, len(optimizer.mse_history) - 1)]
            axes_arr[r, c].set_title(f"Iter {iteration} | MSE {mse_val:.4f}")
            axes_arr[r, c].axis("off")

        total_slots = rows * cols
        for idx in range(len(items), total_slots):
            r = idx // cols
            c = idx % cols
            axes_arr[r, c].axis("off")

        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_log_evolution_grid.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    return saved


def quality_vs_budget_analysis(
    optimizer: HillClimbingOptimizer,
    *,
    output_dir: Path,
    prefix: str,
    budgets: tuple[int, ...] = BUDGET_COUNTS,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(optimizer.accepted_polygons)
    usable_budgets = sorted({n for n in budgets if n > 0 and n <= total})
    if not usable_budgets:
        usable_budgets = [total] if total > 0 else []

    points: list[tuple[int, float]] = []
    sample_images: list[tuple[int, np.ndarray]] = []

    if total == 0:
        curve_path = output_dir / f"{prefix}_quality_vs_budget.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title("Quality vs Polygon Budget")
        ax.text(0.5, 0.5, "No accepted polygons", ha="center", va="center")
        ax.axis("off")
        fig.savefig(curve_path, dpi=170, bbox_inches="tight")
        plt.close(fig)

        csv_path = output_dir / f"{prefix}_quality_vs_budget.csv"
        csv_path.write_text("polygon_count,perceptual_mse\n", encoding="utf-8")
        return curve_path, csv_path

    for n in usable_budgets:
        partial = optimizer.accepted_polygons[:n]
        canvas = render_polygons(optimizer.blank_canvas, partial)
        mse = perceptual_mse_lab(canvas, optimizer.target)
        points.append((n, float(mse)))
        sample_images.append((n, canvas))

    x = np.array([p[0] for p in points], dtype=np.int32)
    y = np.array([p[1] for p in points], dtype=np.float64)

    curve_path = output_dir / f"{prefix}_quality_vs_budget.png"
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(x, y, marker="o", linewidth=2.0, color="tab:blue")
    ax.set_xlabel("Polygon Count")
    ax.set_ylabel("Perceptual MSE (LAB)")
    ax.set_title("Quality vs Polygon Budget")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(curve_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / f"{prefix}_quality_vs_budget.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("polygon_count,perceptual_mse\n")
        for n, mse in points:
            f.write(f"{n},{mse:.8f}\n")

    cols = min(4, len(sample_images))
    rows = int(np.ceil(len(sample_images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))
    axes_arr = np.atleast_2d(axes)
    for idx, (n, canvas) in enumerate(sample_images):
        r = idx // cols
        c = idx % cols
        axes_arr[r, c].imshow(canvas)
        mse = points[idx][1]
        axes_arr[r, c].set_title(f"N={n} | MSE {mse:.4f}")
        axes_arr[r, c].axis("off")

    total_slots = rows * cols
    for idx in range(len(sample_images), total_slots):
        r = idx // cols
        c = idx % cols
        axes_arr[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_budget_gallery.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    return curve_path, csv_path
