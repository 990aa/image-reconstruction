from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.image_loader import load_target_image
from src.optimizer import (
    HillClimbingOptimizer,
    get_phase_name,
    phase_transition_iterations,
)


TARGET_ORDER = ["heart", "logo", "face"]


@dataclass
class RunResult:
    target_name: str
    target: np.ndarray
    final_canvas: np.ndarray
    mse_history: list[float]
    initial_mse: float
    final_mse: float
    accepted_count: int
    iterations_ran: int
    runtime_seconds: float


def canvas_to_uint8(canvas: np.ndarray) -> np.ndarray:
    clipped = np.clip(canvas, 0.0, 1.0)
    return (clipped * 255.0).round().astype(np.uint8)


def save_canvas_jpeg(canvas: np.ndarray, output_path: Path, quality: int = 95) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas_to_uint8(canvas), mode="RGB").save(
        output_path,
        format="JPEG",
        quality=quality,
        optimize=True,
    )


def rolling_mean(
    values: list[float], window: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if arr.size < window:
        x = np.arange(arr.size)
        return x, arr

    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(arr, kernel, mode="valid")
    x = np.arange(window - 1, window - 1 + smoothed.size)
    return x, smoothed


def make_live_panel_snapshot(
    *,
    target: np.ndarray,
    canvas: np.ndarray,
    error_map: np.ndarray,
    mse_history: list[float],
    iteration: int,
    max_iterations: int,
    accepted_count: int,
    acceptance_rate: float,
    phase_name: str,
    panel_out: Path,
    error_out: Path,
) -> None:
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 4, height_ratios=[2.0, 1.0], hspace=0.30, wspace=0.25)

    ax_target = fig.add_subplot(grid[0, 0])
    ax_error = fig.add_subplot(grid[0, 1])
    ax_canvas = fig.add_subplot(grid[0, 2])
    ax_stats = fig.add_subplot(grid[0, 3])
    ax_mse = fig.add_subplot(grid[1, :])

    ax_target.imshow(target)
    ax_target.set_title("Target")
    ax_target.set_xticks([])
    ax_target.set_yticks([])

    ax_error.imshow(error_map, cmap="hot")
    ax_error.set_title("Error Map")
    ax_error.set_xticks([])
    ax_error.set_yticks([])

    ax_canvas.imshow(canvas)
    ax_canvas.set_title("Evolving Canvas")
    ax_canvas.set_xticks([])
    ax_canvas.set_yticks([])

    current_mse = mse_history[-1] if mse_history else 0.0
    progress = 1.0 - (current_mse / max(mse_history[0], 1e-12))
    progress = float(np.clip(progress, 0.0, 1.0))
    border_color = (1.0 - progress, progress, 0.0)
    for spine in ax_canvas.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color(border_color)

    ax_stats.set_title("Live Stats")
    ax_stats.axis("off")
    ax_stats.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"iteration      : {iteration}/{max_iterations}",
                f"mse            : {current_mse:.4f}",
                f"acceptance     : {acceptance_rate * 100.0:6.2f}%",
                f"accepted polys : {accepted_count}",
                f"phase          : {phase_name}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        transform=ax_stats.transAxes,
    )

    ax_mse.set_title("MSE Decay")
    ax_mse.set_xlabel("Iteration")
    ax_mse.set_ylabel("MSE")
    ax_mse.set_yscale("log")
    ax_mse.grid(True, alpha=0.2)

    x_raw = np.arange(len(mse_history))
    y_raw = np.asarray(mse_history, dtype=np.float64)
    ax_mse.plot(
        x_raw, y_raw, color="tab:blue", alpha=0.25, linewidth=1.5, label="Raw MSE"
    )

    x_smooth, y_smooth = rolling_mean(mse_history, window=50)
    ax_mse.plot(
        x_smooth,
        y_smooth,
        color="tab:blue",
        linewidth=2.5,
        label="Smoothed (window=50)",
    )

    if y_raw.size:
        ax_mse.plot(
            [x_raw[-1]],
            [y_raw[-1]],
            marker="o",
            color="tab:red",
            markersize=6,
            label="Current",
        )

    transition_a, transition_b = phase_transition_iterations(max_iterations)
    ax_mse.axvline(transition_a, linestyle="--", color="gray", alpha=0.6)
    ax_mse.axvline(transition_b, linestyle="--", color="gray", alpha=0.6)
    ax_mse.legend(loc="upper right")

    panel_out.parent.mkdir(parents=True, exist_ok=True)
    error_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(panel_out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(5, 5))
    plt.imshow(error_map, cmap="hot")
    plt.axis("off")
    plt.title("Error Map (Mid-Run)")
    plt.savefig(error_out, dpi=150, bbox_inches="tight")
    plt.close()


def generate_replay_gif(
    frame_paths: list[Path], gif_path: Path, frame_delay_ms: int = 50
) -> None:
    if not frame_paths:
        return
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    images[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=frame_delay_ms,
        loop=0,
    )
    for img in images:
        img.close()


def generate_comparison_grid(results: list[RunResult], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    for col, result in enumerate(results):
        axes[0, col].imshow(result.target)
        axes[0, col].set_title(f"Target: {result.target_name}")
        axes[0, col].axis("off")

        axes[1, col].imshow(result.final_canvas)
        axes[1, col].set_title(f"Final Canvas: {result.target_name}")
        axes[1, col].axis("off")

        x = np.arange(len(result.mse_history))
        y = np.asarray(result.mse_history, dtype=np.float64)
        x_smooth, y_smooth = rolling_mean(result.mse_history, window=50)

        axes[2, col].plot(x, y, color="tab:blue", alpha=0.25, linewidth=1.5)
        axes[2, col].plot(x_smooth, y_smooth, color="tab:blue", linewidth=2.0)
        axes[2, col].set_yscale("log")
        axes[2, col].set_title(f"MSE Decay: {result.target_name}")
        axes[2, col].set_xlabel("Iteration")
        axes[2, col].set_ylabel("MSE")
        axes[2, col].grid(True, alpha=0.25)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_formula_image(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        r"$MSE = \frac{1}{n}\sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2$",
        ha="center",
        va="center",
        fontsize=24,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def sorted_frame_paths(output_dir: Path, target_name: str) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(target_name)}_(\d{{4}})\.jpg$")
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{target_name}_*.jpg"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def run_single_target(
    target_name: str,
    target_path: Path,
    output_dir: Path,
    iterations: int,
    seconds_limit: float,
    frame_interval_accepted: int,
    seed: int,
) -> RunResult:
    target = load_target_image(target_path)
    optimizer = HillClimbingOptimizer(
        target_image=target,
        max_iterations=iterations,
        random_seed=seed,
    )

    for stale in output_dir.glob(f"{target_name}_*.jpg"):
        stale.unlink(missing_ok=True)
    for stale in output_dir.glob(f"{target_name}_*.gif"):
        stale.unlink(missing_ok=True)

    save_canvas_jpeg(
        optimizer.canvas, output_dir / f"{target_name}_0000.jpg", quality=95
    )

    coarse_end, structural_end = phase_transition_iterations(iterations)
    phase_coarse_saved = False
    phase_structural_saved = False
    mid_snapshot_saved = False

    start_time = time.perf_counter()
    while optimizer.iteration < iterations:
        if seconds_limit > 0 and (time.perf_counter() - start_time) >= seconds_limit:
            break

        accepted = optimizer.step()

        if not phase_coarse_saved and optimizer.iteration >= max(1, coarse_end):
            save_canvas_jpeg(
                optimizer.canvas,
                output_dir / f"{target_name}_phase_coarse.jpg",
                quality=95,
            )
            phase_coarse_saved = True
        if not phase_structural_saved and optimizer.iteration >= max(1, structural_end):
            save_canvas_jpeg(
                optimizer.canvas,
                output_dir / f"{target_name}_phase_structural.jpg",
                quality=95,
            )
            phase_structural_saved = True

        if accepted and optimizer.accepted_count % frame_interval_accepted == 0:
            save_canvas_jpeg(
                optimizer.canvas,
                output_dir / f"{target_name}_{optimizer.accepted_count:04d}.jpg",
                quality=95,
            )

        if not mid_snapshot_saved and optimizer.iteration >= max(1, iterations // 2):
            make_live_panel_snapshot(
                target=target,
                canvas=optimizer.canvas,
                error_map=optimizer.current_error_map,
                mse_history=optimizer.mse_history,
                iteration=optimizer.iteration,
                max_iterations=iterations,
                accepted_count=optimizer.accepted_count,
                acceptance_rate=optimizer.acceptance_rate,
                phase_name=get_phase_name(optimizer.iteration, iterations),
                panel_out=output_dir / f"{target_name}_live_panel.jpg",
                error_out=output_dir / f"{target_name}_error_map_mid.jpg",
            )
            mid_snapshot_saved = True

    runtime = time.perf_counter() - start_time

    save_canvas_jpeg(
        optimizer.canvas, output_dir / f"{target_name}_phase_detail.jpg", quality=95
    )
    save_canvas_jpeg(
        optimizer.canvas, output_dir / f"{target_name}_final.jpg", quality=100
    )

    frame_paths = sorted_frame_paths(output_dir, target_name)
    generate_replay_gif(
        frame_paths, output_dir / f"{target_name}_replay.gif", frame_delay_ms=50
    )

    return RunResult(
        target_name=target_name,
        target=target,
        final_canvas=np.array(optimizer.canvas, copy=True),
        mse_history=list(optimizer.mse_history),
        initial_mse=float(optimizer.initial_mse),
        final_mse=float(optimizer.current_mse),
        accepted_count=optimizer.accepted_count,
        iterations_ran=optimizer.iteration,
        runtime_seconds=runtime,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-target evolutionary art demo runner."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Maximum iterations per target. Defaults to 5000 (or 500 with --fast).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Seconds limit per target run. Defaults to 60 (or 0 with --fast).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 500 iterations and no time limit unless explicitly set.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=2.0,
        help="Pause duration between targets.",
    )
    parser.add_argument(
        "--frame-interval-accepted",
        type=int,
        default=50,
        help="Save one JPEG every N accepted polygons.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base RNG seed for deterministic demos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for images, GIFs, and stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    iterations = args.iterations
    if iterations is None:
        iterations = 500 if args.fast else 5000

    seconds_limit = args.seconds
    if seconds_limit is None:
        seconds_limit = 0.0 if args.fast else 60.0

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.frame_interval_accepted <= 0:
        raise ValueError("frame_interval_accepted must be positive")

    base_dir = Path(__file__).resolve().parent
    targets_dir = base_dir / "targets"
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []

    for index, target_name in enumerate(TARGET_ORDER):
        target_path = targets_dir / f"{target_name}.png"
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target: {target_path}")

        print(f"[demo] Running target '{target_name}'...")
        result = run_single_target(
            target_name=target_name,
            target_path=target_path,
            output_dir=output_dir,
            iterations=iterations,
            seconds_limit=seconds_limit,
            frame_interval_accepted=args.frame_interval_accepted,
            seed=args.seed + index,
        )
        results.append(result)

        print(
            f"[demo] {target_name}: iterations={result.iterations_ran}, "
            f"accepted={result.accepted_count}, final_mse={result.final_mse:.6f}, "
            f"runtime={result.runtime_seconds:.2f}s"
        )

        if index < len(TARGET_ORDER) - 1 and args.pause_seconds > 0:
            time.sleep(args.pause_seconds)

    generate_comparison_grid(results, output_dir / "comparison_grid.jpg")
    generate_formula_image(output_dir / "mse_formula.png")

    best = min(results, key=lambda item: item.final_mse)
    stats_payload = {
        "best_run": {
            "target": best.target_name,
            "total_iterations": best.iterations_ran,
            "accepted_polygons": best.accepted_count,
            "final_mse": best.final_mse,
            "runtime_seconds": best.runtime_seconds,
        },
        "all_runs": [
            {
                "target": result.target_name,
                "iterations": result.iterations_ran,
                "accepted_polygons": result.accepted_count,
                "initial_mse": result.initial_mse,
                "final_mse": result.final_mse,
                "runtime_seconds": result.runtime_seconds,
            }
            for result in results
        ],
    }
    (output_dir / "run_stats.json").write_text(
        json.dumps(stats_payload, indent=2),
        encoding="utf-8",
    )

    print(f"[demo] Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
