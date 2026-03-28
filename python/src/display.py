from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.image_loader import load_target_image
from src.optimizer import (
    HillClimbingOptimizer,
    get_phase_name,
    phase_transition_iterations,
)


@dataclass
class SharedDisplayState:
    canvas: np.ndarray
    error_map: np.ndarray
    mse_history: list[float]
    iteration: int
    acceptance_rate: float
    accepted_count: int
    phase_name: str
    running: bool
    last_phase_change_iteration: int | None = None


def _rolling_mean(
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


def _optimizer_worker(
    optimizer: HillClimbingOptimizer,
    shared: SharedDisplayState,
    lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    previous_phase = get_phase_name(0, optimizer.max_iterations)

    while not optimizer.is_done and not stop_event.is_set():
        optimizer.step()
        phase_name = optimizer.current_phase
        phase_change_iteration: int | None = None
        if phase_name != previous_phase:
            phase_change_iteration = optimizer.iteration
            previous_phase = phase_name

        with lock:
            shared.canvas = np.array(optimizer.canvas, copy=True)
            shared.error_map = np.array(optimizer.current_error_map, copy=True)
            shared.mse_history = list(optimizer.mse_history)
            shared.iteration = optimizer.iteration
            shared.acceptance_rate = optimizer.acceptance_rate
            shared.accepted_count = optimizer.accepted_count
            shared.phase_name = phase_name
            if phase_change_iteration is not None:
                shared.last_phase_change_iteration = phase_change_iteration

    with lock:
        shared.running = False


def run_live_display(
    target_image: np.ndarray,
    max_iterations: int = 5000,
    update_interval_ms: int = 100,
    capture_iteration: int | None = None,
    capture_path: str = "screenshot.jpg",
    close_after_seconds: float | None = None,
    random_seed: int | None = None,
    target_pyramid: list[np.ndarray] | None = None,
    structure_map: np.ndarray | None = None,
    size_schedule: dict[str, float] | None = None,
    max_polygons: int | None = None,
) -> HillClimbingOptimizer:
    optimizer = HillClimbingOptimizer(
        target_image=target_image,
        max_iterations=max_iterations,
        random_seed=random_seed,
        target_pyramid=target_pyramid,
        structure_map=structure_map,
        size_schedule=size_schedule,
        max_polygons=max_polygons,
    )

    lock = threading.Lock()
    stop_event = threading.Event()
    shared = SharedDisplayState(
        canvas=np.array(optimizer.canvas, copy=True),
        error_map=np.array(optimizer.current_error_map, copy=True),
        mse_history=list(optimizer.mse_history),
        iteration=optimizer.iteration,
        acceptance_rate=optimizer.acceptance_rate,
        accepted_count=optimizer.accepted_count,
        phase_name=optimizer.current_phase,
        running=True,
        last_phase_change_iteration=None,
    )

    worker = threading.Thread(
        target=_optimizer_worker,
        args=(optimizer, shared, lock, stop_event),
        daemon=True,
    )
    worker.start()

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 4, height_ratios=[2.0, 1.0], hspace=0.30, wspace=0.25)

    ax_target = fig.add_subplot(grid[0, 0])
    ax_error = fig.add_subplot(grid[0, 1])
    ax_canvas = fig.add_subplot(grid[0, 2])
    ax_stats = fig.add_subplot(grid[0, 3])
    ax_mse = fig.add_subplot(grid[1, :])

    target_im = ax_target.imshow(target_image)
    target_im.set_interpolation("nearest")
    ax_target.set_title("Target")
    ax_target.set_xticks([])
    ax_target.set_yticks([])
    for spine in ax_target.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color((0.6, 0.6, 0.6, 0.5))

    error_im = ax_error.imshow(shared.error_map, cmap="hot")
    error_im.set_interpolation("nearest")
    ax_error.set_title("Error Map")
    ax_error.set_xticks([])
    ax_error.set_yticks([])

    canvas_im = ax_canvas.imshow(shared.canvas)
    canvas_im.set_interpolation("nearest")
    ax_canvas.set_title("Evolving Canvas")
    ax_canvas.set_xticks([])
    ax_canvas.set_yticks([])

    phase_overlay = ax_canvas.text(
        0.5,
        0.5,
        "",
        transform=ax_canvas.transAxes,
        ha="center",
        va="center",
        fontsize=28,
        color="white",
        alpha=0.0,
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.65, "pad": 12},
    )

    ax_stats.set_title("Live Stats")
    ax_stats.axis("off")
    stats_text = ax_stats.text(
        0.02,
        0.98,
        "",
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

    (raw_line,) = ax_mse.plot(
        [], [], color="tab:blue", alpha=0.25, linewidth=1.5, label="Raw MSE"
    )
    (smooth_line,) = ax_mse.plot(
        [], [], color="tab:blue", linewidth=2.5, label="Smoothed (window=50)"
    )
    (tip_dot,) = ax_mse.plot(
        [], [], marker="o", markersize=6, color="tab:red", label="Current"
    )
    ax_mse.legend(loc="upper right")

    transition_a, transition_b = phase_transition_iterations(max_iterations)
    ax_mse.axvline(transition_a, linestyle="--", color="gray", alpha=0.6)
    ax_mse.axvline(transition_b, linestyle="--", color="gray", alpha=0.6)
    ax_mse.set_xlim(0, max_iterations)
    ax_mse.set_ylim(max(1e-6, optimizer.initial_mse * 0.1), optimizer.initial_mse * 1.2)

    capture_done = False
    overlay_start: float | None = None
    last_announced_iteration: int | None = None

    def _on_close(_: object) -> None:
        stop_event.set()

    fig.canvas.mpl_connect("close_event", _on_close)

    if close_after_seconds is not None and close_after_seconds > 0:
        timer = fig.canvas.new_timer(interval=int(close_after_seconds * 1000))
        timer.add_callback(lambda: plt.close(fig))
        timer.start()

    def update(_: int):
        nonlocal capture_done, overlay_start, last_announced_iteration

        with lock:
            canvas = np.array(shared.canvas, copy=True)
            error_map = np.array(shared.error_map, copy=True)
            mse_history = list(shared.mse_history)
            iteration = int(shared.iteration)
            acceptance_rate = float(shared.acceptance_rate)
            accepted_count = int(shared.accepted_count)
            phase_name = str(shared.phase_name)
            running = bool(shared.running)
            phase_change_iteration = shared.last_phase_change_iteration

        current_mse = float(mse_history[-1]) if mse_history else optimizer.initial_mse
        progress = 1.0 - (current_mse / max(optimizer.initial_mse, 1e-12))
        progress = float(np.clip(progress, 0.0, 1.0))

        border_color = (1.0 - progress, progress, 0.0)
        for spine in ax_canvas.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color(border_color)

        error_im.set_data(error_map)
        canvas_im.set_data(canvas)

        stats_text.set_text(
            "\n".join(
                [
                    f"iteration      : {iteration}/{max_iterations}",
                    f"mse            : {current_mse:.4f}",
                    f"acceptance     : {acceptance_rate * 100.0:6.2f}%",
                    f"accepted polys : {accepted_count}",
                    (
                        f"polygon budget : {accepted_count}/{optimizer.max_polygons}"
                        if optimizer.max_polygons is not None
                        else "polygon budget : unlimited"
                    ),
                    f"phase          : {phase_name}",
                ]
            )
        )

        x_raw = np.arange(len(mse_history))
        y_raw = np.asarray(mse_history, dtype=np.float64)
        raw_line.set_data(x_raw, y_raw)

        x_smooth, y_smooth = _rolling_mean(mse_history, window=50)
        smooth_line.set_data(x_smooth, y_smooth)

        if y_raw.size > 0:
            tip_dot.set_data([x_raw[-1]], [y_raw[-1]])

            y_min = max(1e-8, float(np.min(y_raw)) * 0.9)
            y_max = max(float(np.max(y_raw)) * 1.1, y_min * 10.0)
            ax_mse.set_ylim(y_min, y_max)

        if (
            phase_change_iteration is not None
            and phase_change_iteration != last_announced_iteration
        ):
            phase_overlay.set_text(f"{phase_name} Phase")
            overlay_start = time.monotonic()
            last_announced_iteration = phase_change_iteration

        if overlay_start is not None:
            elapsed = time.monotonic() - overlay_start
            if elapsed <= 2.0:
                phase_overlay.set_alpha(1.0 - (elapsed / 2.0))
            else:
                phase_overlay.set_alpha(0.0)
                overlay_start = None

        if (
            capture_iteration is not None
            and not capture_done
            and iteration >= capture_iteration
        ):
            fig.savefig(capture_path, dpi=150, bbox_inches="tight")
            capture_done = True

        if not running:
            anim.event_source.stop()

        fig.canvas.draw_idle()
        return (
            raw_line,
            smooth_line,
            tip_dot,
            error_im,
            canvas_im,
            phase_overlay,
            stats_text,
        )

    anim = FuncAnimation(
        fig,
        update,
        interval=update_interval_ms,
        blit=False,
        cache_frame_data=False,
    )
    _ = anim

    plt.show()
    stop_event.set()
    worker.join(timeout=2.0)
    return optimizer


def run_live_display_from_path(
    target_path: str,
    max_iterations: int = 5000,
    update_interval_ms: int = 100,
    capture_iteration: int | None = None,
    capture_path: str = "screenshot.jpg",
    close_after_seconds: float | None = None,
    random_seed: int | None = None,
) -> HillClimbingOptimizer:
    target = load_target_image(target_path)
    return run_live_display(
        target_image=target,
        max_iterations=max_iterations,
        update_interval_ms=update_interval_ms,
        capture_iteration=capture_iteration,
        capture_path=capture_path,
        close_after_seconds=close_after_seconds,
        random_seed=random_seed,
    )
