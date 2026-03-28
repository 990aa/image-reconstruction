from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

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
from matplotlib.patches import Rectangle

from src.optimizer import HillClimbingOptimizer, phase_transition_iterations
from src.population import PopulationHillClimber


@dataclass
class UIState:
    paused: bool = False
    show_segmentation_overlay: bool = False
    error_mode_index: int = 2
    display_variant_index: int = 0
    quit_requested: bool = False


def handle_control_key(
    key: str,
    *,
    ui: UIState,
    population: PopulationHillClimber,
    screenshot_callback,
    quit_callback,
) -> str:
    normalized = (key or "").lower()

    if normalized == "p":
        ui.paused = population.toggle_pause()
        return "pause"
    if normalized == "s":
        ui.show_segmentation_overlay = not ui.show_segmentation_overlay
        return "segmentation-toggle"
    if normalized == "e":
        ui.error_mode_index = (ui.error_mode_index + 1) % len(ERROR_MODES)
        return "error-mode-cycle"
    if normalized == "r":
        screenshot_callback()
        return "screenshot"
    if normalized == "q":
        ui.quit_requested = True
        quit_callback()
        return "quit"
    if normalized in {"1", "2", "3"}:
        ui.display_variant_index = int(normalized) - 1
        population.set_display_variant(ui.display_variant_index)
        return "variant-switch"

    return "noop"


ERROR_MODES: tuple[tuple[str, str], ...] = (
    ("rgb", "Raw RGB MSE"),
    ("structure", "Structure-Weighted Error"),
    ("perceptual", "Perceptual LAB Error"),
)


def _safe_log_ylim(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 1e-6, 1.0
    clipped = np.maximum(values, 1e-9)
    y_min = max(1e-9, float(np.min(clipped) * 0.9))
    y_max = max(float(np.max(clipped) * 1.1), y_min * 10.0)
    return y_min, y_max


def _estimate_time_to_target(
    progress_window: deque[tuple[int, float, float]],
    target_mse: float,
) -> float | None:
    if len(progress_window) < 2:
        return None

    last_iter, last_time, last_mse = progress_window[-1]
    if last_mse <= target_mse:
        return 0.0

    target_old_iter = max(0, last_iter - 200)
    oldest = progress_window[0]
    for point in progress_window:
        if point[0] <= target_old_iter:
            oldest = point
        else:
            break

    old_iter, old_time, old_mse = oldest
    delta_iter = last_iter - old_iter
    delta_time = last_time - old_time
    if delta_iter <= 0 or delta_time <= 0.0:
        return None

    iter_rate = delta_iter / delta_time
    mse_per_iter = (last_mse - old_mse) / float(delta_iter)
    if mse_per_iter >= 0.0:
        return None

    remaining_iters = (target_mse - last_mse) / mse_per_iter
    if remaining_iters <= 0.0:
        return 0.0
    return float(remaining_iters / max(iter_rate, 1e-6))


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
    gradient_angle_map: np.ndarray | None = None,
    segmentation_map: np.ndarray | None = None,
    cluster_centroids_lab: np.ndarray | None = None,
    cluster_variances_lab: np.ndarray | None = None,
    size_schedule: dict[str, float] | None = None,
    max_polygons: int | None = None,
    target_mse: float = 0.01,
) -> HillClimbingOptimizer:
    del max_polygons  # population phase uses iteration-bound runs.

    population = PopulationHillClimber(
        target_image=target_image,
        max_iterations=max_iterations,
        target_pyramid=target_pyramid,
        structure_map=structure_map,
        gradient_angle_map=gradient_angle_map,
        segmentation_map=segmentation_map,
        cluster_centroids_lab=cluster_centroids_lab,
        cluster_variances_lab=cluster_variances_lab,
        size_schedule=size_schedule,
        random_seed=random_seed,
    )
    population.start()

    ui = UIState()
    snapshot = population.snapshot()

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 4, height_ratios=[2.0, 1.0], hspace=0.30, wspace=0.25)

    ax_target = fig.add_subplot(grid[0, 0])
    ax_error = fig.add_subplot(grid[0, 1])
    ax_canvas = fig.add_subplot(grid[0, 2])
    ax_stats = fig.add_subplot(grid[0, 3])
    ax_mse = fig.add_subplot(grid[1, :])
    ax_acc = ax_mse.twinx()

    target_im = ax_target.imshow(snapshot.target)
    target_im.set_interpolation("nearest")
    ax_target.set_title("Target")
    ax_target.set_xticks([])
    ax_target.set_yticks([])

    seg_overlay = None
    if snapshot.segmentation_map is not None:
        seg_overlay = ax_target.imshow(
            snapshot.segmentation_map,
            cmap="tab20",
            alpha=0.30,
            interpolation="nearest",
            visible=False,
        )

    error_maps = population.get_error_maps(ui.display_variant_index)
    mode_key, mode_title = ERROR_MODES[ui.error_mode_index]
    error_im = ax_error.imshow(error_maps[mode_key], cmap="hot")
    error_im.set_interpolation("nearest")
    ax_error.set_title(mode_title)
    ax_error.set_xticks([])
    ax_error.set_yticks([])

    canvas_im = ax_canvas.imshow(snapshot.canvas)
    canvas_im.set_interpolation("nearest")
    ax_canvas.set_title("Evolving Canvas")
    ax_canvas.set_xticks([])
    ax_canvas.set_yticks([])

    decision_bar = Rectangle(
        (0.0, 0.0),
        1.0,
        0.035,
        transform=ax_canvas.transAxes,
        color="gray",
        alpha=0.0,
        zorder=6,
    )
    ax_canvas.add_patch(decision_bar)

    ax_stats.set_title("Live Stats")
    ax_stats.axis("off")
    stats_text = ax_stats.text(
        0.02,
        0.98,
        "",
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        transform=ax_stats.transAxes,
    )

    ax_mse.set_title("Population MSE + Acceptance")
    ax_mse.set_xlabel("Checkpoint")
    ax_mse.set_ylabel("MSE (log)")
    ax_mse.set_yscale("log")
    ax_mse.grid(True, alpha=0.25)

    (primary_line,) = ax_mse.plot(
        [], [], color="tab:blue", linewidth=2.0, label="Primary"
    )
    (best_line,) = ax_mse.plot(
        [], [], color="tab:green", linewidth=2.0, label="Best Variant"
    )

    ax_acc.set_ylabel("Acceptance Rate (%)")
    ax_acc.set_ylim(0.0, 100.0)
    (acc_line,) = ax_acc.plot(
        [], [], color="tab:orange", alpha=0.65, linewidth=1.5, label="Acceptance %"
    )

    lines = [primary_line, best_line, acc_line]
    labels: list[str] = [str(line.get_label()) for line in lines]
    ax_mse.legend(lines, labels, loc="upper right")

    transition_a, transition_b = phase_transition_iterations(max_iterations)
    ax_mse.axvline(
        transition_a // 500 if transition_a > 0 else 0,
        linestyle="--",
        color="gray",
        alpha=0.4,
    )
    ax_mse.axvline(
        transition_b // 500 if transition_b > 0 else 0,
        linestyle="--",
        color="gray",
        alpha=0.4,
    )

    stop_event = threading.Event()
    progress_window: deque[tuple[int, float, float]] = deque(maxlen=600)
    eta_seconds: float | None = None

    flash_alpha = 0.0
    flash_color = "gray"
    last_seen_iteration = -1

    capture_done = False

    def _save_screenshot(prefix: str = "live") -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{prefix}_{timestamp}.png"
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        return out_path

    def _on_close(_: object) -> None:
        stop_event.set()
        population.stop()

    def _on_key(event) -> None:
        nonlocal ui
        handle_control_key(
            event.key,
            ui=ui,
            population=population,
            screenshot_callback=lambda: _save_screenshot(prefix="manual_capture"),
            quit_callback=lambda: (
                stop_event.set(),
                population.stop(),
                plt.close(fig),
            ),
        )

    fig.canvas.mpl_connect("close_event", _on_close)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    if close_after_seconds is not None and close_after_seconds > 0:
        timer = fig.canvas.new_timer(interval=int(close_after_seconds * 1000))
        timer.add_callback(lambda: plt.close(fig))
        timer.start()

    def update(_: int):
        nonlocal \
            flash_alpha, \
            flash_color, \
            last_seen_iteration, \
            eta_seconds, \
            capture_done

        if stop_event.is_set():
            anim.event_source.stop()
            return (
                primary_line,
                best_line,
                acc_line,
                error_im,
                canvas_im,
                stats_text,
                decision_bar,
            )

        population.set_display_variant(ui.display_variant_index)
        snap = population.snapshot()

        mode_key, mode_title = ERROR_MODES[ui.error_mode_index]
        maps = population.get_error_maps(snap.variant_index)
        error_im.set_data(maps[mode_key])
        ax_error.set_title(mode_title)

        if seg_overlay is not None:
            seg_overlay.set_visible(ui.show_segmentation_overlay)

        canvas_im.set_data(snap.canvas)

        if snap.iteration != last_seen_iteration:
            last_seen_iteration = snap.iteration
            flash_alpha = 0.9
            flash_color = "#19a974" if snap.last_step_accepted else "#ff4136"
        else:
            flash_alpha *= 0.85

        decision_bar.set_facecolor(flash_color)
        decision_bar.set_alpha(float(np.clip(flash_alpha, 0.0, 0.9)))

        primary_hist = np.asarray(snap.primary_mse_history, dtype=np.float64)
        best_hist = np.asarray(snap.best_mse_history, dtype=np.float64)
        acc_hist = np.asarray(snap.acceptance_history, dtype=np.float64)

        x_vals = np.arange(primary_hist.size)
        primary_line.set_data(x_vals, primary_hist)
        best_line.set_data(np.arange(best_hist.size), best_hist)
        acc_line.set_data(np.arange(acc_hist.size), acc_hist)

        if primary_hist.size > 0:
            y_min, y_max = _safe_log_ylim(primary_hist)
            ax_mse.set_ylim(y_min, y_max)
            ax_mse.set_xlim(0, max(10, primary_hist.size + 5))

            now = time.monotonic()
            progress_window.append(
                (snap.primary_iteration, now, float(primary_hist[-1]))
            )
            if snap.primary_iteration % 100 == 0:
                eta_seconds = _estimate_time_to_target(progress_window, target_mse)

        eta_text = "~unknown"
        if eta_seconds is not None:
            if eta_seconds == 0.0:
                eta_text = "~reached"
            elif eta_seconds > 3600:
                eta_text = f"~{eta_seconds / 3600.0:4.2f} h"
            else:
                eta_text = f"~{eta_seconds:5.1f} s"

        stats_text.set_text(
            "\n".join(
                [
                    f"display variant : {snap.variant_index + 1}",
                    f"primary variant : {snap.primary_index + 1}",
                    f"best variant    : {snap.best_index + 1}",
                    f"iteration       : {snap.iteration}/{max_iterations}",
                    f"primary iter    : {snap.primary_iteration}",
                    f"mse(current)    : {snap.current_mse:.5f}",
                    f"mse(best)       : {snap.best_mse:.5f}",
                    f"acceptance      : {snap.acceptance_rate * 100.0:6.2f}%",
                    f"accepted polys  : {snap.accepted_count}",
                    f"phase           : {snap.phase_name}",
                    f"target mse      : {target_mse:.4f}",
                    f"eta             : {eta_text}",
                    f"paused          : {population.paused}",
                    "keys: P pause | S seg | E error | R shot | Q quit | 1/2/3 variant",
                ]
            )
        )

        if (
            capture_iteration is not None
            and not capture_done
            and snap.primary_iteration >= capture_iteration
        ):
            fig.savefig(capture_path, dpi=150, bbox_inches="tight")
            capture_done = True

        if not snap.running:
            anim.event_source.stop()

        fig.canvas.draw_idle()
        return (
            primary_line,
            best_line,
            acc_line,
            error_im,
            canvas_im,
            stats_text,
            decision_bar,
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
    population.stop()

    return population.optimizers[population.primary_index]
