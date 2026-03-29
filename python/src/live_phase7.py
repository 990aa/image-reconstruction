from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
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
from scipy.ndimage import gaussian_filter

from src.live_optimizer import LiveJointOptimizer, LiveOptimizerConfig
from src.live_renderer import SHAPE_ELLIPSE, SHAPE_QUAD, LivePolygonBatch, SoftRasterizer


@dataclass(frozen=True)
class Phase7Plan:
    stage_a_initial_polygons: int
    stage_a_steps: int
    stage_b_batches: int
    stage_b_batch_size: int
    stage_b_steps_per_batch: int
    stage_b_size_start: float
    stage_b_size_end: float
    stage_c_batches: int
    stage_c_batch_size: int
    stage_c_steps_per_batch: int
    stage_c_size_start: float
    stage_c_size_end: float
    stage_d_steps: int


@dataclass(frozen=True)
class Phase7ExecutionResult:
    final_canvas: np.ndarray
    final_loss: float
    polygon_count: int
    iterations: int
    loss_history: list[float]
    resolution_markers: list[int]
    batch_markers: list[int]
    stage_markers: list[tuple[str, int]]


@dataclass
class Phase7ControlState:
    paused: bool = False
    quit_requested: bool = False


@dataclass
class _SharedViewState:
    target: np.ndarray
    canvas: np.ndarray
    signed_residual: np.ndarray
    loss_history: list[float]
    stage_markers: list[tuple[str, int]]
    polygon_count: int
    iteration: int
    stage_name: str
    running: bool
    status_line: str
    polygon_sizes: np.ndarray


def build_phase7_plan(
    *,
    base_resolution: int,
    polygon_budget: int,
    complexity_score: float,
) -> Phase7Plan:
    del base_resolution
    del complexity_score

    budget = max(120, int(polygon_budget))
    scale = float(np.clip(budget / 240.0, 0.7, 3.0))

    a_count = int(round(0.33 * budget))
    b_total = int(round(0.33 * budget))
    c_total = max(40, budget - a_count - b_total)

    b_batches = 8
    c_batches = 12

    return Phase7Plan(
        stage_a_initial_polygons=max(20, a_count),
        stage_a_steps=500,
        stage_b_batches=b_batches,
        stage_b_batch_size=max(1, int(np.ceil(b_total / b_batches))),
        stage_b_steps_per_batch=200,
        stage_b_size_start=18.0,
        stage_b_size_end=6.0,
        stage_c_batches=c_batches,
        stage_c_batch_size=max(1, int(np.ceil(c_total / c_batches))),
        stage_c_steps_per_batch=50,
        stage_c_size_start=6.0,
        stage_c_size_end=2.0,
        stage_d_steps=1000,
    )


def _rgb_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2, dtype=np.float32))


def _signed_residual(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    scalar = np.mean(target - canvas, axis=2, dtype=np.float32)
    return np.clip(scalar, -1.0, 1.0).astype(np.float32, copy=False)


def _region_mean_color(
    target: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> tuple[float, float, float]:
    h, w = target.shape[:2]
    x0 = max(0, int(cx - radius))
    x1 = min(w, int(cx + radius + 1))
    y0 = max(0, int(cy - radius))
    y1 = min(h, int(cy + radius + 1))
    patch = target[y0:y1, x0:x1]
    if patch.size == 0:
        px = target[cy, cx]
        return (float(px[0]), float(px[1]), float(px[2]))
    avg = patch.mean(axis=(0, 1), dtype=np.float32)
    return (float(avg[0]), float(avg[1]), float(avg[2]))


def _grid_initialized_batch(
    target: np.ndarray,
    *,
    count: int,
    alpha: float,
) -> LivePolygonBatch:
    h, w = target.shape[:2]
    cols = int(np.ceil(np.sqrt(count * w / max(h, 1))))
    rows = int(np.ceil(count / max(cols, 1)))

    cell_w = max(1.0, float(w) / max(cols, 1))
    cell_h = max(1.0, float(h) / max(rows, 1))

    centers = np.zeros((count, 2), dtype=np.float32)
    sizes = np.zeros((count, 2), dtype=np.float32)
    rotations = np.zeros((count,), dtype=np.float32)
    colors = np.zeros((count, 3), dtype=np.float32)
    alphas = np.full((count,), float(alpha), dtype=np.float32)
    shape_types = np.full((count,), SHAPE_QUAD, dtype=np.int32)
    shape_params = np.zeros((count, 6), dtype=np.float32)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= count:
                break
            x0 = int(np.floor(c * w / cols))
            x1 = int(np.floor((c + 1) * w / cols))
            y0 = int(np.floor(r * h / rows))
            y1 = int(np.floor((r + 1) * h / rows))

            cx = int(np.clip((x0 + x1) // 2, 0, w - 1))
            cy = int(np.clip((y0 + y1) // 2, 0, h - 1))

            centers[idx] = np.array([cx, cy], dtype=np.float32)
            sx = max(1.0, 0.5 * float(max(1, x1 - x0)))
            sy = max(1.0, 0.5 * float(max(1, y1 - y0)))
            sizes[idx] = np.array([sx, sy], dtype=np.float32)
            patch = target[y0:y1, x0:x1]
            if patch.size == 0:
                colors[idx] = np.array(
                    _region_mean_color(
                        target,
                        cx,
                        cy,
                        max(2, int(round(max(cell_w, cell_h) * 0.5))),
                    ),
                    dtype=np.float32,
                )
            else:
                colors[idx] = patch.mean(axis=(0, 1), dtype=np.float32)
            idx += 1

    return LivePolygonBatch(
        centers=centers,
        sizes=sizes,
        rotations=rotations,
        colors=colors,
        alphas=alphas,
        shape_types=shape_types,
        shape_params=shape_params,
    )


def _top_error_centers(
    error_map: np.ndarray, *, k: int, radius: int
) -> list[tuple[int, int]]:
    work = np.array(error_map, copy=True)
    h, w = work.shape
    centers: list[tuple[int, int]] = []

    for _ in range(max(0, int(k))):
        flat_idx = int(np.argmax(work))
        y, x = divmod(flat_idx, w)
        if work[y, x] <= 1e-12:
            break
        centers.append((x, y))

        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        work[y0:y1, x0:x1] = 0.0

    return centers


def _high_frequency_error_map(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    residual = np.mean(np.abs(target - canvas), axis=2, dtype=np.float32)
    smooth = gaussian_filter(residual, sigma=3.0, mode="reflect")
    hf = np.clip(residual - smooth, 0.0, None)
    return gaussian_filter(hf, sigma=1.0, mode="reflect").astype(np.float32, copy=False)


def _add_targeted_batch(
    optimizer: LiveJointOptimizer,
    *,
    target: np.ndarray,
    batch_size: int,
    size_px: float,
    alpha: float,
    high_frequency: bool,
) -> None:
    if batch_size <= 0:
        return

    if high_frequency:
        err = _high_frequency_error_map(target, optimizer.current_canvas)
    else:
        err = np.mean(
            (target - optimizer.current_canvas) ** 2, axis=2, dtype=np.float32
        )

    radius = max(2, int(round(size_px * 0.8)))
    centers = _top_error_centers(err, k=batch_size, radius=radius)

    for cx, cy in centers:
        color_hint = _region_mean_color(
            target, cx, cy, max(2, int(round(size_px * 0.6)))
        )
        optimizer.add_polygon(
            center_x=float(cx),
            center_y=float(cy),
            size_x=float(size_px),
            size_y=float(size_px),
            color=color_hint,
            alpha=float(alpha),
            shape_type=SHAPE_QUAD,
            rotation=0.0,
        )


def handle_phase7_control_key(
    key: str,
    *,
    controls: Phase7ControlState,
    screenshot_callback,
    quit_callback,
) -> str:
    normalized = (key or "").lower()
    if normalized == "p":
        controls.paused = not controls.paused
        return "pause"
    if normalized == "r":
        screenshot_callback()
        return "screenshot"
    if normalized == "q":
        controls.quit_requested = True
        quit_callback()
        return "quit"
    return "noop"


def _run_stage_steps(
    *,
    optimizer: LiveJointOptimizer,
    controls: Phase7ControlState,
    stage_name: str,
    steps: int,
    start_softness: float,
    end_softness: float,
    deadline_reached,
    emit_update,
    max_total_steps: int | None,
    total_points: int,
) -> int:
    executed = 0
    for i in range(max(0, steps)):
        if controls.quit_requested or deadline_reached():
            break
        if max_total_steps is not None and total_points + executed >= max_total_steps:
            break

        while (
            controls.paused and not controls.quit_requested and not deadline_reached()
        ):
            time.sleep(0.05)

        if controls.quit_requested or deadline_reached():
            break

        t = i / max(steps - 1, 1)
        softness = start_softness + (end_softness - start_softness) * t
        optimizer.step(float(softness))
        executed += 1

        if executed % 10 == 0:
            emit_update(stage_name)

    emit_update(stage_name)
    return executed


def execute_phase7_schedule(
    *,
    target_image: np.ndarray,
    plan: Phase7Plan,
    random_seed: int,
    minutes: float,
    hard_timeout_seconds: float | None,
    controls: Phase7ControlState,
    shared_update_callback,
    max_total_steps: int | None,
) -> Phase7ExecutionResult:
    if target_image.ndim != 3 or target_image.shape[2] != 3:
        raise ValueError("target_image must have shape (H, W, 3)")

    target = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)
    h, w = target.shape[:2]

    start_time = time.monotonic()
    soft_deadline = (
        start_time + max(0.0, float(minutes) * 60.0) if minutes > 0.0 else None
    )
    hard_deadline = (
        start_time + float(hard_timeout_seconds)
        if hard_timeout_seconds is not None and hard_timeout_seconds > 0.0
        else None
    )

    def _deadline_reached() -> bool:
        now = time.monotonic()
        return bool(
            (soft_deadline is not None and now >= soft_deadline)
            or (hard_deadline is not None and now >= hard_deadline)
        )

    rng = np.random.default_rng(random_seed)
    del rng

    init_batch = _grid_initialized_batch(
        target,
        count=int(plan.stage_a_initial_polygons),
        alpha=1.0,
    )

    # Preserve stage ordering while adapting work volume for shorter runtime budgets.
    runtime_scale = 1.0
    if minutes > 0.0:
        runtime_scale = float(np.clip((minutes * 60.0) / 300.0, 0.10, 2.0))

    def _scaled_count(value: int, *, minimum: int) -> int:
        if value <= 0:
            return 0
        return max(minimum, int(round(float(value) * runtime_scale)))

    stage_a_steps = _scaled_count(int(plan.stage_a_steps), minimum=20)
    stage_b_batches = _scaled_count(int(plan.stage_b_batches), minimum=1)
    stage_b_batch_size = _scaled_count(int(plan.stage_b_batch_size), minimum=1)
    stage_b_steps_per_batch = _scaled_count(int(plan.stage_b_steps_per_batch), minimum=12)
    stage_c_batches = _scaled_count(int(plan.stage_c_batches), minimum=1)
    stage_c_batch_size = _scaled_count(int(plan.stage_c_batch_size), minimum=1)
    stage_c_steps_per_batch = _scaled_count(int(plan.stage_c_steps_per_batch), minimum=8)
    stage_d_steps = _scaled_count(int(plan.stage_d_steps), minimum=40)

    optimizer = LiveJointOptimizer(
        target_image=target,
        rasterizer=SoftRasterizer(height=h, width=w),
        polygons=init_batch,
        config=LiveOptimizerConfig(
            color_lr=0.03,
            position_lr=0.004,
            size_lr=0.001,
            alpha_lr=0.0,
            position_update_interval=100,
            size_update_interval=500,
            max_fd_polygons=30,
            max_size_fd_polygons=20,
            render_chunk_size=50,
            allow_loss_increase=True,
        ),
    )

    loss_history: list[float] = [float(optimizer.loss_history[-1])]
    stage_markers: list[tuple[str, int]] = []
    batch_markers: list[int] = []

    def _emit(stage_name: str) -> None:
        canvas = np.array(optimizer.current_canvas, copy=True)
        loss = _rgb_mse(target, canvas)
        status = (
            f"stage={stage_name} polygons={optimizer.polygons.count} rgb_mse={loss:.6f}"
        )
        shared_update_callback(
            canvas,
            optimizer.polygons.copy(),
            w,
            list(loss_history),
            [],
            list(batch_markers),
            stage_name,
            len(loss_history),
            loss,
            True,
            status,
            list(stage_markers),
        )

    def _extend_history(count: int) -> None:
        if count <= 0:
            return
        start = max(1, len(optimizer.loss_history) - count)
        for idx in range(start, len(optimizer.loss_history)):
            loss_history.append(float(optimizer.loss_history[idx]))

    # Stage A: grid init + color-only ADAM.
    stage_markers.append(("A", len(loss_history)))
    optimizer.config = replace(
        optimizer.config,
        position_update_interval=0,
        size_update_interval=0,
        max_fd_polygons=0,
    )
    done = _run_stage_steps(
        optimizer=optimizer,
        controls=controls,
        stage_name="A",
        steps=int(stage_a_steps),
        start_softness=0.4,
        end_softness=0.05,
        deadline_reached=_deadline_reached,
        emit_update=_emit,
        max_total_steps=max_total_steps,
        total_points=len(loss_history),
    )
    _extend_history(done)

    # Stage B: targeted medium additions.
    if not controls.quit_requested and not _deadline_reached():
        stage_markers.append(("B", len(loss_history)))
    for batch_idx in range(stage_b_batches):
        if controls.quit_requested or _deadline_reached():
            break
        if max_total_steps is not None and len(loss_history) >= max_total_steps:
            break

        t = batch_idx / max(stage_b_batches - 1, 1)
        size_px = float(
            plan.stage_b_size_start
            + (plan.stage_b_size_end - plan.stage_b_size_start) * t
        )
        checkpoint_len = len(loss_history)
        checkpoint_loss = float(optimizer.loss_history[-1])
        checkpoint_polygons = optimizer.polygons.copy()
        checkpoint_canvas = np.array(optimizer.current_canvas, copy=True)
        _add_targeted_batch(
            optimizer,
            target=target,
            batch_size=int(stage_b_batch_size),
            size_px=size_px,
            alpha=0.50,
            high_frequency=False,
        )
        batch_markers.append(len(loss_history))
        optimizer.config = replace(
            optimizer.config,
            position_update_interval=100,
            size_update_interval=500,
            max_fd_polygons=30,
            max_size_fd_polygons=20,
        )
        done = _run_stage_steps(
            optimizer=optimizer,
            controls=controls,
            stage_name="B",
            steps=int(stage_b_steps_per_batch),
            start_softness=0.22,
            end_softness=0.04,
            deadline_reached=_deadline_reached,
            emit_update=_emit,
            max_total_steps=max_total_steps,
            total_points=len(loss_history),
        )
        _extend_history(done)
        if float(optimizer.loss_history[-1]) > checkpoint_loss:
            optimizer.restore_state(
                checkpoint_polygons,
                checkpoint_canvas,
                checkpoint_loss,
                record_loss=False,
            )
            del loss_history[checkpoint_len:]
            if batch_markers and batch_markers[-1] >= checkpoint_len:
                batch_markers.pop()
            _emit("B")

    # Stage C: high-frequency detail additions.
    if not controls.quit_requested and not _deadline_reached():
        stage_markers.append(("C", len(loss_history)))
    for batch_idx in range(stage_c_batches):
        if controls.quit_requested or _deadline_reached():
            break
        if max_total_steps is not None and len(loss_history) >= max_total_steps:
            break

        t = batch_idx / max(stage_c_batches - 1, 1)
        size_px = float(
            plan.stage_c_size_start
            + (plan.stage_c_size_end - plan.stage_c_size_start) * t
        )
        checkpoint_len = len(loss_history)
        checkpoint_loss = float(optimizer.loss_history[-1])
        checkpoint_polygons = optimizer.polygons.copy()
        checkpoint_canvas = np.array(optimizer.current_canvas, copy=True)
        _add_targeted_batch(
            optimizer,
            target=target,
            batch_size=int(stage_c_batch_size),
            size_px=size_px,
            alpha=0.40,
            high_frequency=True,
        )
        batch_markers.append(len(loss_history))
        optimizer.config = replace(
            optimizer.config,
            position_update_interval=100,
            size_update_interval=500,
            max_fd_polygons=30,
            max_size_fd_polygons=20,
        )
        done = _run_stage_steps(
            optimizer=optimizer,
            controls=controls,
            stage_name="C",
            steps=int(stage_c_steps_per_batch),
            start_softness=0.16,
            end_softness=0.03,
            deadline_reached=_deadline_reached,
            emit_update=_emit,
            max_total_steps=max_total_steps,
            total_points=len(loss_history),
        )
        _extend_history(done)
        if float(optimizer.loss_history[-1]) > checkpoint_loss:
            optimizer.restore_state(
                checkpoint_polygons,
                checkpoint_canvas,
                checkpoint_loss,
                record_loss=False,
            )
            del loss_history[checkpoint_len:]
            if batch_markers and batch_markers[-1] >= checkpoint_len:
                batch_markers.pop()
            _emit("C")

    # Stage D: global refinement with position updates for all polygons.
    if not controls.quit_requested and not _deadline_reached():
        stage_markers.append(("D", len(loss_history)))
    optimizer.config = replace(
        optimizer.config,
        position_update_interval=50,
        size_update_interval=0,
        max_fd_polygons=None,
    )
    done = _run_stage_steps(
        optimizer=optimizer,
        controls=controls,
        stage_name="D",
        steps=int(stage_d_steps),
        start_softness=0.12,
        end_softness=0.02,
        deadline_reached=_deadline_reached,
        emit_update=_emit,
        max_total_steps=max_total_steps,
        total_points=len(loss_history),
    )
    _extend_history(done)

    final_canvas = np.array(optimizer.current_canvas, copy=True)
    final_rgb_mse = _rgb_mse(target, final_canvas)

    shared_update_callback(
        final_canvas,
        optimizer.polygons.copy(),
        w,
        list(loss_history),
        [],
        list(batch_markers),
        "done",
        len(loss_history),
        final_rgb_mse,
        False,
        "complete",
        list(stage_markers),
    )

    return Phase7ExecutionResult(
        final_canvas=final_canvas,
        final_loss=final_rgb_mse,
        polygon_count=optimizer.polygons.count,
        iterations=len(loss_history),
        loss_history=list(loss_history),
        resolution_markers=[],
        batch_markers=list(batch_markers),
        stage_markers=list(stage_markers),
    )


def run_phase7_headless(
    *,
    target_image: np.ndarray,
    segmentation_map: np.ndarray | None,
    plan: Phase7Plan,
    random_seed: int,
    minutes: float,
    hard_timeout_seconds: float | None = None,
    max_total_steps: int | None = None,
) -> Phase7ExecutionResult:
    del segmentation_map
    controls = Phase7ControlState()

    def _noop_update(
        _canvas: np.ndarray,
        _polygons: LivePolygonBatch,
        _polygon_resolution: int,
        _losses: list[float],
        _resolution_markers: list[int],
        _batch_markers: list[int],
        _stage_name: str,
        _iteration: int,
        _loss: float,
        _running: bool,
        _status: str,
        _stage_markers: list[tuple[str, int]],
    ) -> None:
        return

    return execute_phase7_schedule(
        target_image=target_image,
        plan=plan,
        random_seed=random_seed,
        minutes=minutes,
        hard_timeout_seconds=hard_timeout_seconds,
        controls=controls,
        shared_update_callback=_noop_update,
        max_total_steps=max_total_steps,
    )


def run_phase7_live_display(
    *,
    target_image: np.ndarray,
    segmentation_map: np.ndarray | None,
    plan: Phase7Plan,
    random_seed: int,
    minutes: float,
    hard_timeout_seconds: float | None = None,
    update_interval_ms: int = 5000,
    close_after_seconds: float | None = None,
    max_total_steps: int | None = None,
) -> Phase7ExecutionResult:
    del segmentation_map

    target = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)
    h, w = target.shape[:2]
    blank = np.ones_like(target, dtype=np.float32)

    shared = _SharedViewState(
        target=np.array(target, copy=True),
        canvas=np.array(blank, copy=True),
        signed_residual=_signed_residual(target, blank),
        loss_history=[_rgb_mse(target, blank)],
        stage_markers=[],
        polygon_count=0,
        iteration=0,
        stage_name="init",
        running=True,
        status_line="initializing",
        polygon_sizes=np.zeros((0,), dtype=np.float32),
    )

    controls = Phase7ControlState()
    lock = threading.Lock()
    result_holder: dict[str, Phase7ExecutionResult] = {}

    def _update_shared(
        canvas: np.ndarray,
        polygons: LivePolygonBatch,
        _polygon_resolution: int,
        losses: list[float],
        _resolution_markers: list[int],
        _batch_markers: list[int],
        stage_name: str,
        iteration: int,
        _loss: float,
        running: bool,
        status: str,
        stage_markers: list[tuple[str, int]],
    ) -> None:
        sizes = np.maximum(polygons.sizes[:, 0], polygons.sizes[:, 1]).astype(
            np.float32
        )
        with lock:
            shared.canvas = np.array(canvas, copy=True)
            shared.signed_residual = _signed_residual(shared.target, shared.canvas)
            shared.loss_history = list(losses)
            shared.stage_markers = list(stage_markers)
            shared.polygon_count = int(polygons.count)
            shared.iteration = int(iteration)
            shared.stage_name = str(stage_name)
            shared.running = bool(running)
            shared.status_line = str(status)
            shared.polygon_sizes = sizes

    def _worker() -> None:
        result_holder["result"] = execute_phase7_schedule(
            target_image=target,
            plan=plan,
            random_seed=random_seed,
            minutes=minutes,
            hard_timeout_seconds=hard_timeout_seconds,
            controls=controls,
            shared_update_callback=_update_shared,
            max_total_steps=max_total_steps,
        )

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.30, wspace=0.24)

    ax_target = fig.add_subplot(gs[0, 0])
    ax_canvas = fig.add_subplot(gs[0, 1])
    ax_error = fig.add_subplot(gs[0, 2])

    ax_curve = fig.add_subplot(gs[1, 0:2])
    right = gs[1, 2].subgridspec(2, 1, height_ratios=[0.62, 0.38], hspace=0.25)
    ax_stats = fig.add_subplot(right[0, 0])
    ax_sizes = fig.add_subplot(right[1, 0])

    im_target = ax_target.imshow(shared.target)
    im_target.set_interpolation("nearest")
    ax_target.set_title("Target")
    ax_target.set_xticks([])
    ax_target.set_yticks([])

    im_canvas = ax_canvas.imshow(shared.canvas)
    im_canvas.set_interpolation("nearest")
    ax_canvas.set_title("Current Canvas")
    ax_canvas.set_xticks([])
    ax_canvas.set_yticks([])

    im_error = ax_error.imshow(
        shared.signed_residual, cmap="coolwarm", vmin=-1.0, vmax=1.0
    )
    im_error.set_interpolation("nearest")
    ax_error.set_title("Signed Residual")
    ax_error.set_xticks([])
    ax_error.set_yticks([])

    ax_curve.set_title("MSE Curve (log) with Stage Transitions")
    ax_curve.set_xlabel("Iteration")
    ax_curve.set_ylabel("MSE (log)")
    ax_curve.set_yscale("log")
    ax_curve.grid(True, alpha=0.25)
    (loss_line,) = ax_curve.plot([], [], color="tab:blue", linewidth=1.8)
    marker_lines: list = []

    ax_stats.axis("off")
    stats_text = ax_stats.text(
        0.01,
        0.98,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=ax_stats.transAxes,
    )

    ax_sizes.set_title("Polygon Size Histogram")
    ax_sizes.set_xlabel("Size")
    ax_sizes.set_ylabel("Count")

    def _save_screenshot() -> Path:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"phase7_single_{ts}.png"
        fig.savefig(out, dpi=170, bbox_inches="tight")
        return out

    def _quit() -> None:
        controls.quit_requested = True

    def _on_key(event) -> None:
        handle_phase7_control_key(
            str(event.key),
            controls=controls,
            screenshot_callback=_save_screenshot,
            quit_callback=_quit,
        )

    def _on_close(_: object) -> None:
        controls.quit_requested = True

    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("close_event", _on_close)

    if close_after_seconds is not None and close_after_seconds > 0:
        timer = fig.canvas.new_timer(interval=int(close_after_seconds * 1000))
        timer.add_callback(lambda: plt.close(fig))
        timer.start()

    def update(_: int):
        with lock:
            canvas = np.array(shared.canvas, copy=True)
            signed = np.array(shared.signed_residual, copy=True)
            losses = np.array(shared.loss_history, dtype=np.float64)
            stage_markers = list(shared.stage_markers)
            stage_name = str(shared.stage_name)
            iteration = int(shared.iteration)
            polygon_count = int(shared.polygon_count)
            running = bool(shared.running)
            status = str(shared.status_line)
            paused = bool(controls.paused)
            sizes = np.array(shared.polygon_sizes, copy=True)

        im_canvas.set_data(canvas)
        im_error.set_data(signed)

        x = np.arange(losses.size)
        if losses.size > 0:
            loss_line.set_data(x, np.maximum(losses, 1e-9))
            y = np.maximum(losses, 1e-9)
            y_min = max(1e-9, float(np.min(y) * 0.9))
            y_max = max(float(np.max(y) * 1.1), y_min * 10.0)
            ax_curve.set_ylim(y_min, y_max)
            ax_curve.set_xlim(0, max(10, int(x[-1]) + 5))

        for line in marker_lines:
            line.remove()
        marker_lines.clear()

        for name, idx in stage_markers:
            marker_lines.append(
                ax_curve.axvline(
                    int(idx), linestyle="--", color="gray", alpha=0.65, linewidth=1.1
                )
            )
            ax_curve.text(
                int(idx),
                ax_curve.get_ylim()[1] * 0.92,
                name,
                fontsize=8,
                ha="center",
                va="top",
                color="gray",
            )

        stats_text.set_text(
            "\n".join(
                [
                    f"stage      : {stage_name}",
                    f"iteration  : {iteration}",
                    f"polygons   : {polygon_count}",
                    f"rgb mse    : {losses[-1]:.6f}"
                    if losses.size
                    else "rgb mse    : n/a",
                    f"paused     : {paused}",
                    f"state      : {status}",
                    "keys: P pause | R screenshot | Q quit",
                ]
            )
        )

        ax_sizes.clear()
        ax_sizes.set_title("Polygon Size Histogram")
        ax_sizes.set_xlabel("Size")
        ax_sizes.set_ylabel("Count")
        if sizes.size > 0:
            bins = min(10, max(3, int(np.sqrt(sizes.size))))
            hist, edges = np.histogram(sizes, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = np.maximum(edges[1:] - edges[:-1], 1e-3)
            ax_sizes.bar(
                centers, hist, width=widths * 0.9, color="tab:orange", alpha=0.8
            )

        if controls.quit_requested and not running:
            anim.event_source.stop()

        fig.canvas.draw_idle()
        return (im_canvas, im_error, loss_line, stats_text)

    anim = FuncAnimation(
        fig,
        update,
        interval=max(200, int(update_interval_ms)),
        blit=False,
        cache_frame_data=False,
    )
    _ = anim

    plt.show()

    controls.quit_requested = True
    worker.join(timeout=5.0)

    if "result" in result_holder:
        return result_holder["result"]

    with lock:
        canvas = np.array(shared.canvas, copy=True)
        losses = list(shared.loss_history)
        stage_marks = list(shared.stage_markers)
        poly_count = int(shared.polygon_count)

    return Phase7ExecutionResult(
        final_canvas=canvas,
        final_loss=float(losses[-1]) if losses else _rgb_mse(target, canvas),
        polygon_count=poly_count,
        iterations=len(losses),
        loss_history=losses,
        resolution_markers=[],
        batch_markers=[],
        stage_markers=stage_marks,
    )
