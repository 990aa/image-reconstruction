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
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, uniform_filter

from src.core_renderer import (
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
    LivePolygonBatch,
    SoftRasterizer,
)
from src.live_optimizer import (
    SequentialHillClimber,
    SequentialStageConfig,
    make_empty_live_batch,
)


@dataclass(frozen=True)
class Phase7Plan:
    polygon_budget: int
    stages: tuple[SequentialStageConfig, ...]


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
    force_growth_requests: int = 0
    force_decompose_requests: int = 0
    view_mode_index: int = 0
    residual_mode_index: int = 0
    show_segmentation_overlay: bool = False
    show_scatter_overlay: bool = True
    softness_scale: float = 1.0
    latest_softness: float = 0.08


@dataclass
class _SharedViewState:
    target: np.ndarray
    canvas: np.ndarray
    loss_history: list[float]
    resolution_markers: list[int]
    batch_markers: list[int]
    stage_markers: list[tuple[str, int]]
    polygon_count: int
    iteration: int
    stage_name: str
    running: bool
    status_line: str


def build_phase7_plan(
    *,
    base_resolution: int,
    polygon_budget: int,
    complexity_score: float,
) -> Phase7Plan:
    budget = max(1, int(polygon_budget))
    _ = float(complexity_score)

    stage_a_count = min(50, budget)
    stage_b_count = min(100, max(0, budget - stage_a_count))
    stage_c_count = max(0, budget - stage_a_count - stage_b_count)

    stage_a_res = max(24, min(50, int(base_resolution)))
    stage_b_res = max(stage_a_res, min(100, int(base_resolution)))
    stage_c_res = int(base_resolution)

    stages = (
        SequentialStageConfig(
            name="foundation",
            resolution=stage_a_res,
            shapes_to_add=stage_a_count,
            candidate_count=50,
            mutation_steps=100,
            size_min=max(1.0, stage_a_res * 0.03),
            size_max=max(3.0, stage_a_res * 0.20),
            alpha_min=0.06,
            alpha_max=0.20,
            softness=0.55,
            allowed_shapes=(SHAPE_ELLIPSE,),
            high_frequency_only=False,
            top_k_regions=50,
            region_window=5,
            mutation_shift_px=1.0,
            mutation_size_ratio=0.10,
            mutation_rotation_deg=5.0,
        ),
        SequentialStageConfig(
            name="structure",
            resolution=stage_b_res,
            shapes_to_add=stage_b_count,
            candidate_count=50,
            mutation_steps=100,
            size_min=max(1.0, stage_b_res * 0.02),
            size_max=max(3.0, stage_b_res * 0.10),
            alpha_min=0.05,
            alpha_max=0.18,
            softness=0.18,
            allowed_shapes=(SHAPE_ELLIPSE, SHAPE_TRIANGLE),
            high_frequency_only=False,
            top_k_regions=50,
            region_window=5,
            mutation_shift_px=1.0,
            mutation_size_ratio=0.10,
            mutation_rotation_deg=5.0,
        ),
        SequentialStageConfig(
            name="detail",
            resolution=stage_c_res,
            shapes_to_add=stage_c_count,
            candidate_count=50,
            mutation_steps=100,
            size_min=1.0,
            size_max=3.0,
            alpha_min=0.05,
            alpha_max=0.15,
            softness=0.035,
            allowed_shapes=(SHAPE_ELLIPSE, SHAPE_TRIANGLE, SHAPE_THIN_STROKE),
            high_frequency_only=True,
            top_k_regions=50,
            region_window=5,
            mutation_shift_px=1.0,
            mutation_size_ratio=0.10,
            mutation_rotation_deg=5.0,
        ),
    )
    return Phase7Plan(polygon_budget=budget, stages=stages)


def _rgb_mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False)
    return float(np.mean(diff * diff, dtype=np.float32))


def _resize_rgb(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image.astype(np.float32, copy=False)
    uint8 = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    pil = Image.fromarray(uint8, mode="RGB")
    resized = pil.resize((width, height), Image.Resampling.LANCZOS)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def _scale_polygons_to_resolution(
    polygons: LivePolygonBatch,
    *,
    old_resolution: int,
    new_resolution: int,
) -> LivePolygonBatch:
    if polygons.count == 0:
        return make_empty_live_batch()
    scale = float(new_resolution) / max(float(old_resolution), 1.0)
    centers = np.array(polygons.centers, copy=True) * scale
    sizes = np.array(polygons.sizes, copy=True) * scale
    rotations = np.array(polygons.rotations, copy=True)
    colors = np.array(polygons.colors, copy=True)
    alphas = np.array(polygons.alphas, copy=True)
    shape_types = np.array(polygons.shape_types, copy=True)
    shape_params = np.array(polygons.shape_params, copy=True)

    centers[:, 0] = np.clip(centers[:, 0], 0.0, new_resolution - 1.0)
    centers[:, 1] = np.clip(centers[:, 1], 0.0, new_resolution - 1.0)
    sizes[:] = np.clip(sizes, 0.5, None)

    stroke_mask = shape_types == SHAPE_THIN_STROKE
    shape_params[stroke_mask, 0:2] *= scale
    shape_params[stroke_mask, 2] *= scale

    return LivePolygonBatch(
        centers=centers,
        sizes=sizes,
        rotations=rotations,
        colors=colors,
        alphas=alphas,
        shape_types=shape_types,
        shape_params=shape_params,
    )


def _compute_structure_maps(
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = np.mean(target, axis=2, dtype=np.float32)
    gx = sobel(gray, axis=1, mode="reflect")
    gy = sobel(gray, axis=0, mode="reflect")
    magnitude = np.hypot(gx, gy).astype(np.float32, copy=False)
    scale = max(float(np.percentile(magnitude, 99.0)), 1e-6)
    structure = np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32, copy=False)
    angle = np.arctan2(gy, gx).astype(np.float32, copy=False)

    ux = np.cos(angle)
    uy = np.sin(angle)
    mean_ux = uniform_filter(ux, size=7, mode="reflect")
    mean_uy = uniform_filter(uy, size=7, mode="reflect")
    linearity = np.sqrt(mean_ux * mean_ux + mean_uy * mean_uy).astype(
        np.float32, copy=False
    )
    linearity = np.clip(linearity, 0.0, 1.0).astype(np.float32, copy=False)
    mean_mag = uniform_filter(structure, size=5, mode="reflect")
    mean_mag_sq = uniform_filter(structure * structure, size=5, mode="reflect")
    grad_variance = np.clip(mean_mag_sq - mean_mag * mean_mag, 0.0, None).astype(
        np.float32, copy=False
    )
    var_scale = max(float(np.percentile(grad_variance, 99.0)), 1e-6)
    grad_variance = np.clip(grad_variance / var_scale, 0.0, 1.0).astype(
        np.float32, copy=False
    )
    return structure, angle, linearity, grad_variance


def _guide_map(
    target: np.ndarray,
    canvas: np.ndarray,
    *,
    edge_map: np.ndarray,
    high_frequency_only: bool,
) -> np.ndarray:
    residual = np.mean(np.abs(target - canvas), axis=2, dtype=np.float32)
    weighted = residual * np.clip(edge_map.astype(np.float32, copy=False), 0.0, 1.0)
    if high_frequency_only:
        weighted = weighted * np.clip(edge_map, 0.0, 1.0)
    if float(np.max(weighted)) <= 1e-8:
        return residual.astype(np.float32, copy=False)
    return weighted.astype(np.float32, copy=False)


def _annealed_size_bounds(
    *,
    resolution: int,
    total_budget: int,
    accepted_shapes: int,
) -> tuple[float, float]:
    progress = float(np.clip(accepted_shapes / max(total_budget, 1), 0.0, 1.0))
    width = float(resolution)
    if progress < 0.25:
        max_size = 0.20 * width
        min_size = max(1.5, 0.35 * max_size)
    elif progress < 0.50:
        max_size = 0.10 * width
        min_size = max(1.2, 0.30 * max_size)
    elif progress < 0.75:
        max_size = 0.05 * width
        min_size = max(1.0, 0.24 * max_size)
    else:
        max_size = float(np.clip(0.015 * width, 1.0, 3.0))
        min_size = 1.0
    return float(min_size), float(max(max_size, min_size))


def handle_phase7_control_key(
    key: str,
    *,
    controls: Phase7ControlState,
    screenshot_callback,
    quit_callback,
) -> str:
    norm = key.lower().strip()
    if norm == "p":
        controls.paused = not controls.paused
        return "pause"
    if norm == "r":
        screenshot_callback()
        return "screenshot"
    if norm == "q":
        controls.quit_requested = True
        quit_callback()
        return "quit"
    if norm == "1":
        controls.view_mode_index = 0
        return "view-set"
    if norm == "2":
        controls.view_mode_index = 1
        return "view-set"
    if norm == "3":
        controls.view_mode_index = 2
        return "view-set"
    if norm == "v":
        controls.view_mode_index = (controls.view_mode_index + 1) % 3
        return "view-cycle"
    if norm == "e":
        controls.residual_mode_index = (controls.residual_mode_index + 1) % 3
        return "residual-mode"
    if norm == "s":
        controls.show_segmentation_overlay = not controls.show_segmentation_overlay
        return "segmentation-toggle"
    if norm == "x":
        controls.show_scatter_overlay = not controls.show_scatter_overlay
        return "scatter-toggle"
    if norm == "g":
        controls.force_growth_requests += 1
        return "force-growth"
    if norm == "d":
        controls.force_decompose_requests += 1
        return "force-decompose"
    if norm == "+":
        controls.softness_scale = float(min(2.5, controls.softness_scale * 1.10))
        return "softness-up"
    if norm == "-":
        controls.softness_scale = float(max(0.2, controls.softness_scale / 1.10))
        return "softness-down"
    return "ignored"


def execute_phase7_schedule(
    *,
    target_image: np.ndarray,
    plan: Phase7Plan,
    random_seed: int,
    minutes: float,
    hard_timeout_seconds: float | None = None,
    controls: Phase7ControlState,
    shared_update_callback,
    max_total_steps: int | None = None,
    stage_checkpoint_callback=None,
) -> Phase7ExecutionResult:
    target = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)
    base_resolution = int(target.shape[0])
    rng = np.random.default_rng(random_seed)
    background_color = np.mean(target, axis=(0, 1), dtype=np.float32).astype(
        np.float32, copy=False
    )

    deadline = None
    if hard_timeout_seconds is not None:
        deadline = time.perf_counter() + float(hard_timeout_seconds)
    elif minutes > 0.0:
        deadline = time.perf_counter() + float(minutes) * 60.0

    polygons = make_empty_live_batch()
    previous_resolution = None
    iteration = 0
    loss_history: list[float] = []
    resolution_markers: list[int] = [0]
    batch_markers: list[int] = []
    stage_markers: list[tuple[str, int]] = []
    final_canvas = np.broadcast_to(background_color, target.shape).copy()
    final_loss = _rgb_mse(final_canvas, target)
    running = True

    def _emit_update(
        canvas: np.ndarray,
        polygon_resolution: int,
        stage_name: str,
        status: str,
    ) -> None:
        shared_update_callback(
            np.array(canvas, copy=True),
            polygons.copy(),
            int(polygon_resolution),
            list(loss_history),
            list(resolution_markers),
            list(batch_markers),
            str(stage_name),
            int(iteration),
            float(loss_history[-1] if loss_history else final_loss),
            int(iteration),
            float(loss_history[-1] if loss_history else final_loss),
            0,
            bool(running),
            str(status),
            list(stage_markers),
        )

    for stage in plan.stages:
        if controls.quit_requested:
            break
        if max_total_steps is not None and iteration >= int(max_total_steps):
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if stage.shapes_to_add <= 0:
            continue

        if previous_resolution is None:
            stage_polygons = make_empty_live_batch()
        else:
            stage_polygons = _scale_polygons_to_resolution(
                polygons,
                old_resolution=previous_resolution,
                new_resolution=stage.resolution,
            )

        stage_target = _resize_rgb(target, width=stage.resolution, height=stage.resolution)
        rasterizer = SoftRasterizer(height=stage.resolution, width=stage.resolution)
        optimizer = SequentialHillClimber(
            target_image=stage_target,
            rasterizer=rasterizer,
            polygons=stage_polygons,
            background_color=background_color,
        )
        (
            structure_map,
            angle_map,
            linearity_map,
            gradient_variance_map,
        ) = _compute_structure_maps(stage_target)

        stage_markers.append((stage.name, len(loss_history)))
        resolution_markers.append(len(loss_history))
        _emit_update(optimizer.current_canvas, stage.resolution, stage.name, "stage-start")

        no_improvement_count = 0
        for local_idx in range(stage.shapes_to_add):
            while controls.paused and not controls.quit_requested:
                time.sleep(0.05)
            if controls.quit_requested:
                break
            if max_total_steps is not None and iteration >= int(max_total_steps):
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break

            size_min, size_max = _annealed_size_bounds(
                resolution=stage.resolution,
                total_budget=plan.polygon_budget,
                accepted_shapes=iteration,
            )
            effective_stage = SequentialStageConfig(
                name=stage.name,
                resolution=stage.resolution,
                shapes_to_add=stage.shapes_to_add,
                candidate_count=stage.candidate_count,
                mutation_steps=stage.mutation_steps,
                size_min=size_min,
                size_max=size_max,
                alpha_min=stage.alpha_min,
                alpha_max=stage.alpha_max,
                softness=float(max(1e-3, stage.softness * controls.softness_scale)),
                allowed_shapes=stage.allowed_shapes,
                high_frequency_only=stage.high_frequency_only,
                top_k_regions=stage.top_k_regions,
                region_window=stage.region_window,
                mutation_shift_px=stage.mutation_shift_px,
                mutation_size_ratio=stage.mutation_size_ratio,
                mutation_rotation_deg=stage.mutation_rotation_deg,
            )
            controls.latest_softness = effective_stage.softness

            guide_map = _guide_map(
                optimizer.target,
                optimizer.current_canvas,
                edge_map=structure_map,
                high_frequency_only=effective_stage.high_frequency_only,
            )
            candidate = optimizer.search_next_shape(
                stage=effective_stage,
                guide_map=guide_map,
                structure_map=structure_map,
                angle_map=angle_map,
                linearity_map=linearity_map,
                gradient_variance_map=gradient_variance_map,
                rng=rng,
            )
            if candidate is None:
                no_improvement_count += 1
                if no_improvement_count >= 10:
                    break
                continue

            optimizer.commit_shape(candidate)
            polygons = optimizer.polygons.copy()
            previous_resolution = stage.resolution
            iteration += 1
            loss_history.append(float(optimizer.current_mse))
            if local_idx == 0 or (local_idx + 1) % 10 == 0:
                batch_markers.append(len(loss_history) - 1)
            no_improvement_count = 0
            _emit_update(
                optimizer.current_canvas,
                stage.resolution,
                stage.name,
                f"accepted-shape-{local_idx + 1}",
            )

        if polygons.count > 0:
            stage_canvas_base = _resize_rgb(
                optimizer.current_canvas,
                width=base_resolution,
                height=base_resolution,
            )
        else:
            stage_canvas_base = np.broadcast_to(background_color, target.shape).copy()

        if stage_checkpoint_callback is not None:
            stage_checkpoint_callback(
                stage.name,
                stage_canvas_base,
                {
                    "resolution": int(stage.resolution),
                    "accepted_shapes": int(polygons.count),
                    "stage_loss": float(optimizer.current_mse),
                    "softness": float(controls.latest_softness),
                },
            )

        final_canvas = stage_canvas_base
        final_loss = _rgb_mse(final_canvas, target)

    running = False
    if polygons.count > 0 and previous_resolution is not None and previous_resolution != base_resolution:
        final_polygons = _scale_polygons_to_resolution(
            polygons,
            old_resolution=previous_resolution,
            new_resolution=base_resolution,
        )
        final_optimizer = SequentialHillClimber(
            target_image=target,
            rasterizer=SoftRasterizer(height=base_resolution, width=base_resolution),
            polygons=final_polygons,
            background_color=background_color,
        )
        final_canvas = final_optimizer.current_canvas
        final_loss = final_optimizer.current_mse
        polygons = final_optimizer.polygons.copy()

    _emit_update(final_canvas, base_resolution, "done", "finished")
    return Phase7ExecutionResult(
        final_canvas=np.array(final_canvas, copy=True),
        final_loss=float(final_loss),
        polygon_count=int(polygons.count),
        iterations=int(iteration),
        loss_history=list(loss_history),
        resolution_markers=list(resolution_markers),
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
    stage_checkpoint_callback=None,
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
        _stage_iteration: int,
        _stage_start_loss: float,
        _stage_position_updates: int,
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
        stage_checkpoint_callback=stage_checkpoint_callback,
    )


def run_phase7_live_display(
    *,
    target_image: np.ndarray,
    segmentation_map: np.ndarray | None,
    plan: Phase7Plan,
    random_seed: int,
    minutes: float,
    hard_timeout_seconds: float | None = None,
    update_interval_ms: int = 750,
    close_after_seconds: float | None = None,
    max_total_steps: int | None = None,
    stage_checkpoint_callback=None,
) -> Phase7ExecutionResult:
    del segmentation_map
    target = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)
    shared = _SharedViewState(
        target=np.array(target, copy=True),
        canvas=np.array(target, copy=True),
        loss_history=[],
        resolution_markers=[0],
        batch_markers=[],
        stage_markers=[],
        polygon_count=0,
        iteration=0,
        stage_name="init",
        running=True,
        status_line="initializing",
    )
    controls = Phase7ControlState()
    lock = threading.Lock()
    result_holder: dict[str, Phase7ExecutionResult] = {}

    def _update_shared(
        canvas: np.ndarray,
        polygons: LivePolygonBatch,
        _polygon_resolution: int,
        losses: list[float],
        resolution_markers: list[int],
        batch_markers: list[int],
        stage_name: str,
        iteration: int,
        _loss: float,
        _stage_iteration: int,
        _stage_start_loss: float,
        _stage_position_updates: int,
        running: bool,
        status: str,
        stage_markers: list[tuple[str, int]],
    ) -> None:
        with lock:
            shared.canvas = np.array(
                _resize_rgb(canvas, width=target.shape[1], height=target.shape[0]),
                copy=True,
            )
            shared.loss_history = list(losses)
            shared.resolution_markers = list(resolution_markers)
            shared.batch_markers = list(batch_markers)
            shared.stage_markers = list(stage_markers)
            shared.polygon_count = int(polygons.count)
            shared.iteration = int(iteration)
            shared.stage_name = str(stage_name)
            shared.running = bool(running)
            shared.status_line = str(status)

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
            stage_checkpoint_callback=stage_checkpoint_callback,
        )

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_target, ax_canvas, ax_curve = axes
    ax_target.imshow(target)
    ax_target.set_title("Target")
    ax_target.set_xticks([])
    ax_target.set_yticks([])

    def _save_screenshot():
        return None

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

    def _refresh(_: int):
        with lock:
            canvas = np.array(shared.canvas, copy=True)
            losses = np.array(shared.loss_history, dtype=np.float64)
            title = f"{shared.stage_name} | shapes={shared.polygon_count} | {shared.status_line}"
            running = bool(shared.running)

        ax_canvas.clear()
        ax_canvas.imshow(canvas)
        ax_canvas.set_title(title)
        ax_canvas.set_xticks([])
        ax_canvas.set_yticks([])

        ax_curve.clear()
        ax_curve.set_title("RGB MSE")
        ax_curve.set_xlabel("Accepted shape")
        ax_curve.set_ylabel("MSE")
        ax_curve.set_yscale("log")
        ax_curve.grid(True, alpha=0.25)
        if losses.size:
            ax_curve.plot(np.arange(losses.size), np.maximum(losses, 1e-9), color="tab:blue")

        if controls.quit_requested and not running:
            anim.event_source.stop()
        return ()

    anim = FuncAnimation(
        fig,
        _refresh,
        interval=max(200, int(update_interval_ms)),
        blit=False,
        cache_frame_data=False,
    )
    _ = anim

    plt.tight_layout()
    plt.show()
    controls.quit_requested = True
    worker.join(timeout=8.0)

    if "result" in result_holder:
        return result_holder["result"]

    return Phase7ExecutionResult(
        final_canvas=np.array(shared.canvas, copy=True),
        final_loss=_rgb_mse(shared.canvas, target),
        polygon_count=int(shared.polygon_count),
        iterations=int(shared.iteration),
        loss_history=list(shared.loss_history),
        resolution_markers=list(shared.resolution_markers),
        batch_markers=list(shared.batch_markers),
        stage_markers=list(shared.stage_markers),
    )
