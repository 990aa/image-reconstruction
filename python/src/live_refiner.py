from __future__ import annotations

import threading
import time
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
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

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
class phasePlan:
    polygon_budget: int
    stages: tuple[SequentialStageConfig, ...]


@dataclass(frozen=True)
class phaseExecutionResult:
    final_canvas: np.ndarray
    final_loss: float
    polygon_count: int
    iterations: int
    loss_history: list[float]
    resolution_markers: list[int]
    batch_markers: list[int]
    stage_markers: list[tuple[str, int]]


@dataclass
class phaseControlState:
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


def build_phase_plan(
    *,
    base_resolution: int,
    polygon_budget: int,
    complexity_score: float,
) -> phasePlan:
    budget = max(1, int(polygon_budget))
    _ = float(complexity_score)

    stage_a_count = min(200, budget // 6)
    stage_b_count = min(400, max(0, budget // 3))
    stage_c_count = max(0, budget - stage_a_count - stage_b_count)

    stage_a_res = max(50, min(100, int(base_resolution)))
    stage_b_res = max(stage_a_res, min(150, int(base_resolution)))
    stage_c_res = int(base_resolution)

    stages = (
        SequentialStageConfig(
            name="foundation",
            resolution=stage_a_res,
            shapes_to_add=stage_a_count,
            candidate_count=80,
            mutation_steps=160,
            size_min=max(2.5, stage_a_res * 0.12),
            size_max=max(4.0, stage_a_res * 0.34),
            alpha_min=0.55,
            alpha_max=0.85,
            softness=0.75,
            allowed_shapes=(SHAPE_ELLIPSE, SHAPE_QUAD),
            high_frequency_only=False,
            top_k_regions=80,
            region_window=5,
            mutation_shift_px=2.5,
            mutation_size_ratio=0.18,
            mutation_rotation_deg=15.0,
        ),
        SequentialStageConfig(
            name="structure",
            resolution=stage_b_res,
            shapes_to_add=stage_b_count,
            candidate_count=64,
            mutation_steps=128,
            size_min=max(1.8, stage_b_res * 0.035),
            size_max=max(3.5, stage_b_res * 0.16),
            alpha_min=0.40,
            alpha_max=0.72,
            softness=0.32,
            allowed_shapes=(SHAPE_ELLIPSE, SHAPE_QUAD, SHAPE_TRIANGLE),
            high_frequency_only=False,
            top_k_regions=60,
            region_window=5,
            mutation_shift_px=4.0,
            mutation_size_ratio=0.18,
            mutation_rotation_deg=15.0,
        ),
        SequentialStageConfig(
            name="detail",
            resolution=stage_c_res,
            shapes_to_add=stage_c_count,
            candidate_count=72,
            mutation_steps=156,
            size_min=max(0.9, stage_c_res * 0.006),
            size_max=max(2.2, stage_c_res * 0.040),
            alpha_min=0.28,
            alpha_max=0.60,
            softness=0.055,
            allowed_shapes=(SHAPE_ELLIPSE, SHAPE_TRIANGLE, SHAPE_THIN_STROKE),
            high_frequency_only=True,
            top_k_regions=70,
            region_window=5,
            mutation_shift_px=6.0,
            mutation_size_ratio=0.18,
            mutation_rotation_deg=15.0,
        ),
    )
    return phasePlan(polygon_budget=budget, stages=stages)


def _rgb_mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False)
    return float(np.mean(diff * diff, dtype=np.float32))


def _resize_rgb(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image.astype(np.float32, copy=False)
    uint8 = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    pil = Image.fromarray(uint8, mode="RGB")
    resized = pil.resize((width, height), Image.Resampling.LANCZOS)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(
        np.float32, copy=False
    )


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = np.mean(target, axis=2, dtype=np.float32)
    gy, gx = np.gradient(gray)
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
    return structure, angle, linearity


def _guide_map(
    target: np.ndarray,
    canvas: np.ndarray,
    *,
    edge_map: np.ndarray,
    high_frequency_only: bool,
) -> np.ndarray:
    residual = np.mean(np.abs(target - canvas), axis=2, dtype=np.float32)
    if not high_frequency_only:
        return residual.astype(np.float32, copy=False)
    smooth = gaussian_filter(residual, sigma=2.5, mode="reflect")
    high = np.clip(residual - smooth, 0.0, None)
    return (high + 0.40 * residual).astype(np.float32, copy=False)


def _normalized_error_map(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    err = np.mean(np.abs(target - canvas), axis=2, dtype=np.float32)
    scale = max(float(np.quantile(err, 0.99)), 1e-6)
    return np.clip(err / scale, 0.0, 1.0).astype(np.float32, copy=False)


def _draw_dashboard(
    *,
    fig,
    axes: np.ndarray,
    target: np.ndarray,
    canvas: np.ndarray,
    losses: np.ndarray,
    resolution_markers: list[int],
    batch_markers: list[int],
    stage_markers: list[tuple[str, int]],
    polygon_count: int,
    iteration: int,
    stage_name: str,
    status_line: str,
) -> None:
    ax_target, ax_canvas, ax_error, ax_curve = axes.reshape(-1)
    error_map = _normalized_error_map(target, canvas)
    current_loss = float(losses[-1]) if losses.size else _rgb_mse(canvas, target)

    ax_target.clear()
    ax_target.imshow(target)
    ax_target.set_title("Target")
    ax_target.set_xticks([])
    ax_target.set_yticks([])

    ax_canvas.clear()
    ax_canvas.imshow(canvas)
    ax_canvas.set_title(f"Reconstruction | shapes={polygon_count}")
    ax_canvas.set_xticks([])
    ax_canvas.set_yticks([])

    ax_error.clear()
    ax_error.imshow(error_map, cmap="magma", vmin=0.0, vmax=1.0)
    ax_error.set_title("Absolute Error")
    ax_error.set_xticks([])
    ax_error.set_yticks([])

    ax_curve.clear()
    ax_curve.set_title("RGB MSE Reduction")
    ax_curve.set_xlabel("Accepted shape")
    ax_curve.set_ylabel("MSE")
    ax_curve.set_yscale("log")
    ax_curve.grid(True, alpha=0.25)
    if losses.size:
        x = np.arange(losses.size, dtype=np.int32)
        ax_curve.plot(x, np.maximum(losses, 1e-9), color="tab:blue", linewidth=1.6)
        for marker in resolution_markers:
            if 0 <= marker < losses.size:
                ax_curve.axvline(marker, color="#aaaaaa", alpha=0.18, linewidth=1.0)
        for marker in batch_markers:
            if 0 <= marker < losses.size:
                ax_curve.axvline(marker, color="#2a9d8f", alpha=0.08, linewidth=0.8)
        for stage, marker in stage_markers:
            if 0 <= marker < losses.size:
                ax_curve.axvline(marker, color="#e76f51", alpha=0.20, linewidth=1.2)
                ax_curve.text(
                    marker,
                    max(float(np.max(losses)), 1e-6),
                    stage,
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=8,
                    color="#6d2e1f",
                )

    fig.suptitle(
        (
            f"Sequential Primitive Reconstruction | stage={stage_name} | "
            f"iteration={iteration} | loss={current_loss:.6f} | {status_line}"
        ),
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))


def _figure_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return np.array(rgba[:, :, :3], copy=True)


def handle_phase_control_key(
    key: str,
    *,
    controls: phaseControlState,
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


def execute_phase_schedule(
    *,
    target_image: np.ndarray,
    plan: phasePlan,
    random_seed: int,
    minutes: float,
    hard_timeout_seconds: float | None = None,
    controls: phaseControlState,
    shared_update_callback,
    max_total_steps: int | None = None,
    stage_checkpoint_callback=None,
) -> phaseExecutionResult:
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

        stage_target = _resize_rgb(
            target, width=stage.resolution, height=stage.resolution
        )
        rasterizer = SoftRasterizer(height=stage.resolution, width=stage.resolution)
        optimizer = SequentialHillClimber(
            target_image=stage_target,
            rasterizer=rasterizer,
            polygons=stage_polygons,
            background_color=background_color,
        )
        structure_map, angle_map, linearity_map = _compute_structure_maps(stage_target)

        stage_markers.append((stage.name, len(loss_history)))
        resolution_markers.append(len(loss_history))
        _emit_update(
            optimizer.current_canvas, stage.resolution, stage.name, "stage-start"
        )

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

            effective_stage = SequentialStageConfig(
                name=stage.name,
                resolution=stage.resolution,
                shapes_to_add=stage.shapes_to_add,
                candidate_count=stage.candidate_count,
                mutation_steps=stage.mutation_steps,
                size_min=stage.size_min,
                size_max=stage.size_max,
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
                rng=rng,
            )
            if candidate is None:
                no_improvement_count += 1
                if no_improvement_count >= 8:
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
    if (
        polygons.count > 0
        and previous_resolution is not None
        and previous_resolution != base_resolution
    ):
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
    return phaseExecutionResult(
        final_canvas=np.array(final_canvas, copy=True),
        final_loss=float(final_loss),
        polygon_count=int(polygons.count),
        iterations=int(iteration),
        loss_history=list(loss_history),
        resolution_markers=list(resolution_markers),
        batch_markers=list(batch_markers),
        stage_markers=list(stage_markers),
    )


def run_phase_headless(
    *,
    target_image: np.ndarray,
    segmentation_map: np.ndarray | None,
    plan: phasePlan,
    random_seed: int,
    minutes: float,
    hard_timeout_seconds: float | None = None,
    max_total_steps: int | None = None,
    stage_checkpoint_callback=None,
) -> phaseExecutionResult:
    del segmentation_map
    controls = phaseControlState()

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

    return execute_phase_schedule(
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


def record_phase_demo_gif(
    *,
    target_image: np.ndarray,
    segmentation_map: np.ndarray | None,
    plan: phasePlan,
    random_seed: int,
    minutes: float,
    output_path: Path,
    hard_timeout_seconds: float | None = None,
    max_total_steps: int | None = None,
    stage_checkpoint_callback=None,
    frame_stride: int = 2,
    frame_duration_ms: int = 120,
) -> tuple[phaseExecutionResult, dict[str, int | float | str]]:
    del segmentation_map
    target = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)
    controls = phaseControlState()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    frames: list[Image.Image] = []
    update_events = 0
    captured_frames = 0
    first_loss: float | None = None
    last_loss: float | None = None

    def _capture(
        canvas: np.ndarray,
        losses: list[float],
        resolution_markers: list[int],
        batch_markers: list[int],
        stage_name: str,
        iteration: int,
        status: str,
        stage_markers: list[tuple[str, int]],
        polygon_count: int,
    ) -> None:
        nonlocal captured_frames, first_loss, last_loss
        display_canvas = _resize_rgb(
            canvas,
            width=target.shape[1],
            height=target.shape[0],
        )
        loss_arr = np.asarray(losses, dtype=np.float64)
        if first_loss is None:
            first_loss = (
                float(loss_arr[-1])
                if loss_arr.size
                else _rgb_mse(display_canvas, target)
            )
        last_loss = (
            float(loss_arr[-1]) if loss_arr.size else _rgb_mse(display_canvas, target)
        )
        _draw_dashboard(
            fig=fig,
            axes=np.asarray(axes),
            target=target,
            canvas=np.clip(display_canvas, 0.0, 1.0).astype(np.float32, copy=False),
            losses=loss_arr,
            resolution_markers=resolution_markers,
            batch_markers=batch_markers,
            stage_markers=stage_markers,
            polygon_count=polygon_count,
            iteration=iteration,
            stage_name=stage_name,
            status_line=status,
        )
        frames.append(Image.fromarray(_figure_to_rgb(fig), mode="RGB"))
        captured_frames += 1

    def _shared_update(
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
        nonlocal update_events
        update_events += 1
        should_capture = (
            update_events == 1
            or status == "finished"
            or status == "stage-start"
            or (int(frame_stride) <= 1)
            or (update_events % int(max(frame_stride, 1)) == 0)
            or not running
        )
        if should_capture:
            _capture(
                canvas,
                losses,
                resolution_markers,
                batch_markers,
                stage_name,
                iteration,
                status,
                stage_markers,
                int(polygons.count),
            )

    result = execute_phase_schedule(
        target_image=target,
        plan=plan,
        random_seed=random_seed,
        minutes=minutes,
        hard_timeout_seconds=hard_timeout_seconds,
        controls=controls,
        shared_update_callback=_shared_update,
        max_total_steps=max_total_steps,
        stage_checkpoint_callback=stage_checkpoint_callback,
    )

    if not frames:
        _capture(
            result.final_canvas,
            result.loss_history,
            result.resolution_markers,
            result.batch_markers,
            "done",
            result.iterations,
            "finished",
            result.stage_markers,
            result.polygon_count,
        )

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(max(frame_duration_ms, 20)),
        loop=0,
        optimize=False,
    )
    plt.close(fig)

    metadata: dict[str, int | float | str] = {
        "output_path": str(output_path),
        "frames_saved": int(captured_frames),
        "update_events": int(update_events),
        "accepted_polygons": int(result.polygon_count),
        "iterations": int(result.iterations),
        "initial_recorded_loss": float(
            first_loss if first_loss is not None else result.final_loss
        ),
        "final_recorded_loss": float(
            last_loss if last_loss is not None else result.final_loss
        ),
        "frame_stride": int(max(frame_stride, 1)),
        "frame_duration_ms": int(max(frame_duration_ms, 20)),
    }
    return result, metadata


def run_phase_live_display(
    *,
    target_image: np.ndarray,
    segmentation_map: np.ndarray | None,
    plan: phasePlan,
    random_seed: int,
    minutes: float,
    hard_timeout_seconds: float | None = None,
    update_interval_ms: int = 750,
    close_after_seconds: float | None = None,
    max_total_steps: int | None = None,
    stage_checkpoint_callback=None,
) -> phaseExecutionResult:
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
    controls = phaseControlState()
    lock = threading.Lock()
    result_holder: dict[str, phaseExecutionResult] = {}

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
        result_holder["result"] = execute_phase_schedule(
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def _save_screenshot():
        return None

    def _quit() -> None:
        controls.quit_requested = True

    def _on_key(event) -> None:
        handle_phase_control_key(
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
            resolution_markers = list(shared.resolution_markers)
            batch_markers = list(shared.batch_markers)
            stage_markers = list(shared.stage_markers)
            polygon_count = int(shared.polygon_count)
            iteration = int(shared.iteration)
            stage_name = str(shared.stage_name)
            status_line = str(shared.status_line)
            running = bool(shared.running)

        _draw_dashboard(
            fig=fig,
            axes=np.asarray(axes),
            target=target,
            canvas=canvas,
            losses=losses,
            resolution_markers=resolution_markers,
            batch_markers=batch_markers,
            stage_markers=stage_markers,
            polygon_count=polygon_count,
            iteration=iteration,
            stage_name=stage_name,
            status_line=status_line,
        )

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

    plt.show()
    controls.quit_requested = True
    worker.join(timeout=8.0)

    if "result" in result_holder:
        return result_holder["result"]

    return phaseExecutionResult(
        final_canvas=np.array(shared.canvas, copy=True),
        final_loss=_rgb_mse(shared.canvas, target),
        polygon_count=int(shared.polygon_count),
        iterations=int(shared.iteration),
        loss_history=list(shared.loss_history),
        resolution_markers=list(shared.resolution_markers),
        batch_markers=list(shared.batch_markers),
        stage_markers=list(shared.stage_markers),
    )
