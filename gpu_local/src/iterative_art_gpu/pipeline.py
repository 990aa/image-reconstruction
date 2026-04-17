from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.filters import sobel_h, sobel_v

from iterative_art_gpu.constants import (
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
)
from iterative_art_gpu.models import (
    LivePolygonBatch,
    PhasePlan,
    PhaseResult,
    PreprocessedTarget,
    SequentialStageConfig,
)
from iterative_art_gpu.optimizer import GPUSequentialHillClimber
from iterative_art_gpu.renderer import GPUCoreRenderer


ProgressCallback = Callable[[str, int, int, np.ndarray, np.ndarray, list[float]], None]
StageCheckpointCallback = Callable[
    [str, np.ndarray, dict[str, float | int | str]], None
]


def make_empty_live_batch() -> LivePolygonBatch:
    return LivePolygonBatch(
        centers=np.zeros((0, 2), dtype=np.float32),
        sizes=np.zeros((0, 2), dtype=np.float32),
        rotations=np.zeros((0,), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        alphas=np.zeros((0,), dtype=np.float32),
        shape_types=np.zeros((0,), dtype=np.int32),
        shape_params=np.zeros((0, 6), dtype=np.float32),
    )


def build_phase_plan(
    base_resolution: int,
    polygon_budget: int,
    complexity_score: float,
) -> PhasePlan:
    del complexity_score

    budget = max(1, int(polygon_budget))
    stage_a_count = min(200, budget // 6)
    stage_b_count = min(400, max(0, budget // 3))
    stage_c_count = max(0, budget - stage_a_count - stage_b_count)

    stage_a_res = max(50, min(100, int(base_resolution)))
    stage_b_res = max(stage_a_res, min(150, int(base_resolution)))
    stage_c_res = int(base_resolution)

    return PhasePlan(
        polygon_budget=budget,
        stages=(
            SequentialStageConfig(
                "foundation",
                stage_a_res,
                stage_a_count,
                80,
                160,
                max(2.5, stage_a_res * 0.12),
                max(4.0, stage_a_res * 0.34),
                0.55,
                0.85,
                0.75,
                (SHAPE_ELLIPSE, SHAPE_QUAD),
                False,
                80,
                5,
                2.5,
                0.18,
                15.0,
            ),
            SequentialStageConfig(
                "structure",
                stage_b_res,
                stage_b_count,
                64,
                128,
                max(1.8, stage_b_res * 0.035),
                max(3.5, stage_b_res * 0.16),
                0.40,
                0.72,
                0.32,
                (SHAPE_ELLIPSE, SHAPE_QUAD, SHAPE_TRIANGLE),
                False,
                60,
                5,
                4.0,
                0.18,
                15.0,
            ),
            SequentialStageConfig(
                "detail",
                stage_c_res,
                stage_c_count,
                72,
                156,
                max(0.9, stage_c_res * 0.006),
                max(2.2, stage_c_res * 0.040),
                0.28,
                0.60,
                0.055,
                (SHAPE_ELLIPSE, SHAPE_TRIANGLE, SHAPE_THIN_STROKE),
                True,
                70,
                5,
                6.0,
                0.18,
                15.0,
            ),
        ),
    )


def _resize_float_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_h, target_w = size
    if image.shape[:2] == (target_h, target_w):
        return image.astype(np.float32, copy=False)

    pil = Image.fromarray(
        (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8),
        mode="RGB",
    )
    resized = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(
        np.float32, copy=False
    )


def prepare_square_image(
    image_path: str | Path,
    *,
    resolution: int,
    fit_mode: str = "crop",
) -> tuple[np.ndarray, tuple[int, int]]:
    fit = str(fit_mode).lower().strip()
    if fit not in {"crop", "letterbox"}:
        raise ValueError("fit_mode must be 'crop' or 'letterbox'")

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        orig_w, orig_h = rgb.size

        if fit == "crop":
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
            paste_x = (resolution - new_w) // 2
            paste_y = (resolution - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))
            fitted = canvas

    arr = np.asarray(fitted, dtype=np.float32) / 255.0
    return arr.astype(np.float32, copy=False), (orig_w, orig_h)


def preprocess_target_array(
    target_rgb: np.ndarray, base_resolution: int
) -> PreprocessedTarget:
    target = _resize_float_image(target_rgb, (base_resolution, base_resolution))
    gray = np.mean(target, axis=2)
    gx = sobel_h(gray)
    gy = sobel_v(gray)
    grad = np.hypot(gx, gy).astype(np.float32, copy=False)
    structure = np.clip(
        (grad - np.min(grad)) / max(float(np.max(grad) - np.min(grad)), 1e-8),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)

    center = np.mean(target, axis=(0, 1), keepdims=True)
    mean_abs_dev = float(np.mean(np.abs(target - center)))
    raw = float(np.mean(structure) / max(mean_abs_dev, 1e-6))
    complexity = float(np.clip(raw / (1.0 + raw), 0.0, 1.0))

    return PreprocessedTarget(
        base_resolution=int(base_resolution),
        target_rgb=target,
        pyramid=[],
        segmentation_map=np.zeros((1, 1), dtype=np.int32),
        cluster_centroids_lab=np.zeros((1, 1), dtype=np.float32),
        cluster_centroids_rgb=np.zeros((1, 1), dtype=np.float32),
        cluster_variances_lab=np.zeros((1, 1), dtype=np.float32),
        structure_map=structure,
        gradient_angle_map=np.arctan2(gy, gx).astype(np.float32, copy=False),
        complexity_score=complexity,
        recommended_polygons=1500,
        recommended_k=10,
        recommended_size_schedule={},
    )


def _scale_polygons(
    polygons: LivePolygonBatch, old_res: int, new_res: int
) -> LivePolygonBatch:
    if polygons.count == 0:
        return make_empty_live_batch()

    scale = float(new_res) / max(float(old_res), 1.0)
    batch = polygons.copy()
    batch.centers *= scale
    batch.sizes = np.clip(batch.sizes * scale, 0.5, None)
    batch.centers[:, 0] = np.clip(batch.centers[:, 0], 0.0, new_res - 1.0)
    batch.centers[:, 1] = np.clip(batch.centers[:, 1], 0.0, new_res - 1.0)

    stroke_mask = batch.shape_types == SHAPE_THIN_STROKE
    batch.shape_params[stroke_mask, 0:3] *= scale
    return batch


def _compute_stage_maps(
    stage_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = np.mean(stage_target, axis=2)
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    structure_map = np.clip(mag / max(float(np.percentile(mag, 99.0)), 1e-6), 0.0, 1.0)
    angle_map = np.arctan2(gy, gx)
    linearity_map = np.clip(
        np.sqrt(
            uniform_filter(np.cos(angle_map), 7) ** 2
            + uniform_filter(np.sin(angle_map), 7) ** 2
        ),
        0.0,
        1.0,
    )
    return (
        structure_map.astype(np.float32, copy=False),
        angle_map.astype(np.float32, copy=False),
        linearity_map.astype(np.float32, copy=False),
    )


def run_phase_local_gpu(
    *,
    target_rgb: np.ndarray,
    resolution: int,
    polygons: int,
    minutes: float,
    seed: int,
    progress_callback: ProgressCallback | None = None,
    stage_checkpoint_callback: StageCheckpointCallback | None = None,
    progress_interval_seconds: float = 1.0,
) -> PhaseResult:
    prep = preprocess_target_array(target_rgb, resolution)
    plan = build_phase_plan(resolution, polygons, prep.complexity_score)
    rng = np.random.default_rng(seed)
    background = np.mean(prep.target_rgb, axis=(0, 1)).astype(np.float32, copy=False)

    batch = make_empty_live_batch()
    previous_resolution: int | None = None
    loss_history: list[float] = []
    stage_markers: list[tuple[str, int]] = []
    iteration = 0

    start_time = time.perf_counter()
    budget_seconds = max(float(minutes), 0.0) * 60.0
    last_progress_time = start_time

    optimizer: GPUSequentialHillClimber | None = None

    for stage in plan.stages:
        if budget_seconds > 0.0 and (time.perf_counter() - start_time) > budget_seconds:
            break
        if stage.shapes_to_add <= 0 and stage.name != "detail":
            continue

        stage_batch = (
            make_empty_live_batch()
            if previous_resolution is None
            else _scale_polygons(batch, previous_resolution, stage.resolution)
        )
        stage_target = _resize_float_image(
            prep.target_rgb, (stage.resolution, stage.resolution)
        )
        structure_map, angle_map, linearity_map = _compute_stage_maps(stage_target)

        optimizer = GPUSequentialHillClimber(
            target_image=stage_target,
            rasterizer=GPUCoreRenderer(stage.resolution, stage.resolution),
            polygons=stage_batch,
            background_color=background,
        )

        stage_markers.append((stage.name, iteration))
        local_index = 0

        while True:
            if (
                budget_seconds > 0.0
                and (time.perf_counter() - start_time) > budget_seconds
            ):
                break
            if stage.name != "detail" and local_index >= stage.shapes_to_add:
                break
            local_index += 1

            guide_map = np.mean(
                np.abs(optimizer.target_np - optimizer.current_canvas_np),
                axis=2,
            )
            if stage.high_frequency_only:
                guide_map = np.clip(
                    guide_map - gaussian_filter(guide_map, 2.5), 0.0, None
                )
                guide_map = guide_map + 0.40 * np.mean(
                    np.abs(optimizer.target_np - optimizer.current_canvas_np),
                    axis=2,
                )

            candidate = optimizer.search_next_shape(
                stage,
                guide_map,
                structure_map,
                angle_map,
                linearity_map,
                rng,
            )
            if candidate is not None:
                optimizer.commit_shape(candidate)
                batch = optimizer.polygons.copy()
                previous_resolution = stage.resolution
                loss_history.append(float(optimizer.current_mse))
                iteration += 1

            now = time.perf_counter()
            if (
                progress_callback is not None
                and (now - last_progress_time) >= max(progress_interval_seconds, 0.05)
                and optimizer is not None
            ):
                progress_callback(
                    stage.name,
                    iteration,
                    int(batch.count),
                    _resize_float_image(
                        optimizer.current_canvas_np, (resolution, resolution)
                    ),
                    prep.target_rgb,
                    list(loss_history),
                )
                last_progress_time = now

        if optimizer is not None and stage_checkpoint_callback is not None:
            stage_canvas = _resize_float_image(
                optimizer.current_canvas_np, (resolution, resolution)
            )
            stage_checkpoint_callback(
                stage.name,
                stage_canvas,
                {
                    "resolution": int(stage.resolution),
                    "accepted_shapes": int(batch.count),
                    "stage_loss": float(optimizer.current_mse),
                },
            )

    if optimizer is None:
        final_canvas = np.broadcast_to(
            background.reshape(1, 1, 3), prep.target_rgb.shape
        ).copy()
    elif (
        batch.count > 0
        and previous_resolution is not None
        and previous_resolution != resolution
    ):
        final_batch = _scale_polygons(batch, previous_resolution, resolution)
        final_optimizer = GPUSequentialHillClimber(
            target_image=prep.target_rgb,
            rasterizer=GPUCoreRenderer(resolution, resolution),
            polygons=final_batch,
            background_color=background,
        )
        batch = final_optimizer.polygons.copy()
        final_canvas = final_optimizer.current_canvas_np
    else:
        final_canvas = _resize_float_image(
            optimizer.current_canvas_np, (resolution, resolution)
        )

    final_canvas = np.clip(final_canvas, 0.0, 1.0).astype(np.float32, copy=False)

    if progress_callback is not None:
        progress_callback(
            "done",
            iteration,
            int(batch.count),
            final_canvas,
            prep.target_rgb,
            list(loss_history),
        )

    return PhaseResult(
        batch=batch,
        preprocessed=prep,
        background_color=background,
        final_canvas=final_canvas,
        loss_history=loss_history,
        stage_markers=stage_markers,
        iterations=iteration,
    )
