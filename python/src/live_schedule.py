from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from src.live_optimizer import LiveJointOptimizer, LiveOptimizerConfig
from src.live_renderer import SHAPE_ELLIPSE, LivePolygonBatch, SoftRasterizer


@dataclass(frozen=True)
class GrowthEvent:
    cycle_index: int
    target_region_center: tuple[float, float]
    placed_center: tuple[float, float]
    distance_to_target: float


@dataclass(frozen=True)
class GrowthCycleResult:
    cycle_index: int
    batch_size: int
    loss_before_cycle: float
    loss_before_addition: float
    loss_after_cycle: float
    optimization_steps: int
    converged_before_addition: bool


@dataclass(frozen=True)
class MultiResolutionRoundResult:
    round_name: str
    resolution: int
    loss_start: float
    loss_end: float
    boundary_error: float
    polygon_count: int
    optimization_steps: int


@dataclass(frozen=True)
class MultiResolutionResult:
    rounds: list[MultiResolutionRoundResult]
    total_optimization_steps: int
    final_loss: float


def default_growth_batch_schedule() -> list[int]:
    return [20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


def make_empty_live_batch() -> LivePolygonBatch:
    return LivePolygonBatch(
        centers=np.zeros((0, 2), dtype=np.float32),
        sizes=np.zeros((0, 2), dtype=np.float32),
        rotations=np.zeros((0,), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        alphas=np.zeros((0,), dtype=np.float32),
        shape_types=np.zeros((0,), dtype=np.int32),
    )


def make_random_live_batch_with_bounds(
    *,
    count: int,
    height: int,
    width: int,
    min_size: float,
    max_size: float,
    rng: np.random.Generator,
) -> LivePolygonBatch:
    if count < 0:
        raise ValueError("count must be non-negative")
    if min_size <= 0.0 or max_size <= 0.0:
        raise ValueError("size bounds must be positive")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size")

    centers = np.column_stack(
        [
            rng.uniform(0.0, width - 1.0, size=count),
            rng.uniform(0.0, height - 1.0, size=count),
        ]
    ).astype(np.float32)
    sizes = np.column_stack(
        [
            rng.uniform(min_size, max_size, size=count),
            rng.uniform(min_size, max_size, size=count),
        ]
    ).astype(np.float32)
    rotations = rng.uniform(0.0, 2.0 * np.pi, size=count).astype(np.float32)
    colors = rng.uniform(0.0, 1.0, size=(count, 3)).astype(np.float32)
    alphas = rng.uniform(0.2, 0.9, size=count).astype(np.float32)
    shape_types = np.full((count,), SHAPE_ELLIPSE, dtype=np.int32)

    return LivePolygonBatch(
        centers=centers,
        sizes=sizes,
        rotations=rotations,
        colors=colors,
        alphas=alphas,
        shape_types=shape_types,
    )


def _resize_rgb(image: np.ndarray, resolution: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    clipped = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    u8 = np.round(clipped * 255.0).astype(np.uint8)
    pil = Image.fromarray(u8, mode="RGB")
    resized = pil.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def _softness_for_step(
    *,
    step_index: int,
    horizon_steps: int,
    start_softness: float,
    end_softness: float,
) -> float:
    t = float(np.clip(step_index / max(horizon_steps, 1), 0.0, 1.0))
    return float(start_softness + (end_softness - start_softness) * t)


def _relative_loss_improvement(loss_history: Sequence[float], window: int = 100) -> float:
    if window <= 0:
        raise ValueError("window must be positive")
    if len(loss_history) < window + 1:
        return float("inf")

    start = float(loss_history[-(window + 1)])
    end = float(loss_history[-1])
    return float((start - end) / max(abs(start), 1e-8))


def _is_converged(
    loss_history: Sequence[float],
    *,
    window: int = 100,
    relative_threshold: float = 0.001,
) -> bool:
    return _relative_loss_improvement(loss_history, window=window) < relative_threshold


def _region_sum_map(error_map: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    h, w = error_map.shape
    if window > h or window > w:
        raise ValueError("window must fit within error map dimensions")

    integral = np.pad(error_map, ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)

    y2 = np.arange(window, h + 1)
    x2 = np.arange(window, w + 1)

    sums = (
        integral[y2[:, None], x2[None, :]]
        - integral[y2[:, None] - window, x2[None, :]]
        - integral[y2[:, None], x2[None, :] - window]
        + integral[y2[:, None] - window, x2[None, :] - window]
    )
    return sums.astype(np.float32, copy=False)


def highest_error_region_center(
    target: np.ndarray,
    canvas: np.ndarray,
    *,
    window: int = 5,
) -> tuple[tuple[float, float], tuple[int, int, int, int], np.ndarray]:
    if target.shape != canvas.shape:
        raise ValueError("target and canvas must have identical shapes")

    h, w, _ = target.shape
    actual_window = int(min(window, h, w))
    if actual_window <= 0:
        raise ValueError("window must be positive")

    error_map = np.sum((target - canvas) ** 2, axis=2, dtype=np.float32)
    region_scores = _region_sum_map(error_map, actual_window)

    max_index = int(np.argmax(region_scores))
    top_y, top_x = divmod(max_index, region_scores.shape[1])

    y0, x0 = int(top_y), int(top_x)
    y1 = int(y0 + actual_window)
    x1 = int(x0 + actual_window)

    center_x = float(x0 + (actual_window - 1) / 2.0)
    center_y = float(y0 + (actual_window - 1) / 2.0)

    patch_mean = target[y0:y1, x0:x1].mean(axis=(0, 1), dtype=np.float32)
    return (center_x, center_y), (x0, y0, x1, y1), patch_mean


def _refresh_optimizer_canvas(optimizer: LiveJointOptimizer, softness: float) -> float:
    render = optimizer.rasterizer.render(
        optimizer.polygons,
        softness=softness,
        chunk_size=optimizer.config.render_chunk_size,
    )
    optimizer.current_canvas = render.canvas
    loss = optimizer._loss(optimizer.current_canvas, optimizer.target)
    optimizer.loss_history.append(loss)
    return float(loss)


def optimize_until_converged(
    optimizer: LiveJointOptimizer,
    *,
    max_steps: int,
    convergence_window: int = 100,
    convergence_rel_threshold: float = 0.001,
    start_softness: float = 2.0,
    end_softness: float = 0.5,
    softness_horizon: int | None = None,
) -> tuple[int, bool]:
    if max_steps <= 0:
        return 0, _is_converged(
            optimizer.loss_history,
            window=convergence_window,
            relative_threshold=convergence_rel_threshold,
        )

    horizon = max_steps if softness_horizon is None else max(1, int(softness_horizon))
    start_step = optimizer.step_count

    for local_step in range(max_steps):
        softness = _softness_for_step(
            step_index=start_step + local_step,
            horizon_steps=horizon,
            start_softness=start_softness,
            end_softness=end_softness,
        )
        optimizer.step(softness=softness)
        if _is_converged(
            optimizer.loss_history,
            window=convergence_window,
            relative_threshold=convergence_rel_threshold,
        ):
            return local_step + 1, True

    return max_steps, False


def progressive_growth(
    optimizer: LiveJointOptimizer,
    *,
    batch_schedule: Sequence[int],
    max_steps_per_cycle: int = 120,
    post_add_steps: int = 20,
    convergence_window: int = 100,
    convergence_rel_threshold: float = 0.001,
    region_window: int = 5,
    new_polygon_alpha: float = 0.60,
    min_new_size: float = 2.5,
    max_new_size: float | None = None,
    shape_type: int = SHAPE_ELLIPSE,
    start_softness: float = 2.0,
    end_softness: float = 0.5,
) -> tuple[list[GrowthCycleResult], list[GrowthEvent]]:
    cycle_results: list[GrowthCycleResult] = []
    events: list[GrowthEvent] = []

    estimated_steps = max(1, len(batch_schedule) * (max_steps_per_cycle + post_add_steps))

    for cycle_index, batch_size in enumerate(batch_schedule):
        if batch_size <= 0:
            continue

        loss_before_cycle = float(optimizer.loss_history[-1])

        pre_steps, converged = optimize_until_converged(
            optimizer,
            max_steps=max_steps_per_cycle,
            convergence_window=convergence_window,
            convergence_rel_threshold=convergence_rel_threshold,
            start_softness=start_softness,
            end_softness=end_softness,
            softness_horizon=estimated_steps,
        )

        loss_before_add = float(optimizer.loss_history[-1])

        for _ in range(int(batch_size)):
            target_center, _, patch_color = highest_error_region_center(
                optimizer.target,
                optimizer.current_canvas,
                window=region_window,
            )

            target_size = float(region_window) * 0.5
            sx = float(np.clip(target_size, min_new_size, max_new_size or target_size))
            sy = float(np.clip(target_size, min_new_size, max_new_size or target_size))

            optimizer.add_polygon(
                center_x=target_center[0],
                center_y=target_center[1],
                size_x=sx,
                size_y=sy,
                color=(
                    float(np.clip(patch_color[0], 0.0, 1.0)),
                    float(np.clip(patch_color[1], 0.0, 1.0)),
                    float(np.clip(patch_color[2], 0.0, 1.0)),
                ),
                alpha=float(np.clip(new_polygon_alpha, 0.0, 1.0)),
                shape_type=shape_type,
                rotation=0.0,
            )

            current_softness = _softness_for_step(
                step_index=optimizer.step_count,
                horizon_steps=estimated_steps,
                start_softness=start_softness,
                end_softness=end_softness,
            )
            _refresh_optimizer_canvas(optimizer, softness=current_softness)

            placed = optimizer.polygons.centers[-1]
            distance = float(
                np.hypot(
                    float(placed[0]) - target_center[0],
                    float(placed[1]) - target_center[1],
                )
            )
            events.append(
                GrowthEvent(
                    cycle_index=cycle_index,
                    target_region_center=target_center,
                    placed_center=(float(placed[0]), float(placed[1])),
                    distance_to_target=distance,
                )
            )

        post_steps = 0
        for _ in range(max(post_add_steps, 0)):
            softness = _softness_for_step(
                step_index=optimizer.step_count,
                horizon_steps=estimated_steps,
                start_softness=start_softness,
                end_softness=end_softness,
            )
            optimizer.step(softness=softness)
            post_steps += 1

        cycle_results.append(
            GrowthCycleResult(
                cycle_index=cycle_index,
                batch_size=int(batch_size),
                loss_before_cycle=loss_before_cycle,
                loss_before_addition=loss_before_add,
                loss_after_cycle=float(optimizer.loss_history[-1]),
                optimization_steps=int(pre_steps + post_steps),
                converged_before_addition=bool(converged),
            )
        )

    return cycle_results, events


def _boundary_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred_gray = np.mean(pred, axis=2, dtype=np.float32)
    target_gray = np.mean(target, axis=2, dtype=np.float32)

    pred_gy, pred_gx = np.gradient(pred_gray)
    target_gy, target_gx = np.gradient(target_gray)

    pred_mag = np.hypot(pred_gx, pred_gy).astype(np.float32, copy=False)
    target_mag = np.hypot(target_gx, target_gy).astype(np.float32, copy=False)

    return float(np.mean(np.abs(pred_mag - target_mag), dtype=np.float32))


def _scale_polygons_to_resolution(
    polygons: LivePolygonBatch,
    *,
    old_resolution: int,
    new_resolution: int,
) -> LivePolygonBatch:
    scale = float(new_resolution) / max(float(old_resolution), 1.0)

    centers = np.array(polygons.centers, copy=True)
    sizes = np.array(polygons.sizes, copy=True)
    rotations = np.array(polygons.rotations, copy=True)
    colors = np.array(polygons.colors, copy=True)
    alphas = np.array(polygons.alphas, copy=True)
    shape_types = np.array(polygons.shape_types, copy=True)

    centers *= scale
    centers[:, 0] = np.clip(centers[:, 0], 0.0, new_resolution - 1.0)
    centers[:, 1] = np.clip(centers[:, 1], 0.0, new_resolution - 1.0)
    sizes *= scale

    return LivePolygonBatch(
        centers=centers,
        sizes=sizes,
        rotations=rotations,
        colors=colors,
        alphas=alphas,
        shape_types=shape_types,
    )


def run_multi_resolution_schedule(
    target_image: np.ndarray,
    *,
    random_seed: int = 0,
    include_round4: bool = False,
    max_steps_per_cycle: int = 20,
    post_add_steps: int = 6,
    convergence_window: int = 100,
    convergence_rel_threshold: float = 0.001,
    base_config: LiveOptimizerConfig | None = None,
) -> MultiResolutionResult:
    if target_image.ndim != 3 or target_image.shape[2] != 3:
        raise ValueError("target_image must have shape (H, W, 3)")

    cfg = LiveOptimizerConfig() if base_config is None else base_config
    rng = np.random.default_rng(random_seed)

    rounds = [
        ("round-1-50", 50, [20, 20, 20], (2.5, 12.5)),
        ("round-2-100", 100, [20, 20, 20, 20], (3.0, 14.0)),
        ("round-3-200", 200, [20, 20, 20, 20, 20], (2.0, 10.0)),
    ]
    if include_round4:
        rounds.append(("round-4-detail", 200, [20] * 10, (1.5, 4.0)))

    previous_resolution = None
    polygons: LivePolygonBatch | None = None
    round_results: list[MultiResolutionRoundResult] = []
    total_steps = 0

    for round_name, resolution, batch_schedule, size_bounds in rounds:
        target_level = _resize_rgb(target_image, resolution)
        rasterizer = SoftRasterizer(height=resolution, width=resolution)

        if polygons is None:
            polygons = make_random_live_batch_with_bounds(
                count=20,
                height=resolution,
                width=resolution,
                min_size=size_bounds[0],
                max_size=size_bounds[1],
                rng=rng,
            )
        else:
            polygons = _scale_polygons_to_resolution(
                polygons,
                old_resolution=int(previous_resolution),
                new_resolution=resolution,
            )

        round_cfg = replace(
            cfg,
            min_size=size_bounds[0],
            max_size=size_bounds[1],
        )

        optimizer = LiveJointOptimizer(
            target_image=target_level,
            rasterizer=rasterizer,
            polygons=polygons,
            config=round_cfg,
        )

        loss_start = float(optimizer.loss_history[-1])
        cycle_results, _ = progressive_growth(
            optimizer,
            batch_schedule=batch_schedule,
            max_steps_per_cycle=max_steps_per_cycle,
            post_add_steps=post_add_steps,
            convergence_window=convergence_window,
            convergence_rel_threshold=convergence_rel_threshold,
            region_window=5,
            new_polygon_alpha=0.60,
            min_new_size=size_bounds[0],
            max_new_size=size_bounds[1],
            shape_type=SHAPE_ELLIPSE,
            start_softness=2.0,
            end_softness=0.5,
        )

        round_steps = int(sum(c.optimization_steps for c in cycle_results))
        total_steps += round_steps

        loss_end = float(optimizer.loss_history[-1])
        boundary = _boundary_error(optimizer.current_canvas, target_level)

        round_results.append(
            MultiResolutionRoundResult(
                round_name=round_name,
                resolution=resolution,
                loss_start=loss_start,
                loss_end=loss_end,
                boundary_error=boundary,
                polygon_count=optimizer.polygons.count,
                optimization_steps=round_steps,
            )
        )

        previous_resolution = resolution
        polygons = optimizer.polygons.copy()

    final_loss = float(round_results[-1].loss_end)
    return MultiResolutionResult(
        rounds=round_results,
        total_optimization_steps=total_steps,
        final_loss=final_loss,
    )


def load_square_target(image_path: Path, resolution: int) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        square = rgb.crop((left, top, left + side, top + side))
        resized = square.resize((resolution, resolution), Image.Resampling.LANCZOS)
        return (np.asarray(resized, dtype=np.float32) / 255.0).astype(np.float32, copy=False)
