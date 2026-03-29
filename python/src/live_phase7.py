from __future__ import annotations

import threading
import textwrap
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
from PIL import Image, ImageDraw

from src.live_optimizer import LiveJointOptimizer, LiveOptimizerConfig
from src.live_renderer import (
    SHAPE_ANNULAR_SEGMENT,
    SHAPE_BEZIER_PATCH,
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
    LivePolygonBatch,
    SoftRasterizer,
)
from src.live_schedule import (
    apply_low_frequency_color_correction,
    make_random_live_batch_with_bounds,
    progressive_growth,
)


_TRIANGLE_LOCAL = np.array(
    [
        [1.0, 0.0],
        [-0.5, 0.8660254],
        [-0.5, -0.8660254],
    ],
    dtype=np.float32,
)

_QUAD_LOCAL = np.array(
    [
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Phase7RoundConfig:
    name: str
    resolution: int
    batch_schedule: list[int]
    min_size: float
    max_size: float
    max_steps_per_cycle: int
    post_add_steps: int
    start_softness: float
    end_softness: float


@dataclass(frozen=True)
class Phase7Plan:
    rounds: list[Phase7RoundConfig]
    polygon_budget: int


@dataclass(frozen=True)
class Phase7ExecutionResult:
    final_canvas: np.ndarray
    final_loss: float
    polygon_count: int
    iterations: int
    loss_history: list[float]
    resolution_markers: list[int]
    batch_markers: list[int]


@dataclass
class Phase7ControlState:
    paused: bool = False
    quit_requested: bool = False
    show_segmentation_overlay: bool = False
    view_mode: int = 0
    residual_mode: int = 0
    force_growth_requested: bool = False
    correction_requested: bool = False
    softness_scale: float = 1.0


@dataclass
class _SharedViewState:
    target: np.ndarray
    segmentation_map: np.ndarray | None
    canvas: np.ndarray
    signed_residual: np.ndarray
    abs_residual: np.ndarray
    mse_residual: np.ndarray
    polygon_preview: np.ndarray
    loss_history: list[float]
    resolution_markers: list[int]
    batch_markers: list[int]
    round_name: str
    current_resolution: int
    polygon_count: int
    iteration: int
    running: bool
    status_line: str


def _resize_rgb(image: np.ndarray, resolution: int) -> np.ndarray:
    clipped = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    pil = Image.fromarray(np.round(clipped * 255.0).astype(np.uint8), mode="RGB")
    out = pil.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return (np.asarray(out, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def _scale_polygons(
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
    shape_params = np.array(polygons.shape_params, copy=True)

    centers *= scale
    centers[:, 0] = np.clip(centers[:, 0], 0.0, new_resolution - 1.0)
    centers[:, 1] = np.clip(centers[:, 1], 0.0, new_resolution - 1.0)
    sizes *= scale

    stroke_idx = np.where(shape_types == SHAPE_THIN_STROKE)[0]
    if stroke_idx.size > 0:
        shape_params[stroke_idx, :3] *= scale

    return LivePolygonBatch(
        centers=centers,
        sizes=sizes,
        rotations=rotations,
        colors=colors,
        alphas=alphas,
        shape_types=shape_types,
        shape_params=shape_params,
    )


def _build_batch_schedule(total: int, typical_batch: int) -> list[int]:
    total_left = max(0, int(total))
    batch = max(1, int(typical_batch))
    schedule: list[int] = []
    while total_left > 0:
        take = min(batch, total_left)
        schedule.append(int(take))
        total_left -= take
    return schedule


def _allocate_counts(total: int, fractions: list[float]) -> list[int]:
    if total <= 0:
        return [0 for _ in fractions]

    frac = np.asarray(fractions, dtype=np.float64)
    frac = frac / max(np.sum(frac), 1e-9)
    raw = frac * float(total)
    base = np.floor(raw).astype(np.int32)
    rem = int(total - int(np.sum(base)))

    if rem > 0:
        order = np.argsort(-(raw - base))
        for i in range(rem):
            base[int(order[i % len(order)])] += 1

    return [int(v) for v in base.tolist()]


def build_phase7_plan(
    *,
    base_resolution: int,
    polygon_budget: int,
    complexity_score: float,
) -> Phase7Plan:
    complexity = float(np.clip(complexity_score, 0.0, 1.0))
    budget = max(1, int(polygon_budget))

    levels = [
        max(40, int(round(base_resolution * 0.25))),
        max(64, int(round(base_resolution * 0.5))),
        int(base_resolution),
    ]
    unique_levels: list[int] = []
    for level in levels:
        if not unique_levels or level != unique_levels[-1]:
            unique_levels.append(level)

    if len(unique_levels) == 1:
        unique_levels = [max(40, unique_levels[0] // 2), unique_levels[0]]

    if len(unique_levels) == 2:
        fractions = [0.38, 0.62]
    else:
        fractions = [0.22, 0.33, 0.45]

    counts = _allocate_counts(budget, fractions)

    rounds: list[Phase7RoundConfig] = []
    for idx, (resolution, count) in enumerate(zip(unique_levels, counts, strict=False)):
        typical_batch = int(
            np.clip(round((10.0 - 4.0 * complexity) * (1.0 + 0.15 * idx)), 2, 24)
        )
        schedule = _build_batch_schedule(count, typical_batch)

        max_size = max(
            4.0, float(resolution) * (0.22 - 0.07 * complexity) * (0.90**idx)
        )
        min_size = max(1.2, max_size * (0.20 + 0.06 * idx))

        rounds.append(
            Phase7RoundConfig(
                name=f"round-{idx + 1}-{resolution}",
                resolution=int(resolution),
                batch_schedule=schedule,
                min_size=float(min_size),
                max_size=float(max_size),
                max_steps_per_cycle=int(
                    np.clip(round(18.0 + 34.0 * complexity - 2.0 * idx), 8, 48)
                ),
                post_add_steps=int(np.clip(round(4.0 + 8.0 * complexity), 2, 14)),
                start_softness=float(max(0.7, 2.2 - 0.35 * idx)),
                end_softness=float(max(0.25, 0.70 - 0.10 * idx)),
            )
        )

    return Phase7Plan(rounds=rounds, polygon_budget=budget)


def _signed_residual_rgb(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    residual = np.mean(target - canvas, axis=2, dtype=np.float32)
    denom = float(np.quantile(np.abs(residual), 0.98))
    scale = max(denom, 1e-5)
    norm = np.clip(residual / scale, -1.0, 1.0)

    out = np.zeros_like(target, dtype=np.float32)
    out[..., 0] = np.clip(norm, 0.0, 1.0)
    out[..., 2] = np.clip(-norm, 0.0, 1.0)
    return out


def _absolute_residual_rgb(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    residual = np.mean(np.abs(target - canvas), axis=2, dtype=np.float32)
    denom = float(np.quantile(residual, 0.98))
    scale = max(denom, 1e-5)
    norm = np.clip(residual / scale, 0.0, 1.0)
    return np.repeat(norm[:, :, None], 3, axis=2).astype(np.float32, copy=False)


def _mse_residual_rgb(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    residual = np.mean((target - canvas) ** 2, axis=2, dtype=np.float32)
    denom = float(np.quantile(residual, 0.98))
    scale = max(denom, 1e-5)
    norm = np.clip(residual / scale, 0.0, 1.0)

    out = np.zeros_like(target, dtype=np.float32)
    out[..., 0] = norm
    out[..., 1] = np.clip(norm * 0.75, 0.0, 1.0)
    return out


def _local_polygon_points(
    shape_type: int,
    *,
    center_x: float,
    center_y: float,
    size_x: float,
    size_y: float,
    rotation: float,
) -> np.ndarray:
    if shape_type == SHAPE_TRIANGLE:
        base = _TRIANGLE_LOCAL
    else:
        base = _QUAD_LOCAL

    local = np.array(base, copy=True)
    local[:, 0] *= float(size_x)
    local[:, 1] *= float(size_y)

    c = float(np.cos(rotation))
    s = float(np.sin(rotation))
    x = local[:, 0] * c - local[:, 1] * s + float(center_x)
    y = local[:, 0] * s + local[:, 1] * c + float(center_y)
    return np.column_stack([x, y]).astype(np.float32)


def _outline_palette_for_sizes(sizes: np.ndarray) -> list[tuple[int, int, int]]:
    if sizes.size == 0:
        return []
    q1 = float(np.quantile(sizes, 0.33))
    q2 = float(np.quantile(sizes, 0.66))

    colors: list[tuple[int, int, int]] = []
    for value in sizes:
        if value >= q2:
            colors.append((40, 120, 240))
        elif value >= q1:
            colors.append((40, 170, 70))
        else:
            colors.append((220, 60, 50))
    return colors


def make_polygon_outline_preview(
    polygons: LivePolygonBatch,
    *,
    polygon_resolution: int,
    output_resolution: int,
) -> np.ndarray:
    canvas = Image.new(
        "RGB", (output_resolution, output_resolution), color=(255, 255, 255)
    )
    draw = ImageDraw.Draw(canvas)

    if polygons.count == 0:
        return (np.asarray(canvas, dtype=np.float32) / 255.0).astype(
            np.float32, copy=False
        )

    scale = float(output_resolution) / max(float(polygon_resolution), 1.0)
    major_sizes = np.maximum(polygons.sizes[:, 0], polygons.sizes[:, 1]) * scale
    palette = _outline_palette_for_sizes(major_sizes)

    for idx in range(polygons.count):
        color = palette[idx]
        cx = float(polygons.centers[idx, 0] * scale)
        cy = float(polygons.centers[idx, 1] * scale)
        sx = float(polygons.sizes[idx, 0] * scale)
        sy = float(polygons.sizes[idx, 1] * scale)
        rot = float(polygons.rotations[idx])
        shape_type = int(polygons.shape_types[idx])
        params = polygons.shape_params[idx] * scale

        if shape_type in {SHAPE_TRIANGLE, SHAPE_QUAD, SHAPE_BEZIER_PATCH}:
            pts = _local_polygon_points(
                SHAPE_QUAD if shape_type == SHAPE_BEZIER_PATCH else shape_type,
                center_x=cx,
                center_y=cy,
                size_x=sx,
                size_y=sy,
                rotation=rot,
            )
            points = [(float(p[0]), float(p[1])) for p in pts]
            draw.polygon(points, outline=color, width=1)
            continue

        if shape_type == SHAPE_ELLIPSE:
            draw.ellipse((cx - sx, cy - sy, cx + sx, cy + sy), outline=color, width=1)
            continue

        if shape_type == SHAPE_THIN_STROKE:
            x2 = float(params[0])
            y2 = float(params[1])
            width = int(max(1, round(float(params[2]))))
            draw.line((cx, cy, x2, y2), fill=color, width=width)
            continue

        if shape_type == SHAPE_ANNULAR_SEGMENT:
            inner_r = float(min(sx, sy))
            outer_r = float(max(sx, sy))
            start_deg = float(np.degrees(polygons.shape_params[idx, 0]))
            end_deg = float(np.degrees(polygons.shape_params[idx, 1]))

            draw.arc(
                (cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r),
                start=start_deg,
                end=end_deg,
                fill=color,
                width=1,
            )
            draw.arc(
                (cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r),
                start=start_deg,
                end=end_deg,
                fill=color,
                width=1,
            )

            a0 = float(polygons.shape_params[idx, 0])
            a1 = float(polygons.shape_params[idx, 1])
            for a in (a0, a1):
                x_in = cx + inner_r * float(np.cos(a))
                y_in = cy + inner_r * float(np.sin(a))
                x_out = cx + outer_r * float(np.cos(a))
                y_out = cy + outer_r * float(np.sin(a))
                draw.line((x_in, y_in, x_out, y_out), fill=color, width=1)

    return (np.asarray(canvas, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


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
    if normalized == "s":
        controls.show_segmentation_overlay = not controls.show_segmentation_overlay
        return "segmentation-toggle"
    if normalized == "e":
        controls.residual_mode = (controls.residual_mode + 1) % 3
        return "error-mode-cycle"
    if normalized == "r":
        screenshot_callback()
        return "screenshot"
    if normalized == "q":
        controls.quit_requested = True
        quit_callback()
        return "quit"
    if normalized in {"1", "2", "3"}:
        controls.view_mode = int(normalized) - 1
        return "variant-switch"
    if normalized == "v":
        controls.view_mode = (controls.view_mode + 1) % 3
        return "view-cycle"
    if normalized == "g":
        controls.force_growth_requested = True
        return "force-growth"
    if normalized == "d":
        controls.correction_requested = True
        return "residual-correction"
    if normalized in {"+", "="}:
        controls.softness_scale = float(
            np.clip(controls.softness_scale * 1.1, 0.2, 3.0)
        )
        return "softness-up"
    if normalized in {"-", "_"}:
        controls.softness_scale = float(
            np.clip(controls.softness_scale / 1.1, 0.2, 3.0)
        )
        return "softness-down"

    return "noop"


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

    base_resolution = int(target_image.shape[0])
    deadline = (
        time.monotonic() + max(0.0, float(minutes) * 60.0) if minutes > 0.0 else None
    )
    hard_deadline = (
        time.monotonic() + float(hard_timeout_seconds)
        if hard_timeout_seconds is not None and hard_timeout_seconds > 0.0
        else None
    )

    def _deadline_reached() -> bool:
        now = time.monotonic()
        return bool(
            (deadline is not None and now >= deadline)
            or (hard_deadline is not None and now >= hard_deadline)
        )

    def _remaining_seconds() -> float | None:
        now = time.monotonic()
        candidates: list[float] = []
        if deadline is not None:
            candidates.append(float(deadline - now))
        if hard_deadline is not None:
            candidates.append(float(hard_deadline - now))
        if not candidates:
            return None
        return float(min(candidates))

    rng = np.random.default_rng(random_seed)
    cfg = LiveOptimizerConfig(
        color_lr=0.05,
        position_lr=0.002,
        size_lr=0.0008,
        alpha_lr=0.02,
        render_chunk_size=50,
        position_update_interval=8,
        size_update_interval=12,
        max_fd_polygons=8,
    )

    global_losses: list[float] = []
    resolution_markers: list[int] = []
    batch_markers: list[int] = []
    total_iteration_points = 0

    polygons: LivePolygonBatch | None = None
    previous_resolution: int | None = None
    final_canvas = np.ones_like(target_image, dtype=np.float32)
    final_loss = float(np.mean((target_image - final_canvas) ** 2, dtype=np.float32))

    for round_idx, round_cfg in enumerate(plan.rounds):
        if controls.quit_requested:
            break
        if _deadline_reached():
            break

        target_level = _resize_rgb(target_image, round_cfg.resolution)
        rasterizer = SoftRasterizer(
            height=round_cfg.resolution, width=round_cfg.resolution
        )

        if polygons is None:
            polygons = make_random_live_batch_with_bounds(
                count=max(12, min(24, max(1, plan.polygon_budget // 20))),
                height=round_cfg.resolution,
                width=round_cfg.resolution,
                min_size=round_cfg.min_size,
                max_size=round_cfg.max_size,
                rng=rng,
            )
            if not global_losses:
                preview_optimizer = LiveJointOptimizer(
                    target_image=target_level,
                    rasterizer=rasterizer,
                    polygons=polygons.copy(),
                    config=cfg,
                )
                global_losses.append(float(preview_optimizer.loss_history[-1]))
                total_iteration_points = len(global_losses)
                final_canvas = _resize_rgb(
                    preview_optimizer.current_canvas, base_resolution
                )
                final_loss = float(
                    np.mean((target_image - final_canvas) ** 2, dtype=np.float32)
                )
                shared_update_callback(
                    final_canvas,
                    polygons,
                    round_cfg.resolution,
                    global_losses,
                    resolution_markers,
                    batch_markers,
                    round_cfg.name,
                    total_iteration_points,
                    final_loss,
                    True,
                    f"round {round_idx + 1}/{len(plan.rounds)} initialized",
                )
        else:
            if previous_resolution is None:
                raise RuntimeError("previous_resolution missing during round scaling")
            polygons = _scale_polygons(
                polygons,
                old_resolution=previous_resolution,
                new_resolution=round_cfg.resolution,
            )

        optimizer = LiveJointOptimizer(
            target_image=target_level,
            rasterizer=rasterizer,
            polygons=polygons.copy(),
            config=cfg,
        )

        if round_idx > 0:
            resolution_markers.append(len(global_losses))

        for batch_size in round_cfg.batch_schedule:
            if controls.quit_requested:
                break
            if _deadline_reached():
                break
            if (
                max_total_steps is not None
                and total_iteration_points >= max_total_steps
            ):
                break

            while controls.paused and not controls.quit_requested:
                if _deadline_reached():
                    controls.quit_requested = True
                    break
                time.sleep(0.05)
                if controls.correction_requested:
                    apply_low_frequency_color_correction(
                        optimizer,
                        sigma=10.0,
                        strength=0.7,
                        softness=max(
                            0.2, round_cfg.end_softness * controls.softness_scale
                        ),
                    )
                    controls.correction_requested = False

            if controls.quit_requested:
                break

            if controls.correction_requested:
                apply_low_frequency_color_correction(
                    optimizer,
                    sigma=10.0,
                    strength=0.7,
                    softness=max(0.2, round_cfg.end_softness * controls.softness_scale),
                )
                controls.correction_requested = False

            before_len = len(optimizer.loss_history)
            batch_markers.append(len(global_losses))

            force_growth = controls.force_growth_requested
            controls.force_growth_requested = False

            max_steps_per_cycle = int(round_cfg.max_steps_per_cycle)
            post_add_steps = int(round_cfg.post_add_steps)
            remaining = _remaining_seconds()
            if remaining is not None:
                if remaining <= 0.0:
                    break
                throttle = float(np.clip(remaining / 12.0, 0.20, 1.0))
                if not force_growth:
                    max_steps_per_cycle = max(
                        1,
                        min(
                            max_steps_per_cycle,
                            int(np.ceil(max_steps_per_cycle * throttle)),
                        ),
                    )
                post_add_steps = max(
                    1,
                    min(post_add_steps, int(np.ceil(post_add_steps * throttle))),
                )

            start_soft = max(0.2, round_cfg.start_softness * controls.softness_scale)
            end_soft = max(0.15, round_cfg.end_softness * controls.softness_scale)

            last_emit_time = 0.0

            def _emit_intermediate(opt: LiveJointOptimizer) -> None:
                nonlocal last_emit_time
                now = time.monotonic()
                if (now - last_emit_time) < 0.6:
                    return

                partial_losses = [float(v) for v in opt.loss_history[before_len:]]
                if not partial_losses:
                    partial_losses = [float(opt.loss_history[-1])]
                losses_view = list(global_losses) + partial_losses
                if max_total_steps is not None and len(losses_view) > max_total_steps:
                    losses_view = losses_view[:max_total_steps]

                full_canvas = _resize_rgb(opt.current_canvas, base_resolution)
                loss_value = float(
                    np.mean((target_image - full_canvas) ** 2, dtype=np.float32)
                )
                remaining_for_status = _remaining_seconds()
                remaining_text = (
                    "n/a"
                    if remaining_for_status is None
                    else f"{remaining_for_status:.1f}s"
                )

                status = (
                    f"{round_cfg.name} | polygons={opt.polygons.count} | "
                    f"loss={loss_value:.6f} | softness={controls.softness_scale:.2f} | "
                    f"remaining={remaining_text}"
                )
                shared_update_callback(
                    full_canvas,
                    opt.polygons.copy(),
                    round_cfg.resolution,
                    losses_view,
                    resolution_markers,
                    batch_markers,
                    round_cfg.name,
                    len(losses_view),
                    loss_value,
                    True,
                    status,
                )
                last_emit_time = now

            progressive_growth(
                optimizer,
                batch_schedule=[int(batch_size)],
                max_steps_per_cycle=0 if force_growth else max_steps_per_cycle,
                post_add_steps=post_add_steps,
                convergence_window=80,
                convergence_rel_threshold=0.001,
                region_window=5,
                new_polygon_alpha=0.60,
                min_new_size=float(round_cfg.min_size),
                max_new_size=float(round_cfg.max_size),
                use_content_aware_shapes=True,
                use_high_frequency_targeting=True,
                residual_sigma=10.0,
                low_frequency_correction_strength=0.35,
                max_add_attempts=2,
                enforce_cycle_improvement=False,
                max_recovery_steps=0,
                start_softness=start_soft,
                end_softness=end_soft,
                progress_callback=_emit_intermediate,
                progress_every_steps=4,
            )

            if controls.correction_requested:
                apply_low_frequency_color_correction(
                    optimizer,
                    sigma=10.0,
                    strength=0.7,
                    softness=max(0.2, end_soft),
                )
                controls.correction_requested = False

            new_losses = [float(v) for v in optimizer.loss_history[before_len:]]
            if not new_losses:
                new_losses = [float(optimizer.loss_history[-1])]

            global_losses.extend(new_losses)
            total_iteration_points = len(global_losses)

            if max_total_steps is not None and total_iteration_points > max_total_steps:
                global_losses = global_losses[:max_total_steps]
                total_iteration_points = len(global_losses)

            full_canvas = _resize_rgb(optimizer.current_canvas, base_resolution)
            loss_value = float(
                np.mean((target_image - full_canvas) ** 2, dtype=np.float32)
            )
            remaining_for_status = _remaining_seconds()
            remaining_text = (
                "n/a"
                if remaining_for_status is None
                else f"{remaining_for_status:.1f}s"
            )

            status = (
                f"{round_cfg.name} | polygons={optimizer.polygons.count} | "
                f"loss={loss_value:.6f} | softness={controls.softness_scale:.2f} | "
                f"remaining={remaining_text}"
            )
            shared_update_callback(
                full_canvas,
                optimizer.polygons.copy(),
                round_cfg.resolution,
                global_losses,
                resolution_markers,
                batch_markers,
                round_cfg.name,
                total_iteration_points,
                loss_value,
                True,
                status,
            )

        polygons = optimizer.polygons.copy()
        previous_resolution = int(round_cfg.resolution)
        final_canvas = _resize_rgb(optimizer.current_canvas, base_resolution)
        final_loss = float(
            np.mean((target_image - final_canvas) ** 2, dtype=np.float32)
        )

        if controls.quit_requested:
            break
        if _deadline_reached():
            break
        if max_total_steps is not None and total_iteration_points >= max_total_steps:
            break

    shared_update_callback(
        final_canvas,
        polygons
        if polygons is not None
        else LivePolygonBatch(
            centers=np.zeros((0, 2), dtype=np.float32),
            sizes=np.zeros((0, 2), dtype=np.float32),
            rotations=np.zeros((0,), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            alphas=np.zeros((0,), dtype=np.float32),
            shape_types=np.zeros((0,), dtype=np.int32),
            shape_params=np.zeros((0, 6), dtype=np.float32),
        ),
        previous_resolution
        if previous_resolution is not None
        else target_image.shape[0],
        global_losses,
        resolution_markers,
        batch_markers,
        "complete",
        len(global_losses),
        final_loss,
        False,
        "complete",
    )

    return Phase7ExecutionResult(
        final_canvas=final_canvas,
        final_loss=final_loss,
        polygon_count=0 if polygons is None else polygons.count,
        iterations=len(global_losses),
        loss_history=list(global_losses),
        resolution_markers=list(resolution_markers),
        batch_markers=list(batch_markers),
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
    controls = Phase7ControlState()

    def _noop_update(
        _canvas: np.ndarray,
        _polygons: LivePolygonBatch,
        _polygon_resolution: int,
        _losses: list[float],
        _resolution_markers: list[int],
        _batch_markers: list[int],
        _round_name: str,
        _iteration: int,
        _loss: float,
        _running: bool,
        _status: str,
    ) -> None:
        del _canvas
        del _polygons
        del _polygon_resolution
        del _losses
        del _resolution_markers
        del _batch_markers
        del _round_name
        del _iteration
        del _loss
        del _running
        del _status

    del segmentation_map
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
    update_interval_ms: int = 2000,
    close_after_seconds: float | None = None,
    max_total_steps: int | None = None,
) -> Phase7ExecutionResult:
    resolution = int(target_image.shape[0])
    empty_canvas = np.ones_like(target_image, dtype=np.float32)
    empty_poly = LivePolygonBatch(
        centers=np.zeros((0, 2), dtype=np.float32),
        sizes=np.zeros((0, 2), dtype=np.float32),
        rotations=np.zeros((0,), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        alphas=np.zeros((0,), dtype=np.float32),
        shape_types=np.zeros((0,), dtype=np.int32),
        shape_params=np.zeros((0, 6), dtype=np.float32),
    )

    shared = _SharedViewState(
        target=np.array(target_image, copy=True),
        segmentation_map=None
        if segmentation_map is None
        else np.array(segmentation_map, copy=True),
        canvas=np.array(empty_canvas, copy=True),
        signed_residual=_signed_residual_rgb(target_image, empty_canvas),
        abs_residual=_absolute_residual_rgb(target_image, empty_canvas),
        mse_residual=_mse_residual_rgb(target_image, empty_canvas),
        polygon_preview=make_polygon_outline_preview(
            empty_poly, polygon_resolution=resolution, output_resolution=resolution
        ),
        loss_history=[
            float(np.mean((target_image - empty_canvas) ** 2, dtype=np.float32))
        ],
        resolution_markers=[],
        batch_markers=[],
        round_name="initializing",
        current_resolution=resolution,
        polygon_count=0,
        iteration=0,
        running=True,
        status_line="initializing",
    )

    controls = Phase7ControlState()
    lock = threading.Lock()
    result_holder: dict[str, Phase7ExecutionResult] = {}

    def _update_shared(
        canvas: np.ndarray,
        polygons: LivePolygonBatch,
        polygon_resolution: int,
        losses: list[float],
        resolution_markers: list[int],
        batch_markers: list[int],
        round_name: str,
        iteration: int,
        _loss: float,
        running: bool,
        status: str,
    ) -> None:
        preview = make_polygon_outline_preview(
            polygons,
            polygon_resolution=int(polygon_resolution),
            output_resolution=resolution,
        )
        signed = _signed_residual_rgb(shared.target, canvas)
        abs_res = _absolute_residual_rgb(shared.target, canvas)
        mse_res = _mse_residual_rgb(shared.target, canvas)

        with lock:
            shared.canvas = np.array(canvas, copy=True)
            shared.signed_residual = signed
            shared.abs_residual = abs_res
            shared.mse_residual = mse_res
            shared.polygon_preview = preview
            shared.loss_history = list(losses)
            shared.resolution_markers = list(resolution_markers)
            shared.batch_markers = list(batch_markers)
            shared.round_name = str(round_name)
            shared.current_resolution = int(polygon_resolution)
            shared.polygon_count = int(polygons.count)
            shared.iteration = int(iteration)
            shared.running = bool(running)
            shared.status_line = str(status)

    def _worker() -> None:
        result = execute_phase7_schedule(
            target_image=target_image,
            plan=plan,
            random_seed=random_seed,
            minutes=minutes,
            hard_timeout_seconds=hard_timeout_seconds,
            controls=controls,
            shared_update_callback=_update_shared,
            max_total_steps=max_total_steps,
        )
        result_holder["result"] = result

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.25, wspace=0.22)

    ax_target = fig.add_subplot(grid[0, 0])
    ax_current = fig.add_subplot(grid[0, 1])
    ax_residual = fig.add_subplot(grid[0, 2])
    ax_polygons = fig.add_subplot(grid[1, 0])
    ax_loss = fig.add_subplot(grid[1, 1:])

    target_im = ax_target.imshow(shared.target)
    target_im.set_interpolation("nearest")
    ax_target.set_title("Panel 1 - Target")
    ax_target.set_xticks([])
    ax_target.set_yticks([])

    seg_overlay = None
    if shared.segmentation_map is not None:
        seg_overlay = ax_target.imshow(
            shared.segmentation_map,
            cmap="tab20",
            alpha=0.28,
            interpolation="nearest",
            visible=False,
        )

    current_im = ax_current.imshow(shared.canvas)
    current_im.set_interpolation("nearest")
    ax_current.set_title("Panel 2 - Reconstruction")
    ax_current.set_xticks([])
    ax_current.set_yticks([])

    residual_im = ax_residual.imshow(shared.signed_residual)
    residual_im.set_interpolation("nearest")
    ax_residual.set_title("Panel 3 - Residual (signed)")
    ax_residual.set_xticks([])
    ax_residual.set_yticks([])

    polygons_im = ax_polygons.imshow(shared.polygon_preview)
    polygons_im.set_interpolation("nearest")
    ax_polygons.set_title("Panel 4 - Polygon Outlines")
    ax_polygons.set_xticks([])
    ax_polygons.set_yticks([])

    ax_loss.set_title("Panel 5 - Log MSE Curve")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("MSE (log)")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.25)
    (loss_line,) = ax_loss.plot([], [], color="tab:blue", linewidth=1.8)
    status_text = ax_loss.text(
        0.01,
        0.98,
        "",
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        transform=ax_loss.transAxes,
        wrap=True,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.8},
    )

    marker_artists: list = []

    def _save_screenshot() -> Path:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"phase7_live_{ts}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
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
        timer = fig.canvas.new_timer(interval=int(close_after_seconds * 1000.0))
        timer.add_callback(lambda: plt.close(fig))
        timer.start()

    def update(_: int):
        with lock:
            canvas = np.array(shared.canvas, copy=True)
            signed = np.array(shared.signed_residual, copy=True)
            abs_res = np.array(shared.abs_residual, copy=True)
            mse_res = np.array(shared.mse_residual, copy=True)
            poly_preview = np.array(shared.polygon_preview, copy=True)
            losses = np.array(shared.loss_history, dtype=np.float64)
            res_markers = list(shared.resolution_markers)
            batch_marks = list(shared.batch_markers)
            iteration = int(shared.iteration)
            polygon_count = int(shared.polygon_count)
            round_name = str(shared.round_name)
            running = bool(shared.running)
            status = str(shared.status_line)
            show_seg = bool(controls.show_segmentation_overlay)
            view_mode = int(controls.view_mode)
            residual_mode = int(controls.residual_mode)
            softness = float(controls.softness_scale)
            paused = bool(controls.paused)

        if seg_overlay is not None:
            seg_overlay.set_visible(show_seg)

        if view_mode == 0:
            current_im.set_data(canvas)
            ax_current.set_title("Panel 2 - Reconstruction")
        elif view_mode == 1:
            current_im.set_data(signed)
            ax_current.set_title("Panel 2 - Focus View: Residual")
        else:
            current_im.set_data(poly_preview)
            ax_current.set_title("Panel 2 - Focus View: Polygon Outlines")

        if residual_mode == 0:
            residual_im.set_data(signed)
            ax_residual.set_title("Panel 3 - Residual (signed)")
        elif residual_mode == 1:
            residual_im.set_data(abs_res)
            ax_residual.set_title("Panel 3 - Residual (absolute)")
        else:
            residual_im.set_data(mse_res)
            ax_residual.set_title("Panel 3 - Residual (MSE)")

        polygons_im.set_data(poly_preview)

        x = np.arange(losses.size)
        if losses.size > 0:
            loss_line.set_data(x, losses)
            y = np.maximum(losses, 1e-9)
            y_min = max(1e-9, float(np.min(y) * 0.9))
            y_max = max(float(np.max(y) * 1.1), y_min * 10.0)
            ax_loss.set_ylim(y_min, y_max)
            ax_loss.set_xlim(0, max(10, int(x[-1]) + 5))

        for artist in marker_artists:
            artist.remove()
        marker_artists.clear()

        for idx in res_markers:
            marker_artists.append(
                ax_loss.axvline(
                    int(idx), linestyle="--", color="gray", alpha=0.8, linewidth=1.1
                )
            )
        for idx in batch_marks:
            marker_artists.append(
                ax_loss.axvline(
                    int(idx), linestyle="-", color="#ff8c42", alpha=0.20, linewidth=0.8
                )
            )

        keys_help = textwrap.fill(
            "keys: P pause | S seg | E residual mode | R shot | Q quit | 1/2/3 view | V cycle view | G grow | D correct | +/- softness",
            width=76,
            break_long_words=False,
        )
        state_line = textwrap.fill(
            f"state           : {status}",
            width=76,
            break_long_words=False,
        )
        status_text.set_text(
            "\n".join(
                [
                    f"round           : {round_name}",
                    f"iteration       : {iteration}",
                    f"polygons        : {polygon_count}",
                    f"live softness   : {softness:.2f}",
                    f"paused          : {paused}",
                    state_line,
                    keys_help,
                ]
            )
        )

        if controls.quit_requested and not running:
            anim.event_source.stop()

        if not running and not worker.is_alive():
            if controls.quit_requested:
                anim.event_source.stop()

        fig.canvas.draw_idle()
        return (current_im, residual_im, polygons_im, loss_line, status_text)

    anim = FuncAnimation(
        fig,
        update,
        interval=max(50, int(update_interval_ms)),
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
        markers_r = list(shared.resolution_markers)
        markers_b = list(shared.batch_markers)
        poly_count = int(shared.polygon_count)

    return Phase7ExecutionResult(
        final_canvas=canvas,
        final_loss=float(losses[-1])
        if losses
        else float(np.mean((target_image - canvas) ** 2, dtype=np.float32)),
        polygon_count=poly_count,
        iterations=len(losses),
        loss_history=losses,
        resolution_markers=markers_r,
        batch_markers=markers_b,
    )
