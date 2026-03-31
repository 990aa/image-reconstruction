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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from skimage import color as skcolor
from scipy.ndimage import gaussian_filter

from src.core_renderer import (
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
    LivePolygonBatch,
    SoftRasterizer,
)
from src.live_optimizer import LiveJointOptimizer, LiveOptimizerConfig
from src.live_schedule import (
    _compute_structure_maps,
    _initialize_shape_params,
    make_grid_seeded_batch,
    sample_diverse_error_regions,
)


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
    force_growth_requests: int = 0
    force_decompose_requests: int = 0
    view_mode_index: int = 0
    residual_mode_index: int = 0
    show_segmentation_overlay: bool = False
    show_scatter_overlay: bool = True
    softness_scale: float = 1.0
    latest_softness: float = 0.12


@dataclass
class _SharedViewState:
    target: np.ndarray
    canvas: np.ndarray
    signed_residual: np.ndarray
    abs_residual: np.ndarray
    loss_history: list[float]
    resolution_markers: list[int]
    batch_markers: list[int]
    stage_markers: list[tuple[str, int]]
    polygon_count: int
    iteration: int
    stage_name: str
    stage_iteration: int
    stage_start_loss: float
    stage_position_updates: int
    running: bool
    status_line: str
    polygon_sizes: np.ndarray
    polygon_centers: np.ndarray
    polygon_shape_types: np.ndarray
    polygon_rotations: np.ndarray


@dataclass(frozen=True)
class _StructureGuidance:
    gradient: np.ndarray
    anisotropy: np.ndarray
    cornerness: np.ndarray
    orientation: np.ndarray


def build_phase7_plan(
    *,
    base_resolution: int,
    polygon_budget: int,
    complexity_score: float,
) -> Phase7Plan:
    del base_resolution
    del polygon_budget
    del complexity_score

    return Phase7Plan(
        stage_a_initial_polygons=24,
        stage_a_steps=200,
        stage_b_batches=10,
        stage_b_batch_size=8,
        stage_b_steps_per_batch=200,
        stage_b_size_start=18.0,
        stage_b_size_end=8.0,
        stage_c_batches=8,
        stage_c_batch_size=6,
        stage_c_steps_per_batch=150,
        stage_c_size_start=15.0,
        stage_c_size_end=5.0,
        stage_d_steps=80,
    )


def _rgb_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2, dtype=np.float32))


def _signed_residual(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    return np.clip(target - canvas, -1.0, 1.0).astype(np.float32, copy=False)


def _abs_residual(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(target - canvas), axis=2, dtype=np.float32).astype(
        np.float32,
        copy=False,
    )


def _resize_rgb(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image.astype(np.float32, copy=False)

    uint8 = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    pil = Image.fromarray(uint8, mode="RGB")
    resized = pil.resize((width, height), Image.Resampling.LANCZOS)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(
        np.float32, copy=False
    )


def _resize_labels(labels: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if labels.shape[:2] == (height, width):
        return labels.astype(np.int32, copy=False)

    pil = Image.fromarray(labels.astype(np.int32, copy=False), mode="I")
    resized = pil.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.int32)


def _scale_polygons_to_resolution(
    polygons: LivePolygonBatch,
    *,
    old_width: int,
    old_height: int,
    new_width: int,
    new_height: int,
) -> LivePolygonBatch:
    sx = float(new_width) / max(float(old_width), 1.0)
    sy = float(new_height) / max(float(old_height), 1.0)

    centers = np.array(polygons.centers, copy=True)
    centers[:, 0] *= sx
    centers[:, 1] *= sy
    centers[:, 0] = np.clip(centers[:, 0], 0.0, new_width - 1.0)
    centers[:, 1] = np.clip(centers[:, 1], 0.0, new_height - 1.0)

    sizes = np.array(polygons.sizes, copy=True)
    sizes[:, 0] *= sx
    sizes[:, 1] *= sy

    return LivePolygonBatch(
        centers=centers,
        sizes=sizes,
        rotations=np.array(polygons.rotations, copy=True),
        colors=np.array(polygons.colors, copy=True),
        alphas=np.array(polygons.alphas, copy=True),
        shape_types=np.array(polygons.shape_types, copy=True),
        shape_params=np.array(polygons.shape_params, copy=True),
    )


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


def _top_error_centers(
    error_map: np.ndarray,
    *,
    k: int,
    radius: int,
) -> list[tuple[int, int]]:
    work = np.array(error_map, copy=True)
    _, w = work.shape
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
        y1 = min(work.shape[0], y + radius + 1)
        work[y0:y1, x0:x1] = 0.0

    return centers


def _lab_residual_map(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    target_lab = skcolor.rgb2lab(np.clip(target, 0.0, 1.0)).astype(np.float32, copy=False)
    canvas_lab = skcolor.rgb2lab(np.clip(canvas, 0.0, 1.0)).astype(np.float32, copy=False)
    residual = np.mean(np.abs(target_lab - canvas_lab), axis=2, dtype=np.float32)
    return np.clip(residual / 120.0, 0.0, 1.0).astype(np.float32, copy=False)


def _select_round_shape_type(
    *,
    stage_name: str,
    x: int,
    y: int,
    structure_map: np.ndarray,
    circularity_map: np.ndarray,
) -> int:
    structure = float(structure_map[y, x])
    circularity = float(circularity_map[y, x])

    if stage_name == "A":
        return SHAPE_ELLIPSE
    if stage_name == "B":
        return SHAPE_TRIANGLE if structure >= 0.30 else SHAPE_ELLIPSE
    if stage_name == "C":
        if structure >= 0.60 and circularity < 0.30:
            return SHAPE_THIN_STROKE
        return SHAPE_TRIANGLE if structure >= 0.30 else SHAPE_ELLIPSE
    return SHAPE_TRIANGLE if structure >= 0.30 else SHAPE_ELLIPSE


def _mean_lab_rgb(weighted_lab: np.ndarray) -> np.ndarray:
    rgb = skcolor.lab2rgb(weighted_lab.reshape(1, 1, 3)).reshape(3)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def _build_structure_guidance(target: np.ndarray) -> _StructureGuidance:
    luma = (
        0.2126 * target[..., 0]
        + 0.7152 * target[..., 1]
        + 0.0722 * target[..., 2]
    ).astype(np.float32, copy=False)
    smoothed = gaussian_filter(luma, sigma=1.1, mode="reflect")
    gy, gx = np.gradient(smoothed)
    gradient = np.sqrt(gx * gx + gy * gy, dtype=np.float32).astype(np.float32, copy=False)

    jxx = gaussian_filter(gx * gx, sigma=1.4, mode="reflect")
    jyy = gaussian_filter(gy * gy, sigma=1.4, mode="reflect")
    jxy = gaussian_filter(gx * gy, sigma=1.4, mode="reflect")

    trace = jxx + jyy
    det = np.clip(jxx * jyy - jxy * jxy, 0.0, None)
    disc = np.clip(0.25 * trace * trace - det, 0.0, None)
    root = np.sqrt(disc, dtype=np.float32)
    eig1 = 0.5 * trace + root
    eig2 = 0.5 * trace - root
    anisotropy = np.clip((eig1 - eig2) / (eig1 + eig2 + 1e-6), 0.0, 1.0).astype(
        np.float32, copy=False
    )
    cornerness = np.clip(det - 0.04 * trace * trace, 0.0, None).astype(
        np.float32, copy=False
    )
    cornerness = cornerness / max(float(np.percentile(cornerness, 99.0)), 1e-6)
    cornerness = np.clip(cornerness, 0.0, 1.0).astype(np.float32, copy=False)

    orientation = np.arctan2(gy, gx).astype(np.float32, copy=False)

    return _StructureGuidance(
        gradient=np.clip(gradient / max(float(np.percentile(gradient, 98.0)), 1e-6), 0.0, 1.0),
        anisotropy=anisotropy,
        cornerness=cornerness,
        orientation=orientation,
    )


def _clustered_error_centers(
    error_map: np.ndarray,
    *,
    k: int,
    rng: np.random.Generator,
    pool_factor: int = 8,
    suppression_radius: int = 4,
) -> list[tuple[int, int]]:
    flat = error_map.reshape(-1)
    if k <= 0 or flat.size == 0:
        return []

    pool = int(min(flat.size, max(k * pool_factor, k)))
    idx_pool = np.argpartition(flat, -pool)[-pool:]
    weights = np.clip(flat[idx_pool], 0.0, None).astype(np.float32, copy=False)
    if not np.any(weights > 0.0):
        return []

    h, w = error_map.shape
    ys, xs = np.divmod(idx_pool, w)
    coords = np.stack([xs, ys], axis=1).astype(np.float32, copy=False)

    n_clusters = int(max(1, min(k, coords.shape[0])))
    if n_clusters == coords.shape[0]:
        order = np.argsort(weights)[::-1]
        ranked_points = [(int(coords[i, 0]), int(coords[i, 1])) for i in order]
    else:
        seed = int(rng.integers(0, 2**31 - 1))
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=min(1024, coords.shape[0]),
            n_init=3,
        )
        try:
            labels = model.fit_predict(coords, sample_weight=weights + 1e-6)
        except TypeError:
            labels = model.fit_predict(coords)

        ranked_weighted: list[tuple[int, int, float]] = []
        for cluster_id in range(n_clusters):
            members = np.where(labels == cluster_id)[0]
            if members.size == 0:
                continue
            best_local = members[int(np.argmax(weights[members]))]
            ranked_weighted.append(
                (
                    int(coords[best_local, 0]),
                    int(coords[best_local, 1]),
                    float(weights[best_local]),
                )
            )
        ranked_weighted.sort(key=lambda item: item[2], reverse=True)
        ranked_points = [(x, y) for x, y, _ in ranked_weighted]

    selected: list[tuple[int, int]] = []
    radius = int(max(1, suppression_radius))
    for cx, cy in ranked_points:
        if len(selected) >= k:
            break
        if any((abs(cx - sx) <= radius and abs(cy - sy) <= radius) for sx, sy in selected):
            continue
        selected.append((cx, cy))

    if len(selected) < k:
        fallback = _top_error_centers(error_map, k=k - len(selected), radius=radius)
        for center in fallback:
            if len(selected) >= k:
                break
            if center not in selected:
                selected.append(center)

    return selected


def _shape_for_structure(
    guidance: _StructureGuidance,
    *,
    x: int,
    y: int,
) -> tuple[int, float]:
    grad = float(guidance.gradient[y, x])
    anis = float(guidance.anisotropy[y, x])
    corner = float(guidance.cornerness[y, x])
    orient = float(guidance.orientation[y, x])

    if corner > 0.42 and grad > 0.10:
        return SHAPE_TRIANGLE, orient
    if anis > 0.35 and grad > 0.08:
        return SHAPE_QUAD, orient
    return SHAPE_ELLIPSE, orient


def _high_frequency_error_map(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    residual = np.mean(np.abs(target - canvas), axis=2, dtype=np.float32)
    smooth = gaussian_filter(residual, sigma=3.0, mode="reflect")
    hf = np.clip(residual - smooth, 0.0, None)
    return gaussian_filter(hf, sigma=1.0, mode="reflect").astype(np.float32, copy=False)


def _apply_residual_color_correction(
    optimizer: LiveJointOptimizer,
    *,
    target: np.ndarray,
    strength: float,
    softness: float,
) -> float:
    if optimizer.polygons.count == 0:
        return 0.0

    before_loss = float(optimizer.loss_history[-1])
    before_colors = np.array(optimizer.polygons.colors, copy=True)
    before_alphas = np.array(optimizer.polygons.alphas, copy=True)
    before_canvas = np.array(optimizer.current_canvas, copy=True)
    before_len = len(optimizer.loss_history)

    residual = target - optimizer.current_canvas
    centers = np.round(optimizer.polygons.centers).astype(np.int32)
    centers[:, 0] = np.clip(centers[:, 0], 0, target.shape[1] - 1)
    centers[:, 1] = np.clip(centers[:, 1], 0, target.shape[0] - 1)

    correction = residual[centers[:, 1], centers[:, 0]]
    optimizer.polygons.colors = np.clip(
        optimizer.polygons.colors + float(strength) * correction,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)

    optimizer.polygons.alphas = np.clip(
        optimizer.polygons.alphas
        + 0.12
        * np.mean(np.abs(correction), axis=1, dtype=np.float32),
        optimizer.config.min_alpha,
        optimizer.config.max_alpha,
    ).astype(np.float32, copy=False)

    trial_loss = float(optimizer.step(float(max(softness, 0.05))))
    if trial_loss > before_loss:
        optimizer.polygons.colors = before_colors
        optimizer.polygons.alphas = before_alphas
        optimizer.current_canvas = before_canvas
        del optimizer.loss_history[before_len:]
        return 0.0

    return float(before_loss - trial_loss)


def _add_targeted_batch(
    optimizer: LiveJointOptimizer,
    *,
    stage_name: str,
    target: np.ndarray,
    batch_size: int,
    size_min_px: float,
    size_max_px: float,
    alpha_min: float,
    alpha_max: float,
    high_frequency: bool,
    rng: np.random.Generator,
    error_map: np.ndarray | None = None,
    structure_map: np.ndarray | None = None,
    circularity_map: np.ndarray | None = None,
    angle_map: np.ndarray | None = None,
) -> None:
    if batch_size <= 0:
        return

    if error_map is not None:
        err = np.clip(error_map, 0.0, None).astype(np.float32, copy=False)
    elif high_frequency:
        err = _high_frequency_error_map(target, optimizer.current_canvas)
    else:
        err = _lab_residual_map(target, optimizer.current_canvas)

    sampled_regions = sample_diverse_error_regions(
        target,
        optimizer.current_canvas,
        count=batch_size,
        window=5,
        top_k=50,
        guide_map=err,
        rng=rng,
    )
    if not sampled_regions:
        return

    centers = [
        (
            int(np.clip(round(center[0]), 0, target.shape[1] - 1)),
            int(np.clip(round(center[1]), 0, target.shape[0] - 1)),
            region_box,
            patch_color,
        )
        for center, region_box, patch_color in sampled_regions
    ]
    values = np.array([err[cy, cx] for cx, cy, _, _ in centers], dtype=np.float32)
    vmax = float(np.max(values)) if values.size else 1.0
    vmin = float(np.min(values)) if values.size else 0.0
    denom = max(vmax - vmin, 1e-6)

    for idx, (cx, cy, region_box, patch_color) in enumerate(centers):
        score = float((values[idx] - vmin) / denom)
        if high_frequency:
            size_px = float(size_max_px - (size_max_px - size_min_px) * score)
            alpha = float(alpha_min + (alpha_max - alpha_min) * np.sqrt(score))
        else:
            size_px = float(size_min_px + (size_max_px - size_min_px) * score)
            alpha = float(alpha_min + (alpha_max - alpha_min) * score)
        alpha = float(np.clip(alpha + rng.normal(0.0, 0.03), alpha_min, alpha_max))
        size_px = float(max(size_min_px, min(size_max_px, size_px)))

        if structure_map is not None and circularity_map is not None:
            shape_type = _select_round_shape_type(
                stage_name=stage_name,
                x=int(cx),
                y=int(cy),
                structure_map=structure_map,
                circularity_map=circularity_map,
            )
        else:
            shape_type = SHAPE_ELLIPSE

        if stage_name == "D":
            size_px = min(size_px, 7.0)

        if angle_map is not None and structure_map is not None:
            rotation, shape_params, size_x, size_y = _initialize_shape_params(
                shape_type=shape_type,
                center_x=float(cx),
                center_y=float(cy),
                size_x=float(size_px),
                size_y=float(size_px),
                angle_map=angle_map,
                magnitude_map=structure_map,
                x0=region_box[0],
                y0=region_box[1],
                x1=region_box[2],
                y1=region_box[3],
            )
        else:
            rotation = float(rng.uniform(-np.pi, np.pi))
            shape_params = np.zeros((6,), dtype=np.float32)
            size_x = size_px
            size_y = size_px

        color_hint = np.asarray(patch_color, dtype=np.float32)
        optimizer.add_polygon(
            center_x=float(cx),
            center_y=float(cy),
            size_x=float(size_x),
            size_y=float(size_y),
            color=(
                float(np.clip(color_hint[0], 0.0, 1.0)),
                float(np.clip(color_hint[1], 0.0, 1.0)),
                float(np.clip(color_hint[2], 0.0, 1.0)),
            ),
            alpha=float(alpha),
            shape_type=int(shape_type),
            rotation=float(rotation),
            shape_params=shape_params,
        )


def _nudge_recent_polygons_toward_error(
    optimizer: LiveJointOptimizer,
    *,
    error_map: np.ndarray,
    recent_count: int,
    max_shift: float,
) -> None:
    count = optimizer.polygons.count
    if count <= 0 or recent_count <= 0:
        return

    h, w = error_map.shape
    start = max(0, count - int(recent_count))
    shift = float(max(0.25, max_shift))
    radius = max(2, int(round(shift * 2.0)))

    for idx in range(start, count):
        cx = int(np.clip(round(float(optimizer.polygons.centers[idx, 0])), 0, w - 1))
        cy = int(np.clip(round(float(optimizer.polygons.centers[idx, 1])), 0, h - 1))

        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        patch = error_map[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
        tx = float(x0 + px)
        ty = float(y0 + py)
        dx = float(np.clip(tx - float(cx), -shift, shift))
        dy = float(np.clip(ty - float(cy), -shift, shift))

        optimizer.polygons.centers[idx, 0] = np.clip(
            float(optimizer.polygons.centers[idx, 0]) + dx,
            0.0,
            float(w - 1),
        )
        optimizer.polygons.centers[idx, 1] = np.clip(
            float(optimizer.polygons.centers[idx, 1]) + dy,
            0.0,
            float(h - 1),
        )


def _progressive_resolutions(width: int) -> list[int]:
    full = max(16, int(width))
    candidates = [
        max(48, int(round(full * 0.25))),
        max(80, int(round(full * 0.50))),
        full,
    ]

    schedule: list[int] = []
    for value in candidates:
        clipped = min(full, max(16, int(value)))
        if not schedule or clipped != schedule[-1]:
            schedule.append(clipped)

    if schedule[-1] != full:
        schedule.append(full)

    return schedule


def handle_phase7_control_key(
    key: str,
    *,
    controls: Phase7ControlState,
    screenshot_callback,
    quit_callback,
) -> str:
    normalized = (key or "").lower().strip()

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

    if normalized == "s":
        controls.show_segmentation_overlay = not controls.show_segmentation_overlay
        return "segmentation-toggle"
    if normalized == "x":
        controls.show_scatter_overlay = not controls.show_scatter_overlay
        return "scatter-toggle"
    if normalized == "e":
        controls.residual_mode_index = (controls.residual_mode_index + 1) % 3
        return "residual-mode"

    if normalized == "v":
        controls.view_mode_index = (controls.view_mode_index + 1) % 3
        return "view-cycle"
    if normalized in {"1", "2", "3"}:
        controls.view_mode_index = int(normalized) - 1
        return "view-set"

    if normalized == "g":
        controls.force_growth_requests += 1
        return "force-growth"
    if normalized == "d":
        controls.force_decompose_requests += 1
        return "force-decompose"

    if normalized in {"+", "=", "plus"}:
        controls.softness_scale = float(
            np.clip(controls.softness_scale * 1.10, 0.35, 3.00)
        )
        return "softness-up"
    if normalized in {"-", "_", "minus"}:
        controls.softness_scale = float(
            np.clip(controls.softness_scale / 1.10, 0.35, 3.00)
        )
        return "softness-down"

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
    on_forced_actions,
    max_total_steps: int | None,
    loss_history: list[float],
    diagnostics_every: int | None = None,
    diagnostics_callback=None,
    step_callback=None,
) -> int:
    executed = 0

    for i in range(max(0, int(steps))):
        if controls.quit_requested or deadline_reached():
            break
        if max_total_steps is not None and len(loss_history) >= max_total_steps:
            break

        while (
            controls.paused and not controls.quit_requested and not deadline_reached()
        ):
            time.sleep(0.05)

        if controls.quit_requested or deadline_reached():
            break

        t = i / max(int(steps) - 1, 1)
        softness_base = start_softness + (end_softness - start_softness) * t
        softness = float(np.clip(softness_base * controls.softness_scale, 0.01, 5.0))
        controls.latest_softness = softness

        forced_changed = on_forced_actions(stage_name, softness)
        if forced_changed:
            emit_update(stage_name)
            if max_total_steps is not None and len(loss_history) >= max_total_steps:
                break
            if controls.quit_requested or deadline_reached():
                break

        position_triggered = bool(
            optimizer.config.position_update_interval > 0
            and (
                optimizer.step_count % optimizer.config.position_update_interval == 0
            )
        )
        loss = optimizer.step(softness)
        loss_history.append(float(loss))
        executed += 1

        if step_callback is not None:
            step_callback(stage_name, position_triggered)

        if (
            diagnostics_every is not None
            and diagnostics_every > 0
            and diagnostics_callback is not None
            and (executed % diagnostics_every == 0)
        ):
            diagnostics_callback(
                stage_name,
                executed,
                softness,
                position_triggered,
            )

        if executed % 10 == 0:
            emit_update(stage_name)

    emit_update(stage_name)
    return executed


def _apply_palette_refinement(
    optimizer: LiveJointOptimizer,
    *,
    target_level: np.ndarray,
    softness: float,
) -> float:
    if optimizer.polygons.count <= 0:
        return float(optimizer.loss_history[-1])

    target_lab = skcolor.rgb2lab(np.clip(target_level, 0.0, 1.0)).astype(
        np.float32, copy=False
    )
    refined_colors = np.array(optimizer.polygons.colors, copy=True)

    for idx in range(optimizer.polygons.count):
        coverage = optimizer.rasterizer.single_coverage(
            optimizer.polygons,
            idx,
            max(float(softness), 0.3),
        )
        weights = np.clip(coverage, 0.0, None).astype(np.float32, copy=False)
        total = float(np.sum(weights, dtype=np.float32))
        if total <= 1e-6:
            continue

        mean_lab = (
            np.einsum("hw,hwc->c", weights, target_lab, dtype=np.float32)
            / max(total, 1e-6)
        ).astype(np.float32, copy=False)
        local_rgb = _mean_lab_rgb(mean_lab)
        refined_colors[idx] = np.clip(
            0.70 * local_rgb + 0.30 * optimizer.polygons.colors[idx],
            0.0,
            1.0,
        )

    optimizer.polygons.colors = refined_colors.astype(np.float32, copy=False)
    render = optimizer.rasterizer.render(
        optimizer.polygons,
        softness=max(float(softness), 0.3),
        chunk_size=optimizer.config.render_chunk_size,
    )
    optimizer.current_canvas = render.canvas
    loss = float(optimizer._loss(optimizer.current_canvas, optimizer.target))
    optimizer.loss_history.append(loss)
    return loss


def _execute_phase7_round_schedule(
    *,
    target_full: np.ndarray,
    plan: Phase7Plan,
    random_seed: int,
    controls: Phase7ControlState,
    shared_update_callback,
    max_total_steps: int | None,
    stage_checkpoint_callback,
    start_time: float,
    soft_deadline: float | None,
    hard_deadline: float | None,
) -> Phase7ExecutionResult:
    full_h, full_w = target_full.shape[:2]
    full_resolution = int(min(full_w, full_h))
    rng = np.random.default_rng(random_seed)

    def _deadline_reached() -> bool:
        now = time.monotonic()
        return bool(
            (soft_deadline is not None and now >= soft_deadline)
            or (hard_deadline is not None and now >= hard_deadline)
        )

    def _remaining_seconds() -> float:
        deadlines = [d for d in (soft_deadline, hard_deadline) if d is not None]
        if not deadlines:
            return float("inf")
        return float(min(deadlines) - time.monotonic())

    def _target_at_resolution(resolution: int) -> np.ndarray:
        if resolution == full_resolution and full_h == full_w:
            return target_full
        return _resize_rgb(target_full, width=resolution, height=resolution)

    round_specs = [
        {
            "stage": "A",
            "resolution": min(50, full_resolution),
            "seed_count": int(plan.stage_a_initial_polygons),
            "batches": 10,
            "batch_size": 8,
            "steps_per_batch": 200,
            "size_min": 8.0,
            "size_max": 18.0,
            "color_lr": 0.05,
            "position_lr": 0.80,
            "size_lr": 0.30,
            "max_fd_polygons": None,
            "high_frequency": False,
            "alpha_min": 0.72,
            "alpha_max": 0.88,
            "softness_start": 1.8,
            "softness_end": 0.55,
        },
        {
            "stage": "B",
            "resolution": min(100, full_resolution),
            "seed_count": 0,
            "batches": 8,
            "batch_size": 6,
            "steps_per_batch": 150,
            "size_min": 5.0,
            "size_max": 15.0,
            "color_lr": 0.04,
            "position_lr": 0.60,
            "size_lr": 0.20,
            "max_fd_polygons": None,
            "high_frequency": False,
            "alpha_min": 0.55,
            "alpha_max": 0.80,
            "softness_start": 1.3,
            "softness_end": 0.45,
        },
        {
            "stage": "C",
            "resolution": min(200, full_resolution),
            "seed_count": 0,
            "batches": 10,
            "batch_size": 5,
            "steps_per_batch": 100,
            "size_min": 3.0,
            "size_max": 10.0,
            "color_lr": 0.03,
            "position_lr": 0.40,
            "size_lr": 0.15,
            "max_fd_polygons": 40,
            "high_frequency": False,
            "alpha_min": 0.38,
            "alpha_max": 0.62,
            "softness_start": 0.95,
            "softness_end": 0.32,
        },
        {
            "stage": "D",
            "resolution": min(200, full_resolution),
            "seed_count": 0,
            "batches": 10,
            "batch_size": 3,
            "steps_per_batch": 80,
            "size_min": 3.0,
            "size_max": 7.0,
            "color_lr": 0.02,
            "position_lr": 0.20,
            "size_lr": 0.08,
            "max_fd_polygons": 40,
            "high_frequency": True,
            "alpha_min": 0.28,
            "alpha_max": 0.50,
            "softness_start": 0.70,
            "softness_end": 0.24,
        },
    ]
    if full_resolution <= 50:
        round_specs = [round_specs[0]]
    elif full_resolution <= 100:
        round_specs = round_specs[:2]

    def _round_config(spec: dict[str, object]) -> LiveOptimizerConfig:
        return LiveOptimizerConfig(
            color_lr=float(spec["color_lr"]),
            position_lr=float(spec["position_lr"]),
            size_lr=float(spec["size_lr"]),
            rotation_lr=0.0,
            alpha_lr=0.0,
            position_update_interval=1,
            size_update_interval=3,
            max_fd_polygons=spec["max_fd_polygons"],  # type: ignore[arg-type]
            render_chunk_size=50,
            checkpoint_stride=10,
            min_size=float(spec["size_min"]),
            max_size=float(spec["size_max"]),
            min_alpha=0.05,
            max_alpha=0.98,
            allow_loss_increase=False,
            use_lab_loss=True,
        )

    current_resolution = int(round_specs[0]["resolution"])
    target_level = _target_at_resolution(current_resolution)
    structure_map, circularity_map, angle_map = _compute_structure_maps(target_level)
    optimizer = LiveJointOptimizer(
        target_image=target_level,
        rasterizer=SoftRasterizer(height=current_resolution, width=current_resolution),
        polygons=make_grid_seeded_batch(
            target=target_level,
            count=int(round_specs[0]["seed_count"]),
            alpha=0.85,
        ),
        config=_round_config(round_specs[0]),
    )

    loss_history: list[float] = [float(optimizer.loss_history[-1])]
    stage_markers: list[tuple[str, int]] = []
    batch_markers: list[int] = []
    resolution_markers: list[int] = [0]
    stage_iteration = 0
    stage_start_loss = float(loss_history[-1])
    stage_position_updates = 0
    active_stage_name: str | None = None

    def _display_canvas() -> np.ndarray:
        canvas = np.array(optimizer.current_canvas, copy=True)
        if current_resolution == full_resolution and full_h == full_w:
            return canvas
        return _resize_rgb(canvas, width=full_w, height=full_h)

    def _display_polygons() -> LivePolygonBatch:
        poly = optimizer.polygons.copy()
        if current_resolution == full_resolution and full_h == full_w:
            return poly
        return _scale_polygons_to_resolution(
            poly,
            old_width=current_resolution,
            old_height=current_resolution,
            new_width=full_w,
            new_height=full_h,
        )

    def _emit(stage_name: str, *, running: bool = True, status_override: str | None = None) -> None:
        canvas = _display_canvas()
        rgb_mse = _rgb_mse(target_full, canvas)
        status = (
            status_override
            if status_override is not None
            else f"stage={stage_name} res={current_resolution} polygons={optimizer.polygons.count} rgb_mse={rgb_mse:.6f}"
        )
        shared_update_callback(
            canvas,
            _display_polygons(),
            full_w,
            list(loss_history),
            list(resolution_markers),
            list(batch_markers),
            stage_name,
            len(loss_history),
            rgb_mse,
            stage_iteration,
            stage_start_loss,
            stage_position_updates,
            running,
            status,
            list(stage_markers),
        )

    def _emit_stage_checkpoint(stage_name: str) -> None:
        if stage_checkpoint_callback is None:
            return
        canvas = _display_canvas()
        stage_checkpoint_callback(
            stage_name,
            canvas,
            {
                "stage": stage_name,
                "global_iteration": int(len(loss_history)),
                "stage_iteration": int(stage_iteration),
                "stage_start_loss": float(stage_start_loss),
                "stage_position_updates": int(stage_position_updates),
                "polygon_count": int(optimizer.polygons.count),
                "rgb_mse": float(_rgb_mse(target_full, canvas)),
                "resolution": int(current_resolution),
                "elapsed_seconds": float(time.monotonic() - start_time),
                "softness": float(controls.latest_softness),
            },
        )

    def _begin_stage(stage_name: str) -> None:
        nonlocal stage_iteration
        nonlocal stage_start_loss
        nonlocal stage_position_updates
        nonlocal active_stage_name
        if active_stage_name is not None:
            _emit_stage_checkpoint(active_stage_name)
        stage_markers.append((stage_name, len(loss_history)))
        stage_iteration = 0
        stage_start_loss = float(optimizer.loss_history[-1])
        stage_position_updates = 0
        active_stage_name = stage_name

    def _stage_step_callback(_stage_name: str, position_triggered: bool) -> None:
        nonlocal stage_iteration
        nonlocal stage_position_updates
        stage_iteration += 1
        if position_triggered:
            stage_position_updates += 1

    def _transition_to_round(spec: dict[str, object], stage_name: str) -> None:
        nonlocal optimizer
        nonlocal target_level
        nonlocal current_resolution
        nonlocal structure_map
        nonlocal circularity_map
        nonlocal angle_map

        new_resolution = int(spec["resolution"])
        scaled = optimizer.polygons.copy()
        if new_resolution != current_resolution:
            scaled = _scale_polygons_to_resolution(
                optimizer.polygons,
                old_width=current_resolution,
                old_height=current_resolution,
                new_width=new_resolution,
                new_height=new_resolution,
            )

        current_resolution = new_resolution
        target_level = _target_at_resolution(current_resolution)
        structure_map, circularity_map, angle_map = _compute_structure_maps(target_level)
        optimizer = LiveJointOptimizer(
            target_image=target_level,
            rasterizer=SoftRasterizer(height=current_resolution, width=current_resolution),
            polygons=scaled,
            config=_round_config(spec),
        )
        loss_history.append(float(optimizer.loss_history[-1]))
        resolution_markers.append(len(loss_history) - 1)
        _emit(stage_name, status_override=f"resolution transition -> {current_resolution}x{current_resolution}")

    skip_to_final_pass = False
    last_stage_name = str(round_specs[0]["stage"])

    for round_index, spec in enumerate(round_specs):
        stage_name = str(spec["stage"])
        last_stage_name = stage_name
        if controls.quit_requested or _deadline_reached():
            break
        if max_total_steps is not None and len(loss_history) >= max_total_steps:
            break

        if round_index > 0:
            _apply_palette_refinement(
                optimizer,
                target_level=target_level,
                softness=max(float(spec["softness_end"]), 0.3),
            )
            loss_history.append(float(optimizer.loss_history[-1]))
            _transition_to_round(spec, stage_name)

        _begin_stage(stage_name)
        _emit(stage_name)

        for _batch_idx in range(int(spec["batches"])):
            if controls.quit_requested or _deadline_reached():
                break
            if max_total_steps is not None and len(loss_history) >= max_total_steps:
                break
            if _remaining_seconds() < 30.0:
                skip_to_final_pass = True
                _emit(
                    stage_name,
                    status_override="remaining time under 30s, skipping additions and running final optimization",
                )
                break

            checkpoint_len = len(loss_history)
            checkpoint_loss = float(optimizer.loss_history[-1])
            checkpoint_polygons = optimizer.polygons.copy()
            checkpoint_canvas = np.array(optimizer.current_canvas, copy=True)

            if bool(spec["high_frequency"]):
                error_map = _high_frequency_error_map(target_level, optimizer.current_canvas)
            elif stage_name == "C":
                hf_err = _high_frequency_error_map(target_level, optimizer.current_canvas)
                lab_err = _lab_residual_map(target_level, optimizer.current_canvas)
                error_map = np.clip(0.65 * hf_err + 0.35 * lab_err, 0.0, 1.0)
            else:
                error_map = _lab_residual_map(target_level, optimizer.current_canvas)

            _add_targeted_batch(
                optimizer,
                stage_name=stage_name,
                target=target_level,
                batch_size=int(spec["batch_size"]),
                size_min_px=float(spec["size_min"]),
                size_max_px=float(spec["size_max"]),
                alpha_min=float(spec["alpha_min"]),
                alpha_max=float(spec["alpha_max"]),
                high_frequency=bool(spec["high_frequency"]),
                rng=rng,
                error_map=error_map,
                structure_map=structure_map,
                circularity_map=circularity_map,
                angle_map=angle_map,
            )
            batch_markers.append(len(loss_history))
            _emit(stage_name)

            _run_stage_steps(
                optimizer=optimizer,
                controls=controls,
                stage_name=stage_name,
                steps=int(spec["steps_per_batch"]),
                start_softness=float(spec["softness_start"]),
                end_softness=float(spec["softness_end"]),
                deadline_reached=_deadline_reached,
                emit_update=_emit,
                on_forced_actions=lambda _stage, _soft: False,
                max_total_steps=max_total_steps,
                loss_history=loss_history,
                step_callback=_stage_step_callback,
            )

            if float(optimizer.loss_history[-1]) > checkpoint_loss + 1e-6:
                optimizer.restore_state(
                    checkpoint_polygons,
                    checkpoint_canvas,
                    checkpoint_loss,
                    record_loss=False,
                )
                del loss_history[checkpoint_len:]
                if batch_markers and batch_markers[-1] >= checkpoint_len:
                    batch_markers.pop()
                _emit(stage_name)

        if skip_to_final_pass:
            break

    if skip_to_final_pass and not controls.quit_requested and not _deadline_reached():
        remaining = max(0.0, _remaining_seconds())
        final_steps = int(max(1, min(240, np.floor(max(remaining - 2.0, 1.0) * 3.0))))
        _run_stage_steps(
            optimizer=optimizer,
            controls=controls,
            stage_name=last_stage_name,
            steps=final_steps,
            start_softness=0.40,
            end_softness=0.20,
            deadline_reached=_deadline_reached,
            emit_update=_emit,
            on_forced_actions=lambda _stage, _soft: False,
            max_total_steps=max_total_steps,
            loss_history=loss_history,
            step_callback=_stage_step_callback,
        )

    final_canvas = _display_canvas()
    final_rgb_mse = _rgb_mse(target_full, final_canvas)
    if active_stage_name is not None:
        _emit_stage_checkpoint(active_stage_name)
    if stage_checkpoint_callback is not None:
        stage_checkpoint_callback(
            "final",
            final_canvas,
            {
                "stage": "final",
                "global_iteration": int(len(loss_history)),
                "stage_iteration": int(stage_iteration),
                "stage_start_loss": float(stage_start_loss),
                "stage_position_updates": int(stage_position_updates),
                "polygon_count": int(optimizer.polygons.count),
                "rgb_mse": float(final_rgb_mse),
                "resolution": int(current_resolution),
                "elapsed_seconds": float(time.monotonic() - start_time),
                "softness": float(controls.latest_softness),
            },
        )

    shared_update_callback(
        final_canvas,
        _display_polygons(),
        full_w,
        list(loss_history),
        list(resolution_markers),
        list(batch_markers),
        "done",
        len(loss_history),
        final_rgb_mse,
        stage_iteration,
        stage_start_loss,
        stage_position_updates,
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
        resolution_markers=list(resolution_markers),
        batch_markers=list(batch_markers),
        stage_markers=list(stage_markers),
    )


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
    stage_checkpoint_callback=None,
) -> Phase7ExecutionResult:
    if target_image.ndim != 3 or target_image.shape[2] != 3:
        raise ValueError("target_image must have shape (H, W, 3)")

    target_full = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)

    start_time = time.monotonic()
    soft_deadline = (
        start_time + max(0.0, float(minutes) * 60.0) if minutes > 0.0 else None
    )
    hard_deadline = (
        start_time + float(hard_timeout_seconds)
        if hard_timeout_seconds is not None and hard_timeout_seconds > 0.0
        else None
    )

    return _execute_phase7_round_schedule(
        target_full=target_full,
        plan=plan,
        random_seed=random_seed,
        controls=controls,
        shared_update_callback=shared_update_callback,
        max_total_steps=max_total_steps,
        stage_checkpoint_callback=stage_checkpoint_callback,
        start_time=start_time,
        soft_deadline=soft_deadline,
        hard_deadline=hard_deadline,
    )

    full_h, full_w = target_full.shape[:2]

    runtime_scale = 1.0
    if minutes > 0.0:
        runtime_scale = float(np.clip((minutes * 60.0) / 240.0, 0.08, 1.80))

    def _scaled_count(value: int, *, minimum: int) -> int:
        if value <= 0:
            return 0
        return max(minimum, int(round(float(value) * runtime_scale)))

    stage_a_steps = _scaled_count(int(plan.stage_a_steps), minimum=24)
    stage_a_steps = max(10, int(round(stage_a_steps * 0.25)))
    stage_a_pos_steps = max(4, int(round(stage_a_steps * 0.25)))
    stage_a_color_steps = max(6, stage_a_steps - stage_a_pos_steps)

    stage_b_batches = _scaled_count(int(plan.stage_b_batches), minimum=1)
    stage_b_batch_size = _scaled_count(int(plan.stage_b_batch_size), minimum=6)
    stage_b_steps_per_batch = _scaled_count(
        int(plan.stage_b_steps_per_batch), minimum=60
    )

    stage_c_batches = _scaled_count(int(plan.stage_c_batches), minimum=1)
    stage_c_batch_size = _scaled_count(int(plan.stage_c_batch_size), minimum=6)
    stage_c_steps_per_batch = _scaled_count(
        int(plan.stage_c_steps_per_batch), minimum=60
    )

    if runtime_scale < 1.0:
        stage_d_steps = _scaled_count(int(plan.stage_d_steps), minimum=120)
    else:
        stage_d_steps = int(max(120, int(plan.stage_d_steps)))

    resolution_schedule = _progressive_resolutions(full_w)

    def _max_fd_for_resolution(resolution: int) -> int | None:
        return None if int(resolution) <= 100 else 40

    def _target_at_resolution(resolution: int) -> np.ndarray:
        if resolution == full_w and full_h == full_w:
            return target_full
        return _resize_rgb(target_full, width=resolution, height=resolution)

    current_resolution = int(resolution_schedule[0])
    target_level = _target_at_resolution(current_resolution)
    structure_guidance = _build_structure_guidance(target_level)
    initial_seed_count = int(plan.stage_a_initial_polygons)
    if current_resolution <= 50:
        initial_seed_count = min(initial_seed_count, 24)

    optimizer = LiveJointOptimizer(
        target_image=target_level,
        rasterizer=SoftRasterizer(height=current_resolution, width=current_resolution),
        polygons=make_grid_seeded_batch(
            target=target_level,
            count=initial_seed_count,
            alpha=0.85,
        ),
        config=LiveOptimizerConfig(
            color_lr=0.05,
            position_lr=0.50,
            size_lr=0.30,
            rotation_lr=0.02,
            alpha_lr=0.01,
            position_update_interval=1,
            size_update_interval=3,
            max_fd_polygons=_max_fd_for_resolution(current_resolution),
            render_chunk_size=50,
            checkpoint_stride=10,
            min_size=3.0,
            max_size=60.0,
            min_alpha=0.05,
            max_alpha=0.98,
            allow_loss_increase=True,
            use_lab_loss=True,
        ),
    )

    loss_history: list[float] = [float(optimizer.loss_history[-1])]
    stage_markers: list[tuple[str, int]] = []
    batch_markers: list[int] = []
    resolution_markers: list[int] = [0]
    stage_iteration = 0
    stage_start_loss = float(loss_history[-1])
    stage_position_updates = 0
    active_stage_name: str | None = None

    def _display_canvas() -> np.ndarray:
        canvas = np.array(optimizer.current_canvas, copy=True)
        if current_resolution == full_w and full_h == full_w:
            return canvas
        return _resize_rgb(canvas, width=full_w, height=full_h)

    def _display_polygons() -> LivePolygonBatch:
        poly = optimizer.polygons.copy()
        if current_resolution == full_w and full_h == full_w:
            return poly
        return _scale_polygons_to_resolution(
            poly,
            old_width=current_resolution,
            old_height=current_resolution,
            new_width=full_w,
            new_height=full_h,
        )

    def _emit(
        stage_name: str, *, running: bool = True, status_override: str | None = None
    ) -> None:
        canvas = _display_canvas()
        loss = _rgb_mse(target_full, canvas)
        status = (
            status_override
            if status_override is not None
            else (
                "stage="
                + stage_name
                + f" res={current_resolution} polygons={optimizer.polygons.count} rgb_mse={loss:.6f}"
            )
        )
        shared_update_callback(
            canvas,
            _display_polygons(),
            full_w,
            list(loss_history),
            list(resolution_markers),
            list(batch_markers),
            stage_name,
            len(loss_history),
            loss,
            stage_iteration,
            stage_start_loss,
            stage_position_updates,
            running,
            status,
            list(stage_markers),
        )

    def _begin_stage(stage_name: str) -> None:
        nonlocal stage_iteration
        nonlocal stage_start_loss
        nonlocal stage_position_updates
        nonlocal active_stage_name

        if active_stage_name is not None:
            _emit_stage_checkpoint(active_stage_name)

        stage_markers.append((stage_name, len(loss_history)))
        stage_iteration = 0
        stage_start_loss = float(optimizer.loss_history[-1])
        stage_position_updates = 0
        active_stage_name = stage_name

    def _stage_step_callback(_stage_name: str, position_triggered: bool) -> None:
        nonlocal stage_iteration
        nonlocal stage_position_updates
        stage_iteration += 1
        if position_triggered:
            stage_position_updates += 1

    def _emit_stage_checkpoint(stage_name: str) -> None:
        if stage_checkpoint_callback is None:
            return
        canvas = _display_canvas()
        metrics = {
            "stage": stage_name,
            "global_iteration": int(len(loss_history)),
            "stage_iteration": int(stage_iteration),
            "stage_start_loss": float(stage_start_loss),
            "stage_position_updates": int(stage_position_updates),
            "polygon_count": int(optimizer.polygons.count),
            "rgb_mse": float(_rgb_mse(target_full, canvas)),
            "resolution": int(current_resolution),
            "elapsed_seconds": float(time.monotonic() - start_time),
            "softness": float(controls.latest_softness),
        }
        stage_checkpoint_callback(stage_name, canvas, metrics)

    def _shape_distribution_summary() -> str:
        if optimizer.polygons.count <= 0:
            return "E:0 Q:0 T:0"
        shape_types = optimizer.polygons.shape_types
        e = int(np.count_nonzero(shape_types == SHAPE_ELLIPSE))
        q = int(np.count_nonzero(shape_types == SHAPE_QUAD))
        t = int(np.count_nonzero(shape_types == SHAPE_TRIANGLE))
        return f"E:{e} Q:{q} T:{t}"

    def _stage_diagnostics(
        stage_name: str,
        executed_steps: int,
        softness: float,
        position_triggered: bool,
    ) -> None:
        if stage_name != "A":
            return
        print(
            "[phase7:A]"
            f" step={executed_steps:04d}"
            f" softness={softness:.4f}"
            f" pos_trigger={'yes' if position_triggered else 'no '}"
            f" shapes={_shape_distribution_summary()}"
        )

    def _forced_size_hint(stage_name: str) -> float:
        if stage_name == "A":
            return float(plan.stage_b_size_start)
        if stage_name == "B":
            return float(0.5 * (plan.stage_b_size_start + plan.stage_b_size_end))
        if stage_name == "C":
            return float(0.5 * (plan.stage_c_size_start + plan.stage_c_size_end))
        return float(plan.stage_c_size_end)

    def _forced_batch_hint(stage_name: str) -> int:
        if stage_name in {"A", "B"}:
            return max(1, stage_b_batch_size // 2)
        return max(1, stage_c_batch_size // 2)

    def _on_forced_actions(stage_name: str, softness: float) -> bool:
        changed = False

        while controls.force_growth_requests > 0:
            if controls.quit_requested or _deadline_reached():
                break
            if max_total_steps is not None and len(loss_history) >= max_total_steps:
                break

            controls.force_growth_requests -= 1

            checkpoint_len = len(loss_history)
            checkpoint_loss = float(optimizer.loss_history[-1])
            checkpoint_polygons = optimizer.polygons.copy()
            checkpoint_canvas = np.array(optimizer.current_canvas, copy=True)

            size_px = _forced_size_hint(stage_name)
            batch_size = _forced_batch_hint(stage_name)
            hf = stage_name in {"C", "D"}

            _add_targeted_batch(
                optimizer,
                target=target_level,
                batch_size=batch_size,
                size_min_px=(3.0 if hf else 8.0),
                size_max_px=(10.0 if hf else 25.0),
                alpha_min=(0.40 if hf else 0.55),
                alpha_max=(0.60 if hf else 0.85),
                high_frequency=hf,
                rng=rng,
            )
            batch_markers.append(len(loss_history))

            settle_steps = max(4, min(14, batch_size * 2))
            for _ in range(settle_steps):
                if controls.quit_requested or _deadline_reached():
                    break
                if max_total_steps is not None and len(loss_history) >= max_total_steps:
                    break
                loss = optimizer.step(float(np.clip(softness * 0.90, 0.01, 5.0)))
                loss_history.append(float(loss))

            if float(optimizer.loss_history[-1]) > checkpoint_loss + 1e-4:
                optimizer.restore_state(
                    checkpoint_polygons,
                    checkpoint_canvas,
                    checkpoint_loss,
                    record_loss=False,
                )
                del loss_history[checkpoint_len:]
                if batch_markers and batch_markers[-1] >= checkpoint_len:
                    batch_markers.pop()
            else:
                changed = True

        while controls.force_decompose_requests > 0:
            if controls.quit_requested or _deadline_reached():
                break
            if max_total_steps is not None and len(loss_history) >= max_total_steps:
                break

            controls.force_decompose_requests -= 1

            checkpoint_len = len(loss_history)
            checkpoint_loss = float(optimizer.loss_history[-1])
            checkpoint_polygons = optimizer.polygons.copy()
            checkpoint_canvas = np.array(optimizer.current_canvas, copy=True)

            before_correction_len = len(optimizer.loss_history)
            gain = _apply_residual_color_correction(
                optimizer,
                target=target_level,
                strength=0.55,
                softness=float(np.clip(softness * 0.90, 0.01, 5.0)),
            )
            for idx in range(before_correction_len, len(optimizer.loss_history)):
                loss_history.append(float(optimizer.loss_history[idx]))

            size_px = max(1.5, _forced_size_hint(stage_name) * 0.65)
            batch_size = max(1, _forced_batch_hint(stage_name))
            _add_targeted_batch(
                optimizer,
                target=target_level,
                batch_size=batch_size,
                size_min_px=3.0,
                size_max_px=max(4.0, size_px),
                alpha_min=0.35,
                alpha_max=0.60,
                high_frequency=True,
                rng=rng,
            )
            batch_markers.append(len(loss_history))

            for _ in range(8):
                if controls.quit_requested or _deadline_reached():
                    break
                if max_total_steps is not None and len(loss_history) >= max_total_steps:
                    break
                loss = optimizer.step(float(np.clip(softness * 0.80, 0.01, 5.0)))
                loss_history.append(float(loss))

            if float(optimizer.loss_history[-1]) > checkpoint_loss and gain <= 0.0:
                optimizer.restore_state(
                    checkpoint_polygons,
                    checkpoint_canvas,
                    checkpoint_loss,
                    record_loss=False,
                )
                del loss_history[checkpoint_len:]
                if batch_markers and batch_markers[-1] >= checkpoint_len:
                    batch_markers.pop()
            else:
                changed = True

        return changed

    def _transition_to_resolution(new_resolution: int, stage_name: str) -> None:
        nonlocal optimizer
        nonlocal target_level
        nonlocal current_resolution
        nonlocal structure_guidance

        if new_resolution == current_resolution:
            return

        scaled = _scale_polygons_to_resolution(
            optimizer.polygons,
            old_width=current_resolution,
            old_height=current_resolution,
            new_width=new_resolution,
            new_height=new_resolution,
        )

        current_resolution = int(new_resolution)
        target_level = _target_at_resolution(current_resolution)
        structure_guidance = _build_structure_guidance(target_level)

        optimizer = LiveJointOptimizer(
            target_image=target_level,
            rasterizer=SoftRasterizer(
                height=current_resolution, width=current_resolution
            ),
            polygons=scaled,
            config=replace(
                optimizer.config,
                position_update_interval=1,
                size_update_interval=3,
                max_fd_polygons=_max_fd_for_resolution(current_resolution),
            ),
        )

        loss_history.append(float(optimizer.loss_history[-1]))
        resolution_markers.append(len(loss_history) - 1)
        _emit(
            stage_name,
            status_override=(
                f"resolution transition -> {current_resolution}x{current_resolution}"
            ),
        )

    _begin_stage("A")
    optimizer.config = replace(
        optimizer.config,
        position_update_interval=1,
        size_update_interval=3,
        max_fd_polygons=_max_fd_for_resolution(current_resolution),
    )
    _run_stage_steps(
        optimizer=optimizer,
        controls=controls,
        stage_name="A",
        steps=int(stage_a_color_steps),
        start_softness=3.0,
        end_softness=1.2,
        deadline_reached=_deadline_reached,
        emit_update=_emit,
        on_forced_actions=_on_forced_actions,
        max_total_steps=max_total_steps,
        loss_history=loss_history,
        diagnostics_every=10,
        diagnostics_callback=_stage_diagnostics,
        step_callback=_stage_step_callback,
    )

    optimizer.config = replace(
        optimizer.config,
        color_lr=0.015,
        position_update_interval=1,
        size_update_interval=3,
        max_fd_polygons=_max_fd_for_resolution(current_resolution),
    )
    _run_stage_steps(
        optimizer=optimizer,
        controls=controls,
        stage_name="A",
        steps=int(stage_a_pos_steps),
        start_softness=1.2,
        end_softness=0.8,
        deadline_reached=_deadline_reached,
        emit_update=_emit,
        on_forced_actions=_on_forced_actions,
        max_total_steps=max_total_steps,
        loss_history=loss_history,
        diagnostics_every=10,
        diagnostics_callback=_stage_diagnostics,
        step_callback=_stage_step_callback,
    )

    if (
        len(resolution_schedule) >= 2
        and not controls.quit_requested
        and not _deadline_reached()
    ):
        _transition_to_resolution(resolution_schedule[1], "B")

    if not controls.quit_requested and not _deadline_reached():
        _begin_stage("B")

    for batch_idx in range(stage_b_batches):
        if controls.quit_requested or _deadline_reached():
            break
        if max_total_steps is not None and len(loss_history) >= max_total_steps:
            break

        t = batch_idx / max(stage_b_batches - 1, 1)
        size_max = float(
            plan.stage_b_size_start
            + (plan.stage_b_size_end - plan.stage_b_size_start) * t
        )
        size_min = max(8.0, size_max * 0.35)

        checkpoint_len = len(loss_history)
        checkpoint_loss = float(optimizer.loss_history[-1])
        checkpoint_polygons = optimizer.polygons.copy()
        checkpoint_canvas = np.array(optimizer.current_canvas, copy=True)

        lab_err = _lab_residual_map(target_level, optimizer.current_canvas)
        structure_weight = np.clip(
            0.55 * structure_guidance.gradient + 0.45 * structure_guidance.cornerness,
            0.0,
            1.0,
        )
        routing_error = np.clip(0.80 * lab_err + 0.20 * structure_weight, 0.0, 1.0)

        _add_targeted_batch(
            optimizer,
            target=target_level,
            batch_size=int(stage_b_batch_size),
            size_min_px=float(size_min),
            size_max_px=float(max(size_max, size_min + 0.5)),
            alpha_min=0.55,
            alpha_max=0.85,
            high_frequency=False,
            rng=rng,
            error_map=routing_error,
            structure_guidance=structure_guidance,
            clustered=True,
            allow_shape_routing=True,
        )
        batch_markers.append(len(loss_history))

        optimizer.config = replace(
            optimizer.config,
            color_lr=0.04,
            position_update_interval=1,
            size_update_interval=3,
            max_fd_polygons=_max_fd_for_resolution(current_resolution),
        )
        _run_stage_steps(
            optimizer=optimizer,
            controls=controls,
            stage_name="B",
            steps=max(1, int(stage_b_steps_per_batch - 1)),
            start_softness=1.1,
            end_softness=0.45,
            deadline_reached=_deadline_reached,
            emit_update=_emit,
            on_forced_actions=_on_forced_actions,
            max_total_steps=max_total_steps,
            loss_history=loss_history,
            step_callback=_stage_step_callback,
        )

        optimizer.config = replace(
            optimizer.config,
            position_update_interval=1,
            size_update_interval=3,
            max_fd_polygons=_max_fd_for_resolution(current_resolution),
        )
        _run_stage_steps(
            optimizer=optimizer,
            controls=controls,
            stage_name="B",
            steps=1,
            start_softness=0.40,
            end_softness=0.40,
            deadline_reached=_deadline_reached,
            emit_update=_emit,
            on_forced_actions=_on_forced_actions,
            max_total_steps=max_total_steps,
            loss_history=loss_history,
            step_callback=_stage_step_callback,
        )

        _nudge_recent_polygons_toward_error(
            optimizer,
            error_map=lab_err,
            recent_count=max(6, stage_b_batch_size),
            max_shift=1.75,
        )
        _run_stage_steps(
            optimizer=optimizer,
            controls=controls,
            stage_name="B",
            steps=2,
            start_softness=0.38,
            end_softness=0.34,
            deadline_reached=_deadline_reached,
            emit_update=_emit,
            on_forced_actions=_on_forced_actions,
            max_total_steps=max_total_steps,
            loss_history=loss_history,
            step_callback=_stage_step_callback,
        )

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

    if (
        len(resolution_schedule) >= 3
        and not controls.quit_requested
        and not _deadline_reached()
    ):
        _transition_to_resolution(resolution_schedule[-1], "C")

    if not controls.quit_requested and not _deadline_reached():
        _begin_stage("C")

    for batch_idx in range(stage_c_batches):
        if controls.quit_requested or _deadline_reached():
            break
        if max_total_steps is not None and len(loss_history) >= max_total_steps:
            break

        t = batch_idx / max(stage_c_batches - 1, 1)
        size_max = float(
            plan.stage_c_size_start
            + (plan.stage_c_size_end - plan.stage_c_size_start) * t
        )
        size_min = max(3.0, size_max * 0.35)

        checkpoint_len = len(loss_history)
        checkpoint_loss = float(optimizer.loss_history[-1])
        checkpoint_polygons = optimizer.polygons.copy()
        checkpoint_canvas = np.array(optimizer.current_canvas, copy=True)

        hf_err = _high_frequency_error_map(target_level, optimizer.current_canvas)
        lab_err = _lab_residual_map(target_level, optimizer.current_canvas)
        detail_error = np.clip(0.65 * hf_err + 0.35 * lab_err, 0.0, 1.0)

        _add_targeted_batch(
            optimizer,
            target=target_level,
            batch_size=int(stage_c_batch_size),
            size_min_px=float(size_min),
            size_max_px=float(max(size_max, size_min + 0.4)),
            alpha_min=0.35,
            alpha_max=0.60,
            high_frequency=True,
            rng=rng,
            error_map=detail_error,
            structure_guidance=structure_guidance,
            clustered=True,
            allow_shape_routing=True,
        )
        batch_markers.append(len(loss_history))

        optimizer.config = replace(
            optimizer.config,
            color_lr=0.03,
            position_update_interval=1,
            size_update_interval=3,
            max_fd_polygons=_max_fd_for_resolution(current_resolution),
        )
        _run_stage_steps(
            optimizer=optimizer,
            controls=controls,
            stage_name="C",
            steps=int(stage_c_steps_per_batch),
            start_softness=0.90,
            end_softness=0.30,
            deadline_reached=_deadline_reached,
            emit_update=_emit,
            on_forced_actions=_on_forced_actions,
            max_total_steps=max_total_steps,
            loss_history=loss_history,
            step_callback=_stage_step_callback,
        )

        if batch_idx % 2 == 0:
            _nudge_recent_polygons_toward_error(
                optimizer,
                error_map=detail_error,
                recent_count=max(8, stage_c_batch_size * 2),
                max_shift=1.25,
            )
            _run_stage_steps(
                optimizer=optimizer,
                controls=controls,
                stage_name="C",
                steps=2,
                start_softness=0.36,
                end_softness=0.30,
                deadline_reached=_deadline_reached,
                emit_update=_emit,
                on_forced_actions=_on_forced_actions,
                max_total_steps=max_total_steps,
                loss_history=loss_history,
                step_callback=_stage_step_callback,
            )

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

    if not controls.quit_requested and not _deadline_reached():
        _begin_stage("D")

    optimizer.config = replace(
        optimizer.config,
        color_lr=0.015,
        position_lr=0.80,
        position_update_interval=1,
        size_update_interval=3,
        max_fd_polygons=_max_fd_for_resolution(current_resolution),
        allow_loss_increase=False,
    )

    softness_schedule = [1.5, 1.2, 0.9, 0.6, 0.4, 0.3]
    stage_d_target_steps = int(max(1, stage_d_steps))
    if stage_d_target_steps >= 600:
        per_level = [100] * len(softness_schedule)
    else:
        base = stage_d_target_steps // len(softness_schedule)
        rem = stage_d_target_steps % len(softness_schedule)
        per_level = [base + (1 if i < rem else 0) for i in range(len(softness_schedule))]

    stage_d_executed = 0

    for softness_level, segment_steps in zip(softness_schedule, per_level, strict=True):
        if segment_steps <= 0:
            continue
        if controls.quit_requested or _deadline_reached():
            break
        if max_total_steps is not None and len(loss_history) >= max_total_steps:
            break

        done = _run_stage_steps(
            optimizer=optimizer,
            controls=controls,
            stage_name="D",
            steps=int(segment_steps),
            start_softness=float(softness_level),
            end_softness=float(softness_level),
            deadline_reached=_deadline_reached,
            emit_update=_emit,
            on_forced_actions=_on_forced_actions,
            max_total_steps=max_total_steps,
            loss_history=loss_history,
            step_callback=_stage_step_callback,
        )
        stage_d_executed += int(done)

        if stage_d_executed < 200:
            window = min(40, max(8, stage_d_executed // 2))
            if len(loss_history) >= window + 1:
                prev = float(loss_history[-(window + 1)])
                curr = float(loss_history[-1])
                rel_improvement = (prev - curr) / max(abs(prev), 1e-8)
                if rel_improvement < 1e-4:
                    _emit(
                        "D",
                        status_override=(
                            f"stage D converged early at step {stage_d_executed}"
                        ),
                    )
                    break

    final_canvas = _display_canvas()
    final_rgb_mse = _rgb_mse(target_full, final_canvas)

    if active_stage_name is not None:
        _emit_stage_checkpoint(active_stage_name)

    if stage_checkpoint_callback is not None:
        final_metrics = {
            "stage": "final",
            "global_iteration": int(len(loss_history)),
            "stage_iteration": int(stage_iteration),
            "stage_start_loss": float(stage_start_loss),
            "stage_position_updates": int(stage_position_updates),
            "polygon_count": int(optimizer.polygons.count),
            "rgb_mse": float(final_rgb_mse),
            "resolution": int(current_resolution),
            "elapsed_seconds": float(time.monotonic() - start_time),
            "softness": float(controls.latest_softness),
        }
        stage_checkpoint_callback("final", final_canvas, final_metrics)

    shared_update_callback(
        final_canvas,
        _display_polygons(),
        full_w,
        list(loss_history),
        list(resolution_markers),
        list(batch_markers),
        "done",
        len(loss_history),
        final_rgb_mse,
        stage_iteration,
        stage_start_loss,
        stage_position_updates,
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


def _segmentation_overlay_rgba(
    segmentation_map: np.ndarray | None,
    *,
    width: int,
    height: int,
) -> np.ndarray | None:
    if segmentation_map is None:
        return None

    seg = _resize_labels(
        segmentation_map.astype(np.int32, copy=False), width=width, height=height
    )
    edges = np.zeros((height, width), dtype=bool)
    edges[:-1, :] |= seg[:-1, :] != seg[1:, :]
    edges[:, :-1] |= seg[:, :-1] != seg[:, 1:]

    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 1] = 0.92
    rgba[..., 2] = 0.10
    rgba[..., 3] = edges.astype(np.float32) * 0.65
    return rgba


def _quad_outline_vertices(
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    rotation: float,
) -> np.ndarray:
    local = np.array(
        [
            [sx, sy],
            [sx, -sy],
            [-sx, -sy],
            [-sx, sy],
        ],
        dtype=np.float32,
    )
    cos_r = float(np.cos(rotation))
    sin_r = float(np.sin(rotation))
    rot = np.array(
        [
            [cos_r, -sin_r],
            [sin_r, cos_r],
        ],
        dtype=np.float32,
    )
    verts = local @ rot.T
    verts[:, 0] += float(cx)
    verts[:, 1] += float(cy)
    return verts


def _draw_outline_panel(
    ax,
    *,
    centers: np.ndarray,
    sizes: np.ndarray,
    rotations: np.ndarray,
    shape_types: np.ndarray,
    width: int,
    height: int,
) -> None:
    ax.imshow(np.ones((height, width, 3), dtype=np.float32))
    ax.set_xticks([])
    ax.set_yticks([])

    count = int(centers.shape[0])
    if count == 0:
        return

    max_draw = 1200
    if count > max_draw:
        draw_indices = np.linspace(0, count - 1, max_draw, dtype=np.int32)
    else:
        draw_indices = np.arange(count, dtype=np.int32)

    draw_sizes = np.maximum(sizes[draw_indices, 0], sizes[draw_indices, 1]).astype(
        np.float32
    )
    if draw_sizes.size >= 3:
        q1, q2 = np.quantile(draw_sizes, [0.33, 0.66])
    elif draw_sizes.size == 2:
        q1, q2 = float(np.min(draw_sizes)), float(np.max(draw_sizes))
    else:
        q1 = q2 = float(draw_sizes[0])

    for idx in draw_indices:
        cx = float(centers[idx, 0])
        cy = float(centers[idx, 1])
        sx = float(max(sizes[idx, 0], 0.1))
        sy = float(max(sizes[idx, 1], 0.1))
        rot = float(rotations[idx])
        size_key = float(max(sx, sy))

        if size_key >= q2:
            color = "#1f77b4"
        elif size_key >= q1:
            color = "#2ca02c"
        else:
            color = "#d62728"

        if int(shape_types[idx]) == SHAPE_ELLIPSE:
            patch = mpatches.Ellipse(
                (cx, cy),
                width=2.0 * sx,
                height=2.0 * sy,
                angle=float(np.degrees(rot)),
                fill=False,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.85,
            )
        else:
            patch = mpatches.Polygon(
                _quad_outline_vertices(cx, cy, sx, sy, rot),
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.85,
            )

        ax.add_patch(patch)


def _draw_shape_scatter_overlay(
    ax,
    *,
    centers: np.ndarray,
    sizes: np.ndarray,
    shape_types: np.ndarray,
) -> None:
    if centers.size == 0:
        return

    major = np.maximum(sizes[:, 0], sizes[:, 1]).astype(np.float32, copy=False)
    marker_sizes = np.clip(major * 1.8, 6.0, 45.0)

    color_map = np.empty((centers.shape[0], 4), dtype=np.float32)
    color_map[:] = np.array([0.30, 0.30, 0.30, 0.80], dtype=np.float32)
    color_map[shape_types == SHAPE_ELLIPSE] = np.array([0.15, 0.47, 0.85, 0.80], dtype=np.float32)
    color_map[shape_types == SHAPE_TRIANGLE] = np.array([0.80, 0.18, 0.22, 0.80], dtype=np.float32)
    color_map[shape_types == SHAPE_QUAD] = np.array([0.18, 0.62, 0.24, 0.80], dtype=np.float32)

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=marker_sizes,
        c=color_map,
        marker="o",
        linewidths=0.25,
        edgecolors="white",
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
    stage_checkpoint_callback=None,
) -> Phase7ExecutionResult:
    target = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)
    h, w = target.shape[:2]
    blank = np.ones_like(target, dtype=np.float32)

    shared = _SharedViewState(
        target=np.array(target, copy=True),
        canvas=np.array(blank, copy=True),
        signed_residual=_signed_residual(target, blank),
        abs_residual=_abs_residual(target, blank),
        loss_history=[_rgb_mse(target, blank)],
        resolution_markers=[0],
        batch_markers=[],
        stage_markers=[],
        polygon_count=0,
        iteration=0,
        stage_name="init",
        stage_iteration=0,
        stage_start_loss=_rgb_mse(target, blank),
        stage_position_updates=0,
        running=True,
        status_line="initializing",
        polygon_sizes=np.zeros((0, 2), dtype=np.float32),
        polygon_centers=np.zeros((0, 2), dtype=np.float32),
        polygon_shape_types=np.zeros((0,), dtype=np.int32),
        polygon_rotations=np.zeros((0,), dtype=np.float32),
    )

    seg_overlay = _segmentation_overlay_rgba(segmentation_map, width=w, height=h)

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
        stage_iteration: int,
        stage_start_loss: float,
        stage_position_updates: int,
        running: bool,
        status: str,
        stage_markers: list[tuple[str, int]],
    ) -> None:
        sizes = np.array(polygons.sizes, copy=True).astype(np.float32, copy=False)
        with lock:
            shared.canvas = np.array(canvas, copy=True)
            shared.signed_residual = _signed_residual(shared.target, shared.canvas)
            shared.abs_residual = _abs_residual(shared.target, shared.canvas)
            shared.loss_history = list(losses)
            shared.resolution_markers = list(resolution_markers)
            shared.batch_markers = list(batch_markers)
            shared.stage_markers = list(stage_markers)
            shared.polygon_count = int(polygons.count)
            shared.iteration = int(iteration)
            shared.stage_name = str(stage_name)
            shared.stage_iteration = int(stage_iteration)
            shared.stage_start_loss = float(stage_start_loss)
            shared.stage_position_updates = int(stage_position_updates)
            shared.running = bool(running)
            shared.status_line = str(status)
            shared.polygon_sizes = sizes
            shared.polygon_centers = np.array(polygons.centers, copy=True)
            shared.polygon_shape_types = np.array(polygons.shape_types, copy=True)
            shared.polygon_rotations = np.array(polygons.rotations, copy=True)

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

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.08], hspace=0.26, wspace=0.22)

    ax_target = fig.add_subplot(gs[0, 0])
    ax_focus = fig.add_subplot(gs[0, 1])
    ax_residual = fig.add_subplot(gs[0, 2])
    ax_poly = fig.add_subplot(gs[1, 0])
    ax_curve = fig.add_subplot(gs[1, 1:3])

    ax_target.imshow(shared.target)
    ax_target.set_title("Panel 1 - Target")
    ax_target.set_xticks([])
    ax_target.set_yticks([])

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
            abs_res = np.array(shared.abs_residual, copy=True)
            losses = np.array(shared.loss_history, dtype=np.float64)
            stage_markers = list(shared.stage_markers)
            resolution_markers = list(shared.resolution_markers)
            batch_markers = list(shared.batch_markers)
            stage_name = str(shared.stage_name)
            iteration = int(shared.iteration)
            stage_iteration = int(shared.stage_iteration)
            stage_start_loss = float(shared.stage_start_loss)
            stage_position_updates = int(shared.stage_position_updates)
            polygon_count = int(shared.polygon_count)
            running = bool(shared.running)
            status = str(shared.status_line)
            paused = bool(controls.paused)
            centers = np.array(shared.polygon_centers, copy=True)
            sizes = np.array(shared.polygon_sizes, copy=True)
            rotations = np.array(shared.polygon_rotations, copy=True)
            shape_types = np.array(shared.polygon_shape_types, copy=True)
            view_mode = int(controls.view_mode_index)
            residual_mode = int(controls.residual_mode_index)
            softness = float(controls.latest_softness)
            softness_scale = float(controls.softness_scale)
            show_seg = bool(controls.show_segmentation_overlay)
            show_scatter = bool(controls.show_scatter_overlay)

        signed_scalar = np.mean(signed, axis=2, dtype=np.float32)
        signed_display = np.clip(0.5 * signed_scalar + 0.5, 0.0, 1.0)

        ax_focus.clear()
        ax_focus.set_xticks([])
        ax_focus.set_yticks([])
        if view_mode == 0:
            ax_focus.imshow(canvas)
            ax_focus.set_title("Panel 2 - Current Reconstruction")
            if show_seg and seg_overlay is not None:
                ax_focus.imshow(seg_overlay)
            if show_scatter:
                _draw_shape_scatter_overlay(
                    ax_focus,
                    centers=centers,
                    sizes=sizes,
                    shape_types=shape_types,
                )
        elif view_mode == 1:
            ax_focus.imshow(signed_display, cmap="RdBu_r", vmin=0.0, vmax=1.0)
            ax_focus.set_title("Panel 2 - Focus View: Signed Residual")
        else:
            ax_focus.set_title("Panel 2 - Focus View: Polygon Outlines")
            _draw_outline_panel(
                ax_focus,
                centers=centers,
                sizes=sizes,
                rotations=rotations,
                shape_types=shape_types,
                width=w,
                height=h,
            )

        ax_residual.clear()
        ax_residual.set_xticks([])
        ax_residual.set_yticks([])
        if residual_mode == 0:
            ax_residual.imshow(signed_display, cmap="RdBu_r", vmin=0.0, vmax=1.0)
            ax_residual.set_title("Panel 3 - Signed Residual")
        elif residual_mode == 1:
            ax_residual.imshow(abs_res, cmap="magma", vmin=0.0, vmax=1.0)
            ax_residual.set_title("Panel 3 - Absolute Residual")
        else:
            ax_residual.imshow(
                np.clip(signed * signed, 0.0, 1.0), cmap="viridis", vmin=0.0, vmax=1.0
            )
            ax_residual.set_title("Panel 3 - Squared Residual")

        ax_poly.clear()
        ax_poly.set_title(
            "Panel 4 - Polygon Outlines (Blue/Large, Green/Medium, Red/Small)"
        )
        _draw_outline_panel(
            ax_poly,
            centers=centers,
            sizes=sizes,
            rotations=rotations,
            shape_types=shape_types,
            width=w,
            height=h,
        )
        if show_scatter:
            _draw_shape_scatter_overlay(
                ax_poly,
                centers=centers,
                sizes=sizes,
                shape_types=shape_types,
            )

        ax_curve.clear()
        ax_curve.set_title("Panel 5 - Log MSE vs Iteration")
        ax_curve.set_xlabel("Iteration")
        ax_curve.set_ylabel("MSE (log)")
        ax_curve.set_yscale("log")
        ax_curve.grid(True, alpha=0.25)

        if losses.size > 0:
            x = np.arange(losses.size)
            y = np.maximum(losses, 1e-9)
            ax_curve.plot(x, y, color="tab:blue", linewidth=1.8, label="MSE")
            ax_curve.set_xlim(0, max(10, int(x[-1]) + 4))
            y_min = max(1e-9, float(np.min(y) * 0.9))
            y_max = max(float(np.max(y) * 1.1), y_min * 10.0)
            ax_curve.set_ylim(y_min, y_max)

            for idx in resolution_markers:
                ax_curve.axvline(
                    int(idx),
                    linestyle="-.",
                    color="black",
                    alpha=0.65,
                    linewidth=1.0,
                )

            for idx in batch_markers:
                ax_curve.axvline(
                    int(idx),
                    linestyle=":",
                    color="tab:orange",
                    alpha=0.75,
                    linewidth=1.0,
                )

            for name, idx in stage_markers:
                ax_curve.axvline(
                    int(idx),
                    linestyle="--",
                    color="gray",
                    alpha=0.40,
                    linewidth=1.0,
                )
                ax_curve.text(
                    int(idx),
                    y_max * 0.90,
                    name,
                    fontsize=8,
                    ha="center",
                    va="top",
                    color="gray",
                )

            ax_curve.legend(loc="upper right")

        curr_loss = float(losses[-1]) if losses.size else float("nan")
        stage_delta = max(stage_start_loss - curr_loss, 0.0) if losses.size else 0.0
        stage_pct = (
            100.0 * stage_delta / max(abs(stage_start_loss), 1e-9)
            if losses.size
            else 0.0
        )
        e_count = int(np.count_nonzero(shape_types == SHAPE_ELLIPSE))
        t_count = int(np.count_nonzero(shape_types == SHAPE_TRIANGLE))
        q_count = int(np.count_nonzero(shape_types == SHAPE_QUAD))

        stat_lines = [
            "STAGE PROGRESS",
            f"stage: {stage_name} | iter in stage: {stage_iteration}",
            (
                f"stage loss: {stage_start_loss:.6f} -> {curr_loss:.6f} "
                f"({stage_pct:.2f}% better)"
                if losses.size
                else "stage loss: n/a"
            ),
            f"shape counts: ellipse={e_count} triangle={t_count} quad={q_count}",
            f"position updates (stage): {stage_position_updates}",
            f"total iter: {iteration} | polygons: {polygon_count}",
            f"paused: {paused} | softness: {softness:.4f} (x{softness_scale:.2f})",
            f"state: {status}",
            "keys: P/S/E/R/Q | 1/2/3/V | G/D | +/- | X(scatter)",
        ]
        ax_curve.text(
            0.01,
            0.98,
            "\n".join(stat_lines),
            transform=ax_curve.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none"},
        )

        if controls.quit_requested and not running:
            anim.event_source.stop()

        fig.canvas.draw_idle()
        return ()

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
    worker.join(timeout=8.0)

    if "result" in result_holder:
        return result_holder["result"]

    with lock:
        canvas = np.array(shared.canvas, copy=True)
        losses = list(shared.loss_history)
        stage_marks = list(shared.stage_markers)
        res_marks = list(shared.resolution_markers)
        batch_marks = list(shared.batch_markers)
        poly_count = int(shared.polygon_count)

    return Phase7ExecutionResult(
        final_canvas=canvas,
        final_loss=float(losses[-1]) if losses else _rgb_mse(target, canvas),
        polygon_count=poly_count,
        iterations=len(losses),
        loss_history=losses,
        resolution_markers=res_marks,
        batch_markers=batch_marks,
        stage_markers=stage_marks,
    )
