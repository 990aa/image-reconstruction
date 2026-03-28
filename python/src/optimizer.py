from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Sequence

import numpy as np
from skimage import color
from skimage.transform import resize

from src.canvas import create_white_canvas
from src.mse import (
    per_pixel_perceptual_error_map,
    process_error_map,
    rgb_to_lab_image,
)
from src.polygon import Polygon, ShapeType, generate_shape, polygon_center
from src.renderer import polygon_mask, render_polygon, render_polygons


PHASE_COARSE_END = 0.30
PHASE_STRUCTURAL_END = 0.70

SHAPE_CYCLE: tuple[ShapeType, ShapeType, ShapeType] = (
    ShapeType.TRIANGLE,
    ShapeType.QUADRILATERAL,
    ShapeType.ELLIPSE,
)


ALPHA_CANDIDATES: tuple[float, float, float] = (0.15, 0.40, 0.70)


def phase_transition_iterations(max_iterations: int) -> tuple[int, int]:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    return int(max_iterations * PHASE_COARSE_END), int(
        max_iterations * PHASE_STRUCTURAL_END
    )


def get_phase_name(iteration: int, max_iterations: int) -> str:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    progress = float(np.clip(iteration / max_iterations, 0.0, 1.0))
    if progress < PHASE_COARSE_END:
        return "Coarse"
    if progress < PHASE_STRUCTURAL_END:
        return "Structural"
    return "Detail"


def _linear(start: float, end: float, t: float) -> float:
    return start + (end - start) * float(np.clip(t, 0.0, 1.0))


def _default_size_schedule() -> dict[str, float]:
    return {
        "coarse_start": 30.0,
        "coarse_end": 15.0,
        "structural_end": 8.0,
        "detail_end": 3.0,
    }


def _validate_size_schedule(size_schedule: dict[str, float]) -> dict[str, float]:
    required = ("coarse_start", "coarse_end", "structural_end", "detail_end")
    missing = [key for key in required if key not in size_schedule]
    if missing:
        raise ValueError(f"size_schedule missing keys: {missing}")

    normalized = {k: float(size_schedule[k]) for k in required}
    if any(value <= 0.0 for value in normalized.values()):
        raise ValueError("All size schedule values must be positive.")

    c0 = normalized["coarse_start"]
    c1 = min(normalized["coarse_end"], c0)
    c2 = min(normalized["structural_end"], c1)
    c3 = min(normalized["detail_end"], c2)

    normalized["coarse_end"] = c1
    normalized["structural_end"] = c2
    normalized["detail_end"] = c3
    return normalized


def get_current_size(
    iteration: int,
    max_iterations: int,
    size_schedule: dict[str, float] | None = None,
) -> float:
    """Compute scheduled shape size for the current optimization phase."""
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    schedule = _default_size_schedule() if size_schedule is None else size_schedule
    schedule = _validate_size_schedule(schedule)

    progress = float(np.clip(iteration / max_iterations, 0.0, 1.0))
    if progress < PHASE_COARSE_END:
        local_t = progress / PHASE_COARSE_END
        return _linear(schedule["coarse_start"], schedule["coarse_end"], local_t)

    if progress < PHASE_STRUCTURAL_END:
        local_t = (progress - PHASE_COARSE_END) / (
            PHASE_STRUCTURAL_END - PHASE_COARSE_END
        )
        return _linear(schedule["coarse_end"], schedule["structural_end"], local_t)

    local_t = (progress - PHASE_STRUCTURAL_END) / (1.0 - PHASE_STRUCTURAL_END)
    return _linear(schedule["structural_end"], schedule["detail_end"], local_t)


def _wrap_axis_angle(angle: float) -> float:
    return float(np.mod(angle, np.pi))


def _axis_angle_delta(a: float, b: float) -> float:
    diff = abs(_wrap_axis_angle(a) - _wrap_axis_angle(b))
    return min(diff, np.pi - diff)


def select_shape_type(
    row: int,
    col: int,
    structure_map: np.ndarray | None,
    phase: str,
) -> ShapeType:
    if structure_map is None:
        if phase == "Coarse":
            return ShapeType.QUADRILATERAL
        if phase == "Structural":
            return ShapeType.TRIANGLE
        return ShapeType.ELLIPSE

    mag = float(structure_map[row, col])
    if mag >= 0.60:
        if phase == "Coarse":
            return ShapeType.QUADRILATERAL
        return ShapeType.TRIANGLE

    if mag <= 0.22:
        if phase == "Detail":
            return ShapeType.QUADRILATERAL
        return ShapeType.ELLIPSE

    return ShapeType.TRIANGLE


def _region_profile(magnitude: float) -> str:
    if magnitude >= 0.60:
        return "edge"
    if magnitude <= 0.22:
        return "flat"
    return "texture"


class HillClimbingOptimizer:
    """Main optimization loop with guided sampling and accept/reject updates."""

    def __init__(
        self,
        target_image: np.ndarray,
        max_iterations: int = 5000,
        snapshot_interval: int = 100,
        error_sigma: float = 3.0,
        random_seed: int | None = None,
        target_pyramid: Sequence[np.ndarray] | None = None,
        structure_map: np.ndarray | None = None,
        gradient_angle_map: np.ndarray | None = None,
        segmentation_map: np.ndarray | None = None,
        cluster_centroids_lab: np.ndarray | None = None,
        cluster_variances_lab: np.ndarray | None = None,
        size_schedule: dict[str, float] | None = None,
        max_polygons: int | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive.")
        if target_image.ndim != 3 or target_image.shape[2] != 3:
            raise ValueError("target_image must have shape (H, W, 3).")
        if max_polygons is not None and max_polygons <= 0:
            raise ValueError("max_polygons must be positive when provided.")

        self.max_iterations = max_iterations
        self.snapshot_interval = snapshot_interval
        self.error_sigma = error_sigma
        self.max_polygons = max_polygons

        self.rng = np.random.default_rng(random_seed)
        if random_seed is not None:
            np.random.seed(random_seed)

        self.target = target_image.astype(np.float32, copy=False)
        self.target_lab = rgb_to_lab_image(self.target)
        self.height, self.width, _ = self.target.shape
        self.blank_canvas = create_white_canvas(width=self.width, height=self.height)
        self.canvas = np.array(self.blank_canvas, copy=True)

        self.target_pyramid = self._prepare_target_pyramid(target_pyramid)
        self.target_pyramid_lab = [
            rgb_to_lab_image(level) for level in self.target_pyramid
        ]

        self.structure_map = self._prepare_structure_map(structure_map)
        self.gradient_angle_map = self._prepare_angle_map(gradient_angle_map)
        self.segmentation_map = self._prepare_segmentation_map(segmentation_map)
        self.cluster_centroids_lab = self._prepare_cluster_centroids(
            cluster_centroids_lab
        )
        self.cluster_variances_lab = self._prepare_cluster_variances(
            cluster_variances_lab
        )

        self.size_schedule = _validate_size_schedule(
            _default_size_schedule() if size_schedule is None else size_schedule
        )

        self._fine_indices, self._coarse_indices = self._split_pyramid_groups(
            len(self.target_pyramid)
        )

        self.current_mse = self._evaluate_multiscale_loss(self.canvas, iteration=0)
        self.initial_mse = self.current_mse
        self.current_error_map = self._compute_guided_error_map(self.canvas, 0)

        self.iteration = 0
        self.mse_history: list[float] = [self.current_mse]
        self.accepted_polygons: list[Polygon] = []
        self.snapshots: list[np.ndarray] = []

        self.acceptance_window: deque[bool] = deque(maxlen=100)
        self.acceptance_history: list[bool] = []

    @property
    def acceptance_rate(self) -> float:
        if not self.acceptance_window:
            return 0.0
        return float(np.mean(self.acceptance_window, dtype=np.float32))

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_polygons)

    @property
    def current_phase(self) -> str:
        return get_phase_name(self.iteration, self.max_iterations)

    @property
    def is_done(self) -> bool:
        if self.iteration >= self.max_iterations:
            return True
        if self.max_polygons is not None and self.accepted_count >= self.max_polygons:
            return True
        return False

    def _prepare_target_pyramid(
        self, target_pyramid: Sequence[np.ndarray] | None
    ) -> list[np.ndarray]:
        if target_pyramid is None or len(target_pyramid) == 0:
            return [self.target]

        pyramid: list[np.ndarray] = []
        for i, level in enumerate(target_pyramid):
            if level.ndim != 3 or level.shape[2] != 3:
                raise ValueError("Each pyramid level must have shape (H, W, 3).")

            level32 = level.astype(np.float32, copy=False)
            if i == 0 and level32.shape[:2] != self.target.shape[:2]:
                pyramid.append(self.target)
            else:
                pyramid.append(level32)

        if pyramid[0].shape[:2] != self.target.shape[:2]:
            pyramid[0] = self.target

        return pyramid

    def _resize_2d(
        self,
        data: np.ndarray,
        order: int,
        clip: bool = False,
    ) -> np.ndarray:
        resized = resize(
            data.astype(np.float32, copy=False),
            self.target.shape[:2],
            order=order,
            mode="reflect",
            anti_aliasing=(order > 0),
            preserve_range=True,
        ).astype(np.float32, copy=False)
        if clip:
            resized = np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)
        return resized

    def _prepare_structure_map(
        self, structure_map: np.ndarray | None
    ) -> np.ndarray | None:
        if structure_map is None:
            return None
        if structure_map.ndim != 2:
            raise ValueError("structure_map must have shape (H, W).")
        if structure_map.shape != self.target.shape[:2]:
            return self._resize_2d(structure_map, order=1, clip=True)
        return np.clip(structure_map, 0.0, 1.0).astype(np.float32, copy=False)

    def _prepare_angle_map(self, angle_map: np.ndarray | None) -> np.ndarray | None:
        if angle_map is None:
            return None
        if angle_map.ndim != 2:
            raise ValueError("gradient_angle_map must have shape (H, W).")
        if angle_map.shape != self.target.shape[:2]:
            return self._resize_2d(angle_map, order=1, clip=False)
        return angle_map.astype(np.float32, copy=False)

    def _prepare_segmentation_map(
        self, segmentation_map: np.ndarray | None
    ) -> np.ndarray | None:
        if segmentation_map is None:
            return None
        if segmentation_map.ndim != 2:
            raise ValueError("segmentation_map must have shape (H, W).")
        if segmentation_map.shape != self.target.shape[:2]:
            resized = self._resize_2d(segmentation_map, order=0, clip=False)
            return np.round(resized).astype(np.int32, copy=False)
        return segmentation_map.astype(np.int32, copy=False)

    def _prepare_cluster_centroids(
        self,
        cluster_centroids_lab: np.ndarray | None,
    ) -> np.ndarray | None:
        if cluster_centroids_lab is None:
            return None
        if cluster_centroids_lab.ndim != 2 or cluster_centroids_lab.shape[1] != 3:
            raise ValueError("cluster_centroids_lab must have shape (K, 3).")
        return cluster_centroids_lab.astype(np.float32, copy=False)

    def _prepare_cluster_variances(
        self,
        cluster_variances_lab: np.ndarray | None,
    ) -> np.ndarray | None:
        if cluster_variances_lab is None:
            return None
        if cluster_variances_lab.ndim != 1:
            raise ValueError("cluster_variances_lab must have shape (K,).")
        return np.maximum(cluster_variances_lab.astype(np.float32, copy=False), 1e-6)

    def _split_pyramid_groups(self, count: int) -> tuple[list[int], list[int]]:
        if count <= 1:
            return [0], [0]

        split = max(1, count // 2)
        fine = list(range(0, split))
        coarse = list(range(split, count))
        if not coarse:
            coarse = fine
        return fine, coarse

    def _edge_weight(self, iteration: int) -> float:
        if self.structure_map is None:
            return 0.0

        phase = get_phase_name(iteration, self.max_iterations)
        if phase == "Coarse":
            return 0.10
        if phase == "Structural":
            return 0.75
        return 0.30

    def _loss_weights(self, iteration: int) -> tuple[float, float]:
        phase = get_phase_name(iteration, self.max_iterations)
        if phase == "Coarse":
            return 0.8, 0.2
        if phase == "Detail":
            return 0.2, 0.8

        coarse_end, structural_end = phase_transition_iterations(self.max_iterations)
        span = max(1, structural_end - coarse_end)
        t = (iteration - coarse_end) / span
        coarse_weight = _linear(0.8, 0.2, t)
        return coarse_weight, 1.0 - coarse_weight

    def _compute_level_mse(
        self,
        canvas: np.ndarray,
        target_level_lab: np.ndarray,
    ) -> float:
        target_h, target_w, _ = target_level_lab.shape
        if canvas.shape[:2] == (target_h, target_w):
            canvas_level = canvas
        else:
            canvas_level = resize(
                canvas,
                (target_h, target_w, 3),
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32, copy=False)

        canvas_level_lab = rgb_to_lab_image(canvas_level)
        diff = canvas_level_lab - target_level_lab
        return float(np.mean(np.square(diff), dtype=np.float32))

    def _evaluate_multiscale_loss(self, canvas: np.ndarray, iteration: int) -> float:
        level_mses = [
            self._compute_level_mse(canvas, target_level_lab)
            for target_level_lab in self.target_pyramid_lab
        ]
        fine_mse = float(np.mean([level_mses[i] for i in self._fine_indices]))
        coarse_mse = float(np.mean([level_mses[i] for i in self._coarse_indices]))

        coarse_weight, fine_weight = self._loss_weights(iteration)
        return coarse_weight * coarse_mse + fine_weight * fine_mse

    def _compute_guided_error_map(
        self, canvas: np.ndarray, iteration: int
    ) -> np.ndarray:
        raw_error = per_pixel_perceptual_error_map(canvas, self.target)
        if self.structure_map is None:
            return raw_error

        edge_weight = self._edge_weight(iteration)
        if edge_weight <= 0.0:
            return raw_error

        guided = raw_error * (1.0 + edge_weight * self.structure_map)
        return guided.astype(np.float32, copy=False)

    def _sample_center_from_error_distribution(self) -> tuple[int, int, int]:
        processed_map = process_error_map(
            self.current_error_map, sigma=self.error_sigma
        )
        flat_error_map = processed_map.reshape(self.height * self.width)
        sampled_index = int(self.rng.choice(flat_error_map.size, p=flat_error_map))
        row, col = divmod(sampled_index, self.width)
        return sampled_index, row, col

    def _shape_profile_and_orientation(
        self, row: int, col: int
    ) -> tuple[ShapeType, str, float]:
        phase = self.current_phase
        if self.structure_map is None:
            shape_type = SHAPE_CYCLE[self.iteration % len(SHAPE_CYCLE)]
            orientation = float(self.rng.uniform(0.0, 2.0 * np.pi))
            return shape_type, "default", orientation

        magnitude = float(self.structure_map[row, col])
        shape_type = select_shape_type(row, col, self.structure_map, phase)
        profile = _region_profile(magnitude)

        if self.gradient_angle_map is not None:
            grad_angle = float(self.gradient_angle_map[row, col])
        else:
            grad_angle = float(self.rng.uniform(0.0, 2.0 * np.pi))

        if profile == "edge":
            orientation = grad_angle + (np.pi / 2.0)
        elif profile == "texture":
            orientation = grad_angle + float(self.rng.normal(0.0, np.deg2rad(18.0)))
        else:
            orientation = grad_angle + float(self.rng.normal(0.0, np.deg2rad(45.0)))

        return shape_type, profile, float(orientation)

    def _make_candidate(self, row: int, col: int, size_px: float) -> Polygon:
        shape_type, profile, orientation = self._shape_profile_and_orientation(row, col)
        center_xy = (col, row)

        return generate_shape(
            shape_type=shape_type,
            probability_map=self.current_error_map,
            target_image=self.target,
            target_lab=self.target_lab,
            segmentation_map=self.segmentation_map,
            cluster_centroids_lab=self.cluster_centroids_lab,
            cluster_variances_lab=self.cluster_variances_lab,
            size_px=size_px,
            center_xy=center_xy,
            orientation=orientation,
            profile=profile,
            alpha=0.40,
            rng=self.rng,
        )

    def select_best_alpha(
        self,
        candidate: Polygon,
        alpha_values: Sequence[float] = ALPHA_CANDIDATES,
    ) -> tuple[Polygon, np.ndarray, float]:
        best_polygon = candidate
        best_canvas = render_polygon(self.canvas, candidate)
        best_mse = self._evaluate_multiscale_loss(best_canvas, self.iteration)

        for alpha in alpha_values:
            test_polygon = replace(candidate, alpha=float(alpha))
            test_canvas = render_polygon(self.canvas, test_polygon)
            test_mse = self._evaluate_multiscale_loss(test_canvas, self.iteration)
            if test_mse < best_mse:
                best_polygon = test_polygon
                best_canvas = test_canvas
                best_mse = test_mse

        return best_polygon, best_canvas, best_mse

    def _estimate_polygon_scale(self, polygon: Polygon) -> float:
        xs = [v[0] for v in polygon.vertices]
        ys = [v[1] for v in polygon.vertices]
        span = max(float(max(xs) - min(xs)), float(max(ys) - min(ys)))
        return max(2.0, span * 0.5)

    def _split_polygon_candidate(
        self, polygon: Polygon
    ) -> tuple[Polygon, Polygon] | None:
        cx, cy = polygon_center(polygon)
        if polygon.orientation is not None:
            axis = float(polygon.orientation)
        elif polygon.ellipse_rotation:
            axis = float(polygon.ellipse_rotation)
        else:
            axis = 0.0

        parent_scale = self._estimate_polygon_scale(polygon)
        child_scale = max(1.5, parent_scale * 0.65)
        offset = max(1.0, parent_scale * 0.35)

        dx = float(np.cos(axis) * offset)
        dy = float(np.sin(axis) * offset)

        c1x = int(np.clip(round(cx + dx), 0, self.width - 1))
        c1y = int(np.clip(round(cy + dy), 0, self.height - 1))
        c2x = int(np.clip(round(cx - dx), 0, self.width - 1))
        c2y = int(np.clip(round(cy - dy), 0, self.height - 1))

        if (c1x, c1y) == (c2x, c2y):
            return None

        child1 = generate_shape(
            shape_type=polygon.shape_type,
            probability_map=self.current_error_map,
            target_image=self.target,
            target_lab=self.target_lab,
            segmentation_map=self.segmentation_map,
            cluster_centroids_lab=self.cluster_centroids_lab,
            cluster_variances_lab=self.cluster_variances_lab,
            size_px=child_scale,
            center_xy=(c1x, c1y),
            orientation=axis,
            profile="texture",
            alpha=polygon.alpha,
            rng=self.rng,
        )
        child2 = generate_shape(
            shape_type=polygon.shape_type,
            probability_map=self.current_error_map,
            target_image=self.target,
            target_lab=self.target_lab,
            segmentation_map=self.segmentation_map,
            cluster_centroids_lab=self.cluster_centroids_lab,
            cluster_variances_lab=self.cluster_variances_lab,
            size_px=child_scale,
            center_xy=(c2x, c2y),
            orientation=axis,
            profile="texture",
            alpha=polygon.alpha,
            rng=self.rng,
        )

        return child1, child2

    def run_palette_refinement_pass(self) -> float:
        if not self.accepted_polygons:
            return 0.0

        refined: list[Polygon] = []
        for polygon in self.accepted_polygons:
            mask = polygon_mask((self.height, self.width), polygon)
            if not np.any(mask):
                refined.append(polygon)
                continue

            local_lab = self.target_lab[mask].mean(axis=0, dtype=np.float32)
            local_rgb = color.lab2rgb(local_lab[np.newaxis, np.newaxis, :])[0, 0]
            local_rgb = np.clip(local_rgb.astype(np.float32, copy=False), 0.0, 1.0)

            old_rgb = np.asarray(polygon.color, dtype=np.float32)
            blended = np.clip(0.25 * old_rgb + 0.75 * local_rgb, 0.0, 1.0)
            refined.append(
                replace(
                    polygon,
                    color=(float(blended[0]), float(blended[1]), float(blended[2])),
                )
            )

        refined_canvas = render_polygons(self.blank_canvas, refined)
        refined_loss = self._evaluate_multiscale_loss(refined_canvas, self.iteration)
        improvement = self.current_mse - refined_loss

        if improvement > 0.0:
            self.accepted_polygons = refined
            self.canvas = refined_canvas
            self.current_mse = refined_loss
            self.current_error_map = self._compute_guided_error_map(
                self.canvas,
                self.iteration,
            )

        return float(max(0.0, improvement))

    def run_polygon_death_and_replacement(
        self,
        contribution_threshold: float = 1e-4,
    ) -> tuple[int, int]:
        if len(self.accepted_polygons) < 5:
            return 0, 0

        full_canvas = render_polygons(self.blank_canvas, self.accepted_polygons)
        full_loss = self._evaluate_multiscale_loss(full_canvas, self.iteration)

        contributions: list[tuple[int, float]] = []
        for idx in range(len(self.accepted_polygons)):
            without = self.accepted_polygons[:idx] + self.accepted_polygons[idx + 1 :]
            without_canvas = render_polygons(self.blank_canvas, without)
            without_loss = self._evaluate_multiscale_loss(
                without_canvas, self.iteration
            )
            delta = float(without_loss - full_loss)
            contributions.append((idx, delta))

        contributions.sort(key=lambda item: item[1])
        bottom_count = max(1, int(round(0.05 * len(self.accepted_polygons))))

        candidate_indices = [
            idx for idx, delta in contributions if delta < contribution_threshold
        ]
        if not candidate_indices:
            candidate_indices = [idx for idx, _ in contributions[:bottom_count]]
        else:
            candidate_indices = candidate_indices[:bottom_count]

        remove_set = set(candidate_indices)
        before_count = len(self.accepted_polygons)
        self.accepted_polygons = [
            polygon
            for i, polygon in enumerate(self.accepted_polygons)
            if i not in remove_set
        ]

        self.canvas = render_polygons(self.blank_canvas, self.accepted_polygons)
        self.current_mse = self._evaluate_multiscale_loss(self.canvas, self.iteration)
        self.current_error_map = self._compute_guided_error_map(
            self.canvas, self.iteration
        )

        removed = before_count - len(self.accepted_polygons)
        added = 0

        for _ in range(removed):
            if (
                self.max_polygons is not None
                and self.accepted_count >= self.max_polygons
            ):
                break

            accepted = self._attempt_single_candidate(allow_split=False)
            if accepted:
                added += 1

        return removed, added

    def _attempt_single_candidate(self, allow_split: bool = True) -> bool:
        _, sampled_row, sampled_col = self._sample_center_from_error_distribution()
        size_px = get_current_size(
            self.iteration,
            self.max_iterations,
            size_schedule=self.size_schedule,
        )

        candidate = self._make_candidate(sampled_row, sampled_col, size_px)
        candidate, parent_canvas, parent_mse = self.select_best_alpha(candidate)

        accepted = parent_mse < self.current_mse
        if not accepted:
            return False

        accepted_polys = [candidate]
        accepted_canvas = parent_canvas
        accepted_mse = parent_mse

        if allow_split and self.current_phase != "Coarse":
            split_children = self._split_polygon_candidate(candidate)
            if split_children is not None:
                split_canvas = render_polygons(
                    self.canvas, [split_children[0], split_children[1]]
                )
                split_mse = self._evaluate_multiscale_loss(split_canvas, self.iteration)
                if split_mse < accepted_mse:
                    accepted_polys = [split_children[0], split_children[1]]
                    accepted_canvas = split_canvas
                    accepted_mse = split_mse

        self.canvas = accepted_canvas
        self.current_mse = accepted_mse
        self.accepted_polygons.extend(accepted_polys)
        return True

    def step(self) -> bool:
        """Run one optimization step; returns True when candidate is accepted."""
        if self.is_done:
            return False

        accepted = self._attempt_single_candidate(allow_split=True)

        self.iteration += 1
        self.acceptance_window.append(accepted)
        self.acceptance_history.append(accepted)

        if accepted and self.accepted_count > 0 and self.accepted_count % 500 == 0:
            self.run_palette_refinement_pass()

        if self.iteration % 1000 == 0:
            self.run_polygon_death_and_replacement(contribution_threshold=1e-4)

        self.current_error_map = self._compute_guided_error_map(
            self.canvas,
            self.iteration,
        )
        self.mse_history.append(self.current_mse)

        if self.iteration % self.snapshot_interval == 0:
            self.snapshots.append(np.array(self.canvas, copy=True))

        return accepted

    def run(self, iterations: int | None = None) -> HillClimbingOptimizer:
        total_steps = self.max_iterations if iterations is None else iterations
        for _ in range(total_steps):
            if self.is_done:
                break
            self.step()
        return self


__all__ = [
    "HillClimbingOptimizer",
    "get_current_size",
    "get_phase_name",
    "phase_transition_iterations",
    "select_shape_type",
    "_axis_angle_delta",
]
