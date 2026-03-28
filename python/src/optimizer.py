from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
from skimage.transform import resize

from src.canvas import create_white_canvas
from src.mse import mean_squared_error, per_pixel_error_map, process_error_map
from src.polygon import Polygon, ShapeType, generate_shape
from src.renderer import render_polygon


PHASE_COARSE_END = 0.30
PHASE_STRUCTURAL_END = 0.70

SHAPE_CYCLE: tuple[ShapeType, ShapeType, ShapeType] = (
    ShapeType.TRIANGLE,
    ShapeType.QUADRILATERAL,
    ShapeType.ELLIPSE,
)


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
        self.height, self.width, _ = self.target.shape
        self.canvas = create_white_canvas(width=self.width, height=self.height)

        self.target_pyramid = self._prepare_target_pyramid(target_pyramid)
        self.structure_map = self._prepare_structure_map(structure_map)
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

    def _prepare_structure_map(self, structure_map: np.ndarray | None) -> np.ndarray | None:
        if structure_map is None:
            return None
        if structure_map.ndim != 2:
            raise ValueError("structure_map must have shape (H, W).")

        if structure_map.shape != self.target.shape[:2]:
            resized = resize(
                structure_map.astype(np.float32, copy=False),
                self.target.shape[:2],
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            )
            structure = resized.astype(np.float32, copy=False)
        else:
            structure = structure_map.astype(np.float32, copy=False)

        return np.clip(structure, 0.0, 1.0).astype(np.float32, copy=False)

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
            return 0.12
        if phase == "Structural":
            return 0.65
        return 0.28

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

    def _compute_level_mse(self, canvas: np.ndarray, target_level: np.ndarray) -> float:
        target_h, target_w, _ = target_level.shape
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

        return mean_squared_error(canvas_level, target_level)

    def _evaluate_multiscale_loss(self, canvas: np.ndarray, iteration: int) -> float:
        level_mses = [
            self._compute_level_mse(canvas, level) for level in self.target_pyramid
        ]
        fine_mse = float(np.mean([level_mses[i] for i in self._fine_indices]))
        coarse_mse = float(np.mean([level_mses[i] for i in self._coarse_indices]))

        coarse_weight, fine_weight = self._loss_weights(iteration)
        return coarse_weight * coarse_mse + fine_weight * fine_mse

    def _compute_guided_error_map(self, canvas: np.ndarray, iteration: int) -> np.ndarray:
        raw_error = per_pixel_error_map(canvas, self.target)
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

    def step(self) -> bool:
        """Run one optimization step; returns True when candidate is accepted."""
        if self.is_done:
            return False

        shape_type = SHAPE_CYCLE[self.iteration % len(SHAPE_CYCLE)]
        size_px = get_current_size(
            self.iteration,
            self.max_iterations,
            size_schedule=self.size_schedule,
        )

        _, sampled_row, sampled_col = self._sample_center_from_error_distribution()
        center_xy = (sampled_col, sampled_row)

        candidate = generate_shape(
            shape_type=shape_type,
            probability_map=self.current_error_map,
            target_image=self.target,
            size_px=size_px,
            center_xy=center_xy,
            rng=self.rng,
        )
        candidate_canvas = render_polygon(self.canvas, candidate)
        candidate_mse = self._evaluate_multiscale_loss(
            candidate_canvas,
            iteration=self.iteration,
        )

        accepted = candidate_mse < self.current_mse
        if accepted:
            self.canvas = candidate_canvas
            self.current_mse = candidate_mse
            self.accepted_polygons.append(candidate)

        self.iteration += 1
        self.acceptance_window.append(accepted)
        self.acceptance_history.append(accepted)
        self.mse_history.append(self.current_mse)
        self.current_error_map = self._compute_guided_error_map(
            self.canvas,
            self.iteration,
        )

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
