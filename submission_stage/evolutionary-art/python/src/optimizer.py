from __future__ import annotations

from collections import deque

import numpy as np

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


def get_current_size(iteration: int, max_iterations: int) -> float:
    """Compute scheduled shape size for the current optimization phase."""
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    progress = float(np.clip(iteration / max_iterations, 0.0, 1.0))
    if progress < PHASE_COARSE_END:
        local_t = progress / PHASE_COARSE_END
        return _linear(30.0, 15.0, local_t)

    if progress < PHASE_STRUCTURAL_END:
        local_t = (progress - PHASE_COARSE_END) / (
            PHASE_STRUCTURAL_END - PHASE_COARSE_END
        )
        return _linear(15.0, 8.0, local_t)

    local_t = (progress - PHASE_STRUCTURAL_END) / (1.0 - PHASE_STRUCTURAL_END)
    return _linear(8.0, 3.0, local_t)


class HillClimbingOptimizer:
    """Main optimization loop with guided sampling and accept/reject updates."""

    def __init__(
        self,
        target_image: np.ndarray,
        max_iterations: int = 5000,
        snapshot_interval: int = 100,
        error_sigma: float = 3.0,
        random_seed: int | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive.")
        if target_image.shape[:2] != (100, 100) or target_image.shape[2] != 3:
            raise ValueError(
                "target_image must have shape (100, 100, 3) for this phase."
            )

        self.max_iterations = max_iterations
        self.snapshot_interval = snapshot_interval
        self.error_sigma = error_sigma

        self.rng = np.random.default_rng(random_seed)
        if random_seed is not None:
            np.random.seed(random_seed)

        self.target = target_image.astype(np.float32, copy=False)
        self.canvas = create_white_canvas(width=100, height=100)

        self.current_mse = mean_squared_error(self.canvas, self.target)
        self.initial_mse = self.current_mse
        self.current_error_map = per_pixel_error_map(self.canvas, self.target)

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

    def _sample_center_from_error_distribution(self) -> tuple[int, int, int]:
        processed_map = process_error_map(
            self.current_error_map, sigma=self.error_sigma
        )
        flat_error_map = processed_map.reshape(10000)
        sampled_index = int(np.random.choice(10000, p=flat_error_map))
        row, col = divmod(sampled_index, 100)
        return sampled_index, row, col

    def step(self) -> bool:
        """Run one optimization step; returns True when candidate is accepted."""
        if self.iteration >= self.max_iterations:
            return False

        shape_type = SHAPE_CYCLE[self.iteration % len(SHAPE_CYCLE)]
        size_px = get_current_size(self.iteration, self.max_iterations)

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
        candidate_mse = mean_squared_error(candidate_canvas, self.target)

        accepted = candidate_mse < self.current_mse
        if accepted:
            self.canvas = candidate_canvas
            self.current_mse = candidate_mse
            self.accepted_polygons.append(candidate)
            self.current_error_map = per_pixel_error_map(self.canvas, self.target)

        self.iteration += 1
        self.acceptance_window.append(accepted)
        self.acceptance_history.append(accepted)
        self.mse_history.append(self.current_mse)

        if self.iteration % self.snapshot_interval == 0:
            self.snapshots.append(np.array(self.canvas, copy=True))

        return accepted

    def run(self, iterations: int | None = None) -> HillClimbingOptimizer:
        total_steps = self.max_iterations if iterations is None else iterations
        for _ in range(total_steps):
            if self.iteration >= self.max_iterations:
                break
            self.step()
        return self
