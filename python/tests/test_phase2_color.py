from __future__ import annotations

import numpy as np
from skimage import color

from src.optimizer import HillClimbingOptimizer
from src.polygon import Polygon, ShapeType
from src.renderer import render_polygons


def _lab_distance(
    rgb_a: tuple[float, float, float], rgb_b: tuple[float, float, float]
) -> float:
    a = np.array(rgb_a, dtype=np.float32).reshape(1, 1, 3)
    b = np.array(rgb_b, dtype=np.float32).reshape(1, 1, 3)
    lab_a = color.rgb2lab(np.clip(a, 0.0, 1.0))[0, 0]
    lab_b = color.rgb2lab(np.clip(b, 0.0, 1.0))[0, 0]
    return float(np.linalg.norm(lab_a - lab_b))


def test_palette_refinement_moves_color_closer_in_lab() -> None:
    target = np.zeros((100, 100, 3), dtype=np.float32)
    target[..., 2] = 1.0

    optimizer = HillClimbingOptimizer(
        target_image=target, max_iterations=10, random_seed=1
    )

    wrong = Polygon(
        vertices=[(34, 50), (50, 34), (66, 50), (50, 66)],
        color=(1.0, 0.0, 0.0),
        alpha=0.70,
        shape_type=ShapeType.ELLIPSE,
        ellipse_center=(50, 50),
        ellipse_axes=(16, 16),
        ellipse_rotation=0.0,
        orientation=0.0,
    )

    optimizer.accepted_polygons = [wrong]
    optimizer.canvas = render_polygons(
        optimizer.blank_canvas, optimizer.accepted_polygons
    )
    optimizer.current_mse = optimizer._evaluate_multiscale_loss(
        optimizer.canvas, optimizer.iteration
    )

    before = _lab_distance(optimizer.accepted_polygons[0].color, (0.0, 0.0, 1.0))
    improvement = optimizer.run_palette_refinement_pass()
    after = _lab_distance(optimizer.accepted_polygons[0].color, (0.0, 0.0, 1.0))

    assert improvement >= 0.0
    assert after < before


def test_adaptive_alpha_prefers_higher_alpha_for_blank_region() -> None:
    target = np.zeros((100, 100, 3), dtype=np.float32)
    target[..., 2] = 1.0

    candidate = Polygon(
        vertices=[(40, 50), (50, 40), (60, 50), (50, 60)],
        color=(0.0, 0.0, 1.0),
        alpha=0.40,
        shape_type=ShapeType.ELLIPSE,
        ellipse_center=(50, 50),
        ellipse_axes=(12, 10),
        ellipse_rotation=0.0,
        orientation=0.0,
    )

    blank_case = HillClimbingOptimizer(
        target_image=target, max_iterations=10, random_seed=2
    )
    blank_case.canvas = np.ones_like(target, dtype=np.float32)
    blank_case.current_mse = blank_case._evaluate_multiscale_loss(
        blank_case.canvas, blank_case.iteration
    )
    best_blank, _, _ = blank_case.select_best_alpha(candidate)

    close_case = HillClimbingOptimizer(
        target_image=target, max_iterations=10, random_seed=3
    )
    close_case.canvas = np.zeros_like(target, dtype=np.float32)
    close_case.canvas[..., 2] = 0.90
    close_case.current_mse = close_case._evaluate_multiscale_loss(
        close_case.canvas, close_case.iteration
    )
    best_close, _, _ = close_case.select_best_alpha(candidate)

    assert best_blank.alpha >= best_close.alpha
    assert best_blank.alpha == 0.70
    assert best_close.alpha in (0.15, 0.40)
