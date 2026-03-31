from __future__ import annotations

import numpy as np

from src.core_renderer import SHAPE_QUAD, SoftRasterizer
from src.live_optimizer import SequentialHillClimber, SequentialStageConfig


def _rect_target(height: int = 48, width: int = 48) -> tuple[np.ndarray, np.ndarray]:
    target = np.ones((height, width, 3), dtype=np.float32) * 0.9
    color = np.array([0.15, 0.70, 0.30], dtype=np.float32)
    target[14:34, 16:36] = color
    return target, color


def test_analytic_color_matches_simple_full_coverage_region() -> None:
    target, color = _rect_target()
    optimizer = SequentialHillClimber(
        target_image=target,
        rasterizer=SoftRasterizer(height=48, width=48),
    )
    candidate = optimizer.random_candidate(
        stage=SequentialStageConfig(
            name="unit",
            resolution=48,
            shapes_to_add=1,
            candidate_count=1,
            mutation_steps=0,
            size_min=10.0,
            size_max=10.0,
            alpha_min=1.0,
            alpha_max=1.0,
            softness=0.05,
            allowed_shapes=(SHAPE_QUAD,),
        ),
        center_x=25.0,
        center_y=24.0,
        structure_map=np.ones((48, 48), dtype=np.float32) * 0.4,
        angle_map=np.zeros((48, 48), dtype=np.float32),
        linearity_map=np.ones((48, 48), dtype=np.float32),
        rng=np.random.default_rng(4),
    )
    candidate.center_x = 25.0
    candidate.center_y = 24.0
    candidate.size_x = 10.0
    candidate.size_y = 10.0
    candidate.rotation = 0.0
    scored = optimizer.evaluate_candidate(candidate, softness=0.05)

    assert np.allclose(scored.color, color, atol=0.08)
    assert scored.mse < optimizer.current_mse


def test_sequential_search_commits_strict_mse_improvement() -> None:
    target, _ = _rect_target()
    optimizer = SequentialHillClimber(
        target_image=target,
        rasterizer=SoftRasterizer(height=48, width=48),
    )
    stage = SequentialStageConfig(
        name="foundation",
        resolution=48,
        shapes_to_add=1,
        candidate_count=20,
        mutation_steps=32,
        size_min=6.0,
        size_max=14.0,
        alpha_min=0.6,
        alpha_max=1.0,
        softness=0.12,
        allowed_shapes=(SHAPE_QUAD,),
    )
    guide = np.mean(np.abs(target - optimizer.current_canvas), axis=2, dtype=np.float32)
    before = optimizer.current_mse
    candidate = optimizer.search_next_shape(
        stage=stage,
        guide_map=guide,
        structure_map=np.ones((48, 48), dtype=np.float32) * 0.4,
        angle_map=np.zeros((48, 48), dtype=np.float32),
        linearity_map=np.ones((48, 48), dtype=np.float32),
        rng=np.random.default_rng(5),
    )

    assert candidate is not None
    assert candidate.mse < before

    optimizer.commit_shape(candidate)
    assert optimizer.polygons.count == 1
    assert optimizer.current_mse < before
    assert optimizer.loss_history[-1] == optimizer.current_mse
