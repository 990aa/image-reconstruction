from __future__ import annotations

import numpy as np

from src.optimizer import HillClimbingOptimizer, _axis_angle_delta
from src.polygon import Polygon, ShapeType, polygon_center
from src.preprocessing import preprocess_target_array
from src.renderer import render_polygons


def test_edge_aware_orientations_align_near_diagonal_edge() -> None:
    h, w = 100, 100
    target = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if x >= y:
                target[y, x] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    prep = preprocess_target_array(target, polygon_override=220, random_seed=8)

    optimizer = HillClimbingOptimizer(
        target_image=prep.target_rgb,
        max_iterations=220,
        target_pyramid=prep.pyramid,
        structure_map=prep.structure_map,
        gradient_angle_map=prep.gradient_angle_map,
        segmentation_map=prep.segmentation_map,
        cluster_centroids_lab=prep.cluster_centroids_lab,
        cluster_variances_lab=prep.cluster_variances_lab,
        size_schedule=prep.recommended_size_schedule,
        max_polygons=220,
        random_seed=8,
    )

    optimizer.run(iterations=220)

    expected_edge_angle = np.pi / 4.0
    near_edge_angles: list[float] = []
    for polygon in optimizer.accepted_polygons:
        if polygon.orientation is None:
            continue
        cx, cy = polygon_center(polygon)
        if abs(cy - cx) <= 14 and float(prep.structure_map[cy, cx]) >= 0.35:
            near_edge_angles.append(float(polygon.orientation))

    assert len(near_edge_angles) >= 2

    aligned = sum(
        1
        for angle in near_edge_angles
        if _axis_angle_delta(angle, expected_edge_angle) <= np.deg2rad(30.0)
    )
    assert (aligned / len(near_edge_angles)) >= 0.5


def test_polygon_death_pass_reduces_count_with_small_mse_change() -> None:
    target = np.ones((100, 100, 3), dtype=np.float32)
    target[..., 0] = 0.20
    target[..., 1] = 0.40
    target[..., 2] = 0.80

    optimizer = HillClimbingOptimizer(
        target_image=target, max_iterations=20, random_seed=9
    )

    redundant: list[Polygon] = []
    for _ in range(18):
        redundant.append(
            Polygon(
                vertices=[(40, 50), (50, 40), (60, 50), (50, 60)],
                color=(0.20, 0.40, 0.80),
                alpha=0.50,
                shape_type=ShapeType.ELLIPSE,
                ellipse_center=(50, 50),
                ellipse_axes=(12, 10),
                ellipse_rotation=0.0,
                orientation=0.0,
            )
        )

    optimizer.accepted_polygons = list(redundant)
    optimizer.canvas = render_polygons(
        optimizer.blank_canvas, optimizer.accepted_polygons
    )
    optimizer.current_mse = optimizer._evaluate_multiscale_loss(
        optimizer.canvas, optimizer.iteration
    )

    before_count = optimizer.accepted_count
    before_mse = optimizer.current_mse

    removed, added = optimizer.run_polygon_death_and_replacement(
        contribution_threshold=1e-4
    )

    after_count = optimizer.accepted_count
    after_mse = optimizer.current_mse

    assert removed >= 1
    assert after_count < before_count
    assert after_mse <= before_mse + 0.02
    assert added <= removed
