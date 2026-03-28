from __future__ import annotations

import numpy as np

from src.population import PopulationHillClimber
from src.polygon import Polygon, ShapeType
from src.preprocessing import preprocess_target_array
from src.renderer import render_polygons


def _make_quad(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[float, float, float],
) -> Polygon:
    return Polygon(
        vertices=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        color=color,
        alpha=0.80,
        shape_type=ShapeType.QUADRILATERAL,
        orientation=0.0,
    )


def test_population_diversity_beats_primary_somewhere() -> None:
    h = w = 120
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    target = np.zeros((h, w, 3), dtype=np.float32)
    target[..., 0] = xx
    target[..., 1] = 0.5 * yy + 0.25 * np.sin(xx * 8.0)
    target[..., 2] = np.where((np.floor(xx * 12) + np.floor(yy * 12)) % 2 == 0, 0.85, 0.15)
    target = np.clip(target, 0.0, 1.0)

    prep = preprocess_target_array(target, random_seed=11, base_resolution=120)

    population = PopulationHillClimber(
        target_image=prep.target_rgb,
        max_iterations=1000,
        target_pyramid=prep.pyramid,
        structure_map=prep.structure_map,
        gradient_angle_map=prep.gradient_angle_map,
        segmentation_map=prep.segmentation_map,
        cluster_centroids_lab=prep.cluster_centroids_lab,
        cluster_variances_lab=prep.cluster_variances_lab,
        size_schedule=prep.recommended_size_schedule,
        random_seed=11,
        recombination_interval=250,
    )

    population.start()
    population.wait_until_complete(timeout=120.0)
    population.stop()

    assert population.non_primary_better_seen


def test_recombination_canvas_beats_both_parents() -> None:
    h = w = 80
    target = np.zeros((h, w, 3), dtype=np.float32)
    target[:, : w // 2, 0] = 1.0
    target[:, w // 2 :, 2] = 1.0

    prep = preprocess_target_array(target, random_seed=5, base_resolution=80)

    population = PopulationHillClimber(
        target_image=prep.target_rgb,
        max_iterations=10,
        target_pyramid=prep.pyramid,
        structure_map=prep.structure_map,
        gradient_angle_map=prep.gradient_angle_map,
        segmentation_map=prep.segmentation_map,
        cluster_centroids_lab=prep.cluster_centroids_lab,
        cluster_variances_lab=prep.cluster_variances_lab,
        size_schedule=prep.recommended_size_schedule,
        random_seed=5,
        recombination_interval=5,
    )

    left_good = _make_quad(0, 0, 39, 79, (1.0, 0.0, 0.0))
    right_good = _make_quad(40, 0, 79, 79, (0.0, 0.0, 1.0))
    left_bad = _make_quad(0, 0, 39, 79, (0.0, 1.0, 0.0))
    right_bad = _make_quad(40, 0, 79, 79, (0.0, 1.0, 0.0))

    parent_a = [left_good, right_bad]
    parent_b = [left_bad, right_good]

    opt_a = population.optimizers[0]
    canvas_a = render_polygons(opt_a.blank_canvas, parent_a)
    mse_a = opt_a.evaluate_canvas_loss(canvas_a)
    opt_a.adopt_solution(parent_a, canvas_a, mse_a)

    opt_b = population.optimizers[1]
    canvas_b = render_polygons(opt_b.blank_canvas, parent_b)
    mse_b = opt_b.evaluate_canvas_loss(canvas_b)
    opt_b.adopt_solution(parent_b, canvas_b, mse_b)

    recombined = population.run_recombination_step()
    primary_after = population.optimizers[0].current_mse

    assert recombined
    assert primary_after <= mse_a + 1e-9
    assert primary_after <= mse_b + 1e-9
