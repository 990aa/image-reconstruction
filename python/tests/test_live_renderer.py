import numpy as np

from src.live_renderer import LivePolygonBatch, SoftRasterizer, make_random_live_batch


def test_soft_triangle_coverage_behavior_and_softness() -> None:
    rasterizer = SoftRasterizer(height=80, width=80)

    vertices = np.array(
        [
            [20.0, 20.0],
            [60.0, 20.0],
            [40.0, 60.0],
        ],
        dtype=np.float32,
    )

    cov_soft = rasterizer.triangle_coverage_from_vertices(vertices, softness=2.0)
    cov_sharp = rasterizer.triangle_coverage_from_vertices(vertices, softness=0.5)

    center_value = float(cov_soft[35, 40])
    edge_value = float(cov_soft[20, 40])
    outside_value = float(cov_soft[5, 40])

    assert center_value > 0.95
    assert 0.45 <= edge_value <= 0.55
    assert outside_value < 0.05

    transition_soft = float(cov_soft[18, 40])
    transition_sharp = float(cov_sharp[18, 40])
    assert transition_soft > transition_sharp


def test_batched_soft_compositing_matches_reference() -> None:
    rng = np.random.default_rng(7)
    rasterizer = SoftRasterizer(height=48, width=48)
    polygons = make_random_live_batch(count=100, height=48, width=48, rng=rng)

    result = rasterizer.render(polygons, softness=1.25)

    reference = np.ones((48, 48, 3), dtype=np.float32)
    for idx in range(polygons.count):
        coverage = rasterizer.single_coverage(polygons, idx, softness=1.25)
        effective_alpha = coverage * float(polygons.alphas[idx])
        color = polygons.colors[idx][None, None, :]
        reference = (
            effective_alpha[:, :, None] * color
            + (1.0 - effective_alpha[:, :, None]) * reference
        )

    assert np.max(np.abs(result.canvas - reference)) < 1e-5


def test_live_batch_validation_rejects_mismatched_shapes() -> None:
    with np.testing.assert_raises(ValueError):
        LivePolygonBatch(
            centers=np.zeros((2, 2), dtype=np.float32),
            sizes=np.ones((1, 2), dtype=np.float32),
            rotations=np.zeros((2,), dtype=np.float32),
            colors=np.ones((2, 3), dtype=np.float32),
            alphas=np.ones((2,), dtype=np.float32),
            shape_types=np.zeros((2,), dtype=np.int32),
        )
