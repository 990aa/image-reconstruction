import numpy as np

from src.live_optimizer import LiveJointOptimizer, LiveOptimizerConfig
from src.live_renderer import LivePolygonBatch, SHAPE_QUAD, SoftRasterizer


def _make_rect_target(height: int, width: int) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
    target = np.ones((height, width, 3), dtype=np.float32)
    color = np.array([0.20, 0.70, 0.40], dtype=np.float32)

    y0, y1 = 14, 36
    x0, x1 = 17, 39
    target[y0:y1, x0:x1] = color

    center = ((x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0)
    return target, center, color


def test_joint_optimizer_color_and_position_converge_on_simple_rectangle() -> None:
    height, width = 50, 50
    target, rect_center, rect_color = _make_rect_target(height, width)

    polygons = LivePolygonBatch(
        centers=np.array([[22.0, 21.0]], dtype=np.float32),
        sizes=np.array([[10.0, 10.0]], dtype=np.float32),
        rotations=np.array([0.0], dtype=np.float32),
        colors=np.array([[0.90, 0.10, 0.10]], dtype=np.float32),
        alphas=np.array([0.85], dtype=np.float32),
        shape_types=np.array([SHAPE_QUAD], dtype=np.int32),
    )

    rasterizer = SoftRasterizer(height=height, width=width)
    optimizer = LiveJointOptimizer(
        target_image=target,
        rasterizer=rasterizer,
        polygons=polygons,
        config=LiveOptimizerConfig(
            color_lr=0.03,
            position_lr=0.03,
            size_lr=0.002,
            alpha_lr=0.0,
            position_eps_px=2.0,
            size_eps_ratio=0.10,
        ),
    )

    initial_center = np.array(polygons.centers[0], copy=True)
    initial_dist = float(np.linalg.norm(initial_center - np.array(rect_center)))

    losses = optimizer.run(100, start_softness=2.0, end_softness=0.5)

    final_color = optimizer.polygons.colors[0]
    final_center = optimizer.polygons.centers[0]
    final_dist = float(np.linalg.norm(final_center - np.array(rect_center)))

    assert np.all(np.abs(final_color - rect_color) <= 0.05)
    assert final_dist < initial_dist

    increases = 0
    for prev, curr in zip(optimizer.loss_history[:-1], optimizer.loss_history[1:], strict=True):
        if curr > prev + 1e-7:
            increases += 1

    assert increases <= 3
    assert losses[-1] < losses[0]
