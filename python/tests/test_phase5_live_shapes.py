import numpy as np

from src.live_optimizer import LiveJointOptimizer, LiveOptimizerConfig
from src.live_renderer import (
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
    LivePolygonBatch,
    SoftRasterizer,
)


def _circle_target(size: int = 64) -> np.ndarray:
    target = np.ones((size, size, 3), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cx = cy = (size - 1) / 2.0
    radius = size * 0.22
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius * radius
    target[mask] = np.array([0.2, 0.4, 0.9], dtype=np.float32)
    return target


def _line_target(size: int = 64) -> np.ndarray:
    target = np.ones((size, size, 3), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dist = np.abs((yy - xx) / np.sqrt(2.0))
    mask = dist <= 1.3
    target[mask] = np.array([0.1, 0.2, 0.1], dtype=np.float32)
    return target


def _run_single_shape(
    target: np.ndarray,
    *,
    shape_type: int,
    shape_params: np.ndarray | None = None,
    size_xy: tuple[float, float] = (14.0, 14.0),
    steps: int = 70,
) -> float:
    h, w = target.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    batch = LivePolygonBatch(
        centers=np.array([[cx, cy]], dtype=np.float32),
        sizes=np.array([[size_xy[0], size_xy[1]]], dtype=np.float32),
        rotations=np.array([0.0], dtype=np.float32),
        colors=np.array([[0.8, 0.1, 0.1]], dtype=np.float32),
        alphas=np.array([0.85], dtype=np.float32),
        shape_types=np.array([shape_type], dtype=np.int32),
        shape_params=(
            np.zeros((1, 6), dtype=np.float32)
            if shape_params is None
            else shape_params.reshape(1, 6).astype(np.float32)
        ),
    )

    optimizer = LiveJointOptimizer(
        target_image=target,
        rasterizer=SoftRasterizer(height=h, width=w),
        polygons=batch,
        config=LiveOptimizerConfig(
            color_lr=0.04,
            position_lr=0.02,
            size_lr=0.003,
            alpha_lr=0.015,
            render_chunk_size=50,
            exact_fd=True,
            max_fd_polygons=1,
        ),
    )
    optimizer.run(steps, start_softness=2.0, end_softness=0.6)
    return float(optimizer.loss_history[-1])


def test_phase5_circle_prefers_ellipse_over_triangle() -> None:
    target = _circle_target(64)
    ellipse_loss = _run_single_shape(
        target, shape_type=SHAPE_ELLIPSE, size_xy=(14.0, 14.0)
    )
    triangle_loss = _run_single_shape(
        target, shape_type=SHAPE_TRIANGLE, size_xy=(14.0, 14.0)
    )
    assert ellipse_loss < triangle_loss


def test_phase5_thin_line_prefers_stroke_over_quadrilateral() -> None:
    target = _line_target(64)

    stroke_params = np.array([60.0, 60.0, 3.2, 0.0, 0.0, 0.0], dtype=np.float32)
    stroke_loss = _run_single_shape(
        target,
        shape_type=SHAPE_THIN_STROKE,
        shape_params=stroke_params,
        size_xy=(8.0, 8.0),
        steps=80,
    )

    quad_loss = _run_single_shape(
        target,
        shape_type=SHAPE_QUAD,
        size_xy=(22.0, 3.0),
        steps=80,
    )

    assert stroke_loss < quad_loss
