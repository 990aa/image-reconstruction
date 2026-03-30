import numpy as np
import pytest

from src.live_optimizer import LiveJointOptimizer, LiveOptimizerConfig
from src.live_renderer import (
    LivePolygonBatch,
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SoftRasterizer,
)


def _make_rect_target(
    height: int, width: int
) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
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
        alphas=np.array([1.00], dtype=np.float32),
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
            exact_fd=True,
        ),
    )

    initial_center = np.array(polygons.centers[0], copy=True)
    initial_dist = float(np.linalg.norm(initial_center - np.array(rect_center)))

    losses = optimizer.run(100, start_softness=2.0, end_softness=0.5)

    final_color = optimizer.polygons.colors[0]
    final_center = optimizer.polygons.centers[0]
    final_dist = float(np.linalg.norm(final_center - np.array(rect_center)))

    assert np.all(np.abs(final_color - rect_color) <= 0.10)
    assert final_dist < initial_dist

    increases = 0
    for prev, curr in zip(
        optimizer.loss_history[:-1], optimizer.loss_history[1:], strict=True
    ):
        if curr > prev + 1e-7:
            increases += 1

    assert increases <= 3
    assert losses[-1] < losses[0]


def test_color_gradient_respects_occlusion_transmittance() -> None:
    height = width = 48
    target = np.zeros((height, width, 3), dtype=np.float32)

    count = 5
    polygons = LivePolygonBatch(
        centers=np.tile(np.array([[24.0, 24.0]], dtype=np.float32), (count, 1)),
        sizes=np.tile(np.array([[11.0, 11.0]], dtype=np.float32), (count, 1)),
        rotations=np.zeros((count,), dtype=np.float32),
        colors=np.tile(np.array([[0.9, 0.2, 0.2]], dtype=np.float32), (count, 1)),
        alphas=np.full((count,), 0.70, dtype=np.float32),
        shape_types=np.full((count,), SHAPE_ELLIPSE, dtype=np.int32),
    )

    rasterizer = SoftRasterizer(height=height, width=width)
    optimizer = LiveJointOptimizer(
        target_image=target,
        rasterizer=rasterizer,
        polygons=polygons,
        config=LiveOptimizerConfig(
            color_lr=0.02,
            position_lr=0.0,
            size_lr=0.0,
            rotation_lr=0.0,
            alpha_lr=0.0,
            position_update_interval=0,
            size_update_interval=0,
            max_fd_polygons=0,
        ),
    )

    for _ in range(50):
        optimizer.step(softness=1.0)

    forward = rasterizer.forward_pass(
        optimizer.polygons,
        softness=1.0,
        chunk_size=50,
        checkpoint_stride=10,
        target=target,
        compute_gradients=True,
    )
    assert forward.grad_colors is not None

    grad_norms = np.linalg.norm(forward.grad_colors, axis=1)
    bottom = float(grad_norms[0])
    top = float(grad_norms[-1])
    assert top > 0.0
    assert bottom < top

    render = rasterizer.render(optimizer.polygons, softness=1.0)
    per_poly_weight = np.sum(render.weights, axis=(1, 2), dtype=np.float32)
    expected_ratio = float(per_poly_weight[0] / max(per_poly_weight[-1], 1e-8))
    observed_ratio = float(bottom / top)

    assert observed_ratio < 0.75
    assert observed_ratio == pytest.approx(expected_ratio, rel=0.35, abs=0.02)


def test_step_rolls_back_geometry_without_reverting_color() -> None:
    height = width = 50
    target, rect_center, _ = _make_rect_target(height, width)

    polygons = LivePolygonBatch(
        centers=np.array([[rect_center[0], rect_center[1]]], dtype=np.float32),
        sizes=np.array([[10.0, 10.0]], dtype=np.float32),
        rotations=np.array([0.0], dtype=np.float32),
        colors=np.array([[0.95, 0.05, 0.05]], dtype=np.float32),
        alphas=np.array([1.0], dtype=np.float32),
        shape_types=np.array([SHAPE_QUAD], dtype=np.int32),
    )

    optimizer = LiveJointOptimizer(
        target_image=target,
        rasterizer=SoftRasterizer(height=height, width=width),
        polygons=polygons,
        config=LiveOptimizerConfig(
            color_lr=0.05,
            alpha_lr=0.0,
            position_lr=25.0,
            size_lr=0.0,
            position_update_interval=1,
            size_update_interval=0,
            max_fd_polygons=1,
            allow_loss_increase=False,
        ),
    )

    start_loss = float(optimizer.loss_history[-1])
    start_centers = np.array(optimizer.polygons.centers, copy=True)
    start_colors = np.array(optimizer.polygons.colors, copy=True)

    def _forced_bad_geometry(
        *, softness: float, checkpoints: dict[int, np.ndarray], indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        del softness
        del checkpoints
        pos = np.zeros_like(optimizer.polygons.centers, dtype=np.float32)
        if indices.size > 0:
            pos[indices] = np.array([250.0, -250.0], dtype=np.float32)
        size = np.zeros_like(optimizer.polygons.sizes, dtype=np.float32)
        return pos, size

    optimizer._fd_geometry_grads = _forced_bad_geometry  # type: ignore[method-assign]
    end_loss = float(optimizer.step(softness=1.0))

    assert end_loss < start_loss
    assert np.allclose(optimizer.polygons.centers, start_centers)
    assert not np.allclose(optimizer.polygons.colors, start_colors)
