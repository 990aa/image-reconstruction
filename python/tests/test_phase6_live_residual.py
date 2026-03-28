import numpy as np

from src.live_optimizer import LiveJointOptimizer, LiveOptimizerConfig
from src.live_renderer import SoftRasterizer
from src.live_schedule import (
    apply_low_frequency_color_correction,
    decompose_residual,
    high_frequency_error_map,
    make_random_live_batch_with_bounds,
    progressive_growth,
)


def _hybrid_target(size: int = 64) -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, size), indexing="ij")
    smooth = np.stack(
        [
            0.3 + 0.5 * xx,
            0.2 + 0.4 * yy,
            0.4 + 0.25 * (1.0 - xx),
        ],
        axis=2,
    ).astype(np.float32)

    checker = np.sign(np.sin(2.0 * np.pi * xx * 10.0) * np.sin(2.0 * np.pi * yy * 10.0)).astype(np.float32)
    blob = np.exp(-(((xx - 0.70) ** 2) + ((yy - 0.38) ** 2)) / (2.0 * 0.10**2)).astype(np.float32)
    local_mask = np.clip(blob, 0.0, 1.0)
    hf = np.zeros_like(smooth)
    hf[..., 0] = 0.08 * checker * local_mask
    hf[..., 1] = -0.06 * checker * local_mask
    hf[..., 2] = 0.05 * checker * local_mask

    target = np.clip(smooth + hf, 0.0, 1.0)
    return target.astype(np.float32, copy=False)


def _make_partial_optimizer(target: np.ndarray) -> LiveJointOptimizer:
    h, w = target.shape[:2]
    config = LiveOptimizerConfig(
        color_lr=0.05,
        position_lr=0.002,
        size_lr=0.001,
        alpha_lr=0.02,
        render_chunk_size=50,
        position_update_interval=8,
        size_update_interval=12,
        max_fd_polygons=8,
    )

    polygons = make_random_live_batch_with_bounds(
        count=30,
        height=h,
        width=w,
        min_size=3.0,
        max_size=12.0,
        rng=np.random.default_rng(11),
    )

    optimizer = LiveJointOptimizer(
        target_image=target,
        rasterizer=SoftRasterizer(height=h, width=w),
        polygons=polygons,
        config=config,
    )
    optimizer.run(40, start_softness=2.0, end_softness=0.6)
    return optimizer


def test_phase6_residual_decomposition_and_targeting() -> None:
    target = _hybrid_target(64)
    optimizer = _make_partial_optimizer(target)

    before = float(optimizer.loss_history[-1])
    gain = apply_low_frequency_color_correction(
        optimizer,
        sigma=10.0,
        strength=0.7,
        softness=0.6,
    )
    after = float(optimizer.loss_history[-1])

    assert gain > 0.0
    assert after <= before * 0.98

    comp = decompose_residual(target, optimizer.current_canvas, sigma=10.0)
    mean_high = float(np.mean(comp.high_frequency, dtype=np.float32))
    assert abs(mean_high) < 1e-3

    base_polygons = optimizer.polygons.copy()
    config = optimizer.config

    raw_optimizer = LiveJointOptimizer(
        target_image=target,
        rasterizer=SoftRasterizer(height=64, width=64),
        polygons=base_polygons.copy(),
        config=config,
    )
    raw_start = float(raw_optimizer.loss_history[-1])
    raw_hf_start = float(np.mean(high_frequency_error_map(target, raw_optimizer.current_canvas, sigma=10.0), dtype=np.float32))
    progressive_growth(
        raw_optimizer,
        batch_schedule=[12],
        max_steps_per_cycle=10,
        post_add_steps=20,
        convergence_window=50,
        convergence_rel_threshold=0.001,
        region_window=3,
        new_polygon_alpha=0.60,
        min_new_size=1.5,
        max_new_size=4.0,
        use_content_aware_shapes=False,
        use_high_frequency_targeting=False,
        residual_sigma=10.0,
        low_frequency_correction_strength=0.0,
        max_add_attempts=1,
        enforce_cycle_improvement=False,
        start_softness=1.0,
        end_softness=0.6,
    )
    raw_drop = raw_start - float(raw_optimizer.loss_history[-1])
    raw_hf_drop = raw_hf_start - float(np.mean(high_frequency_error_map(target, raw_optimizer.current_canvas, sigma=10.0), dtype=np.float32))

    hf_optimizer = LiveJointOptimizer(
        target_image=target,
        rasterizer=SoftRasterizer(height=64, width=64),
        polygons=base_polygons.copy(),
        config=config,
    )
    hf_start = float(hf_optimizer.loss_history[-1])
    hf_hf_start = float(np.mean(high_frequency_error_map(target, hf_optimizer.current_canvas, sigma=10.0), dtype=np.float32))
    progressive_growth(
        hf_optimizer,
        batch_schedule=[12],
        max_steps_per_cycle=0,
        post_add_steps=0,
        convergence_window=50,
        convergence_rel_threshold=0.001,
        region_window=3,
        new_polygon_alpha=0.60,
        min_new_size=1.5,
        max_new_size=4.0,
        use_content_aware_shapes=False,
        use_high_frequency_targeting=True,
        residual_sigma=10.0,
        low_frequency_correction_strength=0.0,
        max_add_attempts=4,
        enforce_cycle_improvement=False,
        start_softness=1.0,
        end_softness=0.6,
    )
    hf_drop = hf_start - float(hf_optimizer.loss_history[-1])
    hf_hf_drop = hf_hf_start - float(np.mean(high_frequency_error_map(target, hf_optimizer.current_canvas, sigma=10.0), dtype=np.float32))

    assert hf_hf_drop >= (raw_hf_drop - 1e-3)
    assert hf_drop > (raw_drop - 0.02)
