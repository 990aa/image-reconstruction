from pathlib import Path

import numpy as np

from src.live_optimizer import LiveJointOptimizer, LiveOptimizerConfig
from src.live_renderer import SoftRasterizer
from src.live_schedule import (
    load_square_target,
    make_empty_live_batch,
    make_random_live_batch_with_bounds,
    progressive_growth,
)


def test_phase3_progressive_growth_on_grape_beats_random_100() -> None:
    target = load_square_target(Path("targets/grape.jpg"), resolution=50)

    config = LiveOptimizerConfig(
        color_lr=0.08,
        position_lr=0.002,
        size_lr=0.0008,
        alpha_lr=0.08,
        render_chunk_size=50,
        position_update_interval=6,
        size_update_interval=10,
        max_fd_polygons=6,
    )

    progressive_optimizer = LiveJointOptimizer(
        target_image=target,
        rasterizer=SoftRasterizer(height=50, width=50),
        polygons=make_empty_live_batch(),
        config=config,
    )

    cycle_results, growth_events = progressive_growth(
        progressive_optimizer,
        batch_schedule=[20, 20, 20, 20, 20],
        max_steps_per_cycle=70,
        post_add_steps=20,
        convergence_window=100,
        convergence_rel_threshold=0.001,
        region_window=5,
        new_polygon_alpha=0.60,
        min_new_size=2.5,
        max_new_size=8.0,
        use_content_aware_shapes=True,
        use_high_frequency_targeting=True,
        residual_sigma=10.0,
        start_softness=2.0,
        end_softness=0.5,
    )
    progressive_start_loss = float(progressive_optimizer.loss_history[0])

    assert len(cycle_results) == 5
    assert progressive_optimizer.polygons.count == 100
    assert growth_events

    improved_cycles = sum(
        1 for cycle in cycle_results if cycle.loss_after_cycle < cycle.loss_before_cycle
    )
    assert improved_cycles >= 4
    assert cycle_results[-1].loss_after_cycle < cycle_results[0].loss_before_cycle

    for event in growth_events:
        assert event.distance_to_target <= 20.0

    total_steps = int(sum(c.optimization_steps for c in cycle_results))
    progressive_optimizer.run(
        max(60, total_steps // 2), start_softness=1.2, end_softness=0.5
    )

    baseline_config = LiveOptimizerConfig(
        color_lr=0.0,
        position_lr=config.position_lr,
        size_lr=config.size_lr,
        alpha_lr=0.0,
        render_chunk_size=config.render_chunk_size,
        position_update_interval=1000,
        size_update_interval=1000,
        max_fd_polygons=0,
    )
    baseline_optimizer = LiveJointOptimizer(
        target_image=target,
        rasterizer=SoftRasterizer(height=50, width=50),
        polygons=make_random_live_batch_with_bounds(
            count=100,
            height=50,
            width=50,
            min_size=2.5,
            max_size=8.0,
            rng=np.random.default_rng(123),
        ),
        config=baseline_config,
    )
    baseline_start_loss = float(baseline_optimizer.loss_history[0])
    baseline_optimizer.run(total_steps, start_softness=2.0, end_softness=0.5)

    progressive_drop = progressive_start_loss - float(
        progressive_optimizer.loss_history[-1]
    )
    baseline_drop = baseline_start_loss - float(baseline_optimizer.loss_history[-1])
    assert progressive_drop > baseline_drop
