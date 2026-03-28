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
    target = load_square_target(Path("targets/internet_portrait.jpg"), resolution=50)

    config = LiveOptimizerConfig(
        color_lr=0.05,
        position_lr=0.002,
        size_lr=0.0008,
        alpha_lr=0.02,
        render_chunk_size=50,
        position_update_interval=3,
        size_update_interval=5,
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
        max_steps_per_cycle=110,
        post_add_steps=14,
        convergence_window=100,
        convergence_rel_threshold=0.001,
        region_window=5,
        new_polygon_alpha=0.60,
        min_new_size=2.5,
        max_new_size=8.0,
        start_softness=2.0,
        end_softness=0.5,
    )

    assert len(cycle_results) == 5
    assert progressive_optimizer.polygons.count == 100
    assert growth_events

    for cycle in cycle_results:
        assert cycle.loss_after_cycle < cycle.loss_before_cycle

    for event in growth_events:
        assert event.distance_to_target <= 20.0

    total_steps = int(sum(c.optimization_steps for c in cycle_results))
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
        config=config,
    )
    baseline_optimizer.run(total_steps, start_softness=2.0, end_softness=0.5)

    assert progressive_optimizer.loss_history[-1] < baseline_optimizer.loss_history[-1]
