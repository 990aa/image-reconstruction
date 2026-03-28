from pathlib import Path

import numpy as np

from src.live_optimizer import LiveJointOptimizer, LiveOptimizerConfig
from src.live_renderer import SoftRasterizer
from src.live_schedule import (
    load_square_target,
    make_random_live_batch_with_bounds,
    run_multi_resolution_schedule,
)


def test_phase4_multi_resolution_schedule_beats_single_resolution() -> None:
    target_200 = load_square_target(Path("targets/grape.jpg"), resolution=200)

    config = LiveOptimizerConfig(
        color_lr=0.05,
        position_lr=0.0015,
        size_lr=0.0006,
        alpha_lr=0.02,
        render_chunk_size=50,
        position_update_interval=50,
        size_update_interval=60,
        max_fd_polygons=0,
    )

    result = run_multi_resolution_schedule(
        target_image=target_200,
        random_seed=77,
        include_round4=False,
        max_steps_per_cycle=1,
        post_add_steps=0,
        convergence_window=100,
        convergence_rel_threshold=0.001,
        base_config=config,
    )

    assert len(result.rounds) == 3
    round_1, round_2, round_3 = result.rounds

    assert round_2.loss_end < round_1.loss_end
    assert round_3.loss_end < round_2.loss_end

    assert round_2.boundary_error < round_1.boundary_error
    assert round_3.boundary_error < round_2.boundary_error

    baseline_optimizer = LiveJointOptimizer(
        target_image=target_200,
        rasterizer=SoftRasterizer(height=200, width=200),
        polygons=make_random_live_batch_with_bounds(
            count=round_3.polygon_count,
            height=200,
            width=200,
            min_size=2.0,
            max_size=10.0,
            rng=np.random.default_rng(991),
        ),
        config=LiveOptimizerConfig(
            color_lr=config.color_lr,
            position_lr=config.position_lr,
            size_lr=config.size_lr,
            alpha_lr=config.alpha_lr,
            render_chunk_size=config.render_chunk_size,
            position_update_interval=config.position_update_interval,
            size_update_interval=config.size_update_interval,
            min_size=2.0,
            max_size=10.0,
        ),
    )

    baseline_optimizer.run(
        result.total_optimization_steps,
        start_softness=2.0,
        end_softness=0.5,
    )

    assert round_3.loss_end < baseline_optimizer.loss_history[-1]
