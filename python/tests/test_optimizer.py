from pathlib import Path

from src.image_loader import load_target_image
from src.optimizer import HillClimbingOptimizer


def test_optimizer_behavior_on_heart_target_500_iterations() -> None:
    target_path = Path(__file__).resolve().parents[1] / "targets" / "heart.png"
    target = load_target_image(target_path)

    optimizer = HillClimbingOptimizer(
        target_image=target,
        max_iterations=500,
        snapshot_interval=100,
        random_seed=1234,
    )
    initial_mse = optimizer.current_mse

    optimizer.run(iterations=500)

    assert optimizer.current_mse < initial_mse
    assert len(optimizer.acceptance_history) == 500
    assert len(optimizer.accepted_polygons) >= 5
    assert optimizer.snapshots
