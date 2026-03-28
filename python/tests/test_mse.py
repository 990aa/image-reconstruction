import numpy as np

from src.canvas import create_white_canvas
from src.mse import mean_squared_error, per_pixel_error_map


def test_mse_white_vs_black_is_one() -> None:
    canvas = create_white_canvas()
    black_target = np.zeros_like(canvas, dtype=np.float32)

    mse = mean_squared_error(canvas, black_target)
    assert np.isclose(mse, 1.0, atol=1e-6)


def test_mse_white_vs_white_is_zero() -> None:
    canvas = create_white_canvas()
    white_target = np.ones_like(canvas, dtype=np.float32)

    mse = mean_squared_error(canvas, white_target)
    assert np.isclose(mse, 0.0, atol=1e-6)


def test_error_map_max_is_single_red_pixel_center() -> None:
    canvas = create_white_canvas()
    target = np.ones_like(canvas, dtype=np.float32)
    target[50, 50] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    error_map = per_pixel_error_map(canvas, target)
    max_pos = np.unravel_index(np.argmax(error_map), error_map.shape)

    assert max_pos == (50, 50)
