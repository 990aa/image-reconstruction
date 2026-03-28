import numpy as np


DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 100


def create_white_canvas(
    width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT
) -> np.ndarray:
    """Create a white float32 canvas with RGB channels."""
    return np.ones((height, width, 3), dtype=np.float32)


def copy_canvas(canvas: np.ndarray) -> np.ndarray:
    """Return a deep copy of an existing canvas."""
    return np.array(canvas, copy=True)
