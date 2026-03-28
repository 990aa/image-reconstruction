from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SIZE = (100, 100)


def load_target_image(
    filepath: str | Path, size: tuple[int, int] = DEFAULT_SIZE
) -> np.ndarray:
    """Load an image as float32 RGB array in the range [0.0, 1.0]."""
    with Image.open(filepath) as img:
        rgb = img.convert("RGB")
        resized = rgb.resize(size, Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    return array.astype(np.float32, copy=False)
