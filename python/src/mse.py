import numpy as np
from scipy.ndimage import gaussian_filter


def mean_squared_error(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Compute scalar MSE across all pixels and channels."""
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have identical shapes.")

    diff = image_a.astype(np.float32, copy=False) - image_b.astype(
        np.float32, copy=False
    )
    return float(np.mean(np.square(diff), dtype=np.float32))


def per_pixel_error_map(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Compute per-pixel squared error summed across RGB channels."""
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have identical shapes.")

    diff = image_a.astype(np.float32, copy=False) - image_b.astype(
        np.float32, copy=False
    )
    return np.sum(np.square(diff), axis=2, dtype=np.float32)


def process_error_map(raw_error_map: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Smooth and normalize an error map into a probability distribution."""
    smoothed = gaussian_filter(
        raw_error_map.astype(np.float32, copy=False), sigma=sigma
    )

    flat = smoothed.ravel()
    total = float(np.sum(flat, dtype=np.float32))
    if total <= 0.0:
        uniform = np.full_like(flat, 1.0 / flat.size, dtype=np.float32)
        return uniform.reshape(smoothed.shape)

    normalized = flat / total
    return normalized.reshape(smoothed.shape).astype(np.float32, copy=False)
