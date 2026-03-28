import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import color


def mean_squared_error(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Compute scalar MSE across all pixels and channels."""
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have identical shapes.")

    diff = image_a.astype(np.float32, copy=False) - image_b.astype(
        np.float32, copy=False
    )
    return float(np.mean(np.square(diff), dtype=np.float32))


def rgb_to_lab_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape (H, W, 3).")

    clipped = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    return color.rgb2lab(clipped).astype(np.float32, copy=False)


def perceptual_mse_lab(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Compute perceptual MSE in LAB space (Delta-E style approximation)."""
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have identical shapes.")

    lab_a = rgb_to_lab_image(image_a)
    lab_b = rgb_to_lab_image(image_b)
    diff = lab_a - lab_b
    return float(np.mean(np.square(diff), dtype=np.float32))


def per_pixel_error_map(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Compute per-pixel squared error summed across RGB channels."""
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have identical shapes.")

    diff = image_a.astype(np.float32, copy=False) - image_b.astype(
        np.float32, copy=False
    )
    return np.sum(np.square(diff), axis=2, dtype=np.float32)


def per_pixel_perceptual_error_map(
    image_a: np.ndarray, image_b: np.ndarray
) -> np.ndarray:
    """Compute per-pixel LAB-space squared error for perceptual guidance."""
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have identical shapes.")

    lab_a = rgb_to_lab_image(image_a)
    lab_b = rgb_to_lab_image(image_b)
    diff = lab_a - lab_b
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
