from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import color
from skimage.filters import sobel_h, sobel_v
from sklearn.cluster import MiniBatchKMeans

from src.image_loader import load_target_image


DEFAULT_BASE_RESOLUTION = 200


def build_level_sizes(base_resolution: int) -> tuple[tuple[int, int], ...]:
    if base_resolution <= 0:
        raise ValueError("base_resolution must be positive.")

    l0 = int(base_resolution)
    l1 = max(2, int(round(base_resolution / 2.0)))
    l2 = max(2, int(round(base_resolution / 4.0)))
    l3 = max(2, int(round(base_resolution / 8.0)))
    return ((l0, l0), (l1, l1), (l2, l2), (l3, l3))


@dataclass(frozen=True)
class PreprocessedTarget:
    base_resolution: int
    target_rgb: np.ndarray
    pyramid: list[np.ndarray]
    segmentation_map: np.ndarray
    cluster_centroids_lab: np.ndarray
    cluster_centroids_rgb: np.ndarray
    cluster_variances_lab: np.ndarray
    structure_map: np.ndarray
    gradient_angle_map: np.ndarray
    complexity_score: float
    recommended_polygons: int
    recommended_k: int
    recommended_size_schedule: dict[str, float]


def _resize_float_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_h, target_w = size
    if image.shape[:2] == (target_h, target_w):
        return image.astype(np.float32, copy=False)

    uint8 = np.clip(image, 0.0, 1.0)
    uint8 = (uint8 * 255.0).round().astype(np.uint8)

    from PIL import Image

    pil = Image.fromarray(uint8, mode="RGB")
    resized = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return arr.astype(np.float32, copy=False)


def build_gaussian_pyramid(
    target_rgb: np.ndarray,
    level_sizes: tuple[tuple[int, int], ...],
) -> list[np.ndarray]:
    if len(level_sizes) != 4:
        raise ValueError("This phase requires exactly four pyramid levels.")

    base = _resize_float_image(target_rgb, level_sizes[0])
    pyramid: list[np.ndarray] = [base]
    previous = base

    for size in level_sizes[1:]:
        blurred = gaussian_filter(previous, sigma=(1.0, 1.0, 0.0), mode="reflect")
        downsampled = _resize_float_image(blurred, size)
        pyramid.append(downsampled)
        previous = downsampled

    return pyramid


def segment_image_lab(
    target_rgb: np.ndarray,
    k_clusters: int,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if k_clusters < 2:
        raise ValueError("k_clusters must be at least 2.")

    h, w, _ = target_rgb.shape
    lab = color.rgb2lab(np.clip(target_rgb, 0.0, 1.0).astype(np.float32, copy=False))
    flat = lab.reshape(-1, 3)

    model = MiniBatchKMeans(
        n_clusters=k_clusters,
        n_init=3,
        random_state=random_seed,
        batch_size=2048,
        max_iter=100,
    )
    labels = model.fit_predict(flat)

    seg_map = labels.reshape(h, w).astype(np.int32, copy=False)

    centroids_lab = model.cluster_centers_.astype(np.float32, copy=False)
    centroids_rgb = color.lab2rgb(centroids_lab[np.newaxis, :, :])[0]
    centroids_rgb = np.clip(centroids_rgb, 0.0, 1.0).astype(np.float32, copy=False)

    variances = np.zeros((k_clusters,), dtype=np.float32)
    for cluster_id in range(k_clusters):
        cluster_points = flat[labels == cluster_id]
        if cluster_points.size == 0:
            variances[cluster_id] = 1.0
            continue
        diffs = cluster_points - centroids_lab[cluster_id]
        sq_dist = np.sum(np.square(diffs, dtype=np.float32), axis=1, dtype=np.float32)
        variances[cluster_id] = float(np.mean(sq_dist, dtype=np.float32))

    return seg_map, centroids_lab, centroids_rgb, variances


def compute_structure_and_direction(
    target_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gray = np.mean(target_rgb.astype(np.float32, copy=False), axis=2)
    gx = sobel_h(gray).astype(np.float32, copy=False)
    gy = sobel_v(gray).astype(np.float32, copy=False)
    grad = np.hypot(gx, gy).astype(np.float32, copy=False)

    g_min = float(np.min(grad))
    g_max = float(np.max(grad))
    if g_max <= g_min + 1e-8:
        structure = np.zeros_like(grad, dtype=np.float32)
    else:
        structure = (grad - g_min) / (g_max - g_min)
        structure = np.clip(structure, 0.0, 1.0).astype(np.float32, copy=False)

    angles = np.arctan2(gy, gx).astype(np.float32, copy=False)
    return structure, angles


def compute_structure_map(target_rgb: np.ndarray) -> np.ndarray:
    structure, _ = compute_structure_and_direction(target_rgb)
    return structure


def compute_complexity_score(
    target_rgb: np.ndarray, structure_map: np.ndarray
) -> float:
    mean_grad = float(np.mean(structure_map, dtype=np.float32))
    center = np.mean(target_rgb, axis=(0, 1), keepdims=True, dtype=np.float32)
    mean_abs_dev = float(
        np.mean(
            np.abs(target_rgb.astype(np.float32, copy=False) - center), dtype=np.float32
        )
    )

    raw_ratio = mean_grad / max(mean_abs_dev, 1e-6)
    score = raw_ratio / (1.0 + raw_ratio)
    return float(np.clip(score, 0.0, 1.0))


def recommend_polygon_count(complexity_score: float) -> int:
    value = 50.0 + 450.0 * float(np.clip(complexity_score, 0.0, 1.0))
    return int(round(value))


def recommend_cluster_count(complexity_score: float) -> int:
    value = 12.0 + 8.0 * float(np.clip(complexity_score, 0.0, 1.0))
    return int(round(value))


def recommend_size_schedule(
    complexity_score: float, max_size_override: float | None = None
) -> dict[str, float]:
    score = float(np.clip(complexity_score, 0.0, 1.0))

    coarse_start = 18.0 + (1.0 - score) * 18.0
    if max_size_override is not None:
        coarse_start = max(4.0, float(max_size_override))

    coarse_end = max(6.0, coarse_start * 0.55)
    structural_end = max(3.5, coarse_end * 0.55)
    detail_end = max(2.0, structural_end * 0.5)

    if coarse_end > coarse_start:
        coarse_end = coarse_start
    if structural_end > coarse_end:
        structural_end = coarse_end
    if detail_end > structural_end:
        detail_end = structural_end

    return {
        "coarse_start": float(coarse_start),
        "coarse_end": float(coarse_end),
        "structural_end": float(structural_end),
        "detail_end": float(detail_end),
    }


def preprocess_target_array(
    target_rgb: np.ndarray,
    polygon_override: int | None = None,
    max_size_override: float | None = None,
    random_seed: int = 0,
    base_resolution: int = DEFAULT_BASE_RESOLUTION,
) -> PreprocessedTarget:
    if target_rgb.ndim != 3 or target_rgb.shape[2] != 3:
        raise ValueError("target_rgb must have shape (H, W, 3).")

    level_sizes = build_level_sizes(base_resolution)

    base_target = _resize_float_image(target_rgb, level_sizes[0])
    pyramid = build_gaussian_pyramid(base_target, level_sizes)
    structure_map, gradient_angle_map = compute_structure_and_direction(base_target)
    complexity = compute_complexity_score(base_target, structure_map)

    recommended_k = recommend_cluster_count(complexity)
    segmentation_map, centroids_lab, centroids_rgb, variances_lab = segment_image_lab(
        base_target,
        k_clusters=recommended_k,
        random_seed=random_seed,
    )

    polygons = recommend_polygon_count(complexity)
    if polygon_override is not None:
        polygons = max(1, int(polygon_override))

    size_schedule = recommend_size_schedule(
        complexity_score=complexity,
        max_size_override=max_size_override,
    )

    return PreprocessedTarget(
        base_resolution=int(base_resolution),
        target_rgb=base_target,
        pyramid=pyramid,
        segmentation_map=segmentation_map,
        cluster_centroids_lab=centroids_lab,
        cluster_centroids_rgb=centroids_rgb,
        cluster_variances_lab=variances_lab,
        structure_map=structure_map,
        gradient_angle_map=gradient_angle_map,
        complexity_score=complexity,
        recommended_polygons=polygons,
        recommended_k=recommended_k,
        recommended_size_schedule=size_schedule,
    )


def preprocess_target_image(
    image_path: str | Path,
    polygon_override: int | None = None,
    max_size_override: float | None = None,
    random_seed: int = 0,
    base_resolution: int = DEFAULT_BASE_RESOLUTION,
) -> PreprocessedTarget:
    level_sizes = build_level_sizes(base_resolution)
    target_rgb = load_target_image(image_path, size=level_sizes[0])
    return preprocess_target_array(
        target_rgb=target_rgb,
        polygon_override=polygon_override,
        max_size_override=max_size_override,
        random_seed=random_seed,
        base_resolution=base_resolution,
    )
