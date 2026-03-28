from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from skimage import color


class ShapeType(Enum):
    TRIANGLE = "triangle"
    QUADRILATERAL = "quadrilateral"
    ELLIPSE = "ellipse"


@dataclass(frozen=True)
class Polygon:
    vertices: list[tuple[int, int]]
    color: tuple[float, float, float]
    alpha: float
    shape_type: ShapeType
    ellipse_center: tuple[int, int] | None = None
    ellipse_axes: tuple[int, int] | None = None
    ellipse_rotation: float = 0.0
    orientation: float | None = None


def sample_center(
    probability_map: np.ndarray, rng: np.random.Generator
) -> tuple[int, int]:
    """Sample a center point from a normalized 2D probability map."""
    h, w = probability_map.shape
    flat = probability_map.ravel().astype(np.float64, copy=False)
    total = flat.sum()
    if total <= 0.0:
        flat = np.full_like(flat, 1.0 / flat.size)
    else:
        flat = flat / total

    index = int(rng.choice(flat.size, p=flat))
    y, x = divmod(index, w)
    return x, y


def _clamp_vertices(
    vertices: Sequence[tuple[float, float]], width: int, height: int
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for x, y in vertices:
        out.append(
            (
                int(np.clip(round(x), 0, width - 1)),
                int(np.clip(round(y), 0, height - 1)),
            )
        )
    return out


def _sample_color_from_bbox(
    target: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    rng: np.random.Generator,
    noise_std: float = 0.15,
) -> tuple[float, float, float]:
    h, w, _ = target.shape
    x0 = int(np.clip(x_min, 0, w - 1))
    x1 = int(np.clip(x_max, 0, w - 1))
    y0 = int(np.clip(y_min, 0, h - 1))
    y1 = int(np.clip(y_max, 0, h - 1))

    patch = target[y0 : y1 + 1, x0 : x1 + 1]
    if patch.size == 0:
        base = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    else:
        base = patch.mean(axis=(0, 1), dtype=np.float32)

    color = np.clip(base + rng.normal(0.0, noise_std, size=3), 0.0, 1.0)
    return float(color[0]), float(color[1]), float(color[2])


def polygon_center(polygon: Polygon) -> tuple[int, int]:
    if polygon.ellipse_center is not None:
        return polygon.ellipse_center

    xs = [v[0] for v in polygon.vertices]
    ys = [v[1] for v in polygon.vertices]
    return int(round(float(np.mean(xs)))), int(round(float(np.mean(ys))))


def _local_lab_average(
    target_lab: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> np.ndarray:
    h, w = target_lab.shape[:2]
    x0 = int(np.clip(x_min, 0, w - 1))
    x1 = int(np.clip(x_max, 0, w - 1))
    y0 = int(np.clip(y_min, 0, h - 1))
    y1 = int(np.clip(y_max, 0, h - 1))

    patch = target_lab[y0 : y1 + 1, x0 : x1 + 1]
    if patch.size == 0:
        return np.array([50.0, 0.0, 0.0], dtype=np.float32)
    return patch.mean(axis=(0, 1), dtype=np.float32)


def _sample_segmentation_aware_color(
    *,
    center_x: int,
    center_y: int,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    target_image: np.ndarray,
    target_lab: np.ndarray | None,
    segmentation_map: np.ndarray | None,
    cluster_centroids_lab: np.ndarray | None,
    cluster_variances_lab: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if (
        segmentation_map is None
        or cluster_centroids_lab is None
        or cluster_variances_lab is None
        or target_lab is None
    ):
        return _sample_color_from_bbox(target_image, x_min, y_min, x_max, y_max, rng)

    h, w = segmentation_map.shape
    cx = int(np.clip(center_x, 0, w - 1))
    cy = int(np.clip(center_y, 0, h - 1))
    cluster_id = int(segmentation_map[cy, cx])
    if cluster_id < 0 or cluster_id >= cluster_centroids_lab.shape[0]:
        return _sample_color_from_bbox(target_image, x_min, y_min, x_max, y_max, rng)

    centroid_lab = cluster_centroids_lab[cluster_id].astype(np.float32, copy=False)
    variance = float(cluster_variances_lab[cluster_id])
    radius = max(1.5, np.sqrt(max(variance, 1e-6)) * 0.35)

    direction = rng.normal(0.0, 1.0, size=3).astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm

    sampled_lab = centroid_lab + direction * float(rng.uniform(0.0, radius))
    local_avg_lab = _local_lab_average(target_lab, x_min, y_min, x_max, y_max)
    blended_lab = 0.6 * sampled_lab + 0.4 * local_avg_lab

    blended_rgb = color.lab2rgb(blended_lab[np.newaxis, np.newaxis, :])[0, 0]
    blended_rgb = np.clip(blended_rgb, 0.0, 1.0).astype(np.float32, copy=False)
    return float(blended_rgb[0]), float(blended_rgb[1]), float(blended_rgb[2])


def _oriented_basis(angle: float) -> tuple[np.ndarray, np.ndarray]:
    major = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    minor = np.array([-major[1], major[0]], dtype=np.float32)
    return major, minor


def _generate_oriented_triangle(
    center_x: int,
    center_y: int,
    size: float,
    orientation: float,
    width: int,
    height: int,
    rng: np.random.Generator,
    thin_factor: float,
) -> list[tuple[int, int]]:
    major, minor = _oriented_basis(orientation)
    length = max(2.0, size)
    half_width = max(1.0, size * thin_factor)

    tip = np.array([center_x, center_y], dtype=np.float32) + major * length
    base_center = np.array([center_x, center_y], dtype=np.float32) - major * (0.6 * length)
    base_a = base_center + minor * half_width
    base_b = base_center - minor * half_width

    return _clamp_vertices([tuple(tip), tuple(base_a), tuple(base_b)], width, height)


def _generate_oriented_quad(
    center_x: int,
    center_y: int,
    size: float,
    orientation: float,
    width: int,
    height: int,
    thin_factor: float,
) -> list[tuple[int, int]]:
    major, minor = _oriented_basis(orientation)
    half_len = max(2.0, size)
    half_w = max(1.0, size * thin_factor)

    c = np.array([center_x, center_y], dtype=np.float32)
    corners = [
        c + major * half_len + minor * half_w,
        c + major * half_len - minor * half_w,
        c - major * half_len - minor * half_w,
        c - major * half_len + minor * half_w,
    ]
    return _clamp_vertices([tuple(p) for p in corners], width, height)


def generate_shape(
    shape_type: ShapeType,
    probability_map: np.ndarray,
    target_image: np.ndarray,
    size_px: float,
    center_xy: tuple[int, int] | None = None,
    rng: np.random.Generator | None = None,
    orientation: float | None = None,
    profile: str = "default",
    segmentation_map: np.ndarray | None = None,
    cluster_centroids_lab: np.ndarray | None = None,
    cluster_variances_lab: np.ndarray | None = None,
    target_lab: np.ndarray | None = None,
    alpha: float | None = None,
) -> Polygon:
    """Generate a triangle, quadrilateral, or ellipse from the guided distribution."""
    if rng is None:
        rng = np.random.default_rng()

    h, w, _ = target_image.shape
    size = max(1.0, float(size_px))
    if center_xy is None:
        center_x, center_y = sample_center(probability_map, rng)
    else:
        center_x, center_y = center_xy

    if orientation is None:
        orientation = float(rng.uniform(0.0, 2.0 * np.pi))
    if alpha is None:
        alpha = 0.40
    alpha = float(np.clip(alpha, 0.0, 1.0))

    if shape_type in (ShapeType.TRIANGLE, ShapeType.QUADRILATERAL):
        if profile == "edge":
            thin_factor = float(rng.uniform(0.10, 0.24))
            if shape_type == ShapeType.TRIANGLE:
                vertices = _generate_oriented_triangle(
                    center_x,
                    center_y,
                    size,
                    orientation,
                    w,
                    h,
                    rng,
                    thin_factor,
                )
            else:
                vertices = _generate_oriented_quad(
                    center_x,
                    center_y,
                    size,
                    orientation,
                    w,
                    h,
                    thin_factor,
                )
        elif profile == "texture":
            jittered = float(orientation + rng.normal(0.0, np.deg2rad(22.0)))
            vertices = _generate_oriented_triangle(
                center_x,
                center_y,
                max(1.5, size * 0.65),
                jittered,
                w,
                h,
                rng,
                thin_factor=0.42,
            )
        else:
            if shape_type == ShapeType.QUADRILATERAL:
                vertices = _generate_oriented_quad(
                    center_x,
                    center_y,
                    size,
                    orientation,
                    w,
                    h,
                    thin_factor=0.55,
                )
            else:
                count = 3
                angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
                radii = rng.uniform(0.4 * size, size, size=count)
                raw_vertices: list[tuple[float, float]] = []
                for angle, radius in zip(angles, radii, strict=True):
                    x = center_x + np.cos(angle) * radius
                    y = center_y + np.sin(angle) * radius
                    raw_vertices.append((x, y))
                vertices = _clamp_vertices(raw_vertices, w, h)

        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        color_rgb = _sample_segmentation_aware_color(
            center_x=center_x,
            center_y=center_y,
            x_min=min(xs),
            y_min=min(ys),
            x_max=max(xs),
            y_max=max(ys),
            target_image=target_image,
            target_lab=target_lab,
            segmentation_map=segmentation_map,
            cluster_centroids_lab=cluster_centroids_lab,
            cluster_variances_lab=cluster_variances_lab,
            rng=rng,
        )

        return Polygon(
            vertices=vertices,
            color=color_rgb,
            alpha=alpha,
            shape_type=shape_type,
            orientation=float(orientation),
        )

    if shape_type == ShapeType.ELLIPSE:
        if profile == "flat":
            semi_major = int(max(3, round(rng.uniform(0.7 * size, 1.25 * size))))
            semi_minor = int(max(2, round(rng.uniform(0.45 * size, 0.95 * size))))
        elif profile == "texture":
            semi_major = int(max(2, round(rng.uniform(0.35 * size, 0.7 * size))))
            semi_minor = int(max(2, round(rng.uniform(0.25 * size, 0.6 * size))))
        else:
            semi_major = int(max(2, round(rng.uniform(0.4 * size, size))))
            semi_minor = int(max(2, round(rng.uniform(0.3 * size, 0.9 * size))))
        rotation = float(orientation)

        vertices = _clamp_vertices(
            [
                (center_x - semi_major, center_y),
                (center_x, center_y - semi_minor),
                (center_x + semi_major, center_y),
                (center_x, center_y + semi_minor),
            ],
            w,
            h,
        )

        color_rgb = _sample_segmentation_aware_color(
            center_x=center_x,
            center_y=center_y,
            x_min=center_x - semi_major,
            y_min=center_y - semi_minor,
            x_max=center_x + semi_major,
            y_max=center_y + semi_minor,
            target_image=target_image,
            target_lab=target_lab,
            segmentation_map=segmentation_map,
            cluster_centroids_lab=cluster_centroids_lab,
            cluster_variances_lab=cluster_variances_lab,
            rng=rng,
        )

        return Polygon(
            vertices=vertices,
            color=color_rgb,
            alpha=alpha,
            shape_type=shape_type,
            ellipse_center=(center_x, center_y),
            ellipse_axes=(semi_major, semi_minor),
            ellipse_rotation=rotation,
            orientation=float(rotation),
        )

    raise ValueError(f"Unsupported shape type: {shape_type}")
