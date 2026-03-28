from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


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


def generate_shape(
    shape_type: ShapeType,
    probability_map: np.ndarray,
    target_image: np.ndarray,
    size_px: float,
    center_xy: tuple[int, int] | None = None,
    rng: np.random.Generator | None = None,
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
    alpha = float(rng.uniform(0.25, 0.65))

    if shape_type in (ShapeType.TRIANGLE, ShapeType.QUADRILATERAL):
        count = 3 if shape_type == ShapeType.TRIANGLE else 4
        angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
        radii = rng.uniform(0.4 * size, size, size=count)

        raw_vertices: list[tuple[float, float]] = []
        for angle, radius in zip(angles, radii, strict=True):
            x = center_x + np.cos(angle) * radius
            y = center_y + np.sin(angle) * radius
            raw_vertices.append((x, y))

        if shape_type == ShapeType.QUADRILATERAL:
            raw_vertices.sort(
                key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x)
            )

        vertices = _clamp_vertices(raw_vertices, w, h)
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        color = _sample_color_from_bbox(
            target_image, min(xs), min(ys), max(xs), max(ys), rng
        )

        return Polygon(
            vertices=vertices, color=color, alpha=alpha, shape_type=shape_type
        )

    if shape_type == ShapeType.ELLIPSE:
        semi_major = int(max(2, round(rng.uniform(0.4 * size, size))))
        semi_minor = int(max(2, round(rng.uniform(0.3 * size, 0.9 * size))))
        rotation = float(rng.uniform(0.0, 2.0 * np.pi))

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

        color = _sample_color_from_bbox(
            target_image,
            center_x - semi_major,
            center_y - semi_minor,
            center_x + semi_major,
            center_y + semi_minor,
            rng,
        )

        return Polygon(
            vertices=vertices,
            color=color,
            alpha=alpha,
            shape_type=shape_type,
            ellipse_center=(center_x, center_y),
            ellipse_axes=(semi_major, semi_minor),
            ellipse_rotation=rotation,
        )

    raise ValueError(f"Unsupported shape type: {shape_type}")
