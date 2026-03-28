from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SHAPE_TRIANGLE = 0
SHAPE_QUAD = 1
SHAPE_ELLIPSE = 2


_TRIANGLE_LOCAL = np.array(
    [
        [1.0, 0.0],
        [-0.5, 0.8660254],
        [-0.5, -0.8660254],
    ],
    dtype=np.float32,
)

_QUAD_LOCAL = np.array(
    [
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass
class LivePolygonBatch:
    centers: np.ndarray
    sizes: np.ndarray
    rotations: np.ndarray
    colors: np.ndarray
    alphas: np.ndarray
    shape_types: np.ndarray

    def __post_init__(self) -> None:
        self.centers = np.ascontiguousarray(self.centers, dtype=np.float32)
        self.sizes = np.ascontiguousarray(self.sizes, dtype=np.float32)
        self.rotations = np.ascontiguousarray(self.rotations, dtype=np.float32)
        self.colors = np.ascontiguousarray(self.colors, dtype=np.float32)
        self.alphas = np.ascontiguousarray(self.alphas, dtype=np.float32)
        self.shape_types = np.ascontiguousarray(self.shape_types, dtype=np.int32)

        if self.centers.ndim != 2 or self.centers.shape[1] != 2:
            raise ValueError("centers must have shape (N, 2).")
        if self.sizes.ndim != 2 or self.sizes.shape[1] != 2:
            raise ValueError("sizes must have shape (N, 2).")
        if self.colors.ndim != 2 or self.colors.shape[1] != 3:
            raise ValueError("colors must have shape (N, 3).")
        if self.rotations.ndim != 1:
            raise ValueError("rotations must have shape (N,).")
        if self.alphas.ndim != 1:
            raise ValueError("alphas must have shape (N,).")
        if self.shape_types.ndim != 1:
            raise ValueError("shape_types must have shape (N,).")

        count = self.centers.shape[0]
        if (
            self.sizes.shape[0] != count
            or self.colors.shape[0] != count
            or self.rotations.shape[0] != count
            or self.alphas.shape[0] != count
            or self.shape_types.shape[0] != count
        ):
            raise ValueError("All parameter arrays must have identical length N.")

        if np.any(self.sizes <= 0.0):
            raise ValueError("sizes must be positive.")

    @property
    def count(self) -> int:
        return int(self.centers.shape[0])

    def copy(self) -> LivePolygonBatch:
        return LivePolygonBatch(
            centers=np.array(self.centers, copy=True),
            sizes=np.array(self.sizes, copy=True),
            rotations=np.array(self.rotations, copy=True),
            colors=np.array(self.colors, copy=True),
            alphas=np.array(self.alphas, copy=True),
            shape_types=np.array(self.shape_types, copy=True),
        )


@dataclass
class SoftRenderResult:
    canvas: np.ndarray
    coverage: np.ndarray
    effective_alpha: np.ndarray
    trans_before: np.ndarray


class SoftRasterizer:
    """Batched soft rasterizer with precomputed pixel grid."""

    def __init__(self, height: int, width: int) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive.")

        self.height = int(height)
        self.width = int(width)

        y_coords, x_coords = np.meshgrid(
            np.arange(self.height, dtype=np.float32),
            np.arange(self.width, dtype=np.float32),
            indexing="ij",
        )
        # Pixel grid is created once and reused across all coverage calls.
        self.pixel_grid = np.stack([x_coords, y_coords], axis=2).astype(
            np.float32, copy=False
        )
        self.grid_x = self.pixel_grid[:, :, 0]
        self.grid_y = self.pixel_grid[:, :, 1]

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -50.0, 50.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)

    def _convex_signed_distance(self, vertices: np.ndarray) -> np.ndarray:
        """Signed distance approximation for convex polygons.

        Args:
            vertices: shape (K, M, 2) where M is the polygon vertex count.

        Returns:
            Signed distance maps with shape (K, H, W).
        """
        if vertices.ndim != 3 or vertices.shape[2] != 2:
            raise ValueError("vertices must have shape (K, M, 2).")

        _, vertex_count, _ = vertices.shape
        if vertex_count < 3:
            raise ValueError("Convex polygon needs at least 3 vertices.")

        starts = vertices
        ends = np.roll(vertices, shift=-1, axis=1)

        edge_x = ends[:, :, 0] - starts[:, :, 0]
        edge_y = ends[:, :, 1] - starts[:, :, 1]
        edge_norm = np.sqrt(edge_x * edge_x + edge_y * edge_y + 1e-6)

        dx = self.grid_x[None, None, :, :] - starts[:, :, 0][:, :, None, None]
        dy = self.grid_y[None, None, :, :] - starts[:, :, 1][:, :, None, None]

        signed_edge_dist = (
            edge_x[:, :, None, None] * dy - edge_y[:, :, None, None] * dx
        ) / edge_norm[:, :, None, None]

        area = 0.5 * np.sum(
            starts[:, :, 0] * ends[:, :, 1] - starts[:, :, 1] * ends[:, :, 0],
            axis=1,
        )
        orientation = np.where(area >= 0.0, 1.0, -1.0).astype(np.float32)
        signed_edge_dist = signed_edge_dist * orientation[:, None, None, None]

        return np.min(signed_edge_dist, axis=1).astype(np.float32, copy=False)

    def _triangle_vertices(
        self,
        centers: np.ndarray,
        sizes: np.ndarray,
        rotations: np.ndarray,
    ) -> np.ndarray:
        sx = sizes[:, 0][:, None]
        sy = sizes[:, 1][:, None]

        local_x = _TRIANGLE_LOCAL[:, 0][None, :] * sx
        local_y = _TRIANGLE_LOCAL[:, 1][None, :] * sy

        cos_r = np.cos(rotations)[:, None]
        sin_r = np.sin(rotations)[:, None]

        rot_x = local_x * cos_r - local_y * sin_r + centers[:, 0][:, None]
        rot_y = local_x * sin_r + local_y * cos_r + centers[:, 1][:, None]

        return np.stack([rot_x, rot_y], axis=2).astype(np.float32, copy=False)

    def _quad_vertices(
        self,
        centers: np.ndarray,
        sizes: np.ndarray,
        rotations: np.ndarray,
    ) -> np.ndarray:
        sx = sizes[:, 0][:, None]
        sy = sizes[:, 1][:, None]

        local_x = _QUAD_LOCAL[:, 0][None, :] * sx
        local_y = _QUAD_LOCAL[:, 1][None, :] * sy

        cos_r = np.cos(rotations)[:, None]
        sin_r = np.sin(rotations)[:, None]

        rot_x = local_x * cos_r - local_y * sin_r + centers[:, 0][:, None]
        rot_y = local_x * sin_r + local_y * cos_r + centers[:, 1][:, None]

        return np.stack([rot_x, rot_y], axis=2).astype(np.float32, copy=False)

    def triangle_coverage_from_vertices(
        self,
        vertices: np.ndarray,
        softness: float,
    ) -> np.ndarray:
        if softness <= 0.0:
            raise ValueError("softness must be positive.")
        signed = self._convex_signed_distance(vertices[None, :, :])[0]
        return self._sigmoid(signed / float(softness))

    def _ellipse_coverage(
        self,
        centers: np.ndarray,
        sizes: np.ndarray,
        rotations: np.ndarray,
        softness: float,
    ) -> np.ndarray:
        cx = centers[:, 0][:, None, None]
        cy = centers[:, 1][:, None, None]
        sx = np.maximum(sizes[:, 0], 1e-3)[:, None, None]
        sy = np.maximum(sizes[:, 1], 1e-3)[:, None, None]

        dx = self.grid_x[None, :, :] - cx
        dy = self.grid_y[None, :, :] - cy

        cos_r = np.cos(rotations)[:, None, None]
        sin_r = np.sin(rotations)[:, None, None]

        xr = cos_r * dx + sin_r * dy
        yr = -sin_r * dx + cos_r * dy

        radial = np.sqrt((xr / sx) ** 2 + (yr / sy) ** 2 + 1e-8)
        scale = np.minimum(sx, sy)
        signed = (1.0 - radial) * scale
        return self._sigmoid(signed / float(softness))

    def coverage_batch(
        self,
        polygons: LivePolygonBatch,
        softness: float,
        chunk_size: int = 128,
    ) -> np.ndarray:
        if softness <= 0.0:
            raise ValueError("softness must be positive.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        n = polygons.count
        coverage = np.zeros((n, self.height, self.width), dtype=np.float32)
        if n == 0:
            return coverage

        for shape_type in (SHAPE_TRIANGLE, SHAPE_QUAD, SHAPE_ELLIPSE):
            indices = np.where(polygons.shape_types == shape_type)[0]
            if indices.size == 0:
                continue

            for start in range(0, indices.size, chunk_size):
                chunk = indices[start : start + chunk_size]
                centers = polygons.centers[chunk]
                sizes = polygons.sizes[chunk]
                rotations = polygons.rotations[chunk]

                if shape_type == SHAPE_TRIANGLE:
                    vertices = self._triangle_vertices(centers, sizes, rotations)
                    signed = self._convex_signed_distance(vertices)
                    coverage[chunk] = self._sigmoid(signed / float(softness))
                elif shape_type == SHAPE_QUAD:
                    vertices = self._quad_vertices(centers, sizes, rotations)
                    signed = self._convex_signed_distance(vertices)
                    coverage[chunk] = self._sigmoid(signed / float(softness))
                else:
                    coverage[chunk] = self._ellipse_coverage(
                        centers,
                        sizes,
                        rotations,
                        softness,
                    )

        return coverage.astype(np.float32, copy=False)

    def single_coverage(
        self,
        polygons: LivePolygonBatch,
        index: int,
        softness: float,
    ) -> np.ndarray:
        if not (0 <= index < polygons.count):
            raise IndexError("index out of range")
        if softness <= 0.0:
            raise ValueError("softness must be positive.")

        shape_type = int(polygons.shape_types[index])
        center = polygons.centers[index : index + 1]
        size = polygons.sizes[index : index + 1]
        rotation = polygons.rotations[index : index + 1]

        if shape_type == SHAPE_TRIANGLE:
            vertices = self._triangle_vertices(center, size, rotation)
            signed = self._convex_signed_distance(vertices)
            return self._sigmoid(signed[0] / float(softness))

        if shape_type == SHAPE_QUAD:
            vertices = self._quad_vertices(center, size, rotation)
            signed = self._convex_signed_distance(vertices)
            return self._sigmoid(signed[0] / float(softness))

        if shape_type == SHAPE_ELLIPSE:
            return self._ellipse_coverage(center, size, rotation, softness)[0]

        raise ValueError(f"Unsupported shape type code: {shape_type}")

    def single_coverage_from_values(
        self,
        shape_type: int,
        center_x: float,
        center_y: float,
        size_x: float,
        size_y: float,
        rotation: float,
        softness: float,
    ) -> np.ndarray:
        batch = LivePolygonBatch(
            centers=np.array([[center_x, center_y]], dtype=np.float32),
            sizes=np.array([[size_x, size_y]], dtype=np.float32),
            rotations=np.array([rotation], dtype=np.float32),
            colors=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            alphas=np.array([1.0], dtype=np.float32),
            shape_types=np.array([shape_type], dtype=np.int32),
        )
        return self.single_coverage(batch, 0, softness)

    def render(
        self,
        polygons: LivePolygonBatch,
        softness: float,
        base_canvas: np.ndarray | None = None,
    ) -> SoftRenderResult:
        if softness <= 0.0:
            raise ValueError("softness must be positive.")

        n = polygons.count
        if base_canvas is None:
            base = np.ones((self.height, self.width, 3), dtype=np.float32)
        else:
            if base_canvas.shape != (self.height, self.width, 3):
                raise ValueError("base_canvas must have shape (H, W, 3).")
            base = base_canvas.astype(np.float32, copy=False)

        coverage = self.coverage_batch(polygons, softness)
        effective_alpha = coverage * polygons.alphas[:, None, None]

        if n == 0:
            empty = np.zeros((0, self.height, self.width), dtype=np.float32)
            return SoftRenderResult(
                canvas=np.array(base, copy=True),
                coverage=empty,
                effective_alpha=empty,
                trans_before=empty,
            )

        one_minus_alpha = np.clip(1.0 - effective_alpha, 0.0, 1.0).astype(
            np.float32, copy=False
        )
        inclusive_trans = np.cumprod(one_minus_alpha, axis=0, dtype=np.float32)

        trans_before = np.empty_like(one_minus_alpha)
        trans_before[0] = 1.0
        if n > 1:
            trans_before[1:] = inclusive_trans[:-1]

        weights = trans_before * effective_alpha
        layer_rgb = np.einsum("nhw,nc->hwc", weights, polygons.colors, optimize=True)
        canvas = layer_rgb + base * inclusive_trans[-1][:, :, None]
        canvas = np.clip(canvas, 0.0, 1.0).astype(np.float32, copy=False)

        return SoftRenderResult(
            canvas=canvas,
            coverage=coverage,
            effective_alpha=effective_alpha,
            trans_before=trans_before,
        )


def make_random_live_batch(
    *,
    count: int,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> LivePolygonBatch:
    if count < 0:
        raise ValueError("count must be non-negative.")

    centers = np.column_stack(
        [
            rng.uniform(0.0, width - 1.0, size=count),
            rng.uniform(0.0, height - 1.0, size=count),
        ]
    ).astype(np.float32)
    sizes = np.column_stack(
        [
            rng.uniform(2.0, max(width * 0.2, 2.5), size=count),
            rng.uniform(2.0, max(height * 0.2, 2.5), size=count),
        ]
    ).astype(np.float32)
    rotations = rng.uniform(0.0, 2.0 * np.pi, size=count).astype(np.float32)
    colors = rng.uniform(0.0, 1.0, size=(count, 3)).astype(np.float32)
    alphas = rng.uniform(0.1, 0.9, size=count).astype(np.float32)
    shape_types = rng.choice(
        np.array([SHAPE_TRIANGLE, SHAPE_QUAD, SHAPE_ELLIPSE], dtype=np.int32),
        size=count,
        replace=True,
    )

    return LivePolygonBatch(
        centers=centers,
        sizes=sizes,
        rotations=rotations,
        colors=colors,
        alphas=alphas,
        shape_types=shape_types,
    )
