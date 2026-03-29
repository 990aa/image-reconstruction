from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


SHAPE_TRIANGLE = 0
SHAPE_QUAD = 1
SHAPE_ELLIPSE = 2
SHAPE_BEZIER_PATCH = 3
SHAPE_THIN_STROKE = 4
SHAPE_ANNULAR_SEGMENT = 5


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
    shape_params: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 6), dtype=np.float32)
    )

    def __post_init__(self) -> None:
        self.centers = np.ascontiguousarray(self.centers, dtype=np.float32)
        self.sizes = np.ascontiguousarray(self.sizes, dtype=np.float32)
        self.rotations = np.ascontiguousarray(self.rotations, dtype=np.float32)
        self.colors = np.ascontiguousarray(self.colors, dtype=np.float32)
        self.alphas = np.ascontiguousarray(self.alphas, dtype=np.float32)
        self.shape_types = np.ascontiguousarray(self.shape_types, dtype=np.int32)
        if self.shape_params.size == 0 and self.centers.shape[0] > 0:
            self.shape_params = np.zeros((self.centers.shape[0], 6), dtype=np.float32)
        self.shape_params = np.ascontiguousarray(self.shape_params, dtype=np.float32)

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
        if self.shape_params.ndim != 2 or self.shape_params.shape[1] != 6:
            raise ValueError("shape_params must have shape (N, 6).")

        count = self.centers.shape[0]
        if (
            self.sizes.shape[0] != count
            or self.colors.shape[0] != count
            or self.rotations.shape[0] != count
            or self.alphas.shape[0] != count
            or self.shape_types.shape[0] != count
            or self.shape_params.shape[0] != count
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
            shape_params=np.array(self.shape_params, copy=True),
        )


@dataclass
class SoftRenderResult:
    canvas: np.ndarray
    coverage: np.ndarray
    effective_alpha: np.ndarray
    trans_after: np.ndarray
    weights: np.ndarray


class SoftRenderer:
    """Minimal soft renderer with correct Porter-Duff compositing weights."""

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
        self.grid_x = x_coords
        self.grid_y = y_coords

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -50.0, 50.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)

    def _convex_signed_distance(self, vertices: np.ndarray) -> np.ndarray:
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

    def _thin_stroke_coverage(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        widths: np.ndarray,
        softness: float,
    ) -> np.ndarray:
        ax = starts[:, 0][:, None, None]
        ay = starts[:, 1][:, None, None]
        bx = ends[:, 0][:, None, None]
        by = ends[:, 1][:, None, None]
        w = np.maximum(widths, 0.5)[:, None, None]

        px = self.grid_x[None, :, :]
        py = self.grid_y[None, :, :]

        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay

        ab_len2 = np.maximum(abx * abx + aby * aby, 1e-6)
        t = np.clip((apx * abx + apy * aby) / ab_len2, 0.0, 1.0)
        cx = ax + t * abx
        cy = ay + t * aby

        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + 1e-8)
        signed = (w * 0.5) - dist
        return self._sigmoid(signed / float(softness))

    def _annular_segment_coverage(
        self,
        centers: np.ndarray,
        sizes: np.ndarray,
        angle_ranges: np.ndarray,
        softness: float,
    ) -> np.ndarray:
        cx = centers[:, 0][:, None, None]
        cy = centers[:, 1][:, None, None]
        inner_r = np.maximum(np.minimum(sizes[:, 0], sizes[:, 1]), 0.5)[:, None, None]
        outer_r = np.maximum(np.maximum(sizes[:, 0], sizes[:, 1]), 0.75)[:, None, None]

        dx = self.grid_x[None, :, :] - cx
        dy = self.grid_y[None, :, :] - cy
        dist = np.sqrt(dx * dx + dy * dy + 1e-8)

        inner_signed = dist - inner_r
        outer_signed = outer_r - dist
        radial_signed = np.minimum(inner_signed, outer_signed)

        ang = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
        start = np.mod(angle_ranges[:, 0], 2.0 * np.pi)[:, None, None]
        end = np.mod(angle_ranges[:, 1], 2.0 * np.pi)[:, None, None]

        sector_mask = np.where(
            start <= end, (ang >= start) & (ang <= end), (ang >= start) | (ang <= end)
        )
        radial_cov = self._sigmoid(radial_signed / float(softness))
        return (radial_cov * sector_mask.astype(np.float32)).astype(
            np.float32, copy=False
        )

    def _bezier_patch_coverage(
        self,
        centers: np.ndarray,
        sizes: np.ndarray,
        rotations: np.ndarray,
        curvatures: np.ndarray,
        softness: float,
    ) -> np.ndarray:
        base = self._quad_vertices(centers, sizes, rotations)
        k = base.shape[0]
        samples_per_edge = 10
        boundary = np.zeros((k, 4 * samples_per_edge, 2), dtype=np.float32)

        for edge_idx in range(4):
            start_v = base[:, edge_idx]
            end_v = base[:, (edge_idx + 1) % 4]
            mid = 0.5 * (start_v + end_v)

            edge = end_v - start_v
            edge_norm = np.sqrt(np.sum(edge * edge, axis=1, keepdims=True) + 1e-8)
            tangent = edge / edge_norm
            normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

            curvature = curvatures[:, edge_idx : edge_idx + 1]
            control = mid + normal * curvature

            t = np.linspace(0.0, 1.0, samples_per_edge, dtype=np.float32)[None, :, None]
            omt = 1.0 - t
            pts = (
                omt * omt * start_v[:, None, :]
                + 2.0 * omt * t * control[:, None, :]
                + t * t * end_v[:, None, :]
            )

            start_i = edge_idx * samples_per_edge
            end_i = start_i + samples_per_edge
            boundary[:, start_i:end_i, :] = pts

        signed = self._convex_signed_distance(boundary)
        return self._sigmoid(signed / float(softness))

    def coverage_batch(
        self,
        polygons: LivePolygonBatch,
        softness: float,
        chunk_size: int = 50,
    ) -> np.ndarray:
        del chunk_size

        if softness <= 0.0:
            raise ValueError("softness must be positive.")

        n = polygons.count
        coverage = np.zeros((n, self.height, self.width), dtype=np.float32)
        if n == 0:
            return coverage

        shape_types = polygons.shape_types
        centers = polygons.centers
        sizes = polygons.sizes
        rotations = polygons.rotations
        params = polygons.shape_params

        tri_idx = np.where(shape_types == SHAPE_TRIANGLE)[0]
        if tri_idx.size > 0:
            tri_vertices = self._triangle_vertices(
                centers[tri_idx],
                sizes[tri_idx],
                rotations[tri_idx],
            )
            tri_signed = self._convex_signed_distance(tri_vertices)
            coverage[tri_idx] = self._sigmoid(tri_signed / float(softness))

        quad_idx = np.where(shape_types == SHAPE_QUAD)[0]
        if quad_idx.size > 0:
            quad_vertices = self._quad_vertices(
                centers[quad_idx],
                sizes[quad_idx],
                rotations[quad_idx],
            )
            quad_signed = self._convex_signed_distance(quad_vertices)
            coverage[quad_idx] = self._sigmoid(quad_signed / float(softness))

        ellipse_idx = np.where(shape_types == SHAPE_ELLIPSE)[0]
        if ellipse_idx.size > 0:
            coverage[ellipse_idx] = self._ellipse_coverage(
                centers[ellipse_idx],
                sizes[ellipse_idx],
                rotations[ellipse_idx],
                softness,
            )

        bezier_idx = np.where(shape_types == SHAPE_BEZIER_PATCH)[0]
        if bezier_idx.size > 0:
            edge_curvatures = params[bezier_idx, :4]
            max_curve = np.minimum(sizes[bezier_idx, 0], sizes[bezier_idx, 1]) * 0.35
            edge_curvatures = np.clip(
                edge_curvatures,
                -max_curve[:, None],
                max_curve[:, None],
            )
            coverage[bezier_idx] = self._bezier_patch_coverage(
                centers[bezier_idx],
                sizes[bezier_idx],
                rotations[bezier_idx],
                edge_curvatures,
                softness,
            )

        stroke_idx = np.where(shape_types == SHAPE_THIN_STROKE)[0]
        if stroke_idx.size > 0:
            stroke_ends = params[stroke_idx, :2]
            stroke_width = params[stroke_idx, 2]
            coverage[stroke_idx] = self._thin_stroke_coverage(
                centers[stroke_idx],
                stroke_ends,
                stroke_width,
                softness,
            )

        annular_idx = np.where(shape_types == SHAPE_ANNULAR_SEGMENT)[0]
        if annular_idx.size > 0:
            angle_ranges = params[annular_idx, :2]
            coverage[annular_idx] = self._annular_segment_coverage(
                centers[annular_idx],
                sizes[annular_idx],
                angle_ranges,
                softness,
            )

        return coverage

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
        params = polygons.shape_params[index : index + 1]

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

        if shape_type == SHAPE_BEZIER_PATCH:
            edge_curvatures = np.clip(
                params[:, :4],
                -np.minimum(size[:, 0], size[:, 1])[:, None] * 0.35,
                np.minimum(size[:, 0], size[:, 1])[:, None] * 0.35,
            )
            return self._bezier_patch_coverage(
                center,
                size,
                rotation,
                edge_curvatures,
                softness,
            )[0]

        if shape_type == SHAPE_THIN_STROKE:
            return self._thin_stroke_coverage(
                center,
                params[:, :2],
                params[:, 2],
                softness,
            )[0]

        if shape_type == SHAPE_ANNULAR_SEGMENT:
            return self._annular_segment_coverage(
                center,
                size,
                params[:, :2],
                softness,
            )[0]

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
        shape_params: np.ndarray | None = None,
    ) -> np.ndarray:
        if shape_params is None:
            shape_params = np.zeros((6,), dtype=np.float32)
        params = np.asarray(shape_params, dtype=np.float32).reshape(6)
        batch = LivePolygonBatch(
            centers=np.array([[center_x, center_y]], dtype=np.float32),
            sizes=np.array([[size_x, size_y]], dtype=np.float32),
            rotations=np.array([rotation], dtype=np.float32),
            colors=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            alphas=np.array([1.0], dtype=np.float32),
            shape_types=np.array([shape_type], dtype=np.int32),
            shape_params=params.reshape(1, 6),
        )
        return self.single_coverage(batch, 0, softness)

    def render(
        self,
        polygons: LivePolygonBatch,
        softness: float,
        base_canvas: np.ndarray | None = None,
        chunk_size: int = 50,
    ) -> SoftRenderResult:
        del chunk_size

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
                trans_after=empty,
                weights=empty,
            )

        one_minus_alpha = np.clip(1.0 - effective_alpha, 0.0, 1.0).astype(
            np.float32, copy=False
        )

        reverse_cumprod = np.cumprod(one_minus_alpha[::-1], axis=0, dtype=np.float32)
        inclusive_trans = reverse_cumprod[::-1]

        trans_after = np.empty_like(one_minus_alpha)
        trans_after[-1] = 1.0
        if n > 1:
            trans_after[:-1] = inclusive_trans[1:]

        weights = (effective_alpha * trans_after).astype(np.float32, copy=False)
        layer_rgb = np.einsum("nhw,nc->hwc", weights, polygons.colors, optimize=True)
        canvas = layer_rgb + base * inclusive_trans[0][:, :, None]
        canvas = np.clip(canvas, 0.0, 1.0).astype(np.float32, copy=False)

        return SoftRenderResult(
            canvas=canvas,
            coverage=coverage,
            effective_alpha=effective_alpha,
            trans_after=trans_after,
            weights=weights,
        )


# Backward-compatible alias used throughout existing code.
SoftRasterizer = SoftRenderer


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
    shape_params = np.zeros((count, 6), dtype=np.float32)

    return LivePolygonBatch(
        centers=centers,
        sizes=sizes,
        rotations=rotations,
        colors=colors,
        alphas=alphas,
        shape_types=shape_types,
        shape_params=shape_params,
    )
