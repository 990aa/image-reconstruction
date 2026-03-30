from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


SHAPE_TRIANGLE = 0
SHAPE_QUAD = 1
SHAPE_ELLIPSE = 2
SHAPE_BEZIER_PATCH = 3
SHAPE_THIN_STROKE = 4
SHAPE_ANNULAR_SEGMENT = 5


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
        if self.rotations.ndim != 1:
            raise ValueError("rotations must have shape (N,).")
        if self.colors.ndim != 2 or self.colors.shape[1] != 3:
            raise ValueError("colors must have shape (N, 3).")
        if self.alphas.ndim != 1:
            raise ValueError("alphas must have shape (N,).")
        if self.shape_types.ndim != 1:
            raise ValueError("shape_types must have shape (N,).")
        if self.shape_params.ndim != 2 or self.shape_params.shape[1] != 6:
            raise ValueError("shape_params must have shape (N, 6).")

        n = self.centers.shape[0]
        if (
            self.sizes.shape[0] != n
            or self.rotations.shape[0] != n
            or self.colors.shape[0] != n
            or self.alphas.shape[0] != n
            or self.shape_types.shape[0] != n
            or self.shape_params.shape[0] != n
        ):
            raise ValueError("All polygon arrays must have identical length N.")

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
    checkpoints: dict[int, np.ndarray] | None = None


@dataclass
class ForwardPassResult:
    canvas: np.ndarray
    checkpoints: dict[int, np.ndarray]
    grad_colors: np.ndarray | None
    grad_alphas: np.ndarray | None


class CoreRenderer:
    def __init__(self, height: int, width: int) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")

        self.height = int(height)
        self.width = int(width)

        yy, xx = np.meshgrid(
            np.arange(self.height, dtype=np.float32),
            np.arange(self.width, dtype=np.float32),
            indexing="ij",
        )
        self.grid_x = xx
        self.grid_y = yy
        self._logged_first_render_softness = False

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -60.0, 60.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)

    def _log_first_softness(self, softness: float, call_name: str) -> None:
        if self._logged_first_render_softness:
            return
        self._logged_first_render_softness = True
        print(f"[core_renderer] first {call_name} softness={float(softness):.4f}")

    def _ellipse_coverage_params(
        self,
        *,
        center_x: float,
        center_y: float,
        axis_x: float,
        axis_y: float,
        rotation: float,
        softness: float,
    ) -> np.ndarray:
        ax = max(float(axis_x), 1e-3)
        ay = max(float(axis_y), 1e-3)

        cos_t = float(np.cos(rotation))
        sin_t = float(np.sin(rotation))

        dx0 = self.grid_x - float(center_x)
        dy0 = self.grid_y - float(center_y)

        dx = dx0 * cos_t + dy0 * sin_t
        dy = -dx0 * sin_t + dy0 * cos_t

        d = np.sqrt((dx / ax) ** 2 + (dy / ay) ** 2 + 1e-8)
        logits = (1.0 - d) / max(float(softness), 1e-6)
        return self._sigmoid(logits)

    def _quad_coverage_params(
        self,
        *,
        center_x: float,
        center_y: float,
        axis_x: float,
        axis_y: float,
        rotation: float,
        softness: float,
    ) -> np.ndarray:
        ax = max(float(axis_x), 1e-3)
        ay = max(float(axis_y), 1e-3)

        cos_t = float(np.cos(rotation))
        sin_t = float(np.sin(rotation))

        dx0 = self.grid_x - float(center_x)
        dy0 = self.grid_y - float(center_y)

        dx = dx0 * cos_t + dy0 * sin_t
        dy = -dx0 * sin_t + dy0 * cos_t

        d = np.maximum(np.abs(dx) / ax, np.abs(dy) / ay)
        logits = (1.0 - d) / max(float(softness), 1e-6)
        return self._sigmoid(logits)

    def _triangle_coverage_params(
        self,
        *,
        center_x: float,
        center_y: float,
        axis_x: float,
        axis_y: float,
        rotation: float,
        softness: float,
    ) -> np.ndarray:
        sx = max(float(axis_x), 1e-3)
        sy = max(float(axis_y), 1e-3)

        local = np.array(
            [
                [sx, 0.0],
                [-0.5 * sx, 0.5 * sy],
                [-0.5 * sx, -0.5 * sy],
            ],
            dtype=np.float32,
        )

        cos_t = float(np.cos(rotation))
        sin_t = float(np.sin(rotation))
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        verts = local @ rot.T
        verts[:, 0] += float(center_x)
        verts[:, 1] += float(center_y)
        return self.triangle_coverage_from_vertices(verts, softness)

    def triangle_coverage_from_vertices(
        self,
        vertices: np.ndarray,
        softness: float,
    ) -> np.ndarray:
        if vertices.shape != (3, 2):
            raise ValueError("vertices must have shape (3, 2)")
        if softness <= 0.0:
            raise ValueError("softness must be positive")

        x = self.grid_x
        y = self.grid_y

        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]

        def _signed_dist(ax: float, ay: float, bx: float, by: float) -> np.ndarray:
            ex = bx - ax
            ey = by - ay
            nx = -ey
            ny = ex
            norm = np.sqrt(nx * nx + ny * ny + 1e-8)
            return ((x - ax) * nx + (y - ay) * ny) / norm

        d1 = _signed_dist(float(x1), float(y1), float(x2), float(y2))
        d2 = _signed_dist(float(x2), float(y2), float(x3), float(y3))
        d3 = _signed_dist(float(x3), float(y3), float(x1), float(y1))

        signed = np.minimum(np.minimum(d1, d2), d3)
        return self._sigmoid(signed / float(softness))

    def _thin_stroke_coverage_params(
        self,
        *,
        center_x: float,
        center_y: float,
        shape_params: np.ndarray,
        softness: float,
    ) -> np.ndarray:
        x0 = float(center_x)
        y0 = float(center_y)
        x1 = float(shape_params[0])
        y1 = float(shape_params[1])
        width = max(float(shape_params[2]), 1e-3)

        dx = x1 - x0
        dy = y1 - y0
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq <= 1e-8:
            dist = np.sqrt(
                (self.grid_x - x0) * (self.grid_x - x0)
                + (self.grid_y - y0) * (self.grid_y - y0)
                + 1e-8
            )
        else:
            t = (
                ((self.grid_x - x0) * dx) + ((self.grid_y - y0) * dy)
            ) / float(seg_len_sq)
            t = np.clip(t, 0.0, 1.0)
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            dist = np.sqrt(
                (self.grid_x - proj_x) * (self.grid_x - proj_x)
                + (self.grid_y - proj_y) * (self.grid_y - proj_y)
                + 1e-8
            )

        half_width = 0.5 * width
        logits = (half_width - dist) / max(float(softness), 1e-6)
        return self._sigmoid(logits)

    def single_coverage(self, polygons: LivePolygonBatch, index: int, softness: float) -> np.ndarray:
        idx = int(index)
        if idx < 0 or idx >= polygons.count:
            raise IndexError("polygon index out of bounds")

        shape_type = int(polygons.shape_types[idx])
        cx = float(polygons.centers[idx, 0])
        cy = float(polygons.centers[idx, 1])
        sx = float(polygons.sizes[idx, 0])
        sy = float(polygons.sizes[idx, 1])
        rot = float(polygons.rotations[idx])

        if shape_type == SHAPE_QUAD:
            return self._quad_coverage_params(
                center_x=cx,
                center_y=cy,
                axis_x=sx,
                axis_y=sy,
                rotation=rot,
                softness=softness,
            )

        if shape_type == SHAPE_TRIANGLE:
            return self._triangle_coverage_params(
                center_x=cx,
                center_y=cy,
                axis_x=sx,
                axis_y=sy,
                rotation=rot,
                softness=softness,
            )

        if shape_type == SHAPE_THIN_STROKE:
            return self._thin_stroke_coverage_params(
                center_x=cx,
                center_y=cy,
                shape_params=polygons.shape_params[idx],
                softness=softness,
            )

        return self._ellipse_coverage_params(
            center_x=cx,
            center_y=cy,
            axis_x=sx,
            axis_y=sy,
            rotation=rot,
            softness=softness,
        )

    def coverage_batch(self, polygons: LivePolygonBatch, softness: float, chunk_size: int = 50) -> np.ndarray:
        n = polygons.count
        out = np.zeros((n, self.height, self.width), dtype=np.float32)
        if n == 0:
            return out

        step = max(1, int(chunk_size))
        for start in range(0, n, step):
            end = min(start + step, n)
            for idx in range(start, end):
                out[idx] = self.single_coverage(polygons, idx, softness)
        return out

    def render(
        self,
        polygons: LivePolygonBatch,
        softness: float,
        chunk_size: int = 50,
    ) -> SoftRenderResult:
        self._log_first_softness(softness, "render")
        n = polygons.count
        cov = self.coverage_batch(polygons, softness=softness, chunk_size=chunk_size)

        canvas = np.ones((self.height, self.width, 3), dtype=np.float32)

        effective_alpha = np.zeros((n, self.height, self.width), dtype=np.float32)
        trans_after = np.zeros((n, self.height, self.width), dtype=np.float32)

        for idx in range(n):
            ea = cov[idx] * float(np.clip(polygons.alphas[idx], 0.0, 1.0))
            effective_alpha[idx] = ea

        trans_suffix = np.ones((self.height, self.width), dtype=np.float32)
        for idx in range(n - 1, -1, -1):
            trans_after[idx] = trans_suffix
            trans_suffix = trans_suffix * (1.0 - effective_alpha[idx])

        weights = effective_alpha * trans_after

        for idx in range(n):
            ea = effective_alpha[idx]
            color = polygons.colors[idx][None, None, :]
            canvas = canvas * (1.0 - ea[:, :, None]) + color * ea[:, :, None]

        return SoftRenderResult(
            canvas=canvas.astype(np.float32, copy=False),
            coverage=cov,
            effective_alpha=effective_alpha,
            trans_after=trans_after,
            weights=weights,
            checkpoints=None,
        )

    def forward_pass(
        self,
        polygons: LivePolygonBatch,
        *,
        softness: float,
        chunk_size: int = 50,
        checkpoint_stride: int = 10,
        target: np.ndarray | None = None,
        compute_gradients: bool = False,
    ) -> ForwardPassResult:
        self._log_first_softness(softness, "forward_pass")
        n = polygons.count
        canvas = np.ones((self.height, self.width, 3), dtype=np.float32)
        checkpoints: dict[int, np.ndarray] = {0: np.array(canvas, copy=True)}

        grad_colors: np.ndarray | None = None
        grad_alphas: np.ndarray | None = None
        effective_alpha = np.zeros((n, self.height, self.width), dtype=np.float32)

        if compute_gradients:
            if target is None:
                raise ValueError("target is required when compute_gradients=True")
            grad_colors = np.zeros((n, 3), dtype=np.float32)
            grad_alphas = np.zeros((n,), dtype=np.float32)

        step = max(1, int(chunk_size))
        for start in range(0, n, step):
            end = min(start + step, n)
            cov_chunk = np.zeros((end - start, self.height, self.width), dtype=np.float32)
            for local in range(end - start):
                idx = start + local
                cov_chunk[local] = self.single_coverage(polygons, idx, softness)

            for local in range(end - start):
                idx = start + local
                cov = cov_chunk[local]
                alpha = float(np.clip(polygons.alphas[idx], 0.0, 1.0))
                color = polygons.colors[idx][None, None, :]

                canvas_before = canvas
                ea = cov * alpha
                effective_alpha[idx] = ea
                canvas = canvas_before * (1.0 - ea[:, :, None]) + color * ea[:, :, None]

                if compute_gradients and target is not None and grad_alphas is not None:
                    residual = canvas - target
                    scale = 2.0 / float(target.size)
                    grad_alphas[idx] = float(
                        scale
                        * np.sum(
                            cov[:, :, None]
                            * (color - canvas_before)
                            * residual,
                            dtype=np.float32,
                        )
                    )

                if (idx + 1) % max(1, int(checkpoint_stride)) == 0 or (idx + 1) == n:
                    checkpoints[idx + 1] = np.array(canvas, copy=True)

        if compute_gradients and target is not None and grad_colors is not None:
            residual_final = canvas - target
            trans_after = np.zeros_like(effective_alpha)
            trans_suffix = np.ones((self.height, self.width), dtype=np.float32)
            for idx in range(n - 1, -1, -1):
                trans_after[idx] = trans_suffix
                trans_suffix = trans_suffix * (1.0 - effective_alpha[idx])

            weights = trans_after * effective_alpha
            grad_colors[:] = (
                (2.0 / float(target.size))
                * np.einsum("nhw,hwc->nc", weights, residual_final, dtype=np.float32)
            ).astype(np.float32, copy=False)

        return ForwardPassResult(
            canvas=canvas.astype(np.float32, copy=False),
            checkpoints=checkpoints,
            grad_colors=grad_colors,
            grad_alphas=grad_alphas,
        )

    def render_suffix(
        self,
        polygons: LivePolygonBatch,
        *,
        start_index: int,
        base_canvas: np.ndarray,
        softness: float,
        override_index: int | None = None,
        override_center: tuple[float, float] | None = None,
        override_size: tuple[float, float] | None = None,
        override_rotation: float | None = None,
    ) -> np.ndarray:
        n = polygons.count
        start = int(np.clip(start_index, 0, n))
        canvas = np.array(base_canvas, copy=True).astype(np.float32, copy=False)

        for idx in range(start, n):
            cx = float(polygons.centers[idx, 0])
            cy = float(polygons.centers[idx, 1])
            sx = float(polygons.sizes[idx, 0])
            sy = float(polygons.sizes[idx, 1])
            rot = float(polygons.rotations[idx])

            if override_index is not None and idx == int(override_index):
                if override_center is not None:
                    cx = float(override_center[0])
                    cy = float(override_center[1])
                if override_size is not None:
                    sx = float(override_size[0])
                    sy = float(override_size[1])
                if override_rotation is not None:
                    rot = float(override_rotation)

            shape_type = int(polygons.shape_types[idx])
            if shape_type == SHAPE_QUAD:
                cov = self._quad_coverage_params(
                    center_x=cx,
                    center_y=cy,
                    axis_x=sx,
                    axis_y=sy,
                    rotation=rot,
                    softness=softness,
                )
            elif shape_type == SHAPE_TRIANGLE:
                cov = self._triangle_coverage_params(
                    center_x=cx,
                    center_y=cy,
                    axis_x=sx,
                    axis_y=sy,
                    rotation=rot,
                    softness=softness,
                )
            elif shape_type == SHAPE_THIN_STROKE:
                cov = self._thin_stroke_coverage_params(
                    center_x=cx,
                    center_y=cy,
                    shape_params=polygons.shape_params[idx],
                    softness=softness,
                )
            else:
                cov = self._ellipse_coverage_params(
                    center_x=cx,
                    center_y=cy,
                    axis_x=sx,
                    axis_y=sy,
                    rotation=rot,
                    softness=softness,
                )

            alpha = float(np.clip(polygons.alphas[idx], 0.0, 1.0))
            ea = cov * alpha
            color = polygons.colors[idx][None, None, :]
            canvas = canvas * (1.0 - ea[:, :, None]) + color * ea[:, :, None]

        return canvas.astype(np.float32, copy=False)


SoftRasterizer = CoreRenderer


def make_random_live_batch(
    *,
    count: int,
    height: int,
    width: int,
    rng: np.random.Generator | None = None,
) -> LivePolygonBatch:
    if count < 0:
        raise ValueError("count must be non-negative")

    generator = np.random.default_rng() if rng is None else rng

    centers = np.column_stack(
        [
            generator.uniform(0.0, max(width - 1, 1), size=count),
            generator.uniform(0.0, max(height - 1, 1), size=count),
        ]
    ).astype(np.float32)

    sizes = np.column_stack(
        [
            generator.uniform(2.0, max(width * 0.10, 3.0), size=count),
            generator.uniform(2.0, max(height * 0.10, 3.0), size=count),
        ]
    ).astype(np.float32)

    rotations = generator.uniform(-np.pi, np.pi, size=count).astype(np.float32)
    colors = generator.uniform(0.0, 1.0, size=(count, 3)).astype(np.float32)
    alphas = generator.uniform(0.25, 0.95, size=count).astype(np.float32)
    shape_types = generator.choice(
        np.array([SHAPE_ELLIPSE, SHAPE_QUAD], dtype=np.int32), size=count
    ).astype(np.int32)

    return LivePolygonBatch(
        centers=centers,
        sizes=sizes,
        rotations=rotations,
        colors=colors,
        alphas=alphas,
        shape_types=shape_types,
        shape_params=np.zeros((count, 6), dtype=np.float32),
    )
