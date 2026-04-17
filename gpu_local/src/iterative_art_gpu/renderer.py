from __future__ import annotations

import math

import numpy as np
import torch


class GPUCoreRenderer:
    """Torch coverage rasterizer used by the notebook-native GPU optimizer."""

    def __init__(self, height: int, width: int) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")

        self.height = int(height)
        self.width = int(width)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        yy, xx = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32, device=self.device),
            torch.arange(self.width, dtype=torch.float32, device=self.device),
            indexing="ij",
        )
        self.grid_x = xx
        self.grid_y = yy

    @staticmethod
    def _sigmoid(values: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(values, -60.0, 60.0)
        return 1.0 / (1.0 + torch.exp(-clipped))

    def _ellipse_coverage_params(
        self,
        center_x: float,
        center_y: float,
        axis_x: float,
        axis_y: float,
        rotation: float,
        softness: float,
    ) -> torch.Tensor:
        ax = max(float(axis_x), 1e-3)
        ay = max(float(axis_y), 1e-3)
        cos_t = math.cos(rotation)
        sin_t = math.sin(rotation)

        dx0 = self.grid_x - float(center_x)
        dy0 = self.grid_y - float(center_y)
        dx = dx0 * cos_t + dy0 * sin_t
        dy = -dx0 * sin_t + dy0 * cos_t

        d = torch.sqrt((dx / ax) ** 2 + (dy / ay) ** 2 + 1e-8)
        return self._sigmoid((1.0 - d) / max(float(softness), 1e-6))

    def _quad_coverage_params(
        self,
        center_x: float,
        center_y: float,
        axis_x: float,
        axis_y: float,
        rotation: float,
        softness: float,
    ) -> torch.Tensor:
        ax = max(float(axis_x), 1e-3)
        ay = max(float(axis_y), 1e-3)
        cos_t = math.cos(rotation)
        sin_t = math.sin(rotation)

        dx0 = self.grid_x - float(center_x)
        dy0 = self.grid_y - float(center_y)
        dx = dx0 * cos_t + dy0 * sin_t
        dy = -dx0 * sin_t + dy0 * cos_t

        d = torch.maximum(torch.abs(dx) / ax, torch.abs(dy) / ay)
        return self._sigmoid((1.0 - d) / max(float(softness), 1e-6))

    def _triangle_coverage_params(
        self,
        center_x: float,
        center_y: float,
        axis_x: float,
        axis_y: float,
        rotation: float,
        softness: float,
    ) -> torch.Tensor:
        sx = max(float(axis_x), 1e-3)
        sy = max(float(axis_y), 1e-3)
        local = torch.tensor(
            [[sx, 0.0], [-0.5 * sx, 0.5 * sy], [-0.5 * sx, -0.5 * sy]],
            dtype=torch.float32,
            device=self.device,
        )
        cos_t = math.cos(rotation)
        sin_t = math.sin(rotation)
        rot = torch.tensor(
            [[cos_t, -sin_t], [sin_t, cos_t]],
            dtype=torch.float32,
            device=self.device,
        )
        verts = torch.matmul(local, rot.T)
        verts[:, 0] += float(center_x)
        verts[:, 1] += float(center_y)

        x1, y1 = verts[0, 0], verts[0, 1]
        x2, y2 = verts[1, 0], verts[1, 1]
        x3, y3 = verts[2, 0], verts[2, 1]

        def _signed_dist(ax: float, ay: float, bx: float, by: float) -> torch.Tensor:
            ex = bx - ax
            ey = by - ay
            nx = -ey
            ny = ex
            norm = math.sqrt(nx * nx + ny * ny + 1e-8)
            return ((self.grid_x - ax) * nx + (self.grid_y - ay) * ny) / norm

        d1 = _signed_dist(float(x1), float(y1), float(x2), float(y2))
        d2 = _signed_dist(float(x2), float(y2), float(x3), float(y3))
        d3 = _signed_dist(float(x3), float(y3), float(x1), float(y1))
        signed = torch.minimum(torch.minimum(d1, d2), d3)
        return self._sigmoid(signed / max(float(softness), 1e-6))

    def _thin_stroke_coverage_params(
        self,
        center_x: float,
        center_y: float,
        shape_params: np.ndarray,
        softness: float,
    ) -> torch.Tensor:
        x0 = float(center_x)
        y0 = float(center_y)
        x1 = float(shape_params[0])
        y1 = float(shape_params[1])
        width = max(float(shape_params[2]), 1e-3)

        dx = x1 - x0
        dy = y1 - y0
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq <= 1e-8:
            dist = torch.sqrt((self.grid_x - x0) ** 2 + (self.grid_y - y0) ** 2 + 1e-8)
        else:
            t = ((self.grid_x - x0) * dx + (self.grid_y - y0) * dy) / seg_len_sq
            t = torch.clamp(t, 0.0, 1.0)
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            dist = torch.sqrt(
                (self.grid_x - proj_x) ** 2 + (self.grid_y - proj_y) ** 2 + 1e-8
            )

        return self._sigmoid((0.5 * width - dist) / max(float(softness), 1e-6))
