from __future__ import annotations

from src.core_renderer import (
    SHAPE_ANNULAR_SEGMENT,
    SHAPE_BEZIER_PATCH,
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
    LivePolygonBatch,
    SoftRenderResult,
    SoftRasterizer,
    make_random_live_batch,
)

SoftRenderer = SoftRasterizer

__all__ = [
    "SHAPE_TRIANGLE",
    "SHAPE_QUAD",
    "SHAPE_ELLIPSE",
    "SHAPE_BEZIER_PATCH",
    "SHAPE_THIN_STROKE",
    "SHAPE_ANNULAR_SEGMENT",
    "LivePolygonBatch",
    "SoftRenderResult",
    "SoftRenderer",
    "SoftRasterizer",
    "make_random_live_batch",
]
