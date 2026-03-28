import numpy as np
from skimage.draw import ellipse as draw_ellipse
from skimage.draw import polygon as draw_polygon

from src.polygon import Polygon, ShapeType


def render_polygon(canvas: np.ndarray, polygon: Polygon) -> np.ndarray:
    """Render a shape onto a copy of the canvas using alpha blending."""
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("Canvas must have shape (H, W, 3).")

    h, w, _ = canvas.shape
    new_canvas = np.array(canvas, copy=True)
    mask = np.zeros((h, w), dtype=bool)

    if polygon.shape_type in (ShapeType.TRIANGLE, ShapeType.QUADRILATERAL):
        if len(polygon.vertices) < 3:
            raise ValueError("Polygon needs at least 3 vertices.")

        ys = np.array([v[1] for v in polygon.vertices], dtype=np.float32)
        xs = np.array([v[0] for v in polygon.vertices], dtype=np.float32)
        rr, cc = draw_polygon(ys, xs, shape=(h, w))
        mask[rr, cc] = True

    elif polygon.shape_type == ShapeType.ELLIPSE:
        if polygon.ellipse_center is None or polygon.ellipse_axes is None:
            raise ValueError("Ellipse polygons must define center and axes.")

        cx, cy = polygon.ellipse_center
        semi_major, semi_minor = polygon.ellipse_axes
        rr, cc = draw_ellipse(
            cy,
            cx,
            semi_minor,
            semi_major,
            rotation=float(polygon.ellipse_rotation),
            shape=(h, w),
        )
        mask[rr, cc] = True

    else:
        raise ValueError(f"Unsupported shape type: {polygon.shape_type}")

    alpha = float(np.clip(polygon.alpha, 0.0, 1.0))
    color = np.asarray(polygon.color, dtype=np.float32)
    new_canvas[mask] = alpha * color + (1.0 - alpha) * new_canvas[mask]
    return new_canvas.astype(np.float32, copy=False)
