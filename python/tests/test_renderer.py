import numpy as np

from src.canvas import create_white_canvas
from src.polygon import Polygon, ShapeType
from src.renderer import render_polygon


def test_triangle_render_changes_center_and_not_corner() -> None:
    canvas = create_white_canvas()
    triangle = Polygon(
        vertices=[(40, 40), (60, 40), (50, 60)],
        color=(1.0, 0.0, 0.0),
        alpha=1.0,
        shape_type=ShapeType.TRIANGLE,
    )

    rendered = render_polygon(canvas, triangle)

    assert np.allclose(rendered[50, 50], np.array([1.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(rendered[5, 5], np.array([1.0, 1.0, 1.0], dtype=np.float32), atol=1e-6)


def test_ellipse_render_changes_center_and_not_corner() -> None:
    canvas = create_white_canvas()
    ellipse = Polygon(
        vertices=[(38, 50), (50, 42), (62, 50), (50, 58)],
        color=(0.0, 0.0, 1.0),
        alpha=1.0,
        shape_type=ShapeType.ELLIPSE,
        ellipse_center=(50, 50),
        ellipse_axes=(12, 8),
        ellipse_rotation=0.0,
    )

    rendered = render_polygon(canvas, ellipse)

    assert np.allclose(rendered[50, 50], np.array([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(rendered[5, 5], np.array([1.0, 1.0, 1.0], dtype=np.float32), atol=1e-6)
