from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from evolutionary_art_gpu.constants import (
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
)
from evolutionary_art_gpu.models import LivePolygonBatch


def save_rgb_image(path: str | Path, image: np.ndarray) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.round(np.clip(image, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(out_path)


def export_svg(
    batch: LivePolygonBatch,
    *,
    width: int,
    height: int,
    background_color: np.ndarray,
    filename: str | Path,
) -> Path:
    svg_path = Path(filename)
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    bg_r, bg_g, bg_b = [int(np.clip(c, 0.0, 1.0) * 255) for c in background_color]

    with svg_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            'width="100%" height="100%">\n'
        )
        handle.write(
            f'  <rect width="100%" height="100%" fill="rgb({bg_r},{bg_g},{bg_b})" />\n'
        )

        for idx in range(batch.count):
            cx, cy = batch.centers[idx]
            sx, sy = batch.sizes[idx]
            rot = batch.rotations[idx]
            rot_deg = math.degrees(float(rot))
            r, g, b = [int(np.clip(c, 0.0, 1.0) * 255) for c in batch.colors[idx]]
            a = float(np.clip(batch.alphas[idx], 0.0, 1.0))
            shape_type = int(batch.shape_types[idx])
            color_str = f"rgba({r},{g},{b},{a:.3f})"

            if shape_type == SHAPE_ELLIPSE:
                handle.write(
                    f'  <ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{sx:.2f}" ry="{sy:.2f}" '
                    f'fill="{color_str}" transform="rotate({rot_deg:.2f} {cx:.2f} {cy:.2f})" />\n'
                )
            elif shape_type == SHAPE_QUAD:
                handle.write(
                    f'  <rect x="{cx - sx:.2f}" y="{cy - sy:.2f}" width="{2 * sx:.2f}" height="{2 * sy:.2f}" '
                    f'fill="{color_str}" transform="rotate({rot_deg:.2f} {cx:.2f} {cy:.2f})" />\n'
                )
            elif shape_type == SHAPE_TRIANGLE:
                points = f"{sx:.2f},0 {-0.5 * sx:.2f},{0.5 * sy:.2f} {-0.5 * sx:.2f},{-0.5 * sy:.2f}"
                handle.write(
                    f'  <polygon points="{points}" fill="{color_str}" '
                    f'transform="translate({cx:.2f} {cy:.2f}) rotate({rot_deg:.2f})" />\n'
                )
            elif shape_type == SHAPE_THIN_STROKE:
                x1 = float(batch.shape_params[idx][0])
                y1 = float(batch.shape_params[idx][1])
                stroke_width = float(batch.shape_params[idx][2])
                handle.write(
                    f'  <line x1="{cx:.2f}" y1="{cy:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                    f'stroke="{color_str}" stroke-width="{stroke_width:.2f}" stroke-linecap="round" />\n'
                )

        handle.write("</svg>\n")

    return svg_path
