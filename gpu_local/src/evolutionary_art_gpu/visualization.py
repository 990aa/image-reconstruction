from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from evolutionary_art_gpu.constants import (
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
)
from evolutionary_art_gpu.models import LivePolygonBatch


def plot_3d_exploded_view(
    batch: LivePolygonBatch,
    *,
    width: int,
    height: int,
):
    """Render a notebook-style exploded layer mesh with Plotly."""

    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []
    all_i: list[int] = []
    all_j: list[int] = []
    all_k: list[int] = []
    facecolors: list[str] = []
    vert_offset = 0

    for idx in range(batch.count):
        shape_type = int(batch.shape_types[idx])
        cx, cy = batch.centers[idx]
        sx, sy = batch.sizes[idx]
        rot = float(batch.rotations[idx])
        z_val = float(idx)

        r, g, b = [int(np.clip(c, 0.0, 1.0) * 255) for c in batch.colors[idx]]
        a = float(np.clip(batch.alphas[idx], 0.15, 1.0))
        color_str = f"rgba({r},{g},{b},{a:.3f})"

        verts: list[tuple[float, float]] = []
        abs_verts: list[tuple[float, float]] = []

        if shape_type == SHAPE_ELLIPSE:
            num_pts = 16
            for p in range(num_pts):
                t = 2.0 * math.pi * p / num_pts
                verts.append((float(sx * math.cos(t)), float(sy * math.sin(t))))
        elif shape_type == SHAPE_QUAD:
            verts = [
                (float(sx), float(sy)),
                (-float(sx), float(sy)),
                (-float(sx), -float(sy)),
                (float(sx), -float(sy)),
            ]
        elif shape_type == SHAPE_TRIANGLE:
            verts = [
                (float(sx), 0.0),
                (-0.5 * float(sx), 0.5 * float(sy)),
                (-0.5 * float(sx), -0.5 * float(sy)),
            ]
        elif shape_type == SHAPE_THIN_STROKE:
            x1 = float(batch.shape_params[idx][0])
            y1 = float(batch.shape_params[idx][1])
            width_stroke = float(batch.shape_params[idx][2])
            dx = x1 - float(cx)
            dy = y1 - float(cy)
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            nx = -dy / length * (width_stroke / 2.0)
            ny = dx / length * (width_stroke / 2.0)
            abs_verts = [
                (float(cx + nx), float(cy + ny)),
                (float(cx - nx), float(cy - ny)),
                (float(x1 - nx), float(y1 - ny)),
                (float(x1 + nx), float(y1 + ny)),
            ]

        if shape_type != SHAPE_THIN_STROKE:
            if not verts:
                continue
            cos_t = math.cos(rot)
            sin_t = math.sin(rot)
            for vx, vy in verts:
                rx = vx * cos_t - vy * sin_t
                ry = vx * sin_t + vy * cos_t
                abs_verts.append((float(cx + rx), float(cy + ry)))

        num_vertices = len(abs_verts)
        if num_vertices < 3:
            continue

        for vx, vy in abs_verts:
            all_x.append(vx)
            all_y.append(vy)
            all_z.append(z_val)

        for t_idx in range(1, num_vertices - 1):
            all_i.append(vert_offset)
            all_j.append(vert_offset + t_idx)
            all_k.append(vert_offset + t_idx + 1)
            facecolors.append(color_str)

        vert_offset += num_vertices

    mesh = go.Mesh3d(
        x=all_x,
        y=all_y,
        z=all_z,
        i=all_i,
        j=all_j,
        k=all_k,
        facecolor=facecolors,
        opacity=1.0,
        flatshading=True,
        hoverinfo="none",
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title="3D Exploded Algorithmic View",
        scene=dict(
            xaxis=dict(range=[0, width], showgrid=False, showticklabels=False),
            yaxis=dict(range=[height, 0], showgrid=False, showticklabels=False),
            zaxis=dict(
                range=[0, max(batch.count, 1)], showgrid=False, showticklabels=False
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=(height / max(width, 1)), z=0.8),
        ),
        scene_camera=dict(eye=dict(x=0, y=0, z=2.2), up=dict(x=0, y=-1, z=0)),
        paper_bgcolor="rgb(20,20,20)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig
