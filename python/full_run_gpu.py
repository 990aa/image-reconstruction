IMAGE_PATH = "/content/internet_portrait.jpg"
POLYGONS = 1500
RESOLUTION = 200
MINUTES = 10.0
SEED = 42

import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage import color
from skimage.filters import sobel_h, sobel_v
from skimage.metrics import structural_similarity
from sklearn.cluster import MiniBatchKMeans
warnings.filterwarnings("ignore")

SHAPE_TRIANGLE = 0
SHAPE_QUAD = 1
SHAPE_ELLIPSE = 2
SHAPE_BEZIER_PATCH = 3
SHAPE_THIN_STROKE = 4
SHAPE_ANNULAR_SEGMENT = 5
DEFAULT_BASE_RESOLUTION = 200

@dataclass
class LivePolygonBatch:
    centers: np.ndarray
    sizes: np.ndarray
    rotations: np.ndarray
    colors: np.ndarray
    alphas: np.ndarray
    shape_types: np.ndarray
    shape_params: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), dtype=np.float32))

    def __post_init__(self):
        self.centers = np.ascontiguousarray(self.centers, dtype=np.float32)
        self.sizes = np.ascontiguousarray(self.sizes, dtype=np.float32)
        self.rotations = np.ascontiguousarray(self.rotations, dtype=np.float32)
        self.colors = np.ascontiguousarray(self.colors, dtype=np.float32)
        self.alphas = np.ascontiguousarray(self.alphas, dtype=np.float32)
        self.shape_types = np.ascontiguousarray(self.shape_types, dtype=np.int32)
        if self.shape_params.size == 0 and self.centers.shape[0] > 0:
            self.shape_params = np.zeros((self.centers.shape[0], 6), dtype=np.float32)
        self.shape_params = np.ascontiguousarray(self.shape_params, dtype=np.float32)

    @property
    def count(self) -> int:
        return int(self.centers.shape[0])

    def copy(self):
        return LivePolygonBatch(
            centers=np.array(self.centers, copy=True),
            sizes=np.array(self.sizes, copy=True),
            rotations=np.array(self.rotations, copy=True),
            colors=np.array(self.colors, copy=True),
            alphas=np.array(self.alphas, copy=True),
            shape_types=np.array(self.shape_types, copy=True),
            shape_params=np.array(self.shape_params, copy=True),
        )

def make_empty_live_batch() -> LivePolygonBatch:
    return LivePolygonBatch(
        centers=np.zeros((0, 2), dtype=np.float32),
        sizes=np.zeros((0, 2), dtype=np.float32),
        rotations=np.zeros((0,), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        alphas=np.zeros((0,), dtype=np.float32),
        shape_types=np.zeros((0,), dtype=np.int32),
        shape_params=np.zeros((0, 6), dtype=np.float32),
    )

@dataclass
class ShapeCandidate:
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    rotation: float
    alpha: float
    shape_type: int
    shape_params: np.ndarray
    color: np.ndarray
    mse: float = float("inf")
    coverage_tensor: torch.Tensor | None = None
    canvas_tensor: torch.Tensor | None = None
    residual_tensor: torch.Tensor | None = None

    def copy(self):
        return ShapeCandidate(
            center_x=float(self.center_x),
            center_y=float(self.center_y),
            size_x=float(self.size_x),
            size_y=float(self.size_y),
            rotation=float(self.rotation),
            alpha=float(self.alpha),
            shape_type=int(self.shape_type),
            shape_params=np.array(self.shape_params, copy=True),
            color=np.array(self.color, copy=True),
            mse=float(self.mse),
            coverage_tensor=self.coverage_tensor,
            canvas_tensor=self.canvas_tensor,
            residual_tensor=self.residual_tensor
        )

@dataclass(frozen=True)
class SequentialStageConfig:
    name: str
    resolution: int
    shapes_to_add: int
    candidate_count: int
    mutation_steps: int
    size_min: float
    size_max: float
    alpha_min: float
    alpha_max: float
    softness: float
    allowed_shapes: tuple[int, ...]
    high_frequency_only: bool = False
    top_k_regions: int = 50
    region_window: int = 5
    mutation_shift_px: float = 1.0
    mutation_size_ratio: float = 0.10
    mutation_rotation_deg: float = 5.0

@dataclass(frozen=True)
class PhasePlan:
    polygon_budget: int
    stages: tuple[SequentialStageConfig, ...]

@dataclass(frozen=True)
class PreprocessedTarget:
    base_resolution: int
    target_rgb: np.ndarray
    pyramid: list[np.ndarray]
    segmentation_map: np.ndarray
    cluster_centroids_lab: np.ndarray
    cluster_centroids_rgb: np.ndarray
    cluster_variances_lab: np.ndarray
    structure_map: np.ndarray
    gradient_angle_map: np.ndarray
    complexity_score: float
    recommended_polygons: int
    recommended_k: int
    recommended_size_schedule: dict[str, float]

class GPUCoreRenderer:
    """A PyTorch implementation of the rasterizer for T4 GPU acceleration."""
    def __init__(self, height: int, width: int):
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

    def _sigmoid(self, values: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(values, -60.0, 60.0)
        return 1.0 / (1.0 + torch.exp(-clipped))

    def _ellipse_coverage_params(self, center_x: float, center_y: float, axis_x: float, axis_y: float, rotation: float, softness: float):
        ax = max(float(axis_x), 1e-3)
        ay = max(float(axis_y), 1e-3)
        cos_t, sin_t = math.cos(rotation), math.sin(rotation)
        dx0, dy0 = self.grid_x - center_x, self.grid_y - center_y
        dx = dx0 * cos_t + dy0 * sin_t
        dy = -dx0 * sin_t + dy0 * cos_t
        d = torch.sqrt((dx / ax)**2 + (dy / ay)**2 + 1e-8)
        return self._sigmoid((1.0 - d) / max(softness, 1e-6))

    def _quad_coverage_params(self, center_x: float, center_y: float, axis_x: float, axis_y: float, rotation: float, softness: float):
        ax = max(float(axis_x), 1e-3)
        ay = max(float(axis_y), 1e-3)
        cos_t, sin_t = math.cos(rotation), math.sin(rotation)
        dx0, dy0 = self.grid_x - center_x, self.grid_y - center_y
        dx = dx0 * cos_t + dy0 * sin_t
        dy = -dx0 * sin_t + dy0 * cos_t
        d = torch.maximum(torch.abs(dx) / ax, torch.abs(dy) / ay)
        return self._sigmoid((1.0 - d) / max(softness, 1e-6))

    def _triangle_coverage_params(self, center_x: float, center_y: float, axis_x: float, axis_y: float, rotation: float, softness: float):
        sx = max(float(axis_x), 1e-3)
        sy = max(float(axis_y), 1e-3)
        local = torch.tensor([[sx, 0.0], [-0.5 * sx, 0.5 * sy], [-0.5 * sx, -0.5 * sy]], dtype=torch.float32, device=self.device)
        cos_t, sin_t = math.cos(rotation), math.sin(rotation)
        rot = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float32, device=self.device)
        verts = torch.matmul(local, rot.T)
        verts[:, 0] += center_x
        verts[:, 1] += center_y

        x1, y1 = verts[0, 0], verts[0, 1]
        x2, y2 = verts[1, 0], verts[1, 1]
        x3, y3 = verts[2, 0], verts[2, 1]

        def _signed_dist(ax, ay, bx, by):
            ex, ey = bx - ax, by - ay
            nx, ny = -ey, ex
            norm = math.sqrt(nx*nx + ny*ny + 1e-8)
            return ((self.grid_x - ax) * nx + (self.grid_y - ay) * ny) / norm

        d1 = _signed_dist(x1, y1, x2, y2)
        d2 = _signed_dist(x2, y2, x3, y3)
        d3 = _signed_dist(x3, y3, x1, y1)
        signed = torch.minimum(torch.minimum(d1, d2), d3)
        return self._sigmoid(signed / softness)

    def _thin_stroke_coverage_params(self, center_x: float, center_y: float, shape_params: np.ndarray, softness: float):
        x0, y0 = float(center_x), float(center_y)
        x1, y1 = float(shape_params[0]), float(shape_params[1])
        width = max(float(shape_params[2]), 1e-3)
        dx, dy = x1 - x0, y1 - y0
        seg_len_sq = dx*dx + dy*dy

        if seg_len_sq <= 1e-8:
            dist = torch.sqrt((self.grid_x - x0)**2 + (self.grid_y - y0)**2 + 1e-8)
        else:
            t = ((self.grid_x - x0)*dx + (self.grid_y - y0)*dy) / seg_len_sq
            t = torch.clamp(t, 0.0, 1.0)
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            dist = torch.sqrt((self.grid_x - proj_x)**2 + (self.grid_y - proj_y)**2 + 1e-8)
        return self._sigmoid((0.5 * width - dist) / max(softness, 1e-6))

class GPUSequentialHillClimber:
    def __init__(self, target_image: np.ndarray, rasterizer: GPUCoreRenderer, polygons: LivePolygonBatch, background_color: np.ndarray):
        self.target_np = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)
        self.rasterizer = rasterizer
        self.height, self.width = self.target_np.shape[:2]
        self.device = rasterizer.device

        self.target_tensor = torch.from_numpy(self.target_np).to(self.device)
        self.bg_tensor = torch.from_numpy(np.clip(background_color, 0.0, 1.0)).to(self.device)

        self.polygons = polygons.copy()

        # Build initial canvas completely on GPU to maintain exact parameter parity
        self.current_canvas_tensor = self.bg_tensor.view(1, 1, 3).expand(self.height, self.width, 3).clone()
        if self.polygons.count > 0:
            for i in range(self.polygons.count):
                candidate = ShapeCandidate(
                    center_x=self.polygons.centers[i, 0], center_y=self.polygons.centers[i, 1],
                    size_x=self.polygons.sizes[i, 0], size_y=self.polygons.sizes[i, 1],
                    rotation=self.polygons.rotations[i], alpha=self.polygons.alphas[i],
                    shape_type=self.polygons.shape_types[i], shape_params=self.polygons.shape_params[i],
                    color=self.polygons.colors[i]
                )
                cov = self._coverage_from_candidate(candidate, softness=1.0)
                weight = cov * float(np.clip(candidate.alpha, 0.0, 1.0))
                col_tensor = torch.from_numpy(candidate.color).to(self.device).view(1, 1, 3)
                self.current_canvas_tensor = self.current_canvas_tensor + weight.unsqueeze(2) * (col_tensor - self.current_canvas_tensor)

        self.current_canvas_tensor = torch.clamp(self.current_canvas_tensor, 0.0, 1.0)
        res = self.current_canvas_tensor - self.target_tensor
        self.current_mse = torch.mean(res * res).item()
        self.loss_history = [self.current_mse]

    @property
    def current_canvas_np(self):
        return self.current_canvas_tensor.cpu().numpy()

    def _coverage_from_candidate(self, candidate: ShapeCandidate, softness: float) -> torch.Tensor:
        if candidate.shape_type == SHAPE_QUAD:
            return self.rasterizer._quad_coverage_params(candidate.center_x, candidate.center_y, candidate.size_x, candidate.size_y, candidate.rotation, softness)
        if candidate.shape_type == SHAPE_TRIANGLE:
            return self.rasterizer._triangle_coverage_params(candidate.center_x, candidate.center_y, candidate.size_x, candidate.size_y, candidate.rotation, softness)
        if candidate.shape_type == SHAPE_THIN_STROKE:
            return self.rasterizer._thin_stroke_coverage_params(candidate.center_x, candidate.center_y, candidate.shape_params, softness)
        return self.rasterizer._ellipse_coverage_params(candidate.center_x, candidate.center_y, candidate.size_x, candidate.size_y, candidate.rotation, softness)

    def evaluate_candidate(self, candidate: ShapeCandidate, softness: float) -> ShapeCandidate:
        # All heavy tensor math happens without CPU bottlenecks
        coverage = self._coverage_from_candidate(candidate, softness)
        weight = coverage * float(np.clip(candidate.alpha, 0.0, 1.0))
        weight3 = weight.unsqueeze(2)

        denom = torch.sum(weight * weight).item()
        if denom <= 1e-8:
            color = torch.mean(self.target_tensor, dim=(0, 1))
        else:
            numer = torch.sum(weight3 * (self.target_tensor - self.current_canvas_tensor * (1.0 - weight3)), dim=(0, 1))
            color = torch.clamp(numer / denom, 0.0, 1.0)

        canvas = self.current_canvas_tensor + weight3 * (color.view(1, 1, 3) - self.current_canvas_tensor)
        canvas = torch.clamp(canvas, 0.0, 1.0)
        residual = canvas - self.target_tensor
        mse = torch.mean(residual * residual).item()

        scored = candidate.copy()
        scored.color = color.cpu().numpy()
        scored.mse = mse
        scored.coverage_tensor = coverage
        scored.canvas_tensor = canvas
        scored.residual_tensor = residual
        return scored

    def _region_sum_map(self, error_map: np.ndarray, window: int) -> np.ndarray:
        h, w = error_map.shape
        actual = int(max(1, min(window, h, w)))
        integral = np.pad(error_map, ((1, 0), (1, 0)), mode="constant")
        integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
        y2, x2 = np.arange(actual, h + 1), np.arange(actual, w + 1)
        return (integral[y2[:, None], x2[None, :]] - integral[y2[:, None] - actual, x2[None, :]]
                - integral[y2[:, None], x2[None, :] - actual] + integral[y2[:, None] - actual, x2[None, :] - actual]).astype(np.float32)

    def sample_error_centers(self, guide_map: np.ndarray, count: int, top_k: int, window: int, rng: np.random.Generator):
        if count <= 0: return []
        work = self._region_sum_map(np.clip(guide_map, 0.0, None), max(1, window))
        candidates = []
        suppression = max(1, window // 2)

        while len(candidates) < top_k:
            flat_idx = int(np.argmax(work))
            score = float(work.reshape(-1)[flat_idx])
            if score <= 1e-12: break
            top_y, top_x = divmod(flat_idx, work.shape[1])
            candidates.append((float(top_x + 0.5 * max(window - 1, 0)), float(top_y + 0.5 * max(window - 1, 0)), score))
            y0, y1 = max(0, top_y - suppression), min(work.shape[0], top_y + suppression + 1)
            x0, x1 = max(0, top_x - suppression), min(work.shape[1], top_x + suppression + 1)
            work[y0:y1, x0:x1] = 0.0

        if not candidates: return []
        weights = np.array([max(c[2], 0.0) for c in candidates], dtype=np.float64)
        if float(np.sum(weights)) <= 0.0: weights[:] = 1.0
        chosen = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False, p=weights/np.sum(weights))
        return [(float(np.clip(candidates[idx][0], 0.0, self.width - 1.0)),
                 float(np.clip(candidates[idx][1], 0.0, self.height - 1.0))) for idx in chosen]

    def _shape_type_for_location(self, stage, structure_map, linearity_map, x, y, rng):
        structure, linearity = float(structure_map[y, x]), float(linearity_map[y, x])
        allowed = stage.allowed_shapes
        if structure < 0.18: return SHAPE_ELLIPSE if SHAPE_ELLIPSE in allowed else allowed[0]
        if SHAPE_THIN_STROKE in allowed and structure >= 0.62 and linearity >= 0.68: return SHAPE_THIN_STROKE
        if SHAPE_TRIANGLE in allowed and structure >= 0.42 and linearity >= 0.45: return SHAPE_TRIANGLE
        if SHAPE_QUAD in allowed and structure >= 0.35 and rng.random() < 0.15: return SHAPE_QUAD
        if SHAPE_ELLIPSE in allowed: return SHAPE_ELLIPSE
        return allowed[int(rng.integers(0, len(allowed)))]

    def _aspect_ratio(self, structure: float) -> float:
        if structure <= 0.10: return 1.10
        if structure >= 0.20: return 3.00
        return float(1.10 + ((structure - 0.10) / 0.10) * (3.00 - 1.10))

    def random_candidate(self, stage, center_x, center_y, structure_map, angle_map, linearity_map, rng):
        px, py = int(np.clip(round(center_x), 0, self.width - 1)), int(np.clip(round(center_y), 0, self.height - 1))
        structure, angle = float(structure_map[py, px]), float(angle_map[py, px])
        shape_type = self._shape_type_for_location(stage, structure_map, linearity_map, px, py, rng)
        jitter = max(0.5, stage.region_window * 0.45)
        x = float(np.clip(center_x + rng.uniform(-jitter, jitter), 0.0, self.width - 1.0))
        y = float(np.clip(center_y + rng.uniform(-jitter, jitter), 0.0, self.height - 1.0))
        alpha = float(rng.uniform(stage.alpha_min, stage.alpha_max))

        params = np.zeros(6, dtype=np.float32)
        if shape_type == SHAPE_THIN_STROKE:
            length = float(np.clip(rng.uniform(stage.size_min, stage.size_max) * (1.1 + 1.2 * structure), stage.size_min, stage.size_max * 1.8))
            width = float(max(1.0, 0.15 * length))
            rotation = float(angle + 0.5 * np.pi + rng.uniform(-0.18, 0.18))
            params[0], params[1], params[2] = x + np.cos(rotation)*length, y + np.sin(rotation)*length, width
            return ShapeCandidate(x, y, 0.5*length, 0.5*width, rotation, alpha, shape_type, params, np.zeros(3, dtype=np.float32))

        major = float(rng.uniform(stage.size_min, stage.size_max))
        minor = float(max(stage.size_min * 0.35, major / max(self._aspect_ratio(structure), 1.0)))
        rotation = float(angle + 0.5 * np.pi + rng.uniform(-0.35, 0.35)) if structure >= 0.12 else float(rng.uniform(0.0, np.pi))
        return ShapeCandidate(x, y, major, minor, rotation, alpha, shape_type, params, np.zeros(3, dtype=np.float32))

    def mutate_candidate(self, candidate: ShapeCandidate, stage, rng):
        mutated = candidate.copy()
        rot_step = float(np.deg2rad(stage.mutation_rotation_deg))
        max_size = float(max(stage.size_max, stage.size_min))

        for _ in range(int(rng.integers(1, 3))):
            op = int(rng.integers(0, 5))
            if op == 0: mutated.center_x = float(np.clip(mutated.center_x + rng.choice([-1.0, 1.0]) * stage.mutation_shift_px, 0.0, self.width - 1.0))
            elif op == 1: mutated.center_y = float(np.clip(mutated.center_y + rng.choice([-1.0, 1.0]) * stage.mutation_shift_px, 0.0, self.height - 1.0))
            elif op == 2: mutated.size_x = float(np.clip(mutated.size_x * (1.0 + rng.choice([-1.0, 1.0]) * stage.mutation_size_ratio), stage.size_min, max_size))
            elif op == 3: mutated.size_y = float(np.clip(mutated.size_y * (1.0 + rng.choice([-1.0, 1.0]) * stage.mutation_size_ratio), max(stage.size_min * 0.25, 0.8), max_size))
            else: mutated.rotation = float(mutated.rotation + rng.choice([-1.0, 1.0]) * rot_step)

        if mutated.shape_type == SHAPE_THIN_STROKE:
            mutated.size_x, mutated.size_y = float(np.clip(mutated.size_x, stage.size_min * 0.5, max_size)), float(np.clip(mutated.size_y, 0.5, max(0.5, stage.size_max * 0.25)))
            length = max(mutated.size_x * 2.0, 2.0)
            mutated.shape_params[0] = mutated.center_x + np.cos(mutated.rotation) * length
            mutated.shape_params[1] = mutated.center_y + np.sin(mutated.rotation) * length
            mutated.shape_params[2] = max(mutated.size_y * 2.0, 1.0)
        else:
            mutated.size_x, mutated.size_y = float(np.clip(mutated.size_x, stage.size_min, max_size)), float(np.clip(mutated.size_y, max(stage.size_min * 0.25, 0.8), max_size))
        return mutated

    def search_next_shape(self, stage, guide_map, structure_map, angle_map, linearity_map, rng):
        centers = self.sample_error_centers(guide_map, stage.candidate_count, stage.top_k_regions, stage.region_window, rng)
        if not centers: return None

        best = None
        for cx, cy in centers:
            scored = self.evaluate_candidate(self.random_candidate(stage, cx, cy, structure_map, angle_map, linearity_map, rng), stage.softness)
            if best is None or scored.mse < best.mse: best = scored

        if best is None or best.mse >= self.current_mse: return None

        stagnation = 0
        for _ in range(stage.mutation_steps):
            scored = self.evaluate_candidate(self.mutate_candidate(best, stage, rng), stage.softness)
            if scored.mse + 1e-12 < best.mse:
                best, stagnation = scored, 0
            else:
                stagnation += 1
                if stagnation >= max(12, stage.mutation_steps // 4): break

        return best if best.mse < self.current_mse else None

    def commit_shape(self, candidate: ShapeCandidate):
        self.polygons.centers = np.vstack([self.polygons.centers, [candidate.center_x, candidate.center_y]])
        self.polygons.sizes = np.vstack([self.polygons.sizes, [candidate.size_x, candidate.size_y]])
        self.polygons.rotations = np.append(self.polygons.rotations, candidate.rotation)
        self.polygons.colors = np.vstack([self.polygons.colors, candidate.color])
        self.polygons.alphas = np.append(self.polygons.alphas, candidate.alpha)
        self.polygons.shape_types = np.append(self.polygons.shape_types, candidate.shape_type)
        self.polygons.shape_params = np.vstack([self.polygons.shape_params, candidate.shape_params])

        self.current_canvas_tensor = candidate.canvas_tensor
        self.current_mse = candidate.mse
        self.loss_history.append(self.current_mse)

def build_phase_plan(base_resolution: int, polygon_budget: int, complexity_score: float) -> PhasePlan:
    budget = max(1, int(polygon_budget))
    stage_a_count = min(200, budget // 6)
    stage_b_count = min(400, max(0, budget // 3))
    stage_c_count = max(0, budget - stage_a_count - stage_b_count)

    stage_a_res = max(50, min(100, int(base_resolution)))
    stage_b_res = max(stage_a_res, min(150, int(base_resolution)))
    stage_c_res = int(base_resolution)

    return PhasePlan(budget, (
        SequentialStageConfig("foundation", stage_a_res, stage_a_count, 80, 160, max(2.5, stage_a_res*0.12), max(4.0, stage_a_res*0.34), 0.55, 0.85, 0.75, (SHAPE_ELLIPSE, SHAPE_QUAD), False, 80, 5, 2.5, 0.18, 15.0),
        SequentialStageConfig("structure", stage_b_res, stage_b_count, 64, 128, max(1.8, stage_b_res*0.035), max(3.5, stage_b_res*0.16), 0.40, 0.72, 0.32, (SHAPE_ELLIPSE, SHAPE_QUAD, SHAPE_TRIANGLE), False, 60, 5, 4.0, 0.18, 15.0),
        SequentialStageConfig("detail", stage_c_res, stage_c_count, 72, 156, max(0.9, stage_c_res*0.006), max(2.2, stage_c_res*0.040), 0.28, 0.60, 0.055, (SHAPE_ELLIPSE, SHAPE_TRIANGLE, SHAPE_THIN_STROKE), True, 70, 5, 6.0, 0.18, 15.0),
    ))

def _resize_float_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == size: return image.astype(np.float32)
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    return (np.asarray(pil.resize((size[1], size[0]), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0)

def preprocess_target_array(target_rgb: np.ndarray, base_resolution: int) -> PreprocessedTarget:
    target = _resize_float_image(target_rgb, (base_resolution, base_resolution))
    gray = np.mean(target, axis=2)
    gx, gy = sobel_h(gray), sobel_v(gray)
    grad = np.hypot(gx, gy).astype(np.float32)
    structure = np.clip((grad - np.min(grad)) / max(np.max(grad) - np.min(grad), 1e-8), 0.0, 1.0)
    complexity = float(np.clip((np.mean(structure) / max(np.mean(np.abs(target - np.mean(target, axis=(0,1)))), 1e-6)) / (1.0 + (np.mean(structure) / max(np.mean(np.abs(target - np.mean(target, axis=(0,1)))), 1e-6))), 0.0, 1.0))
    return PreprocessedTarget(base_resolution, target, [], np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1)), structure, np.arctan2(gy, gx).astype(np.float32), complexity, 1500, 10, {})

def _scale_polygons(polygons: LivePolygonBatch, old_res: int, new_res: int):
    if polygons.count == 0: return make_empty_live_batch()
    scale = float(new_res) / max(float(old_res), 1.0)
    batch = polygons.copy()
    batch.centers *= scale
    batch.sizes = np.clip(batch.sizes * scale, 0.5, None)
    batch.centers[:, 0] = np.clip(batch.centers[:, 0], 0.0, new_res - 1.0)
    batch.centers[:, 1] = np.clip(batch.centers[:, 1], 0.0, new_res - 1.0)
    stroke_mask = batch.shape_types == SHAPE_THIN_STROKE
    batch.shape_params[stroke_mask, 0:3] *= scale
    return batch

def run(image_path, resolution, polygons, minutes, seed):
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        side = min(rgb.size)
        left, top = (rgb.size[0] - side) // 2, (rgb.size[1] - side) // 2
        rgb = rgb.crop((left, top, left + side, top + side))
        arr = np.asarray(rgb.resize((resolution, resolution), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0

    prep = preprocess_target_array(arr, resolution)
    plan = build_phase_plan(resolution, polygons, prep.complexity_score)
    rng = np.random.default_rng(seed)
    bg_color = np.mean(prep.target_rgb, axis=(0, 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    plt.close(fig)

    last_ui_update = time.time()
    batch = make_empty_live_batch()
    prev_res = None
    loss_history, res_markers, stage_markers = [], [0], []
    iteration = 0
    start_time = time.time()

    def _update_ui(canvas, target, losses, stage_name, status):
        axes[0,0].clear(); axes[0,0].imshow(target); axes[0,0].set_title("Target"); axes[0,0].axis('off')
        axes[0,1].clear(); axes[0,1].imshow(canvas); axes[0,1].set_title(f"Reconstruction | {batch.count} shapes"); axes[0,1].axis('off')
        err = np.clip(np.mean(np.abs(target - canvas), axis=2) / max(np.quantile(np.mean(np.abs(target - canvas), axis=2), 0.99), 1e-6), 0.0, 1.0)
        axes[1,0].clear(); axes[1,0].imshow(err, cmap="magma", vmin=0.0, vmax=1.0); axes[1,0].set_title("Abs Error"); axes[1,0].axis('off')
        axes[1,1].clear(); axes[1,1].set_title("MSE Reduction"); axes[1,1].set_yscale("log"); axes[1,1].grid(True, alpha=0.25)
        if losses: axes[1,1].plot(np.maximum(losses, 1e-9), color="tab:blue")
        for m in stage_markers: axes[1,1].axvline(m[1], color="#e76f51", alpha=0.5)
        fig.suptitle(f"Stage: {stage_name} | Iter: {iteration} | {status}", fontsize=13)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        clear_output(wait=True)
        display(fig)

    # 3. Optimization Loop
    for stage in plan.stages:
        if time.time() - start_time > minutes * 60: break
        if stage.shapes_to_add <= 0: continue

        stage_batch = make_empty_live_batch() if prev_res is None else _scale_polygons(batch, prev_res, stage.resolution)
        stage_target = _resize_float_image(prep.target_rgb, (stage.resolution, stage.resolution))

        gray = np.mean(stage_target, axis=2)
        gy, gx = np.gradient(gray)
        mag = np.hypot(gx, gy)
        structure_map = np.clip(mag / max(np.percentile(mag, 99.0), 1e-6), 0.0, 1.0)
        angle_map = np.arctan2(gy, gx)
        linearity_map = np.clip(np.sqrt(uniform_filter(np.cos(angle_map), 7)**2 + uniform_filter(np.sin(angle_map), 7)**2), 0.0, 1.0)

        rasterizer = GPUCoreRenderer(stage.resolution, stage.resolution)
        optimizer = GPUSequentialHillClimber(stage_target, rasterizer, stage_batch, bg_color)

        stage_markers.append((stage.name, iteration))
        res_markers.append(iteration)

        for i in range(stage.shapes_to_add):
            if time.time() - start_time > minutes * 60: break

            guide_map = np.mean(np.abs(optimizer.target_np - optimizer.current_canvas_np), axis=2)
            if stage.high_frequency_only:
                guide_map = np.clip(guide_map - gaussian_filter(guide_map, 2.5), 0.0, None) + 0.40 * guide_map

            candidate = optimizer.search_next_shape(stage, guide_map, structure_map, angle_map, linearity_map, rng)
            if candidate:
                optimizer.commit_shape(candidate)
                batch = optimizer.polygons.copy()
                prev_res = stage.resolution
                loss_history.append(optimizer.current_mse)
                iteration += 1
            if time.time() - last_ui_update > 1.0:
                _update_ui(_resize_float_image(optimizer.current_canvas_np, (RESOLUTION, RESOLUTION)), prep.target_rgb, loss_history, stage.name, "Running on GPU...")
                last_ui_update = time.time()

    if batch.count > 0 and prev_res != RESOLUTION:
        final_batch = _scale_polygons(batch, prev_res, RESOLUTION)
        optimizer = GPUSequentialHillClimber(prep.target_rgb, GPUCoreRenderer(RESOLUTION, RESOLUTION), final_batch, bg_color)
        final_canvas = optimizer.current_canvas_np
    else:
        final_canvas = optimizer.current_canvas_np

    _update_ui(final_canvas, prep.target_rgb, loss_history, "DONE", "Finished")

    print("\n Reconstruct Summary ")
    print(f"Accepted Polygons: {batch.count}")
    print(f"Total Iterations:  {iteration}")
    print(f"Final RGB MSE:     {loss_history[-1]:.6f}")

run(IMAGE_PATH, RESOLUTION, POLYGONS, MINUTES, SEED)