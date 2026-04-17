from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


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
            raise ValueError("centers must have shape (N, 2)")
        if self.sizes.ndim != 2 or self.sizes.shape[1] != 2:
            raise ValueError("sizes must have shape (N, 2)")
        if self.rotations.ndim != 1:
            raise ValueError("rotations must have shape (N,)")
        if self.colors.ndim != 2 or self.colors.shape[1] != 3:
            raise ValueError("colors must have shape (N, 3)")
        if self.alphas.ndim != 1:
            raise ValueError("alphas must have shape (N,)")
        if self.shape_types.ndim != 1:
            raise ValueError("shape_types must have shape (N,)")
        if self.shape_params.ndim != 2 or self.shape_params.shape[1] != 6:
            raise ValueError("shape_params must have shape (N, 6)")

        n = self.centers.shape[0]
        if (
            self.sizes.shape[0] != n
            or self.rotations.shape[0] != n
            or self.colors.shape[0] != n
            or self.alphas.shape[0] != n
            or self.shape_types.shape[0] != n
            or self.shape_params.shape[0] != n
        ):
            raise ValueError("All polygon arrays must have the same length")

    @property
    def count(self) -> int:
        return int(self.centers.shape[0])

    def copy(self) -> "LivePolygonBatch":
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

    def copy(self) -> "ShapeCandidate":
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
            residual_tensor=self.residual_tensor,
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


@dataclass(frozen=True)
class PhaseResult:
    batch: LivePolygonBatch
    preprocessed: PreprocessedTarget
    background_color: np.ndarray
    final_canvas: np.ndarray
    loss_history: list[float]
    stage_markers: list[tuple[str, int]]
    iterations: int
