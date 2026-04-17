from __future__ import annotations


import numpy as np
import torch

from iterative_art_gpu.constants import (
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
)
from iterative_art_gpu.models import LivePolygonBatch, ShapeCandidate
from iterative_art_gpu.renderer import GPUCoreRenderer


class GPUSequentialHillClimber:
    """Notebook-compatible sequential optimizer evaluated on GPU tensors."""

    def __init__(
        self,
        target_image: np.ndarray,
        rasterizer: GPUCoreRenderer,
        polygons: LivePolygonBatch,
        background_color: np.ndarray,
    ) -> None:
        self.target_np = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)
        self.rasterizer = rasterizer
        self.height, self.width = self.target_np.shape[:2]
        self.device = rasterizer.device

        self.target_tensor = torch.from_numpy(self.target_np).to(self.device)
        self.bg_tensor = torch.from_numpy(np.clip(background_color, 0.0, 1.0)).to(
            self.device
        )

        self.polygons = polygons.copy()
        self.current_canvas_tensor = (
            self.bg_tensor.view(1, 1, 3).expand(self.height, self.width, 3).clone()
        )

        if self.polygons.count > 0:
            for idx in range(self.polygons.count):
                candidate = ShapeCandidate(
                    center_x=self.polygons.centers[idx, 0],
                    center_y=self.polygons.centers[idx, 1],
                    size_x=self.polygons.sizes[idx, 0],
                    size_y=self.polygons.sizes[idx, 1],
                    rotation=self.polygons.rotations[idx],
                    alpha=self.polygons.alphas[idx],
                    shape_type=self.polygons.shape_types[idx],
                    shape_params=self.polygons.shape_params[idx],
                    color=self.polygons.colors[idx],
                )
                coverage = self._coverage_from_candidate(candidate, softness=1.0)
                weight = coverage * float(np.clip(candidate.alpha, 0.0, 1.0))
                color_tensor = (
                    torch.from_numpy(candidate.color).to(self.device).view(1, 1, 3)
                )
                self.current_canvas_tensor = (
                    self.current_canvas_tensor
                    + weight.unsqueeze(2) * (color_tensor - self.current_canvas_tensor)
                )

        self.current_canvas_tensor = torch.clamp(self.current_canvas_tensor, 0.0, 1.0)
        residual = self.current_canvas_tensor - self.target_tensor
        self.current_mse = float(torch.mean(residual * residual).item())
        self.loss_history: list[float] = [self.current_mse]

    @property
    def current_canvas_np(self) -> np.ndarray:
        return self.current_canvas_tensor.detach().cpu().numpy()

    def _coverage_from_candidate(
        self,
        candidate: ShapeCandidate,
        softness: float,
    ) -> torch.Tensor:
        if candidate.shape_type == SHAPE_QUAD:
            return self.rasterizer._quad_coverage_params(
                candidate.center_x,
                candidate.center_y,
                candidate.size_x,
                candidate.size_y,
                candidate.rotation,
                softness,
            )
        if candidate.shape_type == SHAPE_TRIANGLE:
            return self.rasterizer._triangle_coverage_params(
                candidate.center_x,
                candidate.center_y,
                candidate.size_x,
                candidate.size_y,
                candidate.rotation,
                softness,
            )
        if candidate.shape_type == SHAPE_THIN_STROKE:
            return self.rasterizer._thin_stroke_coverage_params(
                candidate.center_x,
                candidate.center_y,
                candidate.shape_params,
                softness,
            )
        return self.rasterizer._ellipse_coverage_params(
            candidate.center_x,
            candidate.center_y,
            candidate.size_x,
            candidate.size_y,
            candidate.rotation,
            softness,
        )

    def evaluate_candidate(
        self,
        candidate: ShapeCandidate,
        softness: float,
    ) -> ShapeCandidate:
        coverage = self._coverage_from_candidate(candidate, softness)
        weight = coverage * float(np.clip(candidate.alpha, 0.0, 1.0))
        weight3 = weight.unsqueeze(2)

        denom = float(torch.sum(weight * weight).item())
        if denom <= 1e-8:
            color = torch.mean(self.target_tensor, dim=(0, 1))
        else:
            numer = torch.sum(
                weight3
                * (self.target_tensor - self.current_canvas_tensor * (1.0 - weight3)),
                dim=(0, 1),
            )
            color = torch.clamp(numer / denom, 0.0, 1.0)

        canvas = self.current_canvas_tensor + weight3 * (
            color.view(1, 1, 3) - self.current_canvas_tensor
        )
        canvas = torch.clamp(canvas, 0.0, 1.0)
        residual = canvas - self.target_tensor
        mse = float(torch.mean(residual * residual).item())

        scored = candidate.copy()
        scored.color = color.detach().cpu().numpy()
        scored.mse = mse
        scored.coverage_tensor = coverage
        scored.canvas_tensor = canvas
        scored.residual_tensor = residual
        return scored

    @staticmethod
    def _region_sum_map(error_map: np.ndarray, window: int) -> np.ndarray:
        h, w = error_map.shape
        actual = int(max(1, min(window, h, w)))
        integral = np.pad(error_map, ((1, 0), (1, 0)), mode="constant")
        integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
        y2 = np.arange(actual, h + 1)
        x2 = np.arange(actual, w + 1)
        return (
            integral[y2[:, None], x2[None, :]]
            - integral[y2[:, None] - actual, x2[None, :]]
            - integral[y2[:, None], x2[None, :] - actual]
            + integral[y2[:, None] - actual, x2[None, :] - actual]
        ).astype(np.float32, copy=False)

    def sample_error_centers(
        self,
        guide_map: np.ndarray,
        count: int,
        top_k: int,
        window: int,
        rng: np.random.Generator,
    ) -> list[tuple[float, float]]:
        if count <= 0:
            return []

        work = self._region_sum_map(np.clip(guide_map, 0.0, None), max(1, int(window)))
        candidates: list[tuple[float, float, float]] = []
        suppression = max(1, int(window // 2))

        while len(candidates) < int(top_k):
            flat_idx = int(np.argmax(work))
            score = float(work.reshape(-1)[flat_idx])
            if score <= 1e-12:
                break
            top_y, top_x = divmod(flat_idx, work.shape[1])
            candidates.append(
                (
                    float(top_x + 0.5 * max(window - 1, 0)),
                    float(top_y + 0.5 * max(window - 1, 0)),
                    score,
                )
            )
            y0 = max(0, top_y - suppression)
            y1 = min(work.shape[0], top_y + suppression + 1)
            x0 = max(0, top_x - suppression)
            x1 = min(work.shape[1], top_x + suppression + 1)
            work[y0:y1, x0:x1] = 0.0

        if not candidates:
            return []

        weights = np.array([max(item[2], 0.0) for item in candidates], dtype=np.float64)
        if float(np.sum(weights)) <= 0.0:
            weights[:] = 1.0
        probs = weights / np.sum(weights)
        take = min(int(count), len(candidates))
        chosen = rng.choice(len(candidates), size=take, replace=False, p=probs)
        return [
            (
                float(np.clip(candidates[int(idx)][0], 0.0, self.width - 1.0)),
                float(np.clip(candidates[int(idx)][1], 0.0, self.height - 1.0)),
            )
            for idx in np.atleast_1d(chosen)
        ]

    @staticmethod
    def _shape_type_for_location(
        stage,
        structure_map: np.ndarray,
        linearity_map: np.ndarray,
        x: int,
        y: int,
        rng: np.random.Generator,
    ) -> int:
        structure = float(structure_map[y, x])
        linearity = float(linearity_map[y, x])
        allowed = stage.allowed_shapes
        if structure < 0.18:
            return SHAPE_ELLIPSE if SHAPE_ELLIPSE in allowed else allowed[0]
        if SHAPE_THIN_STROKE in allowed and structure >= 0.62 and linearity >= 0.68:
            return SHAPE_THIN_STROKE
        if SHAPE_TRIANGLE in allowed and structure >= 0.42 and linearity >= 0.45:
            return SHAPE_TRIANGLE
        if SHAPE_QUAD in allowed and structure >= 0.35 and rng.random() < 0.15:
            return SHAPE_QUAD
        if SHAPE_ELLIPSE in allowed:
            return SHAPE_ELLIPSE
        return allowed[int(rng.integers(0, len(allowed)))]

    @staticmethod
    def _aspect_ratio(structure: float) -> float:
        if structure <= 0.10:
            return 1.10
        if structure >= 0.20:
            return 3.00
        return float(1.10 + ((structure - 0.10) / 0.10) * (3.00 - 1.10))

    def random_candidate(
        self,
        stage,
        center_x: float,
        center_y: float,
        structure_map: np.ndarray,
        angle_map: np.ndarray,
        linearity_map: np.ndarray,
        rng: np.random.Generator,
    ) -> ShapeCandidate:
        px = int(np.clip(round(center_x), 0, self.width - 1))
        py = int(np.clip(round(center_y), 0, self.height - 1))
        structure = float(structure_map[py, px])
        angle = float(angle_map[py, px])
        shape_type = self._shape_type_for_location(
            stage,
            structure_map,
            linearity_map,
            px,
            py,
            rng,
        )

        jitter = max(0.5, float(stage.region_window) * 0.45)
        x = float(
            np.clip(center_x + rng.uniform(-jitter, jitter), 0.0, self.width - 1.0)
        )
        y = float(
            np.clip(center_y + rng.uniform(-jitter, jitter), 0.0, self.height - 1.0)
        )
        alpha = float(rng.uniform(stage.alpha_min, stage.alpha_max))

        params = np.zeros((6,), dtype=np.float32)
        if shape_type == SHAPE_THIN_STROKE:
            length = float(
                np.clip(
                    rng.uniform(stage.size_min, stage.size_max)
                    * (1.1 + 1.2 * structure),
                    stage.size_min,
                    stage.size_max * 1.8,
                )
            )
            width = float(max(1.0, 0.15 * length))
            rotation = float(angle + 0.5 * np.pi + rng.uniform(-0.18, 0.18))
            params[0] = x + float(np.cos(rotation) * length)
            params[1] = y + float(np.sin(rotation) * length)
            params[2] = width
            return ShapeCandidate(
                center_x=x,
                center_y=y,
                size_x=0.5 * length,
                size_y=0.5 * width,
                rotation=rotation,
                alpha=alpha,
                shape_type=shape_type,
                shape_params=params,
                color=np.zeros((3,), dtype=np.float32),
            )

        major = float(rng.uniform(stage.size_min, stage.size_max))
        minor = float(
            max(stage.size_min * 0.35, major / max(self._aspect_ratio(structure), 1.0))
        )
        if structure >= 0.12:
            rotation = float(angle + 0.5 * np.pi + rng.uniform(-0.35, 0.35))
        else:
            rotation = float(rng.uniform(0.0, np.pi))
        return ShapeCandidate(
            center_x=x,
            center_y=y,
            size_x=major,
            size_y=minor,
            rotation=rotation,
            alpha=alpha,
            shape_type=shape_type,
            shape_params=params,
            color=np.zeros((3,), dtype=np.float32),
        )

    def mutate_candidate(
        self, candidate: ShapeCandidate, stage, rng: np.random.Generator
    ) -> ShapeCandidate:
        mutated = candidate.copy()
        rot_step = float(np.deg2rad(stage.mutation_rotation_deg))
        max_size = float(max(stage.size_max, stage.size_min))

        for _ in range(int(rng.integers(1, 3))):
            op = int(rng.integers(0, 5))
            if op == 0:
                mutated.center_x = float(
                    np.clip(
                        mutated.center_x
                        + rng.choice([-1.0, 1.0]) * stage.mutation_shift_px,
                        0.0,
                        self.width - 1.0,
                    )
                )
            elif op == 1:
                mutated.center_y = float(
                    np.clip(
                        mutated.center_y
                        + rng.choice([-1.0, 1.0]) * stage.mutation_shift_px,
                        0.0,
                        self.height - 1.0,
                    )
                )
            elif op == 2:
                mutated.size_x = float(
                    np.clip(
                        mutated.size_x
                        * (1.0 + rng.choice([-1.0, 1.0]) * stage.mutation_size_ratio),
                        stage.size_min,
                        max_size,
                    )
                )
            elif op == 3:
                mutated.size_y = float(
                    np.clip(
                        mutated.size_y
                        * (1.0 + rng.choice([-1.0, 1.0]) * stage.mutation_size_ratio),
                        max(stage.size_min * 0.25, 0.8),
                        max_size,
                    )
                )
            else:
                mutated.rotation = float(
                    mutated.rotation + rng.choice([-1.0, 1.0]) * rot_step
                )

        if mutated.shape_type == SHAPE_THIN_STROKE:
            mutated.size_x = float(
                np.clip(mutated.size_x, stage.size_min * 0.5, max_size)
            )
            mutated.size_y = float(
                np.clip(mutated.size_y, 0.5, max(0.5, stage.size_max * 0.25))
            )
            length = max(mutated.size_x * 2.0, 2.0)
            mutated.shape_params[0] = mutated.center_x + float(
                np.cos(mutated.rotation) * length
            )
            mutated.shape_params[1] = mutated.center_y + float(
                np.sin(mutated.rotation) * length
            )
            mutated.shape_params[2] = max(mutated.size_y * 2.0, 1.0)
        else:
            mutated.size_x = float(np.clip(mutated.size_x, stage.size_min, max_size))
            mutated.size_y = float(
                np.clip(mutated.size_y, max(stage.size_min * 0.25, 0.8), max_size)
            )
        return mutated

    def search_next_shape(
        self,
        stage,
        guide_map: np.ndarray,
        structure_map: np.ndarray,
        angle_map: np.ndarray,
        linearity_map: np.ndarray,
        rng: np.random.Generator,
    ) -> ShapeCandidate | None:
        centers = self.sample_error_centers(
            guide_map,
            stage.candidate_count,
            stage.top_k_regions,
            stage.region_window,
            rng,
        )
        if not centers:
            return None

        best: ShapeCandidate | None = None
        for center_x, center_y in centers:
            seed_candidate = self.random_candidate(
                stage,
                center_x,
                center_y,
                structure_map,
                angle_map,
                linearity_map,
                rng,
            )
            scored = self.evaluate_candidate(seed_candidate, stage.softness)
            if best is None or scored.mse < best.mse:
                best = scored

        if best is None or best.mse >= self.current_mse:
            return None

        stagnation = 0
        for _ in range(int(stage.mutation_steps)):
            mutant = self.mutate_candidate(best, stage, rng)
            scored = self.evaluate_candidate(mutant, stage.softness)
            if scored.mse + 1e-12 < best.mse:
                best = scored
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= max(12, int(stage.mutation_steps) // 4):
                    break

        if best.mse >= self.current_mse:
            return None
        return best

    def commit_shape(self, candidate: ShapeCandidate) -> None:
        if candidate.canvas_tensor is None:
            raise ValueError("Candidate must be evaluated before commit")

        self.polygons.centers = np.vstack(
            [self.polygons.centers, [candidate.center_x, candidate.center_y]]
        )
        self.polygons.sizes = np.vstack(
            [self.polygons.sizes, [candidate.size_x, candidate.size_y]]
        )
        self.polygons.rotations = np.append(self.polygons.rotations, candidate.rotation)
        self.polygons.colors = np.vstack([self.polygons.colors, candidate.color])
        self.polygons.alphas = np.append(self.polygons.alphas, candidate.alpha)
        self.polygons.shape_types = np.append(
            self.polygons.shape_types, candidate.shape_type
        )
        self.polygons.shape_params = np.vstack(
            [self.polygons.shape_params, candidate.shape_params]
        )

        self.current_canvas_tensor = candidate.canvas_tensor
        self.current_mse = float(candidate.mse)
        self.loss_history.append(self.current_mse)
