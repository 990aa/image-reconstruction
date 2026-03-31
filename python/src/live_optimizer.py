from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core_renderer import (
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
    LivePolygonBatch,
    SoftRasterizer,
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
    coverage: np.ndarray | None = None
    canvas: np.ndarray | None = None
    residual: np.ndarray | None = None

    def copy(self) -> ShapeCandidate:
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
            coverage=None if self.coverage is None else np.array(self.coverage, copy=True),
            canvas=None if self.canvas is None else np.array(self.canvas, copy=True),
            residual=None if self.residual is None else np.array(self.residual, copy=True),
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


class SequentialHillClimber:
    def __init__(
        self,
        *,
        target_image: np.ndarray,
        rasterizer: SoftRasterizer,
        polygons: LivePolygonBatch | None = None,
        background_color: np.ndarray | None = None,
    ) -> None:
        if target_image.ndim != 3 or target_image.shape[2] != 3:
            raise ValueError("target_image must have shape (H, W, 3)")

        self.target = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)
        self.rasterizer = rasterizer
        self.height, self.width = self.target.shape[:2]

        if (self.height, self.width) != (self.rasterizer.height, self.rasterizer.width):
            raise ValueError("target_image dimensions must match the rasterizer")

        bg = (
            np.mean(self.target, axis=(0, 1), dtype=np.float32)
            if background_color is None
            else np.asarray(background_color, dtype=np.float32).reshape(3)
        )
        self.background_color = np.clip(bg, 0.0, 1.0).astype(np.float32, copy=False)

        self.polygons = make_empty_live_batch() if polygons is None else polygons.copy()
        self.current_canvas = self.render_polygons(
            self.polygons,
            softness=1.0,
        )
        self.current_residual = (self.current_canvas - self.target).astype(
            np.float32, copy=False
        )
        self.current_mse = float(
            np.mean(self.current_residual * self.current_residual, dtype=np.float32)
        )
        self.loss_history: list[float] = [self.current_mse]

    def render_polygons(
        self,
        polygons: LivePolygonBatch,
        *,
        softness: float,
        chunk_size: int = 64,
    ) -> np.ndarray:
        canvas = np.broadcast_to(
            self.background_color.reshape(1, 1, 3),
            (self.height, self.width, 3),
        ).astype(np.float32, copy=True)
        if polygons.count == 0:
            return canvas

        coverage = self.rasterizer.coverage_batch(
            polygons,
            softness=float(max(softness, 1e-3)),
            chunk_size=chunk_size,
        )
        for idx in range(polygons.count):
            weight = coverage[idx] * float(np.clip(polygons.alphas[idx], 0.0, 1.0))
            canvas = canvas + weight[:, :, None] * (
                polygons.colors[idx][None, None, :] - canvas
            )
        return np.clip(canvas, 0.0, 1.0).astype(np.float32, copy=False)

    def _coverage_from_candidate(
        self,
        candidate: ShapeCandidate,
        *,
        softness: float,
    ) -> np.ndarray:
        if candidate.shape_type == SHAPE_QUAD:
            return self.rasterizer._quad_coverage_params(
                center_x=candidate.center_x,
                center_y=candidate.center_y,
                axis_x=candidate.size_x,
                axis_y=candidate.size_y,
                rotation=candidate.rotation,
                softness=softness,
            )
        if candidate.shape_type == SHAPE_TRIANGLE:
            return self.rasterizer._triangle_coverage_params(
                center_x=candidate.center_x,
                center_y=candidate.center_y,
                axis_x=candidate.size_x,
                axis_y=candidate.size_y,
                rotation=candidate.rotation,
                softness=softness,
            )
        if candidate.shape_type == SHAPE_THIN_STROKE:
            return self.rasterizer._thin_stroke_coverage_params(
                center_x=candidate.center_x,
                center_y=candidate.center_y,
                shape_params=candidate.shape_params,
                softness=softness,
            )
        return self.rasterizer._ellipse_coverage_params(
            center_x=candidate.center_x,
            center_y=candidate.center_y,
            axis_x=candidate.size_x,
            axis_y=candidate.size_y,
            rotation=candidate.rotation,
            softness=softness,
        )

    def _exact_color(
        self,
        weight: np.ndarray,
    ) -> np.ndarray:
        weight3 = weight[:, :, None]
        denom = float(np.sum(weight * weight, dtype=np.float32))
        if denom <= 1e-8:
            return np.mean(self.target, axis=(0, 1), dtype=np.float32).astype(
                np.float32, copy=False
            )

        numer = np.sum(
            weight3 * (self.target - self.current_canvas * (1.0 - weight3)),
            axis=(0, 1),
            dtype=np.float32,
        )
        return np.clip(numer / denom, 0.0, 1.0).astype(np.float32, copy=False)

    def evaluate_candidate(
        self,
        candidate: ShapeCandidate,
        *,
        softness: float,
    ) -> ShapeCandidate:
        coverage = self._coverage_from_candidate(candidate, softness=softness)
        weight = coverage * float(np.clip(candidate.alpha, 0.0, 1.0))
        color = self._exact_color(weight)
        canvas = self.current_canvas + weight[:, :, None] * (
            color[None, None, :] - self.current_canvas
        )
        residual = canvas - self.target
        mse = float(np.mean(residual * residual, dtype=np.float32))

        scored = candidate.copy()
        scored.color = color.astype(np.float32, copy=False)
        scored.coverage = coverage.astype(np.float32, copy=False)
        scored.canvas = np.clip(canvas, 0.0, 1.0).astype(np.float32, copy=False)
        scored.residual = residual.astype(np.float32, copy=False)
        scored.mse = mse
        return scored

    def _region_sum_map(self, error_map: np.ndarray, window: int) -> np.ndarray:
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
        *,
        count: int,
        top_k: int,
        window: int,
        rng: np.random.Generator,
    ) -> list[tuple[float, float]]:
        if count <= 0:
            return []

        region_scores = self._region_sum_map(
            np.clip(guide_map.astype(np.float32, copy=False), 0.0, None),
            window=max(1, int(window)),
        )
        work = np.array(region_scores, copy=True)
        candidates: list[tuple[float, float, float]] = []
        suppression = max(1, int(window // 2))

        while len(candidates) < max(1, int(top_k)):
            flat_idx = int(np.argmax(work))
            score = float(work.reshape(-1)[flat_idx])
            if score <= 1e-12:
                break
            top_y, top_x = divmod(flat_idx, work.shape[1])
            center_x = float(top_x + 0.5 * max(window - 1, 0))
            center_y = float(top_y + 0.5 * max(window - 1, 0))
            candidates.append((center_x, center_y, score))
            y0 = max(0, top_y - suppression)
            y1 = min(work.shape[0], top_y + suppression + 1)
            x0 = max(0, top_x - suppression)
            x1 = min(work.shape[1], top_x + suppression + 1)
            work[y0:y1, x0:x1] = 0.0

        if not candidates:
            return []

        take = min(int(count), len(candidates))
        weights = np.array([max(item[2], 0.0) for item in candidates], dtype=np.float64)
        if float(np.sum(weights)) <= 0.0:
            weights[:] = 1.0
        probs = weights / np.sum(weights)
        chosen = rng.choice(len(candidates), size=take, replace=False, p=probs)
        return [
            (
                float(np.clip(candidates[int(idx)][0], 0.0, self.width - 1.0)),
                float(np.clip(candidates[int(idx)][1], 0.0, self.height - 1.0)),
            )
            for idx in np.atleast_1d(chosen)
        ]

    def _shape_type_for_location(
        self,
        *,
        stage: SequentialStageConfig,
        structure_map: np.ndarray,
        linearity_map: np.ndarray,
        gradient_variance_map: np.ndarray,
        x: int,
        y: int,
        rng: np.random.Generator,
    ) -> int:
        structure = float(structure_map[y, x])
        linearity = float(linearity_map[y, x])
        variance = float(gradient_variance_map[y, x])
        allowed = tuple(int(shape) for shape in stage.allowed_shapes)

        # Smooth/curved regions should stay organic and round.
        if structure < 0.18 or variance < 0.012:
            if SHAPE_ELLIPSE in allowed:
                return SHAPE_ELLIPSE
            return allowed[0]

        # Only allow angular primitives on strict, high-variance edges.
        if (
            SHAPE_THIN_STROKE in allowed
            and structure >= 0.62
            and variance >= 0.030
            and linearity >= 0.68
        ):
            return SHAPE_THIN_STROKE
        if (
            SHAPE_TRIANGLE in allowed
            and structure >= 0.42
            and variance >= 0.020
            and linearity >= 0.45
        ):
            return SHAPE_TRIANGLE
        if (
            SHAPE_QUAD in allowed
            and structure >= 0.35
            and variance >= 0.018
            and rng.random() < 0.15
        ):
            return SHAPE_QUAD
        if SHAPE_ELLIPSE in allowed:
            return SHAPE_ELLIPSE
        return allowed[int(rng.integers(0, len(allowed)))]

    def _aspect_ratio(self, structure: float) -> float:
        if structure <= 0.10:
            return 1.10
        if structure >= 0.20:
            return 3.00
        t = (structure - 0.10) / 0.10
        return float(1.10 + t * (3.00 - 1.10))

    def _refresh_shape_params(self, candidate: ShapeCandidate) -> ShapeCandidate:
        updated = candidate.copy()
        params = np.zeros((6,), dtype=np.float32)
        if updated.shape_type == SHAPE_THIN_STROKE:
            length = float(max(updated.size_x * 2.0, 2.0))
            width = float(max(updated.size_y * 2.0, 1.0))
            x1 = float(updated.center_x + np.cos(updated.rotation) * length)
            y1 = float(updated.center_y + np.sin(updated.rotation) * length)
            params[0] = x1
            params[1] = y1
            params[2] = width
        updated.shape_params = params
        return updated

    def random_candidate(
        self,
        *,
        stage: SequentialStageConfig,
        center_x: float,
        center_y: float,
        structure_map: np.ndarray,
        angle_map: np.ndarray,
        linearity_map: np.ndarray,
        gradient_variance_map: np.ndarray,
        rng: np.random.Generator,
    ) -> ShapeCandidate:
        px = int(np.clip(round(center_x), 0, self.width - 1))
        py = int(np.clip(round(center_y), 0, self.height - 1))

        structure = float(structure_map[py, px])
        angle = float(angle_map[py, px])
        shape_type = self._shape_type_for_location(
            stage=stage,
            structure_map=structure_map,
            linearity_map=linearity_map,
            gradient_variance_map=gradient_variance_map,
            x=px,
            y=py,
            rng=rng,
        )

        jitter = max(0.5, float(stage.region_window) * 0.45)
        x = float(np.clip(center_x + rng.uniform(-jitter, jitter), 0.0, self.width - 1.0))
        y = float(
            np.clip(center_y + rng.uniform(-jitter, jitter), 0.0, self.height - 1.0)
        )
        alpha = float(rng.uniform(stage.alpha_min, stage.alpha_max))

        if shape_type == SHAPE_THIN_STROKE:
            length = float(
                np.clip(
                    rng.uniform(stage.size_min, stage.size_max) * (1.1 + 1.2 * structure),
                    stage.size_min,
                    stage.size_max * 1.8,
                )
            )
            width = float(max(1.0, 0.15 * length))
            rotation = float(angle + 0.5 * np.pi + rng.uniform(-0.18, 0.18))
            candidate = ShapeCandidate(
                center_x=x,
                center_y=y,
                size_x=0.5 * length,
                size_y=0.5 * width,
                rotation=rotation,
                alpha=alpha,
                shape_type=shape_type,
                shape_params=np.zeros((6,), dtype=np.float32),
                color=np.zeros((3,), dtype=np.float32),
            )
            return self._refresh_shape_params(candidate)

        major = float(rng.uniform(stage.size_min, stage.size_max))
        aspect = self._aspect_ratio(structure)
        minor = float(max(stage.size_min * 0.35, major / max(aspect, 1.0)))
        rotation = (
            float(angle + 0.5 * np.pi + rng.uniform(-0.35, 0.35))
            if structure >= 0.12
            else float(rng.uniform(0.0, np.pi))
        )

        candidate = ShapeCandidate(
            center_x=x,
            center_y=y,
            size_x=major,
            size_y=minor,
            rotation=rotation,
            alpha=alpha,
            shape_type=shape_type,
            shape_params=np.zeros((6,), dtype=np.float32),
            color=np.zeros((3,), dtype=np.float32),
        )
        return self._refresh_shape_params(candidate)

    def mutate_candidate(
        self,
        candidate: ShapeCandidate,
        *,
        stage: SequentialStageConfig,
        rng: np.random.Generator,
    ) -> ShapeCandidate:
        mutated = candidate.copy()
        rot_step = float(np.deg2rad(stage.mutation_rotation_deg))
        max_size = float(max(stage.size_max, stage.size_min))

        operation_count = int(rng.integers(1, 3))
        for _ in range(operation_count):
            op = int(rng.integers(0, 5))
            if op == 0:
                mutated.center_x = float(
                    np.clip(
                        mutated.center_x + rng.choice([-1.0, 1.0]) * stage.mutation_shift_px,
                        0.0,
                        self.width - 1.0,
                    )
                )
            elif op == 1:
                mutated.center_y = float(
                    np.clip(
                        mutated.center_y + rng.choice([-1.0, 1.0]) * stage.mutation_shift_px,
                        0.0,
                        self.height - 1.0,
                    )
                )
            elif op == 2:
                scale = 1.0 + rng.choice([-1.0, 1.0]) * stage.mutation_size_ratio
                mutated.size_x = float(np.clip(mutated.size_x * scale, stage.size_min, max_size))
            elif op == 3:
                scale = 1.0 + rng.choice([-1.0, 1.0]) * stage.mutation_size_ratio
                mutated.size_y = float(
                    np.clip(mutated.size_y * scale, max(stage.size_min * 0.25, 0.8), max_size)
                )
            else:
                mutated.rotation = float(mutated.rotation + rng.choice([-1.0, 1.0]) * rot_step)

        if mutated.shape_type == SHAPE_THIN_STROKE:
            mutated.size_x = float(np.clip(mutated.size_x, stage.size_min * 0.5, max_size))
            mutated.size_y = float(
                np.clip(mutated.size_y, 0.5, max(0.5, stage.size_max * 0.25))
            )
        else:
            mutated.size_x = float(np.clip(mutated.size_x, stage.size_min, max_size))
            mutated.size_y = float(
                np.clip(mutated.size_y, max(stage.size_min * 0.25, 0.8), max_size)
            )

        return self._refresh_shape_params(mutated)

    def search_next_shape(
        self,
        *,
        stage: SequentialStageConfig,
        guide_map: np.ndarray,
        structure_map: np.ndarray,
        angle_map: np.ndarray,
        linearity_map: np.ndarray,
        gradient_variance_map: np.ndarray,
        rng: np.random.Generator,
    ) -> ShapeCandidate | None:
        centers = self.sample_error_centers(
            guide_map,
            count=stage.candidate_count,
            top_k=stage.top_k_regions,
            window=stage.region_window,
            rng=rng,
        )
        if not centers:
            return None

        best: ShapeCandidate | None = None
        for center_x, center_y in centers:
            candidate = self.random_candidate(
                stage=stage,
                center_x=center_x,
                center_y=center_y,
                structure_map=structure_map,
                angle_map=angle_map,
                linearity_map=linearity_map,
                gradient_variance_map=gradient_variance_map,
                rng=rng,
            )
            scored = self.evaluate_candidate(candidate, softness=stage.softness)
            if best is None or scored.mse < best.mse:
                best = scored

        if best is None or best.mse >= self.current_mse:
            return None

        stagnation = 0
        for _ in range(max(0, int(stage.mutation_steps))):
            mutant = self.mutate_candidate(best, stage=stage, rng=rng)
            scored = self.evaluate_candidate(mutant, softness=stage.softness)
            if scored.mse + 1e-12 < best.mse:
                best = scored
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= max(12, stage.mutation_steps // 4):
                    break

        if best.mse >= self.current_mse:
            return None
        return best

    def commit_shape(self, candidate: ShapeCandidate) -> None:
        if candidate.canvas is None or candidate.residual is None:
            raise ValueError("Candidate must be evaluated before commit.")

        self.polygons.centers = np.concatenate(
            [
                self.polygons.centers,
                np.array([[candidate.center_x, candidate.center_y]], dtype=np.float32),
            ],
            axis=0,
        )
        self.polygons.sizes = np.concatenate(
            [
                self.polygons.sizes,
                np.array([[candidate.size_x, candidate.size_y]], dtype=np.float32),
            ],
            axis=0,
        )
        self.polygons.rotations = np.concatenate(
            [self.polygons.rotations, np.array([candidate.rotation], dtype=np.float32)],
            axis=0,
        )
        self.polygons.colors = np.concatenate(
            [self.polygons.colors, candidate.color.reshape(1, 3).astype(np.float32, copy=False)],
            axis=0,
        )
        self.polygons.alphas = np.concatenate(
            [self.polygons.alphas, np.array([candidate.alpha], dtype=np.float32)],
            axis=0,
        )
        self.polygons.shape_types = np.concatenate(
            [self.polygons.shape_types, np.array([candidate.shape_type], dtype=np.int32)],
            axis=0,
        )
        self.polygons.shape_params = np.concatenate(
            [self.polygons.shape_params, candidate.shape_params.reshape(1, 6)],
            axis=0,
        )

        self.current_canvas = candidate.canvas.astype(np.float32, copy=False)
        self.current_residual = candidate.residual.astype(np.float32, copy=False)
        self.current_mse = float(candidate.mse)
        self.loss_history.append(self.current_mse)
