from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.live_renderer import (
    SHAPE_ELLIPSE,
    LivePolygonBatch,
    SoftRenderResult,
    SoftRasterizer,
)


@dataclass(frozen=True)
class LiveOptimizerConfig:
    color_lr: float = 0.05
    position_lr: float = 0.002
    size_lr: float = 0.0005
    alpha_lr: float = 0.02
    color_decay_steps: int = 200
    position_decay_steps: int = 500
    size_decay_steps: int = 500
    alpha_decay_steps: int = 300
    position_eps_px: float = 2.0
    size_eps_ratio: float = 0.10
    min_size: float = 1.0
    max_alpha: float = 1.0
    min_alpha: float = 0.0


class _AdamState:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def step(self, grad: np.ndarray, lr: float) -> np.ndarray:
        self.t += 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        self.m = beta1 * self.m + (1.0 - beta1) * grad
        self.v = beta2 * self.v + (1.0 - beta2) * (grad * grad)

        bias1 = 1.0 - beta1**self.t
        bias2 = 1.0 - beta2**self.t

        m_hat = self.m / max(bias1, 1e-8)
        v_hat = self.v / max(bias2, 1e-8)
        return (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32, copy=False)


class LiveJointOptimizer:
    """Joint optimizer for all polygons using soft rasterization and grouped updates."""

    def __init__(
        self,
        *,
        target_image: np.ndarray,
        rasterizer: SoftRasterizer,
        polygons: LivePolygonBatch,
        config: LiveOptimizerConfig | None = None,
    ) -> None:
        if target_image.ndim != 3 or target_image.shape[2] != 3:
            raise ValueError("target_image must have shape (H, W, 3).")

        target = target_image.astype(np.float32, copy=False)
        if target.shape[:2] != (rasterizer.height, rasterizer.width):
            raise ValueError(
                "target_image spatial dimensions must match rasterizer grid size."
            )

        self.target = target
        self.rasterizer = rasterizer
        self.polygons = polygons
        self.config = LiveOptimizerConfig() if config is None else config

        self._color_adam = _AdamState(self.polygons.colors.shape)
        self._position_adam = _AdamState(self.polygons.centers.shape)
        self._size_adam = _AdamState(self.polygons.sizes.shape)
        self._alpha_adam = _AdamState(self.polygons.alphas.shape)

        self.step_count = 0
        initial = self.rasterizer.render(self.polygons, softness=2.0)
        self.current_canvas = initial.canvas
        self.loss_history: list[float] = [self._loss(self.current_canvas)]

    @staticmethod
    def _loss(canvas: np.ndarray, target: np.ndarray | None = None) -> float:
        if target is None:
            raise ValueError("target must be provided")
        diff = target - canvas
        return float(np.mean(diff * diff, dtype=np.float32))

    def _current_loss(self) -> float:
        return self.loss_history[-1]

    @staticmethod
    def _decayed_lr(base_lr: float, step: int, decay_steps: int) -> float:
        if decay_steps <= 0:
            return float(base_lr)
        return float(base_lr * (0.5 ** (step // decay_steps)))

    def _color_gradient(self, render: SoftRenderResult) -> np.ndarray:
        residual = self.target - render.canvas
        scale = -2.0 / float(self.target.size)
        grad = scale * np.einsum(
            "nhw,hwc->nc", render.effective_alpha, residual, optimize=True
        )
        return grad.astype(np.float32, copy=False)

    def _alpha_gradient(self, render: SoftRenderResult) -> np.ndarray:
        residual = self.target - render.canvas
        scale = -2.0 / float(self.target.size)
        color_projection = np.einsum(
            "hwc,nc->nhw", residual, self.polygons.colors, optimize=True
        )
        grad = scale * np.sum(render.coverage * color_projection, axis=(1, 2))
        return grad.astype(np.float32, copy=False)

    def _local_trial_loss(
        self,
        *,
        index: int,
        base_render: SoftRenderResult,
        softness: float,
        center_x: float,
        center_y: float,
        size_x: float,
        size_y: float,
    ) -> float:
        coverage_new = self.rasterizer.single_coverage_from_values(
            shape_type=int(self.polygons.shape_types[index]),
            center_x=float(center_x),
            center_y=float(center_y),
            size_x=float(max(size_x, self.config.min_size)),
            size_y=float(max(size_y, self.config.min_size)),
            rotation=float(self.polygons.rotations[index]),
            softness=softness,
        )

        alpha = float(np.clip(self.polygons.alphas[index], 0.0, 1.0))
        old_eff = base_render.effective_alpha[index]
        new_eff = coverage_new * alpha

        delta = (new_eff - old_eff)[:, :, None]
        color = self.polygons.colors[index][None, None, :]
        trial_canvas = base_render.canvas + delta * (color - base_render.canvas)
        trial_canvas = np.clip(trial_canvas, 0.0, 1.0).astype(np.float32, copy=False)
        return self._loss(trial_canvas, self.target)

    def _position_and_size_gradients(
        self,
        render: SoftRenderResult,
        softness: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.polygons.count
        pos_grad = np.zeros((n, 2), dtype=np.float32)
        size_grad = np.zeros((n, 2), dtype=np.float32)

        eps_pos = float(self.config.position_eps_px)

        for i in range(n):
            cx = float(self.polygons.centers[i, 0])
            cy = float(self.polygons.centers[i, 1])
            sx = float(self.polygons.sizes[i, 0])
            sy = float(self.polygons.sizes[i, 1])

            x_plus = self._local_trial_loss(
                index=i,
                base_render=render,
                softness=softness,
                center_x=min(cx + eps_pos, self.rasterizer.width - 1.0),
                center_y=cy,
                size_x=sx,
                size_y=sy,
            )
            x_minus = self._local_trial_loss(
                index=i,
                base_render=render,
                softness=softness,
                center_x=max(cx - eps_pos, 0.0),
                center_y=cy,
                size_x=sx,
                size_y=sy,
            )
            y_plus = self._local_trial_loss(
                index=i,
                base_render=render,
                softness=softness,
                center_x=cx,
                center_y=min(cy + eps_pos, self.rasterizer.height - 1.0),
                size_x=sx,
                size_y=sy,
            )
            y_minus = self._local_trial_loss(
                index=i,
                base_render=render,
                softness=softness,
                center_x=cx,
                center_y=max(cy - eps_pos, 0.0),
                size_x=sx,
                size_y=sy,
            )

            pos_grad[i, 0] = (x_plus - x_minus) / (2.0 * eps_pos)
            pos_grad[i, 1] = (y_plus - y_minus) / (2.0 * eps_pos)

            eps_sx = max(abs(sx) * float(self.config.size_eps_ratio), 0.5)
            eps_sy = max(abs(sy) * float(self.config.size_eps_ratio), 0.5)

            sx_plus_val = sx + eps_sx
            sx_minus_val = max(sx - eps_sx, self.config.min_size)
            sy_plus_val = sy + eps_sy
            sy_minus_val = max(sy - eps_sy, self.config.min_size)

            sx_plus = self._local_trial_loss(
                index=i,
                base_render=render,
                softness=softness,
                center_x=cx,
                center_y=cy,
                size_x=sx_plus_val,
                size_y=sy,
            )
            sx_minus = self._local_trial_loss(
                index=i,
                base_render=render,
                softness=softness,
                center_x=cx,
                center_y=cy,
                size_x=sx_minus_val,
                size_y=sy,
            )
            sy_plus = self._local_trial_loss(
                index=i,
                base_render=render,
                softness=softness,
                center_x=cx,
                center_y=cy,
                size_x=sx,
                size_y=sy_plus_val,
            )
            sy_minus = self._local_trial_loss(
                index=i,
                base_render=render,
                softness=softness,
                center_x=cx,
                center_y=cy,
                size_x=sx,
                size_y=sy_minus_val,
            )

            sx_den = max(sx_plus_val - sx_minus_val, 1e-6)
            sy_den = max(sy_plus_val - sy_minus_val, 1e-6)
            size_grad[i, 0] = (sx_plus - sx_minus) / sx_den
            size_grad[i, 1] = (sy_plus - sy_minus) / sy_den

        return pos_grad, size_grad

    def step(self, softness: float) -> float:
        if softness <= 0.0:
            raise ValueError("softness must be positive.")

        render = self.rasterizer.render(self.polygons, softness=softness)
        previous_loss = self._loss(render.canvas, self.target)

        grad_color = self._color_gradient(render)
        grad_alpha = self._alpha_gradient(render)
        grad_position, grad_size = self._position_and_size_gradients(render, softness)

        lr_color = self._decayed_lr(
            self.config.color_lr,
            self.step_count,
            self.config.color_decay_steps,
        )
        lr_position = self._decayed_lr(
            self.config.position_lr,
            self.step_count,
            self.config.position_decay_steps,
        )
        lr_size = self._decayed_lr(
            self.config.size_lr,
            self.step_count,
            self.config.size_decay_steps,
        )
        lr_alpha = self._decayed_lr(
            self.config.alpha_lr,
            self.step_count,
            self.config.alpha_decay_steps,
        )

        old_centers = np.array(self.polygons.centers, copy=True)
        old_sizes = np.array(self.polygons.sizes, copy=True)
        old_colors = np.array(self.polygons.colors, copy=True)
        old_alphas = np.array(self.polygons.alphas, copy=True)

        self.polygons.colors = np.clip(
            self.polygons.colors - self._color_adam.step(grad_color, lr_color),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        self.polygons.alphas = np.clip(
            self.polygons.alphas - self._alpha_adam.step(grad_alpha, lr_alpha),
            self.config.min_alpha,
            self.config.max_alpha,
        ).astype(np.float32, copy=False)

        self.polygons.centers = (
            self.polygons.centers - self._position_adam.step(grad_position, lr_position)
        ).astype(np.float32, copy=False)
        self.polygons.centers[:, 0] = np.clip(
            self.polygons.centers[:, 0],
            0.0,
            self.rasterizer.width - 1.0,
        )
        self.polygons.centers[:, 1] = np.clip(
            self.polygons.centers[:, 1],
            0.0,
            self.rasterizer.height - 1.0,
        )

        self.polygons.sizes = np.clip(
            self.polygons.sizes - self._size_adam.step(grad_size, lr_size),
            self.config.min_size,
            max(self.rasterizer.width, self.rasterizer.height),
        ).astype(np.float32, copy=False)

        updated_render = self.rasterizer.render(self.polygons, softness=softness)
        updated_loss = self._loss(updated_render.canvas, self.target)

        # Keep updates near-monotonic for stable convergence diagnostics.
        if updated_loss > previous_loss:
            self.polygons.centers = old_centers
            self.polygons.sizes = old_sizes
            self.polygons.colors = old_colors
            self.polygons.alphas = old_alphas
            updated_render = render
            updated_loss = previous_loss

        self.current_canvas = updated_render.canvas
        self.step_count += 1
        self.loss_history.append(updated_loss)
        return updated_loss

    def run(self, steps: int, *, start_softness: float = 2.0, end_softness: float = 0.5) -> list[float]:
        if steps <= 0:
            return []

        losses: list[float] = []
        for i in range(steps):
            t = i / max(steps - 1, 1)
            softness = start_softness + (end_softness - start_softness) * t
            losses.append(self.step(softness=float(softness)))
        return losses

    def locally_converged(self, window: int = 25, min_delta: float = 1e-5) -> bool:
        if len(self.loss_history) < window + 1:
            return False
        recent = np.array(self.loss_history[-(window + 1) :], dtype=np.float32)
        improvements = recent[:-1] - recent[1:]
        return float(np.mean(improvements, dtype=np.float32)) < float(min_delta)

    def add_polygon(
        self,
        *,
        center_x: float,
        center_y: float,
        size_x: float,
        size_y: float,
        color: tuple[float, float, float],
        alpha: float,
        shape_type: int = SHAPE_ELLIPSE,
        rotation: float = 0.0,
    ) -> None:
        self.polygons.centers = np.concatenate(
            [
                self.polygons.centers,
                np.array([[center_x, center_y]], dtype=np.float32),
            ],
            axis=0,
        )
        self.polygons.sizes = np.concatenate(
            [
                self.polygons.sizes,
                np.array([[max(size_x, self.config.min_size), max(size_y, self.config.min_size)]], dtype=np.float32),
            ],
            axis=0,
        )
        self.polygons.rotations = np.concatenate(
            [self.polygons.rotations, np.array([rotation], dtype=np.float32)],
            axis=0,
        )
        self.polygons.colors = np.concatenate(
            [
                self.polygons.colors,
                np.array(
                    [[
                        float(np.clip(color[0], 0.0, 1.0)),
                        float(np.clip(color[1], 0.0, 1.0)),
                        float(np.clip(color[2], 0.0, 1.0)),
                    ]],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        self.polygons.alphas = np.concatenate(
            [self.polygons.alphas, np.array([float(np.clip(alpha, 0.0, 1.0))], dtype=np.float32)],
            axis=0,
        )
        self.polygons.shape_types = np.concatenate(
            [self.polygons.shape_types, np.array([int(shape_type)], dtype=np.int32)],
            axis=0,
        )

        self._color_adam = _AdamState(self.polygons.colors.shape)
        self._position_adam = _AdamState(self.polygons.centers.shape)
        self._size_adam = _AdamState(self.polygons.sizes.shape)
        self._alpha_adam = _AdamState(self.polygons.alphas.shape)

    def run_with_growth(
        self,
        *,
        total_steps: int,
        max_polygons: int,
        start_softness: float = 2.0,
        end_softness: float = 0.5,
        convergence_window: int = 25,
        convergence_delta: float = 1e-5,
    ) -> list[float]:
        if total_steps <= 0:
            return []
        if max_polygons <= 0:
            raise ValueError("max_polygons must be positive.")

        losses: list[float] = []
        for i in range(total_steps):
            t = i / max(total_steps - 1, 1)
            softness = start_softness + (end_softness - start_softness) * t
            losses.append(self.step(softness=float(softness)))

            if self.polygons.count >= max_polygons:
                continue

            if not self.locally_converged(
                window=convergence_window,
                min_delta=convergence_delta,
            ):
                continue

            residual_map = np.sum((self.target - self.current_canvas) ** 2, axis=2)
            max_index = int(np.argmax(residual_map))
            y, x = divmod(max_index, self.rasterizer.width)
            color = tuple(float(v) for v in self.target[y, x])

            self.add_polygon(
                center_x=float(x),
                center_y=float(y),
                size_x=4.0,
                size_y=4.0,
                color=color,
                alpha=0.30,
                shape_type=SHAPE_ELLIPSE,
                rotation=0.0,
            )

        return losses
