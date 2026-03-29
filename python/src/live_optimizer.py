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
    color_lr: float = 0.08
    position_lr: float = 0.004
    size_lr: float = 0.001
    alpha_lr: float = 0.0
    color_decay_steps: int = 300
    position_decay_steps: int = 1000
    size_decay_steps: int = 2000
    alpha_decay_steps: int = 1000
    position_eps_px: float = 1.5
    size_eps_ratio: float = 0.10
    min_size: float = 1.0
    max_size: float | None = None
    max_alpha: float = 1.0
    min_alpha: float = 0.0
    exact_fd: bool = True
    render_chunk_size: int = 50
    position_update_interval: int = 1
    size_update_interval: int = 1
    max_fd_polygons: int | None = 24
    max_size_fd_polygons: int = 20
    allow_loss_increase: bool = False


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

        m_hat = self.m / max(1.0 - beta1**self.t, 1e-8)
        v_hat = self.v / max(1.0 - beta2**self.t, 1e-8)
        return (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32, copy=False)


class LiveJointOptimizer:
    """Joint polygon optimizer using correct compositing-weight gradients.

    Color updates are analytic and exact for the current soft render.
    Position/size updates are finite-difference and sparsified to top-error polygons.
    """

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

        target = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)
        if target.shape[:2] != (rasterizer.height, rasterizer.width):
            raise ValueError(
                "target_image spatial dimensions must match rasterizer grid size."
            )

        self.target = target
        self.rasterizer = rasterizer
        self.polygons = polygons
        self.config = LiveOptimizerConfig() if config is None else config

        self._reset_adam_states()

        self.step_count = 0
        initial = self.rasterizer.render(
            self.polygons,
            softness=0.2,
            chunk_size=self.config.render_chunk_size,
        )
        self.current_canvas = initial.canvas
        self.loss_history: list[float] = [self._loss(self.current_canvas)]

    def _reset_adam_states(self) -> None:
        self._color_adam = _AdamState(self.polygons.colors.shape)
        self._position_adam = _AdamState(self.polygons.centers.shape)
        self._size_adam = _AdamState(self.polygons.sizes.shape)
        self._alpha_adam = _AdamState(self.polygons.alphas.shape)

    def set_polygons(
        self,
        polygons: LivePolygonBatch,
        *,
        softness: float = 0.5,
        record_loss: bool = True,
    ) -> None:
        self.polygons = polygons
        self._reset_adam_states()
        render = self.rasterizer.render(
            self.polygons,
            softness=float(softness),
            chunk_size=self.config.render_chunk_size,
        )
        self.current_canvas = render.canvas
        if record_loss:
            self.loss_history.append(self._loss(self.current_canvas))

    def restore_state(
        self,
        polygons: LivePolygonBatch,
        canvas: np.ndarray,
        loss_value: float,
        *,
        record_loss: bool = True,
    ) -> None:
        if canvas.shape != self.target.shape:
            raise ValueError("canvas shape must match target shape")
        self.polygons = polygons
        self._reset_adam_states()
        self.current_canvas = np.clip(canvas, 0.0, 1.0).astype(np.float32, copy=False)
        if record_loss:
            self.loss_history.append(float(loss_value))

    def remove_last_polygon(
        self, *, softness: float = 0.5, record_loss: bool = True
    ) -> None:
        if self.polygons.count == 0:
            return
        keep = self.polygons.count - 1
        trimmed = LivePolygonBatch(
            centers=np.array(self.polygons.centers[:keep], copy=True),
            sizes=np.array(self.polygons.sizes[:keep], copy=True),
            rotations=np.array(self.polygons.rotations[:keep], copy=True),
            colors=np.array(self.polygons.colors[:keep], copy=True),
            alphas=np.array(self.polygons.alphas[:keep], copy=True),
            shape_types=np.array(self.polygons.shape_types[:keep], copy=True),
            shape_params=np.array(self.polygons.shape_params[:keep], copy=True),
        )
        self.set_polygons(trimmed, softness=softness, record_loss=record_loss)

    def _loss(
        self,
        canvas: np.ndarray,
        target: np.ndarray | None = None,
        *,
        target_lab: np.ndarray | None = None,
    ) -> float:
        del target_lab
        ref = self.target if target is None else target
        diff = np.clip(canvas, 0.0, 1.0) - np.clip(ref, 0.0, 1.0)
        return float(np.mean(diff * diff, dtype=np.float32))

    def _current_loss(self) -> float:
        return self.loss_history[-1]

    @staticmethod
    def _decayed_lr(base_lr: float, step: int, decay_steps: int) -> float:
        if decay_steps <= 0:
            return float(base_lr)
        return float(base_lr * (0.5 ** (step // decay_steps)))

    def _color_gradient(self, render: SoftRenderResult) -> np.ndarray:
        # dL/dc_i = (2/n) * sum(weight_i * (canvas - target))
        residual = render.canvas - self.target
        scale = 2.0 / float(self.target.size)
        grad = scale * np.einsum("nhw,hwc->nc", render.weights, residual, optimize=True)
        return grad.astype(np.float32, copy=False)

    def _alpha_gradient(self, _render: SoftRenderResult) -> np.ndarray:
        # Alpha gradient is intentionally disabled in this simplified optimizer core.
        return np.zeros_like(self.polygons.alphas, dtype=np.float32)

    def _select_fd_indices(
        self, render: SoftRenderResult, topk: int | None
    ) -> np.ndarray:
        n = self.polygons.count
        if n == 0:
            return np.zeros((0,), dtype=np.int32)

        if topk is None or topk >= n:
            return np.arange(n, dtype=np.int32)

        residual_map = np.mean(
            (render.canvas - self.target) ** 2, axis=2, dtype=np.float32
        )
        influence = np.einsum("nhw,hw->n", render.weights, residual_map, optimize=True)
        k = int(max(1, min(topk, n)))
        return np.argpartition(influence, -k)[-k:].astype(np.int32)

    def _trial_loss_for_geometry(
        self,
        *,
        index: int,
        softness: float,
        center_x: float,
        center_y: float,
        size_x: float,
        size_y: float,
    ) -> float:
        old_center = np.array(self.polygons.centers[index], copy=True)
        old_size = np.array(self.polygons.sizes[index], copy=True)
        try:
            self.polygons.centers[index, 0] = float(center_x)
            self.polygons.centers[index, 1] = float(center_y)
            self.polygons.sizes[index, 0] = float(max(size_x, self.config.min_size))
            self.polygons.sizes[index, 1] = float(max(size_y, self.config.min_size))
            trial_render = self.rasterizer.render(
                self.polygons,
                softness=softness,
                chunk_size=self.config.render_chunk_size,
            )
            return self._loss(trial_render.canvas)
        finally:
            self.polygons.centers[index] = old_center
            self.polygons.sizes[index] = old_size

    def _position_gradients(
        self,
        render: SoftRenderResult,
        softness: float,
    ) -> np.ndarray:
        n = self.polygons.count
        pos_grad = np.zeros((n, 2), dtype=np.float32)
        if n == 0:
            return pos_grad

        indices = self._select_fd_indices(render, self.config.max_fd_polygons)
        eps_pos = float(self.config.position_eps_px)

        for i in indices:
            cx = float(self.polygons.centers[i, 0])
            cy = float(self.polygons.centers[i, 1])
            sx = float(self.polygons.sizes[i, 0])
            sy = float(self.polygons.sizes[i, 1])

            x_plus = self._trial_loss_for_geometry(
                index=i,
                softness=softness,
                center_x=min(cx + eps_pos, self.rasterizer.width - 1.0),
                center_y=cy,
                size_x=sx,
                size_y=sy,
            )
            x_minus = self._trial_loss_for_geometry(
                index=i,
                softness=softness,
                center_x=max(cx - eps_pos, 0.0),
                center_y=cy,
                size_x=sx,
                size_y=sy,
            )
            y_plus = self._trial_loss_for_geometry(
                index=i,
                softness=softness,
                center_x=cx,
                center_y=min(cy + eps_pos, self.rasterizer.height - 1.0),
                size_x=sx,
                size_y=sy,
            )
            y_minus = self._trial_loss_for_geometry(
                index=i,
                softness=softness,
                center_x=cx,
                center_y=max(cy - eps_pos, 0.0),
                size_x=sx,
                size_y=sy,
            )

            pos_grad[i, 0] = (x_plus - x_minus) / (2.0 * eps_pos)
            pos_grad[i, 1] = (y_plus - y_minus) / (2.0 * eps_pos)

        return pos_grad

    def _size_gradients(
        self,
        render: SoftRenderResult,
        softness: float,
    ) -> np.ndarray:
        n = self.polygons.count
        size_grad = np.zeros((n, 2), dtype=np.float32)
        if n == 0:
            return size_grad

        indices = self._select_fd_indices(render, int(self.config.max_size_fd_polygons))

        for i in indices:
            cx = float(self.polygons.centers[i, 0])
            cy = float(self.polygons.centers[i, 1])
            sx = float(self.polygons.sizes[i, 0])
            sy = float(self.polygons.sizes[i, 1])

            eps_sx = max(abs(sx) * float(self.config.size_eps_ratio), 0.4)
            eps_sy = max(abs(sy) * float(self.config.size_eps_ratio), 0.4)

            sx_plus_val = sx + eps_sx
            sx_minus_val = max(sx - eps_sx, self.config.min_size)
            sy_plus_val = sy + eps_sy
            sy_minus_val = max(sy - eps_sy, self.config.min_size)

            sx_plus = self._trial_loss_for_geometry(
                index=i,
                softness=softness,
                center_x=cx,
                center_y=cy,
                size_x=sx_plus_val,
                size_y=sy,
            )
            sx_minus = self._trial_loss_for_geometry(
                index=i,
                softness=softness,
                center_x=cx,
                center_y=cy,
                size_x=sx_minus_val,
                size_y=sy,
            )
            sy_plus = self._trial_loss_for_geometry(
                index=i,
                softness=softness,
                center_x=cx,
                center_y=cy,
                size_x=sx,
                size_y=sy_plus_val,
            )
            sy_minus = self._trial_loss_for_geometry(
                index=i,
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

        return size_grad

    def step(self, softness: float) -> float:
        if softness <= 0.0:
            raise ValueError("softness must be positive.")

        render = self.rasterizer.render(
            self.polygons,
            softness=softness,
            chunk_size=self.config.render_chunk_size,
        )
        baseline_loss = float(self.loss_history[-1])

        grad_color = self._color_gradient(render)
        grad_alpha = self._alpha_gradient(render)

        update_position = self.config.position_update_interval > 0 and (
            self.step_count % self.config.position_update_interval == 0
        )
        update_size = self.config.size_update_interval > 0 and (
            self.step_count % self.config.size_update_interval == 0
        )

        if update_position:
            grad_position = self._position_gradients(render, softness)
        else:
            grad_position = np.zeros_like(self.polygons.centers, dtype=np.float32)

        if update_size:
            grad_size = self._size_gradients(render, softness)
        else:
            grad_size = np.zeros_like(self.polygons.sizes, dtype=np.float32)

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

        if lr_alpha > 0.0:
            self.polygons.alphas = np.clip(
                self.polygons.alphas - self._alpha_adam.step(grad_alpha, lr_alpha),
                self.config.min_alpha,
                self.config.max_alpha,
            ).astype(np.float32, copy=False)

        if update_position:
            self.polygons.centers = (
                self.polygons.centers
                - self._position_adam.step(grad_position, lr_position)
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

        if update_size:
            max_size = (
                max(self.rasterizer.width, self.rasterizer.height)
                if self.config.max_size is None
                else float(self.config.max_size)
            )
            self.polygons.sizes = np.clip(
                self.polygons.sizes - self._size_adam.step(grad_size, lr_size),
                self.config.min_size,
                max_size,
            ).astype(np.float32, copy=False)

        updated_render = self.rasterizer.render(
            self.polygons,
            softness=softness,
            chunk_size=self.config.render_chunk_size,
        )
        updated_loss = self._loss(updated_render.canvas)

        if (not self.config.allow_loss_increase) and (updated_loss > baseline_loss):
            self.polygons.centers = old_centers
            self.polygons.sizes = old_sizes
            self.polygons.colors = old_colors
            self.polygons.alphas = old_alphas
            updated_render = render
            updated_loss = baseline_loss

        self.current_canvas = updated_render.canvas
        self.step_count += 1
        self.loss_history.append(updated_loss)
        return updated_loss

    def run(
        self,
        steps: int,
        *,
        start_softness: float = 3.0,
        end_softness: float = 0.3,
    ) -> list[float]:
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
        shape_params: np.ndarray | None = None,
    ) -> None:
        if shape_params is None:
            shape_params = np.zeros((6,), dtype=np.float32)
        param_values = np.asarray(shape_params, dtype=np.float32).reshape(6)

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
                np.array(
                    [
                        [
                            max(size_x, self.config.min_size),
                            max(size_y, self.config.min_size),
                        ]
                    ],
                    dtype=np.float32,
                ),
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
                    [
                        [
                            float(np.clip(color[0], 0.0, 1.0)),
                            float(np.clip(color[1], 0.0, 1.0)),
                            float(np.clip(color[2], 0.0, 1.0)),
                        ]
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        self.polygons.alphas = np.concatenate(
            [
                self.polygons.alphas,
                np.array([float(np.clip(alpha, 0.0, 1.0))], dtype=np.float32),
            ],
            axis=0,
        )
        self.polygons.shape_types = np.concatenate(
            [self.polygons.shape_types, np.array([int(shape_type)], dtype=np.int32)],
            axis=0,
        )
        self.polygons.shape_params = np.concatenate(
            [self.polygons.shape_params, param_values.reshape(1, 6)],
            axis=0,
        )

        self._reset_adam_states()

    def run_with_growth(
        self,
        *,
        total_steps: int,
        max_polygons: int,
        start_softness: float = 3.0,
        end_softness: float = 0.3,
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

            residual_map = np.mean((self.target - self.current_canvas) ** 2, axis=2)
            max_index = int(np.argmax(residual_map))
            y, x = divmod(max_index, self.rasterizer.width)
            color_hint = (
                float(self.target[y, x, 0]),
                float(self.target[y, x, 1]),
                float(self.target[y, x, 2]),
            )

            self.add_polygon(
                center_x=float(x),
                center_y=float(y),
                size_x=4.0,
                size_y=4.0,
                color=color_hint,
                alpha=0.30,
                shape_type=SHAPE_ELLIPSE,
                rotation=0.0,
            )

        return losses
