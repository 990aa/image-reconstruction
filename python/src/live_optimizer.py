from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import color

from src.core_renderer import (
    SHAPE_ELLIPSE,
    ForwardPassResult,
    LivePolygonBatch,
    SoftRenderResult,
    SoftRasterizer,
)


@dataclass(frozen=True)
class LiveOptimizerConfig:
    color_lr: float = 0.04
    position_lr: float = 0.50
    size_lr: float = 0.30
    rotation_lr: float = 0.02
    alpha_lr: float = 0.01
    color_decay_steps: int = 0
    position_decay_steps: int = 0
    size_decay_steps: int = 0
    alpha_decay_steps: int = 0
    position_update_interval: int = 20
    size_update_interval: int = 1
    max_fd_polygons: int = 40
    max_size_fd_polygons: int = 20
    render_chunk_size: int = 50
    checkpoint_stride: int = 10
    position_eps_px: float = 2.0
    size_eps_px: float = 1.0
    size_eps_ratio: float = 0.10
    rotation_eps_rad: float = 0.06
    min_size: float = 3.0
    max_size: float | None = 60.0
    min_alpha: float = 0.05
    max_alpha: float = 0.98
    exact_fd: bool = True
    allow_loss_increase: bool = True
    use_lab_loss: bool = True


class _AdamState:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, int]:
        return (np.array(self.m, copy=True), np.array(self.v, copy=True), int(self.t))

    def restore(self, snapshot: tuple[np.ndarray, np.ndarray, int]) -> None:
        self.m = np.array(snapshot[0], copy=True)
        self.v = np.array(snapshot[1], copy=True)
        self.t = int(snapshot[2])

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

        self.target = np.clip(target_image.astype(np.float32, copy=False), 0.0, 1.0)
        if self.target.shape[:2] != (rasterizer.height, rasterizer.width):
            raise ValueError("target_image dimensions must match rasterizer grid")

        self.target_lab = color.rgb2lab(self.target).astype(np.float32, copy=False)

        self.rasterizer = rasterizer
        self.polygons = polygons
        self.config = LiveOptimizerConfig() if config is None else config

        self.step_count = 0
        self._reset_adam_states()

        first = self.rasterizer.forward_pass(
            self.polygons,
            softness=3.0,
            chunk_size=self.config.render_chunk_size,
            checkpoint_stride=self.config.checkpoint_stride,
            target=None,
            compute_gradients=False,
        )
        self.current_canvas = first.canvas
        self.loss_history: list[float] = [self._loss(self.current_canvas)]

    def _reset_adam_states(self) -> None:
        self._color_adam = _AdamState(self.polygons.colors.shape)
        self._alpha_adam = _AdamState(self.polygons.alphas.shape)
        self._position_adam = _AdamState(self.polygons.centers.shape)
        self._size_adam = _AdamState(self.polygons.sizes.shape)
        self._rotation_adam = _AdamState(self.polygons.rotations.shape)

    def _loss(self, canvas: np.ndarray, target: np.ndarray | None = None) -> float:
        ref = self.target if target is None else np.clip(target, 0.0, 1.0)
        if not self.config.use_lab_loss:
            diff = canvas - ref
            return float(np.mean(diff * diff, dtype=np.float32))

        lab_canvas = color.rgb2lab(np.clip(canvas, 0.0, 1.0)).astype(
            np.float32, copy=False
        )
        ref_lab = self.target_lab if target is None else color.rgb2lab(ref).astype(
            np.float32,
            copy=False,
        )
        diff = lab_canvas - ref_lab
        return float(np.mean(diff * diff, dtype=np.float32))

    def _max_size(self) -> float:
        if self.config.max_size is not None:
            return float(self.config.max_size)
        return float(max(self.rasterizer.width, self.rasterizer.height))

    def _forward(self, softness: float, *, compute_gradients: bool) -> ForwardPassResult:
        return self.rasterizer.forward_pass(
            self.polygons,
            softness=float(max(softness, 1e-3)),
            chunk_size=self.config.render_chunk_size,
            checkpoint_stride=self.config.checkpoint_stride,
            target=self.target if compute_gradients else None,
            compute_gradients=compute_gradients,
        )

    def _render(self, softness: float) -> SoftRenderResult:
        return self.rasterizer.render(
            self.polygons,
            softness=float(max(softness, 1e-3)),
            chunk_size=self.config.render_chunk_size,
        )

    def _color_gradient(self, render: SoftRenderResult) -> np.ndarray:
        if render.canvas.shape != self.target.shape:
            raise ValueError("render canvas shape must match target")
        if render.effective_alpha.shape != render.trans_after.shape:
            raise ValueError("render visibility tensors must have matching shapes")

        residual = render.canvas - self.target
        weights = render.trans_after * render.effective_alpha
        scale = 2.0 / float(self.target.size)
        return (
            scale
            * np.einsum("nhw,hwc->nc", weights, residual, dtype=np.float32)
        ).astype(np.float32, copy=False)

    def _select_fd_indices(self, canvas: np.ndarray) -> np.ndarray:
        n = self.polygons.count
        if n == 0:
            return np.zeros((0,), dtype=np.int32)

        topk = self.config.max_fd_polygons
        if topk is None or topk >= n:
            return np.arange(n, dtype=np.int32)

        residual = np.mean(np.abs(canvas - self.target), axis=2, dtype=np.float32)
        cx = np.clip(
            np.round(self.polygons.centers[:, 0]).astype(np.int32),
            0,
            self.rasterizer.width - 1,
        )
        cy = np.clip(
            np.round(self.polygons.centers[:, 1]).astype(np.int32),
            0,
            self.rasterizer.height - 1,
        )
        footprint = np.maximum(
            self.polygons.sizes[:, 0] * self.polygons.sizes[:, 1],
            1.0,
        )
        scores = residual[cy, cx] * footprint * np.clip(self.polygons.alphas, 0.05, 1.0)

        k = int(max(1, min(topk, n)))
        return np.argpartition(scores, -k)[-k:].astype(np.int32)

    def _candidate_loss(
        self,
        *,
        index: int,
        softness: float,
        checkpoints: dict[int, np.ndarray],
        center: tuple[float, float],
        size: tuple[float, float],
        rotation: float,
    ) -> float:
        stride = max(1, int(self.config.checkpoint_stride))
        start = (int(index) // stride) * stride
        if start not in checkpoints:
            closest = max([key for key in checkpoints if key <= int(index)], default=0)
            start = int(closest)

        base_canvas = checkpoints[start]
        trial = self.rasterizer.render_suffix(
            self.polygons,
            start_index=start,
            base_canvas=base_canvas,
            softness=softness,
            override_index=int(index),
            override_center=center,
            override_size=size,
            override_rotation=rotation,
        )
        return self._loss(trial)

    def _fd_geometry_grads(
        self,
        *,
        softness: float,
        checkpoints: dict[int, np.ndarray],
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_grad = np.zeros_like(self.polygons.centers, dtype=np.float32)
        size_grad = np.zeros_like(self.polygons.sizes, dtype=np.float32)

        eps_pos = float(max(self.config.position_eps_px, 0.25))
        eps_size_base = float(max(self.config.size_eps_px, 0.25))

        max_size = self._max_size()

        for idx in indices:
            i = int(idx)
            cx = float(self.polygons.centers[i, 0])
            cy = float(self.polygons.centers[i, 1])
            sx = float(self.polygons.sizes[i, 0])
            sy = float(self.polygons.sizes[i, 1])
            rot = float(self.polygons.rotations[i])
            eps_size = float(
                max(
                    eps_size_base,
                    self.config.size_eps_ratio * max(abs(sx), abs(sy)),
                )
            )

            x_plus = self._candidate_loss(
                index=i,
                softness=softness,
                checkpoints=checkpoints,
                center=(min(cx + eps_pos, self.rasterizer.width - 1.0), cy),
                size=(sx, sy),
                rotation=rot,
            )
            x_minus = self._candidate_loss(
                index=i,
                softness=softness,
                checkpoints=checkpoints,
                center=(max(cx - eps_pos, 0.0), cy),
                size=(sx, sy),
                rotation=rot,
            )
            y_plus = self._candidate_loss(
                index=i,
                softness=softness,
                checkpoints=checkpoints,
                center=(cx, min(cy + eps_pos, self.rasterizer.height - 1.0)),
                size=(sx, sy),
                rotation=rot,
            )
            y_minus = self._candidate_loss(
                index=i,
                softness=softness,
                checkpoints=checkpoints,
                center=(cx, max(cy - eps_pos, 0.0)),
                size=(sx, sy),
                rotation=rot,
            )

            sx_plus = self._candidate_loss(
                index=i,
                softness=softness,
                checkpoints=checkpoints,
                center=(cx, cy),
                size=(min(sx + eps_size, max_size), sy),
                rotation=rot,
            )
            sx_minus = self._candidate_loss(
                index=i,
                softness=softness,
                checkpoints=checkpoints,
                center=(cx, cy),
                size=(max(sx - eps_size, self.config.min_size), sy),
                rotation=rot,
            )
            sy_plus = self._candidate_loss(
                index=i,
                softness=softness,
                checkpoints=checkpoints,
                center=(cx, cy),
                size=(sx, min(sy + eps_size, max_size)),
                rotation=rot,
            )
            sy_minus = self._candidate_loss(
                index=i,
                softness=softness,
                checkpoints=checkpoints,
                center=(cx, cy),
                size=(sx, max(sy - eps_size, self.config.min_size)),
                rotation=rot,
            )

            pos_grad[i, 0] = (x_plus - x_minus) / (2.0 * eps_pos)
            pos_grad[i, 1] = (y_plus - y_minus) / (2.0 * eps_pos)
            size_grad[i, 0] = (sx_plus - sx_minus) / (2.0 * eps_size)
            size_grad[i, 1] = (sy_plus - sy_minus) / (2.0 * eps_size)

        return pos_grad, size_grad

    def set_polygons(
        self,
        polygons: LivePolygonBatch,
        *,
        softness: float = 1.0,
        record_loss: bool = True,
    ) -> None:
        self.polygons = polygons
        self._reset_adam_states()
        out = self._forward(float(max(softness, 1e-3)), compute_gradients=False)
        self.current_canvas = out.canvas
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
            raise ValueError("canvas shape must match target")
        self.polygons = polygons
        self._reset_adam_states()
        self.current_canvas = np.clip(canvas, 0.0, 1.0).astype(np.float32, copy=False)
        if record_loss:
            self.loss_history.append(float(loss_value))

    def remove_last_polygon(self, *, softness: float = 1.0, record_loss: bool = True) -> None:
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

    def step(self, softness: float) -> float:
        if softness <= 0.0:
            raise ValueError("softness must be positive")

        baseline_render = self._render(float(softness))
        baseline_canvas = baseline_render.canvas
        baseline_loss = self._loss(baseline_canvas)

        old_colors = np.array(self.polygons.colors, copy=True)
        old_alphas = np.array(self.polygons.alphas, copy=True)
        color_adam_snapshot = self._color_adam.snapshot()
        alpha_adam_snapshot = self._alpha_adam.snapshot()

        grad_colors = self._color_gradient(baseline_render)
        grad_alphas: np.ndarray | None = None
        if self.config.alpha_lr > 0.0:
            alpha_state = self._forward(float(softness), compute_gradients=True)
            if alpha_state.grad_alphas is None:
                raise RuntimeError("forward pass did not produce alpha gradients")
            grad_alphas = alpha_state.grad_alphas

        self.polygons.colors = np.clip(
            self.polygons.colors
            - self._color_adam.step(grad_colors, self.config.color_lr),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        if grad_alphas is not None:
            self.polygons.alphas = np.clip(
                self.polygons.alphas
                - self._alpha_adam.step(grad_alphas, self.config.alpha_lr),
                self.config.min_alpha,
                self.config.max_alpha,
            ).astype(np.float32, copy=False)

        color_render = self._render(float(softness))
        color_canvas = color_render.canvas
        color_loss = self._loss(color_canvas)

        if (not self.config.allow_loss_increase) and color_loss > baseline_loss:
            self.polygons.colors = old_colors
            self.polygons.alphas = old_alphas
            self._color_adam.restore(color_adam_snapshot)
            self._alpha_adam.restore(alpha_adam_snapshot)
            color_canvas = baseline_canvas
            color_loss = float(baseline_loss)
            color_render = baseline_render

        run_position = self.config.position_update_interval > 0 and (
            self.step_count % self.config.position_update_interval == 0
        )
        run_size = self.config.size_update_interval > 0 and (
            self.step_count % self.config.size_update_interval == 0
        )
        run_geometry = run_position or run_size

        final_canvas = color_canvas
        final_loss = float(color_loss)

        if run_geometry and self.polygons.count > 0:
            old_centers = np.array(self.polygons.centers, copy=True)
            old_sizes = np.array(self.polygons.sizes, copy=True)
            position_adam_snapshot = self._position_adam.snapshot()
            size_adam_snapshot = self._size_adam.snapshot()

            color_state = self._forward(float(softness), compute_gradients=False)

            fd_indices = self._select_fd_indices(color_canvas)
            pos_grad, size_grad = self._fd_geometry_grads(
                softness=float(softness),
                checkpoints=color_state.checkpoints,
                indices=fd_indices,
            )

            if run_position:
                self.polygons.centers = (
                    self.polygons.centers
                    - self._position_adam.step(pos_grad, self.config.position_lr)
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

            if run_size:
                max_size = self._max_size()
                self.polygons.sizes = np.clip(
                    self.polygons.sizes
                    - self._size_adam.step(size_grad, self.config.size_lr),
                    self.config.min_size,
                    max_size,
                ).astype(np.float32, copy=False)

            geometry_state = self._forward(float(softness), compute_gradients=False)
            geometry_canvas = geometry_state.canvas
            geometry_loss = self._loss(geometry_canvas)

            if (not self.config.allow_loss_increase) and geometry_loss > color_loss:
                self.polygons.centers = old_centers
                self.polygons.sizes = old_sizes
                self._position_adam.restore(position_adam_snapshot)
                self._size_adam.restore(size_adam_snapshot)
                final_canvas = color_canvas
                final_loss = float(color_loss)
            else:
                final_canvas = geometry_canvas
                final_loss = float(geometry_loss)

        self.current_canvas = final_canvas
        self.loss_history.append(float(final_loss))
        self.step_count += 1
        return float(final_loss)

    def run(
        self,
        steps: int,
        *,
        start_softness: float = 3.0,
        end_softness: float = 0.3,
    ) -> list[float]:
        if steps <= 0:
            return []

        out: list[float] = []
        for i in range(steps):
            t = i / max(steps - 1, 1)
            softness = start_softness + (end_softness - start_softness) * t
            out.append(self.step(float(softness)))
        return out

    def locally_converged(self, window: int = 50, min_rel_improvement: float = 5e-4) -> bool:
        if len(self.loss_history) < window + 1:
            return False
        start = float(self.loss_history[-(window + 1)])
        end = float(self.loss_history[-1])
        rel = (start - end) / max(abs(start), 1e-8)
        return rel < float(min_rel_improvement)

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
        params = (
            np.zeros((6,), dtype=np.float32)
            if shape_params is None
            else np.asarray(shape_params, dtype=np.float32).reshape(6)
        )

        self.polygons.centers = np.concatenate(
            [
                self.polygons.centers,
                np.array([[float(center_x), float(center_y)]], dtype=np.float32),
            ],
            axis=0,
        )
        self.polygons.sizes = np.concatenate(
            [
                self.polygons.sizes,
                np.array(
                    [
                        [
                            max(float(size_x), self.config.min_size),
                            max(float(size_y), self.config.min_size),
                        ]
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        self.polygons.rotations = np.concatenate(
            [self.polygons.rotations, np.array([float(rotation)], dtype=np.float32)],
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
                np.array(
                    [float(np.clip(alpha, self.config.min_alpha, self.config.max_alpha))],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        self.polygons.shape_types = np.concatenate(
            [self.polygons.shape_types, np.array([int(shape_type)], dtype=np.int32)],
            axis=0,
        )
        self.polygons.shape_params = np.concatenate(
            [self.polygons.shape_params, params.reshape(1, 6)],
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
        convergence_window: int = 50,
        convergence_delta: float = 5e-4,
    ) -> list[float]:
        if total_steps <= 0:
            return []
        if max_polygons <= 0:
            raise ValueError("max_polygons must be positive")

        losses: list[float] = []
        for i in range(total_steps):
            t = i / max(total_steps - 1, 1)
            softness = start_softness + (end_softness - start_softness) * t
            losses.append(self.step(float(softness)))

            if self.polygons.count >= max_polygons:
                continue
            if not self.locally_converged(
                window=convergence_window,
                min_rel_improvement=convergence_delta,
            ):
                continue

            err = np.mean(np.abs(self.target - self.current_canvas), axis=2, dtype=np.float32)
            y, x = np.unravel_index(int(np.argmax(err)), err.shape)
            hint = self.target[y, x]
            self.add_polygon(
                center_x=float(x),
                center_y=float(y),
                size_x=5.0,
                size_y=5.0,
                color=(float(hint[0]), float(hint[1]), float(hint[2])),
                alpha=0.50,
                shape_type=SHAPE_ELLIPSE,
                rotation=0.0,
            )

        return losses
