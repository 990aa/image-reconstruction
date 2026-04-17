from __future__ import annotations

import gc
import math
import time
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from skimage.metrics import structural_similarity

from iterative_art_gpu.constants import (
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
)
from iterative_art_gpu.models import SequentialStageConfig
from iterative_art_gpu.optimizer import GPUSequentialHillClimber
from iterative_art_gpu.pipeline import (
    _resize_float_image,
    build_phase_plan,
    make_empty_live_batch,
    preprocess_target_array,
)
from iterative_art_gpu.renderer import GPUCoreRenderer


class ConfigurableGPUOptimizer(GPUSequentialHillClimber):
    """Notebook ablation optimizer that toggles routing, color solve, and shape constraints."""

    def __init__(self, target_image, rasterizer, polygons, bg_color, config):
        super().__init__(target_image, rasterizer, polygons, bg_color)
        self.cfg = dict(config)

    def sample_error_centers(self, guide_map, count, top_k, window, rng):
        if not self.cfg.get("use_residual_routing", True):
            return [
                (
                    float(rng.uniform(0, self.width - 1)),
                    float(rng.uniform(0, self.height - 1)),
                )
                for _ in range(int(top_k))
            ]

        amplified = np.clip(guide_map**2, 0.0, None)
        return super().sample_error_centers(amplified, count, top_k, window, rng)

    def random_candidate(
        self, stage, center_x, center_y, structure_map, angle_map, linearity_map, rng
    ):
        candidate = super().random_candidate(
            stage,
            center_x,
            center_y,
            structure_map,
            angle_map,
            linearity_map,
            rng,
        )

        if "force_shape" in self.cfg:
            candidate.shape_type = int(self.cfg["force_shape"])
            if candidate.shape_type == SHAPE_ELLIPSE:
                candidate.size_y = max(candidate.size_y, candidate.size_x * 0.2)
                candidate.shape_params = np.zeros((6,), dtype=np.float32)

        if not self.cfg.get("use_analytic_color", True):
            candidate.color = np.array(
                [rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)],
                dtype=np.float32,
            )
        return candidate

    def mutate_candidate(self, candidate, stage, rng):
        mutated = super().mutate_candidate(candidate, stage, rng)

        if self.cfg.get("force_shape") == SHAPE_ELLIPSE:
            mutated.size_y = max(mutated.size_y, mutated.size_x * 0.2)

        if not self.cfg.get("use_analytic_color", True):
            mutated.color = np.clip(
                mutated.color + rng.uniform(-0.15, 0.15, 3), 0.0, 1.0
            )
            mutated.alpha = float(
                np.clip(mutated.alpha + rng.uniform(-0.1, 0.1), 0.1, 1.0)
            )
        return mutated

    def evaluate_candidate(self, candidate, softness):
        if self.cfg.get("use_analytic_color", True):
            return super().evaluate_candidate(candidate, softness)

        coverage = self._coverage_from_candidate(candidate, softness)
        weight = coverage * float(np.clip(candidate.alpha, 0.0, 1.0))
        weight3 = weight.unsqueeze(2)

        color_tensor = torch.from_numpy(candidate.color).to(self.device).view(1, 1, 3)
        canvas = self.current_canvas_tensor + weight3 * (
            color_tensor - self.current_canvas_tensor
        )
        canvas = torch.clamp(canvas, 0.0, 1.0)

        residual = canvas - self.target_tensor
        mse = float(torch.mean(residual * residual).item())

        scored = candidate.copy()
        scored.mse = mse
        scored.coverage_tensor = coverage
        scored.canvas_tensor = canvas
        return scored


def _run_population_baseline(target, bg, polygons, minutes, seed):
    canvas = np.ones_like(target, dtype=np.float32) * np.asarray(bg, dtype=np.float32)
    mse = float(np.mean((target - canvas) ** 2, dtype=np.float32))
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    shapes = 0
    res = int(target.shape[0])

    while (time.perf_counter() - start) < (minutes * 60.0) and shapes < int(polygons):
        best_candidate = None
        best_mse = mse

        for _ in range(20):
            cx, cy = rng.uniform(0, res), rng.uniform(0, res)
            sx, sy = rng.uniform(2, 20), rng.uniform(2, 20)
            color = rng.uniform(0, 1, 3)

            y, x = np.ogrid[:res, :res]
            mask = ((x - cx) ** 2 / sx**2 + (y - cy) ** 2 / sy**2) <= 1
            trial = canvas.copy()
            trial[mask] = trial[mask] * 0.5 + color * 0.5

            trial_mse = float(np.mean((target - trial) ** 2, dtype=np.float32))
            if trial_mse < best_mse:
                best_mse = trial_mse
                best_candidate = trial

        if best_candidate is not None:
            canvas = best_candidate
            mse = best_mse
            shapes += 1

    psnr = 20 * math.log10(1.0 / math.sqrt(max(mse, 1e-10)))
    ssim = float(structural_similarity(target, canvas, data_range=1.0, channel_axis=2))
    return canvas, psnr, ssim, shapes


def _run_gradient_baseline(target, bg, polygons, minutes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_tensor = torch.from_numpy(target).to(device)

    resolution = int(target.shape[0])
    n_shapes = min(50, int(polygons))
    centers = (torch.rand((n_shapes, 2), device=device) * resolution).requires_grad_(
        True
    )
    sizes = (torch.rand((n_shapes, 2), device=device) * 10.0 + 2.0).requires_grad_(True)
    colors = torch.rand((n_shapes, 3), device=device).requires_grad_(True)
    alphas = torch.rand((n_shapes, 1), device=device).requires_grad_(True)

    optimizer = optim.Adam([centers, sizes, colors, alphas], lr=0.5)
    rasterizer = GPUCoreRenderer(resolution, resolution)
    start = time.perf_counter()

    for _ in range(200):
        if (time.perf_counter() - start) > (minutes * 60.0):
            break

        optimizer.zero_grad()
        canvas = (
            torch.tensor(bg, device=device, dtype=torch.float32)
            .view(1, 1, 3)
            .expand(resolution, resolution, 3)
            .clone()
        )

        for idx in range(n_shapes):
            coverage = rasterizer._ellipse_coverage_params(
                centers[idx, 0],
                centers[idx, 1],
                sizes[idx, 0],
                sizes[idx, 1],
                0.0,
                1.0,
            )
            weight = coverage * torch.sigmoid(alphas[idx])
            weight3 = weight.unsqueeze(2)
            color3 = torch.sigmoid(colors[idx]).view(1, 1, 3)
            canvas = canvas + weight3 * (color3 - canvas)

        loss = torch.mean((canvas - target_tensor) ** 2)
        loss.backward()
        optimizer.step()

    final_canvas = canvas.detach().cpu().numpy()
    mse = float(np.mean((target - final_canvas) ** 2, dtype=np.float32))
    psnr = 20 * math.log10(1.0 / math.sqrt(max(mse, 1e-10)))
    ssim = float(
        structural_similarity(target, final_canvas, data_range=1.0, channel_axis=2)
    )
    return final_canvas, psnr, ssim, n_shapes


def run_ablation_suite(
    image_rgb: np.ndarray,
    *,
    polygons: int = 250,
    minutes: float = 1.0,
    resolution: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[tuple[str, np.ndarray]]]:
    target = _resize_float_image(
        np.asarray(image_rgb, dtype=np.float32), (resolution, resolution)
    )
    prep = preprocess_target_array(target, resolution)

    experiments: dict[str, dict[str, object]] = {
        "0_Ours_Main": {},
        "1_A1_No_Analytic_Color": {"use_analytic_color": False},
        "2_A2_No_Residual_Routing": {"use_residual_routing": False},
        "3_A3_Ellipses_Only": {"force_shape": SHAPE_ELLIPSE},
        "4_A3_Strokes_Only": {"force_shape": SHAPE_THIN_STROKE},
        "5_A4_No_Schedule": {"no_schedule": True},
        "6_A5_White_Init": {"white_bg": True},
        "7_B1_EvoLisa_Style": {
            "use_analytic_color": False,
            "use_residual_routing": False,
            "white_bg": True,
            "no_schedule": True,
            "force_shape": SHAPE_TRIANGLE,
        },
        "8_B2_ProHC_Style": {
            "use_analytic_color": False,
            "use_residual_routing": True,
            "force_shape": SHAPE_TRIANGLE,
        },
        "9_B3_Evolved_Art_Pop": {"is_population": True},
        "10_B4_DiffVG_Upper_Bound": {"is_gradient": True},
    }

    results: list[dict[str, float | int | str]] = []
    canvases: list[tuple[str, np.ndarray]] = []

    for name, cfg in experiments.items():
        torch.cuda.empty_cache()
        gc.collect()

        if cfg.get("white_bg", False):
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            bg_color = np.mean(target, axis=(0, 1)).astype(np.float32)

        if cfg.get("is_population", False):
            canvas, psnr, ssim, shapes = _run_population_baseline(
                target,
                bg_color,
                polygons,
                minutes,
                seed,
            )
        elif cfg.get("is_gradient", False):
            canvas, psnr, ssim, shapes = _run_gradient_baseline(
                target,
                bg_color,
                polygons,
                minutes,
            )
        else:
            if cfg.get("no_schedule", False):
                shape_override = cfg.get("shapes")
                stage_shapes: tuple[int, ...]
                if isinstance(shape_override, tuple) and all(
                    isinstance(value, int) for value in shape_override
                ):
                    stage_shapes = cast(tuple[int, ...], shape_override)
                else:
                    stage_shapes = (SHAPE_ELLIPSE, SHAPE_QUAD, SHAPE_TRIANGLE)

                stages = (
                    SequentialStageConfig(
                        "flat",
                        resolution,
                        polygons,
                        64,
                        128,
                        2.0,
                        20.0,
                        0.5,
                        0.9,
                        1.0,
                        stage_shapes,
                        False,
                        50,
                        5,
                        2.0,
                        0.1,
                        10.0,
                    ),
                )
            else:
                stages = build_phase_plan(
                    resolution, polygons, prep.complexity_score
                ).stages

            optimizer = ConfigurableGPUOptimizer(
                target,
                GPUCoreRenderer(resolution, resolution),
                make_empty_live_batch(),
                bg_color,
                cfg,
            )
            rng = np.random.default_rng(seed + (hash(name) % 1000))
            start = time.perf_counter()

            gray = np.mean(target, axis=2)
            gy, gx = np.gradient(gray)
            structure_map = np.clip(
                np.hypot(gx, gy)
                / max(float(np.percentile(np.hypot(gx, gy), 99.0)), 1e-6),
                0.0,
                1.0,
            )
            angle_map = np.arctan2(gy, gx)
            linearity_map = np.ones_like(angle_map, dtype=np.float32)

            for stage in stages:
                for _ in range(int(stage.shapes_to_add)):
                    if (time.perf_counter() - start) > (minutes * 60.0):
                        break
                    guide = np.mean(
                        np.abs(optimizer.target_np - optimizer.current_canvas_np),
                        axis=2,
                    )
                    candidate = optimizer.search_next_shape(
                        stage,
                        guide,
                        structure_map,
                        angle_map,
                        linearity_map,
                        rng,
                    )
                    if candidate is not None:
                        optimizer.commit_shape(candidate)

            canvas = optimizer.current_canvas_np
            mse = float(np.mean((target - canvas) ** 2, dtype=np.float32))
            psnr = 20 * math.log10(1.0 / math.sqrt(max(mse, 1e-10)))
            ssim = float(
                structural_similarity(target, canvas, data_range=1.0, channel_axis=2)
            )
            shapes = int(optimizer.polygons.count)

        results.append(
            {
                "Method": name[2:],
                "PSNR": float(psnr),
                "SSIM": float(ssim),
                "Shapes Used": int(shapes),
            }
        )
        canvases.append(
            (name[2:], np.clip(canvas, 0.0, 1.0).astype(np.float32, copy=False))
        )

    return pd.DataFrame(results), canvases


def plot_ablation_results(
    results: pd.DataFrame,
    canvases: list[tuple[str, np.ndarray]],
    target_image: np.ndarray,
) -> None:
    sorted_df = results.sort_values(by="SSIM", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.barplot(data=results, y="Method", x="SSIM", palette="magma", ax=axes[0])
    axes[0].set_title("Structural Similarity (SSIM)")
    axes[0].grid(axis="x", linestyle="--", alpha=0.6)

    sns.barplot(data=results, y="Method", x="PSNR", palette="viridis", ax=axes[1])
    axes[1].set_title("Peak Signal-to-Noise Ratio (PSNR)")
    axes[1].grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    fig2, grid = plt.subplots(2, 6, figsize=(20, 7))
    flat_axes = grid.flatten()
    flat_axes[0].imshow(np.clip(target_image, 0.0, 1.0))
    flat_axes[0].set_title("TARGET", fontweight="bold")
    flat_axes[0].axis("off")

    for idx, (name, canvas) in enumerate(canvases):
        if idx + 1 >= len(flat_axes):
            break
        flat_axes[idx + 1].imshow(np.clip(canvas, 0.0, 1.0))
        flat_axes[idx + 1].set_title(name, fontsize=9)
        flat_axes[idx + 1].axis("off")

    for idx in range(len(canvases) + 1, len(flat_axes)):
        flat_axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

    print(sorted_df.to_string(index=False))
