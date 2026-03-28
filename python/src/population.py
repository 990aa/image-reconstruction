from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from src.optimizer import HillClimbingOptimizer
from src.polygon import Polygon
from src.renderer import render_polygon, render_polygons


@dataclass(frozen=True)
class VariantPersonality:
    name: str
    structure_weight_scale: float = 1.0
    structure_bias_mode: str = "normal"
    size_multiplier: float = 1.0
    random_placement_mode: bool = False
    death_interval_iterations: int = 1000


def default_variant_personalities() -> list[VariantPersonality]:
    return [
        VariantPersonality(name="primary-standard"),
        VariantPersonality(
            name="edge-emphasis",
            structure_weight_scale=1.0,
            structure_bias_mode="edge",
        ),
        VariantPersonality(name="flat-bias", structure_bias_mode="flat"),
        VariantPersonality(name="large-polygons", size_multiplier=1.5),
        VariantPersonality(name="small-polygons", size_multiplier=0.5),
        VariantPersonality(name="aggressive-maintenance", death_interval_iterations=200),
    ]


@dataclass
class PopulationSnapshot:
    variant_index: int
    canvas: np.ndarray
    target: np.ndarray
    mse_history: list[float]
    primary_mse_history: list[float]
    best_mse_history: list[float]
    acceptance_history: list[float]
    iteration: int
    primary_iteration: int
    best_iteration: int
    current_mse: float
    best_mse: float
    acceptance_rate: float
    accepted_count: int
    phase_name: str
    running: bool
    last_step_accepted: bool
    segmentation_map: np.ndarray | None
    structure_map: np.ndarray | None
    primary_index: int
    best_index: int


class PopulationHillClimber:
    def __init__(
        self,
        *,
        target_image: np.ndarray,
        max_iterations: int,
        target_pyramid: list[np.ndarray] | None,
        structure_map: np.ndarray | None,
        gradient_angle_map: np.ndarray | None,
        segmentation_map: np.ndarray | None,
        cluster_centroids_lab: np.ndarray | None,
        cluster_variances_lab: np.ndarray | None,
        size_schedule: dict[str, float] | None,
        random_seed: int | None,
        personalities: list[VariantPersonality] | None = None,
        recombination_interval: int = 500,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if recombination_interval <= 0:
            raise ValueError("recombination_interval must be positive")

        self.target_image = target_image.astype(np.float32, copy=False)
        self.max_iterations = int(max_iterations)
        self.recombination_interval = int(recombination_interval)

        self.personalities = (
            default_variant_personalities() if personalities is None else personalities
        )
        if len(self.personalities) != 6:
            raise ValueError("Population must contain exactly 6 variants.")

        self.optimizers: list[HillClimbingOptimizer] = []
        self._variant_locks: list[threading.Lock] = []
        self._variant_running: list[bool] = []
        self._last_step_accepted: list[bool] = []

        for idx, personality in enumerate(self.personalities):
            seed = None if random_seed is None else int(random_seed + idx)
            optimizer = HillClimbingOptimizer(
                target_image=self.target_image,
                max_iterations=self.max_iterations,
                random_seed=seed,
                target_pyramid=target_pyramid,
                structure_map=structure_map,
                gradient_angle_map=gradient_angle_map,
                segmentation_map=segmentation_map,
                cluster_centroids_lab=cluster_centroids_lab,
                cluster_variances_lab=cluster_variances_lab,
                size_schedule=size_schedule,
                max_polygons=None,
                structure_weight_scale=personality.structure_weight_scale,
                structure_bias_mode=personality.structure_bias_mode,
                size_multiplier=personality.size_multiplier,
                random_placement_mode=personality.random_placement_mode,
                death_interval_iterations=personality.death_interval_iterations,
            )
            self.optimizers.append(optimizer)
            self._variant_locks.append(threading.Lock())
            self._variant_running.append(False)
            self._last_step_accepted.append(False)

        self.primary_index = 0
        self.display_variant_index = 0

        self.segmentation_map = (
            None if segmentation_map is None else segmentation_map.astype(np.int32, copy=False)
        )
        self.structure_map = (
            None if structure_map is None else structure_map.astype(np.float32, copy=False)
        )

        self.primary_mse_history: list[float] = []
        self.best_mse_history: list[float] = []
        self.acceptance_rate_history: list[float] = []
        self._history_iterations: list[int] = []

        self.non_primary_better_seen = False
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._barrier = threading.Barrier(len(self.optimizers) + 1)
        self._threads: list[threading.Thread] = []
        self._coordinator_thread: threading.Thread | None = None

    def start(self) -> None:
        for idx in range(len(self.optimizers)):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(idx,),
                daemon=True,
                name=f"variant-worker-{idx}",
            )
            self._threads.append(thread)
            self._variant_running[idx] = True
            thread.start()

        self._coordinator_thread = threading.Thread(
            target=self._coordinator_loop,
            daemon=True,
            name="population-coordinator",
        )
        self._coordinator_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._barrier.abort()
        except Exception:
            pass

        for thread in self._threads:
            thread.join(timeout=2.0)
        if self._coordinator_thread is not None:
            self._coordinator_thread.join(timeout=2.0)

    @property
    def running(self) -> bool:
        return any(self._variant_running)

    def set_display_variant(self, variant_index: int) -> None:
        if 0 <= variant_index < len(self.optimizers):
            self.display_variant_index = int(variant_index)

    def _worker_loop(self, variant_index: int) -> None:
        optimizer = self.optimizers[variant_index]
        lock = self._variant_locks[variant_index]

        while not self._stop_event.is_set() and optimizer.iteration < self.max_iterations:
            with lock:
                accepted = optimizer.step()
                self._last_step_accepted[variant_index] = bool(accepted)

            if (
                optimizer.iteration > 0
                and optimizer.iteration % self.recombination_interval == 0
                and not self._stop_event.is_set()
            ):
                try:
                    self._barrier.wait(timeout=30.0)
                except threading.BrokenBarrierError:
                    break

        self._variant_running[variant_index] = False

    def _coordinator_loop(self) -> None:
        last_checkpoint = -1
        while not self._stop_event.is_set():
            if not self.running:
                break

            try:
                self._barrier.wait(timeout=30.0)
            except threading.BrokenBarrierError:
                break

            primary_iteration = self.optimizers[self.primary_index].iteration
            if primary_iteration == last_checkpoint:
                continue
            last_checkpoint = primary_iteration

            self._update_progress_histories()
            self._update_diversity_flag()
            self.run_recombination_step()

        self._update_progress_histories()

    def _update_progress_histories(self) -> None:
        with self._variant_locks[self.primary_index]:
            primary = self.optimizers[self.primary_index]
            primary_mse = float(primary.current_mse)
            primary_iter = int(primary.iteration)
            acceptance_rate = float(primary.acceptance_rate)

        best_index = self.best_variant_index
        with self._variant_locks[best_index]:
            best_mse = float(self.optimizers[best_index].current_mse)

        with self._state_lock:
            self._history_iterations.append(primary_iter)
            self.primary_mse_history.append(primary_mse)
            self.best_mse_history.append(best_mse)
            self.acceptance_rate_history.append(acceptance_rate * 100.0)

    def _update_diversity_flag(self) -> None:
        with self._variant_locks[self.primary_index]:
            primary_mse = float(self.optimizers[self.primary_index].current_mse)

        for idx in range(len(self.optimizers)):
            if idx == self.primary_index:
                continue
            with self._variant_locks[idx]:
                candidate_mse = float(self.optimizers[idx].current_mse)
            if candidate_mse < primary_mse:
                self.non_primary_better_seen = True
                return

    @property
    def best_variant_index(self) -> int:
        best_idx = 0
        best_mse = float("inf")
        for idx, lock in enumerate(self._variant_locks):
            with lock:
                mse = float(self.optimizers[idx].current_mse)
            if mse < best_mse:
                best_mse = mse
                best_idx = idx
        return best_idx

    def _isolated_polygon_loss(self, polygon: Polygon) -> float:
        primary = self.optimizers[self.primary_index]
        isolated_canvas = render_polygon(primary.blank_canvas, polygon)
        return primary.evaluate_canvas_loss(isolated_canvas)

    def run_recombination_step(self) -> bool:
        mse_pairs: list[tuple[int, float]] = []
        for idx, lock in enumerate(self._variant_locks):
            with lock:
                mse_pairs.append((idx, float(self.optimizers[idx].current_mse)))
        mse_pairs.sort(key=lambda item: item[1])
        if len(mse_pairs) < 2:
            return False

        parent_a_idx, parent_a_mse = mse_pairs[0]
        parent_b_idx, parent_b_mse = mse_pairs[1]

        with self._variant_locks[parent_a_idx]:
            parent_a_polys = list(self.optimizers[parent_a_idx].accepted_polygons)
        with self._variant_locks[parent_b_idx]:
            parent_b_polys = list(self.optimizers[parent_b_idx].accepted_polygons)

        if not parent_a_polys and not parent_b_polys:
            return False

        pair_count = min(len(parent_a_polys), len(parent_b_polys))
        combined: list[Polygon] = []
        for i in range(pair_count):
            poly_a = parent_a_polys[i]
            poly_b = parent_b_polys[i]
            loss_a = self._isolated_polygon_loss(poly_a)
            loss_b = self._isolated_polygon_loss(poly_b)
            combined.append(poly_a if loss_a <= loss_b else poly_b)

        if pair_count < len(parent_a_polys) or pair_count < len(parent_b_polys):
            remainder = parent_a_polys[pair_count:] if parent_a_mse <= parent_b_mse else parent_b_polys[pair_count:]
            combined.extend(remainder)

        primary = self.optimizers[self.primary_index]
        combined_canvas = render_polygons(primary.blank_canvas, combined)
        combined_mse = primary.evaluate_canvas_loss(combined_canvas)

        if combined_mse <= min(parent_a_mse, parent_b_mse):
            with self._variant_locks[self.primary_index]:
                primary.adopt_solution(combined, combined_canvas, combined_mse)
            return True

        return False

    def snapshot(self) -> PopulationSnapshot:
        display_idx = int(self.display_variant_index)
        best_idx = self.best_variant_index

        with self._variant_locks[display_idx]:
            displayed = self.optimizers[display_idx]
            canvas = np.array(displayed.canvas, copy=True)
            mse_history = list(displayed.mse_history)
            iteration = int(displayed.iteration)
            acceptance_rate = float(displayed.acceptance_rate)
            accepted_count = int(displayed.accepted_count)
            phase_name = str(displayed.current_phase)
            current_mse = float(displayed.current_mse)

        with self._variant_locks[self.primary_index]:
            primary = self.optimizers[self.primary_index]
            primary_iteration = int(primary.iteration)

        with self._variant_locks[best_idx]:
            best_variant = self.optimizers[best_idx]
            best_mse = float(best_variant.current_mse)
            best_iteration = int(best_variant.iteration)

        with self._state_lock:
            primary_trace = list(self.primary_mse_history)
            best_trace = list(self.best_mse_history)
            acceptance_trace = list(self.acceptance_rate_history)

        return PopulationSnapshot(
            variant_index=display_idx,
            canvas=canvas,
            target=np.array(self.target_image, copy=True),
            mse_history=mse_history,
            primary_mse_history=primary_trace,
            best_mse_history=best_trace,
            acceptance_history=acceptance_trace,
            iteration=iteration,
            primary_iteration=primary_iteration,
            best_iteration=best_iteration,
            current_mse=current_mse,
            best_mse=best_mse,
            acceptance_rate=acceptance_rate,
            accepted_count=accepted_count,
            phase_name=phase_name,
            running=self.running,
            last_step_accepted=bool(self._last_step_accepted[display_idx]),
            segmentation_map=(
                None if self.segmentation_map is None else np.array(self.segmentation_map, copy=True)
            ),
            structure_map=(
                None if self.structure_map is None else np.array(self.structure_map, copy=True)
            ),
            primary_index=int(self.primary_index),
            best_index=int(best_idx),
        )

    def get_error_maps(self, variant_index: int) -> dict[str, np.ndarray]:
        idx = int(np.clip(variant_index, 0, len(self.optimizers) - 1))
        with self._variant_locks[idx]:
            maps = self.optimizers[idx].compute_error_maps_for_display()
        return {k: np.array(v, copy=True) for k, v in maps.items()}

    def wait_until_complete(self, timeout: float | None = None) -> None:
        start = time.monotonic()
        while self.running:
            if timeout is not None and (time.monotonic() - start) >= timeout:
                break
            time.sleep(0.05)
