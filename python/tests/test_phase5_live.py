from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from run import _estimate_runtime_seconds
from src.display import UIState, handle_control_key
from src.population import PopulationHillClimber
from src.preprocessing import preprocess_target_array


def _make_landscape_image(path: Path) -> None:
    h, w = 240, 360
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = 0.20 + 0.35 * yy
    img[..., 1] = 0.30 + 0.45 * yy
    img[..., 2] = 0.60 + 0.30 * yy
    img[int(h * 0.6) :, :, :] = np.array([0.15, 0.45, 0.20], dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(path)


def _make_portrait_image(path: Path) -> None:
    h, w = 360, 240
    canvas = Image.new("RGB", (w, h), color=(210, 200, 190))
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((60, 50, 180, 190), fill=(230, 185, 160))
    draw.ellipse((90, 95, 105, 110), fill=(30, 30, 30))
    draw.ellipse((135, 95, 150, 110), fill=(30, 30, 30))
    draw.arc((95, 130, 145, 165), start=15, end=165, fill=(120, 40, 40), width=3)
    draw.rectangle((0, 220, w, h), fill=(90, 120, 170))
    canvas.save(path)


def _make_graphic_text_image(path: Path) -> None:
    h, w = 300, 500
    canvas = Image.new("RGB", (w, h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((40, 40, 460, 120), fill=(20, 110, 220))
    draw.rectangle((40, 160, 460, 260), fill=(220, 60, 60))
    draw.text((65, 62), "AI", fill=(255, 255, 255))
    draw.text((65, 182), "ART", fill=(255, 255, 255))
    canvas.save(path)


def test_phase5_three_image_flow_no_display(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    run_script = project_root / "run.py"

    landscape = tmp_path / "landscape.jpg"
    portrait = tmp_path / "portrait.jpg"
    graphic = tmp_path / "graphic.jpg"

    _make_landscape_image(landscape)
    _make_portrait_image(portrait)
    _make_graphic_text_image(graphic)

    for image_path in (landscape, portrait, graphic):
        completed = subprocess.run(
            [
                sys.executable,
                str(run_script),
                str(image_path),
                "--no-display",
                "--fit-mode",
                "crop",
                "--resolution",
                "200",
                "--iterations",
                "500",
                "--seed",
                "7",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert "Input Analysis" in completed.stdout
        assert "estimated_iteration_rate" in completed.stdout


def test_phase5_keyboard_controls_and_clean_q_shutdown() -> None:
    target = np.ones((80, 80, 3), dtype=np.float32)
    prep = preprocess_target_array(target, random_seed=3, base_resolution=80)

    population = PopulationHillClimber(
        target_image=prep.target_rgb,
        max_iterations=50,
        target_pyramid=prep.pyramid,
        structure_map=prep.structure_map,
        gradient_angle_map=prep.gradient_angle_map,
        segmentation_map=prep.segmentation_map,
        cluster_centroids_lab=prep.cluster_centroids_lab,
        cluster_variances_lab=prep.cluster_variances_lab,
        size_schedule=prep.recommended_size_schedule,
        random_seed=3,
        recombination_interval=25,
    )
    population.start()

    ui = UIState()
    screenshot_called = {"value": False}
    quit_called = {"value": False}

    def shot() -> None:
        screenshot_called["value"] = True

    def quit_now() -> None:
        quit_called["value"] = True
        population.stop()

    assert handle_control_key("p", ui=ui, population=population, screenshot_callback=shot, quit_callback=quit_now) == "pause"
    assert ui.paused
    assert handle_control_key("s", ui=ui, population=population, screenshot_callback=shot, quit_callback=quit_now) == "segmentation-toggle"
    assert ui.show_segmentation_overlay
    assert handle_control_key("e", ui=ui, population=population, screenshot_callback=shot, quit_callback=quit_now) == "error-mode-cycle"
    assert handle_control_key("r", ui=ui, population=population, screenshot_callback=shot, quit_callback=quit_now) == "screenshot"
    assert screenshot_called["value"]
    assert handle_control_key("2", ui=ui, population=population, screenshot_callback=shot, quit_callback=quit_now) == "variant-switch"
    assert ui.display_variant_index == 1
    assert handle_control_key("q", ui=ui, population=population, screenshot_callback=shot, quit_callback=quit_now) == "quit"
    assert ui.quit_requested
    assert quit_called["value"]


def test_phase5_iteration_rate_exceeds_500_at_200() -> None:
    h = w = 200
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    target = np.zeros((h, w, 3), dtype=np.float32)
    target[..., 0] = xx
    target[..., 1] = yy
    target[..., 2] = 0.5 * np.sin(xx * 10.0) + 0.5
    target = np.clip(target, 0.0, 1.0)

    prep = preprocess_target_array(target, random_seed=13, base_resolution=200)
    iter_rate, _ = _estimate_runtime_seconds(
        prep,
        max_iterations=800,
        target_mse=0.01,
        seed=13,
    )

    assert iter_rate is not None
    assert iter_rate > 500.0
