from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from src.live_refiner import (
    phaseControlState,
    build_phase_plan,
    handle_phase_control_key,
    run_phase_headless,
)
from src.preprocessing import preprocess_target_array


def test_plan_and_controls() -> None:
    plan = build_phase_plan(
        base_resolution=200,
        polygon_budget=220,
        complexity_score=0.5,
    )
    assert plan.polygon_budget == 220
    assert [stage.name for stage in plan.stages] == [
        "foundation",
        "structure",
        "detail",
    ]
    assert sum(stage.shapes_to_add for stage in plan.stages) == 220

    controls = phaseControlState()
    shot = {"value": False}
    quit_now = {"value": False}

    def screenshot() -> None:
        shot["value"] = True

    def do_quit() -> None:
        quit_now["value"] = True

    assert (
        handle_phase_control_key(
            "p",
            controls=controls,
            screenshot_callback=screenshot,
            quit_callback=do_quit,
        )
        == "pause"
    )
    assert controls.paused

    assert (
        handle_phase_control_key(
            "r",
            controls=controls,
            screenshot_callback=screenshot,
            quit_callback=do_quit,
        )
        == "screenshot"
    )
    assert shot["value"]

    assert (
        handle_phase_control_key(
            "v",
            controls=controls,
            screenshot_callback=screenshot,
            quit_callback=do_quit,
        )
        == "view-cycle"
    )
    assert controls.view_mode_index == 1

    assert (
        handle_phase_control_key(
            "e",
            controls=controls,
            screenshot_callback=screenshot,
            quit_callback=do_quit,
        )
        == "residual-mode"
    )
    assert controls.residual_mode_index == 1

    assert (
        handle_phase_control_key(
            "g",
            controls=controls,
            screenshot_callback=screenshot,
            quit_callback=do_quit,
        )
        == "force-growth"
    )
    assert controls.force_growth_requests == 1

    assert (
        handle_phase_control_key(
            "q",
            controls=controls,
            screenshot_callback=screenshot,
            quit_callback=do_quit,
        )
        == "quit"
    )
    assert controls.quit_requested
    assert quit_now["value"]


def test_headless_refiner_adds_shapes_and_marks_stages() -> None:
    h = w = 96
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    target = np.stack([xx, yy, 0.5 * np.ones_like(xx)], axis=2)

    pre = preprocess_target_array(
        target,
        polygon_override=24,
        random_seed=9,
        base_resolution=96,
    )
    plan = build_phase_plan(
        base_resolution=96,
        polygon_budget=24,
        complexity_score=float(pre.complexity_score),
    )

    result = run_phase_headless(
        target_image=pre.target_rgb,
        segmentation_map=pre.segmentation_map,
        plan=plan,
        random_seed=9,
        minutes=0.05,
        hard_timeout_seconds=10.0,
        max_total_steps=12,
    )

    assert result.iterations > 0
    assert result.polygon_count > 0
    assert result.final_canvas.shape == pre.target_rgb.shape
    assert [name for name, _ in result.stage_markers][0] == "foundation"


def test_cli_no_display_runs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    run_script = project_root / "run.py"

    img = np.zeros((96, 96, 3), dtype=np.float32)
    img[..., 0] = np.linspace(0.0, 1.0, 96, dtype=np.float32)[None, :]
    img[..., 1] = np.linspace(0.0, 1.0, 96, dtype=np.float32)[:, None]
    img[..., 2] = 0.4

    image_path = tmp_path / "refiner_input.png"
    Image.fromarray((img * 255.0).astype(np.uint8), mode="RGB").save(image_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(run_script),
            str(image_path),
            "--no-display",
            "--polygons",
            "24",
            "--minutes",
            "0.05",
            "--resolution",
            "96",
            "--iterations",
            "12",
            "--seed",
            "9",
            "--fit-mode",
            "crop",
            "--no-prompt",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Input Analysis" in completed.stdout
    assert "Run Summary" in completed.stdout
