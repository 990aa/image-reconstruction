from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np

from src.live_phase7 import (
    Phase7ControlState,
    build_phase7_plan,
    handle_phase7_control_key,
)


def test_phase7_plan_allocates_budget() -> None:
    plan = build_phase7_plan(
        base_resolution=200,
        polygon_budget=240,
        complexity_score=0.53,
    )
    assert plan.polygon_budget == 240
    assert plan.rounds
    total = sum(sum(r.batch_schedule) for r in plan.rounds)
    assert total == 240


def test_phase7_keyboard_controls_include_legacy_and_new() -> None:
    controls = Phase7ControlState()
    screenshot_called = {"value": False}
    quit_called = {"value": False}

    def shot() -> None:
        screenshot_called["value"] = True

    def quit_now() -> None:
        quit_called["value"] = True

    assert (
        handle_phase7_control_key(
            "p", controls=controls, screenshot_callback=shot, quit_callback=quit_now
        )
        == "pause"
    )
    assert controls.paused

    assert (
        handle_phase7_control_key(
            "s", controls=controls, screenshot_callback=shot, quit_callback=quit_now
        )
        == "segmentation-toggle"
    )
    assert controls.show_segmentation_overlay

    assert (
        handle_phase7_control_key(
            "e", controls=controls, screenshot_callback=shot, quit_callback=quit_now
        )
        == "error-mode-cycle"
    )
    assert controls.residual_mode == 1

    assert (
        handle_phase7_control_key(
            "r", controls=controls, screenshot_callback=shot, quit_callback=quit_now
        )
        == "screenshot"
    )
    assert screenshot_called["value"]

    assert (
        handle_phase7_control_key(
            "2", controls=controls, screenshot_callback=shot, quit_callback=quit_now
        )
        == "variant-switch"
    )
    assert controls.view_mode == 1

    assert (
        handle_phase7_control_key(
            "v", controls=controls, screenshot_callback=shot, quit_callback=quit_now
        )
        == "view-cycle"
    )
    assert controls.view_mode == 2

    before_softness = controls.softness_scale
    assert (
        handle_phase7_control_key(
            "+", controls=controls, screenshot_callback=shot, quit_callback=quit_now
        )
        == "softness-up"
    )
    assert controls.softness_scale > before_softness

    assert (
        handle_phase7_control_key(
            "g", controls=controls, screenshot_callback=shot, quit_callback=quit_now
        )
        == "force-growth"
    )
    assert controls.force_growth_requested

    assert (
        handle_phase7_control_key(
            "d", controls=controls, screenshot_callback=shot, quit_callback=quit_now
        )
        == "residual-correction"
    )
    assert controls.correction_requested

    assert (
        handle_phase7_control_key(
            "q", controls=controls, screenshot_callback=shot, quit_callback=quit_now
        )
        == "quit"
    )
    assert controls.quit_requested
    assert quit_called["value"]


def test_phase7_run_cli_interface_no_display(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    run_script = project_root / "run.py"

    h = w = 96
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[..., 0] = xx
    image[..., 1] = yy
    image[..., 2] = 0.5

    from PIL import Image

    image_path = tmp_path / "phase7_input.png"
    Image.fromarray(
        (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
    ).save(image_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(run_script),
            str(image_path),
            "--no-display",
            "--polygons",
            "80",
            "--minutes",
            "0.01",
            "--resolution",
            "96",
            "--iterations",
            "25",
            "--seed",
            "9",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Input Analysis" in completed.stdout
    assert "Run Summary" in completed.stdout
