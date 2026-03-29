from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from src.live_phase7 import (
    Phase7ControlState,
    build_phase7_plan,
    handle_phase7_control_key,
    run_phase7_headless,
)
from src.preprocessing import preprocess_target_array


def test_phase7_plan_and_controls() -> None:
    plan = build_phase7_plan(
        base_resolution=200,
        polygon_budget=240,
        complexity_score=0.5,
    )
    assert plan.stage_a_initial_polygons >= 20
    assert plan.stage_b_batches == 8
    assert plan.stage_c_batches == 12

    controls = Phase7ControlState()
    shot = {"value": False}
    quit_now = {"value": False}

    def screenshot() -> None:
        shot["value"] = True

    def do_quit() -> None:
        quit_now["value"] = True

    assert (
        handle_phase7_control_key(
            "p",
            controls=controls,
            screenshot_callback=screenshot,
            quit_callback=do_quit,
        )
        == "pause"
    )
    assert controls.paused

    assert (
        handle_phase7_control_key(
            "r",
            controls=controls,
            screenshot_callback=screenshot,
            quit_callback=do_quit,
        )
        == "screenshot"
    )
    assert shot["value"]

    assert (
        handle_phase7_control_key(
            "1",
            controls=controls,
            screenshot_callback=screenshot,
            quit_callback=do_quit,
        )
        == "noop"
    )

    assert (
        handle_phase7_control_key(
            "q",
            controls=controls,
            screenshot_callback=screenshot,
            quit_callback=do_quit,
        )
        == "quit"
    )
    assert controls.quit_requested
    assert quit_now["value"]


def test_phase7_headless_generates_stage_markers() -> None:
    h = w = 96
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    target = np.stack([xx, yy, 0.5 * np.ones_like(xx)], axis=2)

    pre = preprocess_target_array(
        target, polygon_override=180, random_seed=9, base_resolution=96
    )
    plan = build_phase7_plan(
        base_resolution=96,
        polygon_budget=180,
        complexity_score=float(pre.complexity_score),
    )
    plan = replace(plan, stage_d_steps=120)

    result = run_phase7_headless(
        target_image=pre.target_rgb,
        segmentation_map=pre.segmentation_map,
        plan=plan,
        random_seed=9,
        minutes=0.4,
        hard_timeout_seconds=35.0,
        max_total_steps=400,
    )

    assert result.iterations > 0
    stages = [name for name, _ in result.stage_markers]
    assert "A" in stages
    assert "B" in stages


def test_phase7_cli_no_display_runs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    run_script = project_root / "run.py"

    img = np.zeros((96, 96, 3), dtype=np.float32)
    img[..., 0] = np.linspace(0.0, 1.0, 96, dtype=np.float32)[None, :]
    img[..., 1] = np.linspace(0.0, 1.0, 96, dtype=np.float32)[:, None]
    img[..., 2] = 0.4

    image_path = tmp_path / "phase7_input.png"
    Image.fromarray((img * 255.0).astype(np.uint8), mode="RGB").save(image_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(run_script),
            str(image_path),
            "--no-display",
            "--polygons",
            "180",
            "--minutes",
            "0.1",
            "--resolution",
            "96",
            "--iterations",
            "260",
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
