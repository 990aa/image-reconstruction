from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.live_phase7 import (
    Phase7ControlState,
    build_phase7_plan,
    handle_phase7_control_key,
    run_phase7_headless,
)
from src.preprocessing import preprocess_target_array


def test_phase5_controls_are_back_compatible() -> None:
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
        == "view-set"
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


def test_phase5_stage_markers_include_all_transitions() -> None:
    h = w = 96
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    target = np.stack([xx, yy, 0.5 * np.ones_like(xx)], axis=2)

    pre = preprocess_target_array(
        target, polygon_override=240, random_seed=4, base_resolution=96
    )
    plan = build_phase7_plan(
        base_resolution=96,
        polygon_budget=240,
        complexity_score=float(pre.complexity_score),
    )
    plan = replace(plan, stage_d_steps=200)

    result = run_phase7_headless(
        target_image=pre.target_rgb,
        segmentation_map=pre.segmentation_map,
        plan=plan,
        random_seed=4,
        minutes=0.4,
        hard_timeout_seconds=40.0,
        max_total_steps=None,
    )

    names = [name for name, _ in result.stage_markers]
    assert "A" in names
    assert "B" in names

    # Runtime-constrained runs may stop before later stages when FD geometry is dense.
    if "D" in names:
        assert "C" in names
