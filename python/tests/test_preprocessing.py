from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from src.preprocessing import preprocess_target_image


def test_preprocessing_pipeline_outputs_valid_metadata() -> None:
    project_root = Path(__file__).resolve().parents[1]
    target_path = project_root / "targets" / "face.png"

    result = preprocess_target_image(target_path, random_seed=7)

    assert len(result.pyramid) == 4
    assert [level.shape for level in result.pyramid] == [
        (200, 200, 3),
        (100, 100, 3),
        (50, 50, 3),
        (25, 25, 3),
    ]

    unique_labels = np.unique(result.segmentation_map)
    assert unique_labels.size == result.recommended_k
    assert result.segmentation_map.shape == (200, 200)

    assert result.structure_map.shape == (200, 200)
    assert float(result.structure_map.min()) >= 0.0
    assert float(result.structure_map.max()) <= 1.0

    assert 0.0 <= result.complexity_score <= 1.0
    assert 50 <= result.recommended_polygons <= 500


def test_run_cli_launches_without_error_for_valid_image() -> None:
    project_root = Path(__file__).resolve().parents[1]
    run_script = project_root / "run.py"
    target_path = project_root / "targets" / "face.png"

    completed = subprocess.run(
        [
            sys.executable,
            str(run_script),
            str(target_path),
            "--no-display",
            "--iterations",
            "10",
            "--seed",
            "1",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Input Analysis" in completed.stdout
