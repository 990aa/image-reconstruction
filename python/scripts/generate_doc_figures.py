# Run with:
# uv run python .\scripts\generate_doc_figures.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"


def _box(ax, x: float, y: float, w: float, h: float, text: str, *, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=color,
        edgecolor="#16324f",
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=10)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#16324f",
        )
    )


def pipeline_overview(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.04, 0.60, 0.18, 0.18, "Target Image\nCrop + Resize", color="#d8f3dc")
    _box(ax, 0.29, 0.60, 0.18, 0.18, "Residual Guide Map\n+ Top-K Regions", color="#b7e4c7")
    _box(ax, 0.54, 0.60, 0.18, 0.18, "Candidate Proposal\nGeometry + Alpha", color="#95d5b2")
    _box(ax, 0.79, 0.60, 0.17, 0.18, "Analytic Color Solve\nCandidate Ranking", color="#74c69d")

    _box(ax, 0.18, 0.18, 0.20, 0.20, "Mutation Hill Climb\nPosition / Size / Rotation", color="#ffd6a5")
    _box(ax, 0.46, 0.18, 0.20, 0.20, "Commit One Primitive\nUpdate Canvas", color="#ffcad4")
    _box(ax, 0.74, 0.18, 0.18, 0.20, "Live Dashboard\nTarget / Recon / Error / Loss", color="#cddafd")

    _arrow(ax, (0.22, 0.69), (0.29, 0.69))
    _arrow(ax, (0.47, 0.69), (0.54, 0.69))
    _arrow(ax, (0.72, 0.69), (0.79, 0.69))
    _arrow(ax, (0.87, 0.60), (0.76, 0.38))
    _arrow(ax, (0.54, 0.60), (0.38, 0.38))
    _arrow(ax, (0.38, 0.28), (0.46, 0.28))
    _arrow(ax, (0.66, 0.28), (0.74, 0.28))
    _arrow(ax, (0.56, 0.18), (0.38, 0.10))
    ax.text(0.22, 0.08, "Repeat until time budget expires", fontsize=11, color="#16324f")

    fig.suptitle("Sequential Primitive Reconstruction Pipeline", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def stage_schedule(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(
        ax,
        0.05,
        0.28,
        0.24,
        0.42,
        "Foundation\n100x100\nup to 200 shapes\nalpha 0.55-0.85\ncandidates 80\nmutations 160",
        color="#d8e2dc",
    )
    _box(
        ax,
        0.38,
        0.28,
        0.24,
        0.42,
        "Structure\n150x150\nup to 400 shapes\nalpha 0.40-0.72\ncandidates 64\nmutations 128",
        color="#ffe5d9",
    )
    _box(
        ax,
        0.71,
        0.28,
        0.24,
        0.42,
        "Detail\n200x200\nremaining budget\nalpha 0.28-0.60\ncandidates 72\nmutations 156",
        color="#dbe7ff",
    )
    _arrow(ax, (0.29, 0.49), (0.38, 0.49))
    _arrow(ax, (0.62, 0.49), (0.71, 0.49))
    fig.suptitle("Three-Stage Reconstruction Schedule", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def dashboard_layout(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.05, 0.54, 0.40, 0.34, "Target View\nstatic reference", color="#edf6f9")
    _box(ax, 0.55, 0.54, 0.40, 0.34, "Reconstruction View\nupdated every accepted primitive", color="#e9ecef")
    _box(ax, 0.05, 0.10, 0.40, 0.34, "Absolute Error Map\nbright = unresolved regions", color="#faedcd")
    _box(ax, 0.55, 0.10, 0.40, 0.34, "Log Loss Curve\nstage markers + accepted-shape history", color="#f8edeb")

    fig.suptitle("Live Matplotlib Dashboard Layout", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_overview(FIG_DIR / "pipeline_overview.png")
    stage_schedule(FIG_DIR / "stage_schedule.png")
    dashboard_layout(FIG_DIR / "dashboard_layout.png")
    print("Saved figures to", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
