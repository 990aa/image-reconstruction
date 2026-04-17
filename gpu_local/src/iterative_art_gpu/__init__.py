from evolutionary_art_gpu.constants import (
    DEFAULT_BASE_RESOLUTION,
    SHAPE_ANNULAR_SEGMENT,
    SHAPE_BEZIER_PATCH,
    SHAPE_ELLIPSE,
    SHAPE_QUAD,
    SHAPE_THIN_STROKE,
    SHAPE_TRIANGLE,
)
from evolutionary_art_gpu.exporters import export_svg, save_rgb_image
from evolutionary_art_gpu.ablation import plot_ablation_results, run_ablation_suite
from evolutionary_art_gpu.models import (
    LivePolygonBatch,
    PhasePlan,
    PhaseResult,
    PreprocessedTarget,
    SequentialStageConfig,
    ShapeCandidate,
)
from evolutionary_art_gpu.optimizer import GPUSequentialHillClimber
from evolutionary_art_gpu.pipeline import (
    build_phase_plan,
    make_empty_live_batch,
    prepare_square_image,
    preprocess_target_array,
    run_phase_local_gpu,
)
from evolutionary_art_gpu.renderer import GPUCoreRenderer
from evolutionary_art_gpu.visualization import plot_3d_exploded_view

__all__ = [
    "DEFAULT_BASE_RESOLUTION",
    "SHAPE_TRIANGLE",
    "SHAPE_QUAD",
    "SHAPE_ELLIPSE",
    "SHAPE_BEZIER_PATCH",
    "SHAPE_THIN_STROKE",
    "SHAPE_ANNULAR_SEGMENT",
    "LivePolygonBatch",
    "ShapeCandidate",
    "SequentialStageConfig",
    "PhasePlan",
    "PreprocessedTarget",
    "PhaseResult",
    "GPUCoreRenderer",
    "GPUSequentialHillClimber",
    "make_empty_live_batch",
    "build_phase_plan",
    "prepare_square_image",
    "preprocess_target_array",
    "run_phase_local_gpu",
    "export_svg",
    "save_rgb_image",
    "plot_3d_exploded_view",
    "run_ablation_suite",
    "plot_ablation_results",
]
