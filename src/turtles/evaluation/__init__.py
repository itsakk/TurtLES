"""Full-mesh autoregressive rollout evaluation with the metrics of the paper."""

from __future__ import annotations

from turtles.evaluation.metrics import (
    FrameErrors,
    RolloutRecord,
    aggregate_summaries,
    frame_errors,
    summarize_record,
)
from turtles.evaluation.predict import predict_frame, tile_partition_predict
from turtles.evaluation.rollout import RolloutEvaluator, RolloutSettings
from turtles.evaluation.spectra import (
    SpanwiseRings,
    band_log_rmse,
    build_spanwise_rings,
    spanwise_power,
)

__all__ = [
    "FrameErrors",
    "RolloutEvaluator",
    "RolloutRecord",
    "RolloutSettings",
    "SpanwiseRings",
    "aggregate_summaries",
    "band_log_rmse",
    "build_spanwise_rings",
    "frame_errors",
    "predict_frame",
    "spanwise_power",
    "summarize_record",
    "tile_partition_predict",
]
