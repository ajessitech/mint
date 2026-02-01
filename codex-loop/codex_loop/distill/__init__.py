"""Distill module: Generate Harbor tasks from signals/clusters."""

from codex_loop.distill.segment_selector import select_segment, SegmentWindow
from codex_loop.distill.harbor_generator import (
    distill_signal_to_task,
    distill_cluster_to_dataset,
)

__all__ = [
    "select_segment",
    "SegmentWindow",
    "distill_signal_to_task",
    "distill_cluster_to_dataset",
]
