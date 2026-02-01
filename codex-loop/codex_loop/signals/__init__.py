"""Signals module: Detect friction and delight patterns."""

from codex_loop.signals.types import SignalType, SignalCategory, SignalResult
from codex_loop.signals.mechanical import detect_mechanical_signals
from codex_loop.signals.semantic import detect_semantic_signals

__all__ = [
    "SignalType",
    "SignalCategory",
    "SignalResult",
    "detect_mechanical_signals",
    "detect_semantic_signals",
]
