"""
Segment selector: Choose trace spine segments for distillation.

Selects windows around friction/delight signals that make good
Harbor evaluation tasks.
"""

from dataclasses import dataclass
from typing import Optional, Any

from codex_loop.db.schema import Session, Signal
from codex_loop.ingest.spine_reader import (
    TraceRecord,
    parse_spine,
    get_submissions_by_op,
    get_events_by_type,
    get_compaction_boundaries,
)
from pathlib import Path


@dataclass
class SegmentWindow:
    """
    A window of trace records suitable for distillation.
    """
    start_seq: int
    end_seq: int
    records: list[TraceRecord]
    signal: Optional[Signal] = None
    
    # Context
    user_message: Optional[str] = None
    expected_outcome: Optional[str] = None
    
    @property
    def record_count(self) -> int:
        return len(self.records)
    
    def get_user_turns(self) -> list[TraceRecord]:
        """Get user turn submissions in this window."""
        return [r for r in self.records if r.is_submission() and r.get_op_type() == "user_turn"]
    
    def get_tool_executions(self) -> list[TraceRecord]:
        """Get tool execution events in this window."""
        return [
            r for r in self.records
            if r.is_event() and r.get_event_type() in ["exec_command_end", "patch_apply_end"]
        ]


def select_segment(
    signal: Signal,
    records: Optional[list[TraceRecord]] = None,
    context_before: int = 5,
    context_after: int = 5,
) -> Optional[SegmentWindow]:
    """
    Select a trace segment around a signal for distillation.
    
    Args:
        signal: The signal to center the segment on
        records: Optional pre-parsed records (will load from session if not provided)
        context_before: Number of records to include before the signal
        context_after: Number of records to include after the signal
        
    Returns:
        SegmentWindow if segment can be extracted, None otherwise
    """
    # Load records if not provided
    if records is None:
        if not signal.session or not signal.session.spine_path:
            return None
        spine_path = Path(signal.session.spine_path)
        if not spine_path.exists():
            return None
        records = parse_spine(spine_path)
    
    if not records:
        return None
    
    # Find the signal's position in the trace
    evidence = signal.evidence or {}
    signal_seq = evidence.get("seq_range", [None, None])[0]
    
    if signal_seq is None:
        # Try to find by turn_id
        turn_id = evidence.get("turn_id") or signal.turn_id
        if turn_id:
            for r in records:
                if r.payload.get("id") == turn_id or r.payload.get("turn_id") == turn_id:
                    signal_seq = r.seq
                    break
    
    if signal_seq is None:
        # Default to middle of trace
        signal_seq = records[len(records) // 2].seq
    
    # Build window around signal
    sorted_records = sorted(records, key=lambda r: r.seq)
    
    # Find signal position
    signal_idx = 0
    for i, r in enumerate(sorted_records):
        if r.seq >= signal_seq:
            signal_idx = i
            break
    
    # Calculate window bounds
    start_idx = max(0, signal_idx - context_before)
    end_idx = min(len(sorted_records), signal_idx + context_after + 1)
    
    window_records = sorted_records[start_idx:end_idx]
    
    if not window_records:
        return None
    
    # Extract user message (first user turn in window)
    user_message = None
    for r in window_records:
        if r.is_submission() and r.get_op_type() == "user_turn":
            op = r.payload.get("op", {})
            items = op.get("items", [])
            for item in items:
                if isinstance(item, dict) and item.get("type") == "text":
                    user_message = item.get("text", "")
                    break
            if user_message:
                break
    
    # Determine expected outcome based on signal type
    expected_outcome = _infer_expected_outcome(signal, window_records)
    
    return SegmentWindow(
        start_seq=window_records[0].seq,
        end_seq=window_records[-1].seq,
        records=window_records,
        signal=signal,
        user_message=user_message,
        expected_outcome=expected_outcome,
    )


def select_segments_for_cluster(
    signals: list[Signal],
    max_segments: int = 10,
    records_cache: Optional[dict[str, list[TraceRecord]]] = None,
) -> list[SegmentWindow]:
    """
    Select diverse segments for a cluster of signals.
    
    Args:
        signals: Signals in the cluster
        max_segments: Maximum segments to select
        records_cache: Optional cache of session_id -> records
        
    Returns:
        List of SegmentWindows
    """
    segments = []
    seen_sessions = set()
    
    # Sort by severity (highest first)
    sorted_signals = sorted(signals, key=lambda s: s.severity, reverse=True)
    
    for signal in sorted_signals:
        if len(segments) >= max_segments:
            break
        
        # Prefer diverse sessions
        if signal.session_id in seen_sessions and len(segments) < max_segments // 2:
            continue
        
        # Get cached records or load
        if records_cache and signal.session_id in records_cache:
            records = records_cache[signal.session_id]
        else:
            records = None
        
        segment = select_segment(signal, records)
        if segment:
            segments.append(segment)
            seen_sessions.add(signal.session_id)
    
    return segments


def _infer_expected_outcome(
    signal: Signal,
    records: list[TraceRecord],
) -> str:
    """Infer expected outcome based on signal type."""
    if signal.signal_type == "friction":
        # For friction, the expected outcome is to NOT have the friction
        category = signal.category
        if "denial" in category.lower():
            return "Agent should request approval only when necessary, and commands should be approved"
        elif "failure" in category.lower() or "error" in category.lower():
            return "Agent should execute commands successfully without errors"
        elif "compaction" in category.lower() or "churn" in category.lower():
            return "Agent should manage context efficiently without excessive compaction"
        else:
            return "Agent should complete task without friction"
    else:
        # For delight, we want to replicate the positive outcome
        return "Agent should complete task efficiently with positive user experience"


def filter_by_compaction(
    records: list[TraceRecord],
    include_compacted: bool = False,
) -> list[TraceRecord]:
    """
    Filter records based on compaction boundaries.
    
    By default, only returns records after the last compaction,
    representing the "current" context.
    """
    if include_compacted:
        return records
    
    compactions = get_compaction_boundaries(records)
    if not compactions:
        return records
    
    # Return records after last compaction
    last_compaction_seq = max(c.seq for c in compactions)
    return [r for r in records if r.seq > last_compaction_seq]
