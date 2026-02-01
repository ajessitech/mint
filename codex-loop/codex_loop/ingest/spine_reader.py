"""
Trace spine reader: Parse JSONL trace files from ~/.codex/trace_spine/.

Implements parsing for the codex_trace_spine_v1 format defined in
codex-rs/docs/trace_spine_v1.md.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, Iterator
import json


@dataclass
class TraceRecord:
    """
    A single record from the trace spine.
    
    Maps to the JSONL envelope from trace_spine_v1.md.
    """
    schema_version: str
    thread_id: str
    seq: int
    timestamp: str
    type: str  # session_meta, turn_context, submission, event, artifact_ref, compaction_boundary, bridge
    payload: dict[str, Any]
    
    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TraceRecord":
        """Parse a JSON dict into a TraceRecord."""
        return cls(
            schema_version=data.get("schema_version", "unknown"),
            thread_id=data.get("thread_id", ""),
            seq=data.get("seq", 0),
            timestamp=data.get("timestamp", ""),
            type=data.get("type", "unknown"),
            payload=data.get("payload", {}),
        )
    
    def get_timestamp_dt(self) -> Optional[datetime]:
        """Parse timestamp string to datetime."""
        if not self.timestamp:
            return None
        try:
            # Handle ISO format with milliseconds
            return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    # Convenience accessors for common record types
    
    def is_session_meta(self) -> bool:
        return self.type == "session_meta"
    
    def is_submission(self) -> bool:
        return self.type == "submission"
    
    def is_event(self) -> bool:
        return self.type == "event"
    
    def is_turn_context(self) -> bool:
        return self.type == "turn_context"
    
    def is_compaction_boundary(self) -> bool:
        return self.type == "compaction_boundary"
    
    def is_artifact_ref(self) -> bool:
        return self.type == "artifact_ref"
    
    def is_bridge(self) -> bool:
        return self.type == "bridge"
    
    def get_event_type(self) -> Optional[str]:
        """Get the event message type if this is an event record."""
        if not self.is_event():
            return None
        msg = self.payload.get("msg", {})
        return msg.get("type")
    
    def get_op_type(self) -> Optional[str]:
        """Get the operation type if this is a submission record."""
        if not self.is_submission():
            return None
        op = self.payload.get("op", {})
        return op.get("type")


def find_all_spines(codex_home: Path) -> list[Path]:
    """
    Find all trace spine directories in CODEX_HOME.
    
    Returns:
        List of paths to spine directories (each containing segment-*.jsonl files)
    """
    trace_spine_dir = codex_home / "trace_spine"
    if not trace_spine_dir.exists():
        return []
    
    spine_dirs = []
    for item in trace_spine_dir.iterdir():
        if item.is_dir():
            # Check if it contains segment files
            segment_files = list(item.glob("segment-*.jsonl"))
            if segment_files:
                spine_dirs.append(item)
    
    # Sort by modification time (newest first)
    spine_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return spine_dirs


def find_spines_since(codex_home: Path, hours: int = 24) -> list[Path]:
    """
    Find trace spines modified within the last N hours.
    
    Args:
        codex_home: Path to ~/.codex
        hours: Lookback window in hours
        
    Returns:
        List of spine directory paths
    """
    cutoff = datetime.now() - timedelta(hours=hours)
    cutoff_ts = cutoff.timestamp()
    
    all_spines = find_all_spines(codex_home)
    return [p for p in all_spines if p.stat().st_mtime > cutoff_ts]


def parse_spine(spine_path: Path) -> list[TraceRecord]:
    """
    Parse all records from a trace spine directory.
    
    Args:
        spine_path: Path to the spine directory (containing segment-*.jsonl files)
        
    Returns:
        List of TraceRecords sorted by seq number
    """
    records = []
    
    # Find all segment files
    segment_files = sorted(spine_path.glob("segment-*.jsonl"))
    
    for segment_file in segment_files:
        for record in parse_segment_file(segment_file):
            records.append(record)
    
    # Sort by sequence number for deterministic ordering
    records.sort(key=lambda r: r.seq)
    return records


def parse_segment_file(segment_file: Path) -> Iterator[TraceRecord]:
    """
    Parse a single segment JSONL file.
    
    Yields:
        TraceRecord objects for each line
    """
    try:
        with open(segment_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield TraceRecord.from_json(data)
                except json.JSONDecodeError as e:
                    # Log but continue - don't fail on malformed lines
                    print(f"Warning: Failed to parse line {line_num} in {segment_file}: {e}")
    except IOError as e:
        print(f"Warning: Failed to read {segment_file}: {e}")


def get_thread_id(spine_path: Path) -> Optional[str]:
    """
    Extract the thread ID from a spine directory.
    
    The directory name is typically the thread_id.
    """
    return spine_path.name


# Utility functions for extracting specific record types

def get_session_meta(records: list[TraceRecord]) -> Optional[TraceRecord]:
    """Get the session_meta record from a list of records."""
    for r in records:
        if r.is_session_meta():
            return r
    return None


def get_submissions(records: list[TraceRecord]) -> list[TraceRecord]:
    """Get all submission records."""
    return [r for r in records if r.is_submission()]


def get_events(records: list[TraceRecord]) -> list[TraceRecord]:
    """Get all event records."""
    return [r for r in records if r.is_event()]


def get_turn_contexts(records: list[TraceRecord]) -> list[TraceRecord]:
    """Get all turn_context records."""
    return [r for r in records if r.is_turn_context()]


def get_compaction_boundaries(records: list[TraceRecord]) -> list[TraceRecord]:
    """Get all compaction_boundary records."""
    return [r for r in records if r.is_compaction_boundary()]


def get_events_by_type(records: list[TraceRecord], event_type: str) -> list[TraceRecord]:
    """Get events of a specific type."""
    return [r for r in records if r.is_event() and r.get_event_type() == event_type]


def get_submissions_by_op(records: list[TraceRecord], op_type: str) -> list[TraceRecord]:
    """Get submissions of a specific operation type."""
    return [r for r in records if r.is_submission() and r.get_op_type() == op_type]
