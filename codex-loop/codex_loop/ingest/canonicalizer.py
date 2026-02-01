"""
Canonicalizer: Transform raw trace spine records into canonical SQLite schema.

Implements the normalization from brainstorm-updated.md Stage 3 (Canonicalize).
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import uuid

from codex_loop.db.schema import Session, Turn, ToolCall
from codex_loop.ingest.spine_reader import (
    TraceRecord,
    get_session_meta,
    get_submissions,
    get_events,
    get_turn_contexts,
    get_compaction_boundaries,
    get_events_by_type,
    get_submissions_by_op,
)


def canonicalize_session(
    records: list[TraceRecord],
    spine_path: Optional[Path] = None,
) -> Session:
    """
    Transform trace records into a canonical Session object.
    
    Args:
        records: List of TraceRecords from the spine
        spine_path: Optional path to the spine directory
        
    Returns:
        Populated Session object (not yet committed to DB)
    """
    if not records:
        raise ValueError("Cannot canonicalize empty record list")
    
    # Get session metadata
    meta_record = get_session_meta(records)
    if not meta_record:
        # Fallback: use thread_id from first record
        thread_id = records[0].thread_id
        meta_payload = {}
    else:
        thread_id = meta_record.thread_id
        meta_payload = meta_record.payload
    
    # Extract session-level fields from meta
    meta = meta_payload.get("meta", meta_payload)
    git_info = meta_payload.get("git", {})
    
    # Parse timestamps
    started_at = _parse_timestamp(meta.get("timestamp")) or datetime.utcnow()
    ended_at = _get_session_end_time(records)
    
    # Compute metrics
    compaction_count = len(get_compaction_boundaries(records))
    turn_contexts = get_turn_contexts(records)
    turn_count = len(turn_contexts)
    
    # Count tool calls
    exec_begins = get_events_by_type(records, "exec_command_begin")
    patch_begins = get_events_by_type(records, "patch_apply_begin")
    tool_call_count = len(exec_begins) + len(patch_begins)
    
    # Build facets
    facets = _build_facets(records, meta)
    
    session = Session(
        id=thread_id,
        source=_extract_source(meta.get("source")),
        model_provider=meta.get("model_provider"),
        model=_extract_model(records),
        started_at=started_at,
        ended_at=ended_at,
        cwd=meta.get("cwd"),
        git_repo_url=git_info.get("repository_url"),
        git_commit=git_info.get("commit_hash"),
        git_branch=git_info.get("branch"),
        cli_version=meta.get("cli_version"),
        originator=meta.get("originator"),
        compaction_count=compaction_count,
        turn_count=turn_count,
        tool_call_count=tool_call_count,
        facets=facets,
        signals_computed=False,
        embeddings_computed=False,
        spine_path=str(spine_path) if spine_path else None,
    )
    
    # Add turns
    session.turns = _build_turns(records, thread_id)
    
    return session


def _build_turns(records: list[TraceRecord], session_id: str) -> list[Turn]:
    """Build Turn objects from trace records."""
    turns = []
    turn_contexts = get_turn_contexts(records)
    
    for i, tc in enumerate(turn_contexts):
        # Create unique turn ID using session_id to avoid duplicates across sessions
        base_id = tc.payload.get("turn_id", f"turn_{i}")
        turn_id = f"{session_id[:12]}_tc_{base_id}_{tc.seq}"
        context = tc.payload.get("context", {})
        
        turn = Turn(
            id=turn_id,
            session_id=session_id,
            seq_start=tc.seq,
            seq_end=None,  # Will be computed when next turn starts
            approval_policy=_serialize_policy(context.get("approval_policy")),
            sandbox_policy=_serialize_policy(context.get("sandbox_policy")),
            model=context.get("model"),
            user_message=_extract_user_message(records, turn_id),
            started_at=tc.get_timestamp_dt(),
        )
        turns.append(turn)
    
    # Compute seq_end for each turn
    for i, turn in enumerate(turns):
        if i + 1 < len(turns):
            turn.seq_end = turns[i + 1].seq_start - 1
    
    # If no turn contexts, try to build from submissions
    if not turns:
        turns = _build_turns_from_submissions(records, session_id)
    
    return turns


def _build_turns_from_submissions(records: list[TraceRecord], session_id: str) -> list[Turn]:
    """Build turns from user_turn submissions when turn_context records are missing."""
    turns = []
    user_turns = get_submissions_by_op(records, "user_turn")
    
    for i, sub in enumerate(user_turns):
        # Create unique turn ID using session_id and index to avoid duplicates
        base_id = sub.payload.get("id", str(i))
        turn_id = f"{session_id[:12]}_turn_{base_id}_{sub.seq}"
        op = sub.payload.get("op", {})
        
        # Extract user message from items
        items = op.get("items", [])
        user_message = ""
        for item in items:
            if isinstance(item, dict) and item.get("type") == "text":
                user_message += item.get("text", "")
        
        turn = Turn(
            id=turn_id,
            session_id=session_id,
            seq_start=sub.seq,
            approval_policy=_serialize_policy(op.get("approval_policy")),
            sandbox_policy=_serialize_policy(op.get("sandbox_policy")),
            model=op.get("model"),
            user_message=user_message,
            started_at=sub.get_timestamp_dt(),
        )
        turns.append(turn)
    
    return turns


def _build_facets(records: list[TraceRecord], meta: dict) -> dict[str, Any]:
    """
    Build facets dictionary for the session.
    
    Maps to: brainstorm-updated.md lines 93-99 (spine facets)
    """
    facets = {}
    
    # Spine facets (high-trust)
    facets["session_source"] = _extract_source(meta.get("source"))
    facets["model_provider"] = meta.get("model_provider")
    facets["compaction_count"] = len(get_compaction_boundaries(records))
    
    # Tool usage measures
    exec_events = get_events_by_type(records, "exec_command_end")
    patch_events = get_events_by_type(records, "patch_apply_end")
    facets["exec_count"] = len(exec_events)
    facets["patch_count"] = len(patch_events)
    
    # Approval metrics
    approval_requests = (
        get_events_by_type(records, "exec_approval_request") +
        get_events_by_type(records, "apply_patch_approval_request")
    )
    facets["approval_request_count"] = len(approval_requests)
    
    # Denial count
    exec_approvals = get_submissions_by_op(records, "exec_approval")
    patch_approvals = get_submissions_by_op(records, "patch_approval")
    denial_count = 0
    for approval in exec_approvals + patch_approvals:
        decision = approval.payload.get("op", {}).get("decision", {})
        if isinstance(decision, dict) and decision.get("type") == "denied":
            denial_count += 1
        elif decision == "denied":
            denial_count += 1
    facets["denial_count"] = denial_count
    
    # Error/warning counts
    errors = get_events_by_type(records, "error")
    warnings = get_events_by_type(records, "warning")
    stream_errors = get_events_by_type(records, "stream_error")
    facets["error_count"] = len(errors)
    facets["warning_count"] = len(warnings)
    facets["stream_error_count"] = len(stream_errors)
    
    return facets


def _extract_source(source: Any) -> str:
    """Extract source string from various formats."""
    if source is None:
        return "unknown"
    if isinstance(source, str):
        return source
    if isinstance(source, dict):
        # Handle SubAgent source
        if "SubAgent" in source:
            sub_agent = source["SubAgent"]
            if isinstance(sub_agent, str):
                return f"SubAgent:{sub_agent}"
            elif isinstance(sub_agent, dict):
                return f"SubAgent:{sub_agent.get('type', 'unknown')}"
        # Handle simple type field
        return source.get("type", str(source))
    return str(source)


def _extract_model(records: list[TraceRecord]) -> Optional[str]:
    """Extract the model used from turn contexts."""
    turn_contexts = get_turn_contexts(records)
    for tc in turn_contexts:
        context = tc.payload.get("context", {})
        model = context.get("model")
        if model:
            return model
    return None


def _extract_user_message(records: list[TraceRecord], turn_id: str) -> Optional[str]:
    """Extract user message for a specific turn."""
    # Look for user_turn submission with matching turn_id
    for r in records:
        if r.is_submission():
            if r.payload.get("id") == turn_id:
                op = r.payload.get("op", {})
                if op.get("type") == "user_turn":
                    items = op.get("items", [])
                    texts = []
                    for item in items:
                        if isinstance(item, dict) and item.get("type") == "text":
                            texts.append(item.get("text", ""))
                    return "\n".join(texts) if texts else None
    return None


def _get_session_end_time(records: list[TraceRecord]) -> Optional[datetime]:
    """Get the timestamp of the last record."""
    if not records:
        return None
    last_record = max(records, key=lambda r: r.seq)
    return last_record.get_timestamp_dt()


def _parse_timestamp(ts: Any) -> Optional[datetime]:
    """Parse a timestamp string to datetime."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _serialize_policy(policy: Any) -> Optional[str]:
    """Serialize a policy object to string."""
    if policy is None:
        return None
    if isinstance(policy, str):
        return policy
    if isinstance(policy, dict):
        # Return the policy type or serialize
        return policy.get("type", str(policy))
    return str(policy)
