"""
Mechanical signal detection: Rule-based friction and delight detection.

Implements patterns from brainstorm-updated.md lines 109-118.
These signals are detected directly from trace spine records without LLM.
"""

from typing import Callable, Optional, Any
from collections import Counter

from codex_loop.db.schema import Session
from codex_loop.ingest.spine_reader import (
    TraceRecord,
    parse_spine,
    get_events,
    get_submissions,
    get_events_by_type,
    get_submissions_by_op,
    get_compaction_boundaries,
)
from codex_loop.signals.types import (
    SignalType,
    SignalCategory,
    SignalResult,
    get_default_severity,
)
from pathlib import Path


def _get_event_msg_data(record: TraceRecord) -> dict:
    """
    Extract event message data handling both payload structures:
    1. Old/test: payload.msg.payload.{field}
    2. New/real: payload.msg.{field} (flat structure)
    """
    msg = record.payload.get("msg", {})
    nested_payload = msg.get("payload", {})
    # If nested payload exists and has meaningful data, use it; otherwise use msg directly
    if nested_payload and isinstance(nested_payload, dict) and len(nested_payload) > 0:
        return nested_payload
    return msg


def detect_mechanical_signals(
    session: Session,
    records: Optional[list[TraceRecord]] = None,
) -> list[SignalResult]:
    """
    Detect mechanical friction and delight signals from a session.
    
    Args:
        session: The canonicalized session
        records: Optional pre-parsed records (will load from spine_path if not provided)
        
    Returns:
        List of detected signals
    """
    # Load records if not provided
    if records is None:
        if not session.spine_path:
            return []
        spine_path = Path(session.spine_path)
        if not spine_path.exists():
            return []
        records = parse_spine(spine_path)
    
    if not records:
        return []
    
    signals = []
    
    # Detect friction signals
    signals.extend(_detect_error_signals(session, records))
    signals.extend(_detect_denial_signals(session, records))
    signals.extend(_detect_backtracking_signals(session, records))
    signals.extend(_detect_churn_signals(session, records))
    signals.extend(_detect_failure_signals(session, records))
    
    # Detect delight signals
    signals.extend(_detect_delight_signals(session, records))
    
    return signals


# --- Friction Detection ---

def _detect_error_signals(session: Session, records: list[TraceRecord]) -> list[SignalResult]:
    """Detect error event signals (line 110)."""
    signals = []
    
    # Error events
    errors = get_events_by_type(records, "error")
    for r in errors:
        msg = _get_event_msg_data(r)
        message = msg.get("message", "")
        
        # Check for timeout
        category = SignalCategory.ERROR_EVENT
        if "timeout" in message.lower():
            category = SignalCategory.TIMEOUT
        
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=category,
            severity=get_default_severity(category),
            session_id=session.id,
            turn_id=r.payload.get("id"),
            seq_start=r.seq,
            detector="mechanical",
            description=f"Error: {message[:100]}",
            raw_evidence={"message": message},
        ))
    
    # Stream errors
    stream_errors = get_events_by_type(records, "stream_error")
    for r in stream_errors:
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=SignalCategory.STREAM_ERROR,
            severity=get_default_severity(SignalCategory.STREAM_ERROR),
            session_id=session.id,
            turn_id=r.payload.get("id"),
            seq_start=r.seq,
            detector="mechanical",
            description="Stream error occurred",
        ))
    
    # Warnings
    warnings = get_events_by_type(records, "warning")
    for r in warnings:
        msg = _get_event_msg_data(r)
        message = msg.get("message", "")
        
        if "truncat" in message.lower():
            signals.append(SignalResult(
                signal_type=SignalType.FRICTION,
                category=SignalCategory.TRUNCATION_WARNING,
                severity=get_default_severity(SignalCategory.TRUNCATION_WARNING),
                session_id=session.id,
                turn_id=r.payload.get("id"),
                seq_start=r.seq,
                detector="mechanical",
                description=f"Truncation warning: {message[:100]}",
            ))
    
    return signals


def _detect_denial_signals(session: Session, records: list[TraceRecord]) -> list[SignalResult]:
    """Detect approval denial signals (line 114)."""
    signals = []
    
    # Get all approval submissions
    exec_approvals = get_submissions_by_op(records, "exec_approval")
    patch_approvals = get_submissions_by_op(records, "patch_approval")
    
    denial_count = 0
    denial_seqs = []
    
    for r in exec_approvals:
        op = r.payload.get("op", {})
        decision = op.get("decision", {})
        is_denied = (
            (isinstance(decision, dict) and decision.get("type") == "denied") or
            decision == "denied"
        )
        
        if is_denied:
            denial_count += 1
            denial_seqs.append(r.seq)
            signals.append(SignalResult(
                signal_type=SignalType.FRICTION,
                category=SignalCategory.DENIAL_EXEC,
                severity=get_default_severity(SignalCategory.DENIAL_EXEC),
                session_id=session.id,
                turn_id=r.payload.get("id"),
                seq_start=r.seq,
                detector="mechanical",
                description="Exec command denied",
            ))
    
    for r in patch_approvals:
        op = r.payload.get("op", {})
        decision = op.get("decision", {})
        is_denied = (
            (isinstance(decision, dict) and decision.get("type") == "denied") or
            decision == "denied"
        )
        
        if is_denied:
            denial_count += 1
            denial_seqs.append(r.seq)
            signals.append(SignalResult(
                signal_type=SignalType.FRICTION,
                category=SignalCategory.DENIAL_PATCH,
                severity=get_default_severity(SignalCategory.DENIAL_PATCH),
                session_id=session.id,
                turn_id=r.payload.get("id"),
                seq_start=r.seq,
                detector="mechanical",
                description="Patch application denied",
            ))
    
    # Check for repeated denials
    if denial_count >= 3:
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=SignalCategory.REPEATED_DENIAL,
            severity=get_default_severity(SignalCategory.REPEATED_DENIAL),
            session_id=session.id,
            seq_start=min(denial_seqs) if denial_seqs else None,
            seq_end=max(denial_seqs) if denial_seqs else None,
            detector="mechanical",
            description=f"Repeated denials: {denial_count} denials in session",
            raw_evidence={"denial_count": denial_count},
        ))
    
    return signals


def _detect_backtracking_signals(session: Session, records: list[TraceRecord]) -> list[SignalResult]:
    """Detect backtracking signals (line 115)."""
    signals = []
    
    # Undo operations
    undos = get_submissions_by_op(records, "Undo")
    for r in undos:
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=SignalCategory.UNDO,
            severity=get_default_severity(SignalCategory.UNDO),
            session_id=session.id,
            turn_id=r.payload.get("id"),
            seq_start=r.seq,
            detector="mechanical",
            description="Undo operation",
        ))
    
    # Thread rollback
    rollbacks = get_submissions_by_op(records, "ThreadRollback")
    for r in rollbacks:
        op = r.payload.get("op", {})
        num_turns = op.get("num_turns", 1)
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=SignalCategory.ROLLBACK,
            severity=get_default_severity(SignalCategory.ROLLBACK),
            session_id=session.id,
            turn_id=r.payload.get("id"),
            seq_start=r.seq,
            detector="mechanical",
            description=f"Thread rollback: {num_turns} turns",
            raw_evidence={"num_turns": num_turns},
        ))
    
    # Check user messages for revert language
    user_turns = get_submissions_by_op(records, "user_turn")
    revert_words = ["undo", "revert", "go back", "delete that", "remove that", "cancel"]
    
    for r in user_turns:
        op = r.payload.get("op", {})
        items = op.get("items", [])
        for item in items:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "").lower()
                if any(word in text for word in revert_words):
                    signals.append(SignalResult(
                        signal_type=SignalType.FRICTION,
                        category=SignalCategory.REVERT_IN_MESSAGE,
                        severity=get_default_severity(SignalCategory.REVERT_IN_MESSAGE),
                        session_id=session.id,
                        turn_id=r.payload.get("id"),
                        seq_start=r.seq,
                        detector="mechanical",
                        description="Revert language in user message",
                    ))
                    break
    
    return signals


def _detect_churn_signals(session: Session, records: list[TraceRecord]) -> list[SignalResult]:
    """Detect context churn signals (line 116)."""
    signals = []
    
    # Compaction events
    compactions = get_compaction_boundaries(records)
    if compactions:
        for r in compactions:
            signals.append(SignalResult(
                signal_type=SignalType.FRICTION,
                category=SignalCategory.COMPACTION,
                severity=get_default_severity(SignalCategory.COMPACTION),
                session_id=session.id,
                turn_id=r.payload.get("turn_id"),
                seq_start=r.seq,
                detector="mechanical",
                description="Context compaction occurred",
            ))
    
    # Repeated compaction
    if len(compactions) >= 2:
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=SignalCategory.REPEATED_COMPACTION,
            severity=get_default_severity(SignalCategory.REPEATED_COMPACTION),
            session_id=session.id,
            seq_start=compactions[0].seq,
            seq_end=compactions[-1].seq,
            detector="mechanical",
            description=f"Repeated compaction: {len(compactions)} compactions",
            raw_evidence={"compaction_count": len(compactions)},
        ))
    
    # File churn detection
    file_churn = _detect_file_churn(records)
    if file_churn:
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=SignalCategory.FILE_CHURN,
            severity=get_default_severity(SignalCategory.FILE_CHURN),
            session_id=session.id,
            detector="mechanical",
            description=f"File churn detected: {', '.join(file_churn[:3])}",
            raw_evidence={"churned_files": file_churn},
        ))
    
    # Plan update churn
    plan_updates = get_events_by_type(records, "plan_update")
    if len(plan_updates) >= 5:
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=SignalCategory.PLAN_UPDATE_CHURN,
            severity=get_default_severity(SignalCategory.PLAN_UPDATE_CHURN),
            session_id=session.id,
            seq_start=plan_updates[0].seq,
            seq_end=plan_updates[-1].seq,
            detector="mechanical",
            description=f"Plan instability: {len(plan_updates)} plan updates",
            raw_evidence={"plan_update_count": len(plan_updates)},
        ))
    
    return signals


def _detect_failure_signals(session: Session, records: list[TraceRecord]) -> list[SignalResult]:
    """Detect tool execution failures."""
    signals = []
    
    # Exec failures (non-zero exit code)
    exec_ends = get_events_by_type(records, "exec_command_end")
    for r in exec_ends:
        # Handle both payload structures:
        # 1. Old: payload.msg.payload.exit_code
        # 2. New: payload.msg.exit_code (flat structure)
        msg = r.payload.get("msg", {})
        nested_payload = msg.get("payload", {})
        exit_code = nested_payload.get("exit_code") if nested_payload else msg.get("exit_code")
        call_id = nested_payload.get("call_id") if nested_payload else msg.get("call_id")
        
        if exit_code is not None and exit_code != 0:
            signals.append(SignalResult(
                signal_type=SignalType.FRICTION,
                category=SignalCategory.EXEC_FAILURE,
                severity=get_default_severity(SignalCategory.EXEC_FAILURE),
                session_id=session.id,
                turn_id=r.payload.get("id"),
                call_id=call_id,
                seq_start=r.seq,
                detector="mechanical",
                description=f"Command failed with exit code {exit_code}",
                raw_evidence={"exit_code": exit_code},
            ))
    
    # Patch failures
    patch_ends = get_events_by_type(records, "patch_apply_end")
    for r in patch_ends:
        msg_data = _get_event_msg_data(r)
        success = msg_data.get("success", True)
        
        if not success:
            signals.append(SignalResult(
                signal_type=SignalType.FRICTION,
                category=SignalCategory.PATCH_FAILURE,
                severity=get_default_severity(SignalCategory.PATCH_FAILURE),
                session_id=session.id,
                turn_id=r.payload.get("id"),
                call_id=msg_data.get("call_id"),
                seq_start=r.seq,
                detector="mechanical",
                description="Patch application failed",
            ))
    
    return signals


def _detect_file_churn(records: list[TraceRecord]) -> list[str]:
    """Detect files that were added then removed (or vice versa)."""
    file_actions: dict[str, list[str]] = {}
    
    patch_ends = get_events_by_type(records, "patch_apply_end")
    for r in patch_ends:
        msg_data = _get_event_msg_data(r)
        changes = msg_data.get("changes", {})
        
        for path, change in changes.items():
            if isinstance(change, dict):
                change_type = change.get("type", "update")
                file_actions.setdefault(path, []).append(change_type)
    
    churned_files = []
    for path, actions in file_actions.items():
        if "add" in actions and "delete" in actions:
            churned_files.append(path)
    
    return churned_files


# --- Delight Detection ---

def _detect_delight_signals(session: Session, records: list[TraceRecord]) -> list[SignalResult]:
    """Detect delight signals (line 118)."""
    signals = []
    
    # Fast completion: few turns, successful outcome
    turn_count = session.turn_count or len(get_submissions_by_op(records, "user_turn"))
    if turn_count <= 3 and turn_count > 0:
        # Check for success indicators (exec or patch)
        exec_ends = get_events_by_type(records, "exec_command_end")
        successful_execs = [
            r for r in exec_ends
            if _get_event_msg_data(r).get("exit_code") == 0
        ]
        
        patch_ends = get_events_by_type(records, "patch_apply_end")
        successful_patches = [
            r for r in patch_ends
            if _get_event_msg_data(r).get("success", True)
        ]
        
        if successful_execs or successful_patches:
            signals.append(SignalResult(
                signal_type=SignalType.DELIGHT,
                category=SignalCategory.FAST_COMPLETION,
                severity=get_default_severity(SignalCategory.FAST_COMPLETION),
                session_id=session.id,
                detector="mechanical",
                description=f"Fast completion: {turn_count} turns with success",
                raw_evidence={
                    "turn_count": turn_count, 
                    "successful_execs": len(successful_execs),
                    "successful_patches": len(successful_patches),
                },
            ))
    
    # Zero denial: approvals requested but none denied
    approval_requests = (
        get_events_by_type(records, "exec_approval_request") +
        get_events_by_type(records, "apply_patch_approval_request")
    )
    denials = session.facets.get("denial_count", 0)
    
    if len(approval_requests) > 0 and denials == 0:
        signals.append(SignalResult(
            signal_type=SignalType.DELIGHT,
            category=SignalCategory.ZERO_DENIAL,
            severity=get_default_severity(SignalCategory.ZERO_DENIAL),
            session_id=session.id,
            detector="mechanical",
            description=f"Zero denials: {len(approval_requests)} approvals requested, none denied",
            raw_evidence={"approval_count": len(approval_requests)},
        ))
    
    # Clean execution: all exec commands succeeded
    exec_ends = get_events_by_type(records, "exec_command_end")
    if exec_ends:
        all_succeeded = all(
            _get_event_msg_data(r).get("exit_code") == 0
            for r in exec_ends
        )
        if all_succeeded:
            signals.append(SignalResult(
                signal_type=SignalType.DELIGHT,
                category=SignalCategory.FIRST_ATTEMPT_SUCCESS,
                severity=get_default_severity(SignalCategory.FIRST_ATTEMPT_SUCCESS),
                session_id=session.id,
                detector="mechanical",
                description=f"All {len(exec_ends)} commands succeeded",
                raw_evidence={"exec_count": len(exec_ends)},
            ))
    
    return signals


# Export main detection lists for reference
MECHANICAL_FRICTION_SIGNALS = [
    "error_event", "exec_failure", "patch_failure", "stream_error", "timeout",
    "denial_exec", "denial_patch", "repeated_denial", "abandoned_tool_flow",
    "undo", "rollback", "revert_in_message", "backtracking",
    "compaction", "repeated_compaction", "file_churn", "context_churn",
    "plan_update_churn", "diff_oscillation", "truncation_warning",
]

MECHANICAL_DELIGHT_SIGNALS = [
    "fast_completion", "first_attempt_success", "zero_denial",
    "clean_plan", "rapid_approvals",
]
