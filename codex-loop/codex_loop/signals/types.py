"""
Signal type definitions for friction and delight detection.

Maps to: brainstorm-updated.md lines 109-118 (signal patterns)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class SignalType(str, Enum):
    """Top-level signal classification."""
    FRICTION = "friction"
    DELIGHT = "delight"


class SignalCategory(str, Enum):
    """
    Specific signal categories.
    
    Friction patterns (brainstorm-updated.md lines 109-117):
    """
    # Line 110: "Error events: model errors, tool failures, timeouts"
    ERROR_EVENT = "error_event"
    EXEC_FAILURE = "exec_failure"
    PATCH_FAILURE = "patch_failure"
    STREAM_ERROR = "stream_error"
    TIMEOUT = "timeout"
    
    # Line 111: "Repeated rephrasing: >=3 consecutive restating messages"
    REPEATED_REPHRASING = "repeated_rephrasing"
    
    # Line 112: "Escalation tone: 'broken', 'why isn't', 'frustrating'"
    ESCALATION_TONE = "escalation_tone"
    
    # Line 113: "Platform confusion: questions about Codex features"
    PLATFORM_CONFUSION = "platform_confusion"
    
    # Line 114: "Abandoned tool flow: tool calls rejected or cancelled"
    DENIAL_EXEC = "denial_exec"
    DENIAL_PATCH = "denial_patch"
    REPEATED_DENIAL = "repeated_denial"
    ABANDONED_TOOL_FLOW = "abandoned_tool_flow"
    
    # Line 115: "Backtracking: 'undo', 'revert', deleting code"
    UNDO = "undo"
    ROLLBACK = "rollback"
    REVERT_IN_MESSAGE = "revert_in_message"
    BACKTRACKING = "backtracking"
    
    # Line 116: "Context churn: add/remove the same file repeatedly"
    COMPACTION = "compaction"
    REPEATED_COMPACTION = "repeated_compaction"
    FILE_CHURN = "file_churn"
    CONTEXT_CHURN = "context_churn"
    
    # Additional mechanical indicators
    PLAN_UPDATE_CHURN = "plan_update_churn"
    DIFF_OSCILLATION = "diff_oscillation"
    TRUNCATION_WARNING = "truncation_warning"
    
    # Delight patterns (line 118):
    FAST_COMPLETION = "fast_completion"
    FIRST_ATTEMPT_SUCCESS = "first_attempt_success"
    ZERO_DENIAL = "zero_denial"
    CLEAN_PLAN = "clean_plan"
    RAPID_APPROVALS = "rapid_approvals"
    POSITIVE_FEEDBACK = "positive_feedback"


@dataclass
class SignalResult:
    """
    Result of signal detection.
    
    Contains evidence pointers per brainstorm-updated.md lines 897-901.
    """
    signal_type: SignalType
    category: SignalCategory | str
    severity: float  # 0.0 to 1.0
    
    # Evidence pointers
    session_id: str
    turn_id: Optional[str] = None
    call_id: Optional[str] = None
    seq_start: Optional[int] = None
    seq_end: Optional[int] = None
    
    # Detection metadata
    detector: str = "mechanical"  # "mechanical" or "semantic"
    confidence: float = 1.0
    
    # Additional context
    description: Optional[str] = None
    raw_evidence: dict[str, Any] = field(default_factory=dict)
    
    def to_evidence_dict(self) -> dict[str, Any]:
        """Build evidence pointer dictionary."""
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "call_id": self.call_id,
            "seq_range": [self.seq_start, self.seq_end] if self.seq_start else None,
            "description": self.description,
            "raw": self.raw_evidence,
        }


# Severity guidelines per signal category
SEVERITY_GUIDELINES = {
    # High severity (0.7-1.0): Clear user frustration or system failure
    SignalCategory.ERROR_EVENT: 0.8,
    SignalCategory.STREAM_ERROR: 0.9,
    SignalCategory.TIMEOUT: 0.8,
    SignalCategory.REPEATED_DENIAL: 0.9,
    SignalCategory.ESCALATION_TONE: 0.85,
    SignalCategory.ABANDONED_TOOL_FLOW: 0.8,
    
    # Medium severity (0.4-0.7): Friction but recoverable
    SignalCategory.EXEC_FAILURE: 0.6,
    SignalCategory.PATCH_FAILURE: 0.6,
    SignalCategory.DENIAL_EXEC: 0.5,
    SignalCategory.DENIAL_PATCH: 0.5,
    SignalCategory.REPEATED_REPHRASING: 0.7,
    SignalCategory.PLATFORM_CONFUSION: 0.6,
    SignalCategory.BACKTRACKING: 0.5,
    SignalCategory.REPEATED_COMPACTION: 0.6,
    SignalCategory.FILE_CHURN: 0.5,
    SignalCategory.PLAN_UPDATE_CHURN: 0.6,
    
    # Low severity (0.1-0.4): Minor friction
    SignalCategory.UNDO: 0.3,
    SignalCategory.ROLLBACK: 0.4,
    SignalCategory.COMPACTION: 0.2,
    SignalCategory.TRUNCATION_WARNING: 0.3,
    
    # Delight (positive signals)
    SignalCategory.FAST_COMPLETION: 0.8,
    SignalCategory.FIRST_ATTEMPT_SUCCESS: 0.9,
    SignalCategory.ZERO_DENIAL: 0.7,
    SignalCategory.CLEAN_PLAN: 0.7,
    SignalCategory.RAPID_APPROVALS: 0.6,
    SignalCategory.POSITIVE_FEEDBACK: 0.9,
}


def get_default_severity(category: SignalCategory | str) -> float:
    """Get the default severity for a signal category."""
    if isinstance(category, str):
        try:
            category = SignalCategory(category)
        except ValueError:
            return 0.5  # Default for unknown categories
    return SEVERITY_GUIDELINES.get(category, 0.5)
