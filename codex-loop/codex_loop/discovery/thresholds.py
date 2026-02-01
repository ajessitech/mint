"""
Thresholding: Determine when patterns cross action thresholds.

Implements the decisioning logic from brainstorm-updated.md lines 380-386
for deciding when to create tickets, expand eval coverage, or investigate.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any

from sqlalchemy import func
from sqlalchemy.orm import Session as DBSession

from codex_loop.db.schema import Cluster, Signal


class ActionType(str, Enum):
    """Types of actions that can be taken on a cluster."""
    CREATE_TICKET = "create_ticket"
    EXPAND_EVAL_COVERAGE = "expand_eval_coverage"
    INVESTIGATE = "investigate"
    MONITOR = "monitor"
    IGNORE = "ignore"


@dataclass
class ActionThreshold:
    """
    Thresholds for determining when to take action.
    """
    min_frequency: int = 10  # Minimum occurrences
    min_severity: float = 0.5  # Minimum average severity
    novelty_window_days: int = 7  # For detecting sudden spikes
    
    # Severity tiers
    critical_severity: float = 0.8
    high_severity: float = 0.6
    medium_severity: float = 0.4


@dataclass
class ActionDecision:
    """Result of threshold evaluation."""
    action: ActionType
    reason: str
    priority: int  # 1 = highest, 5 = lowest
    evidence: dict[str, Any]


def should_act(
    cluster: Cluster,
    db: DBSession,
    thresholds: Optional[ActionThreshold] = None,
) -> Optional[ActionDecision]:
    """
    Determine if a cluster crosses action thresholds.
    
    Args:
        cluster: The cluster to evaluate
        db: Database session for querying
        thresholds: Optional custom thresholds
        
    Returns:
        ActionDecision if action should be taken, None otherwise
    """
    if thresholds is None:
        thresholds = ActionThreshold()
    
    # Check minimum frequency
    if cluster.member_count < thresholds.min_frequency:
        return None
    
    # Get signals for severity analysis
    signals = db.query(Signal).filter(Signal.cluster_id == cluster.id).all()
    if not signals:
        return None
    
    # Calculate average severity
    avg_severity = sum(s.severity for s in signals) / len(signals)
    cluster.avg_severity = avg_severity  # Update cluster
    
    # Check for novelty (sudden spike)
    is_novel_spike = _is_novel_spike(cluster, db, thresholds.novelty_window_days)
    
    # Decision logic
    evidence = {
        "member_count": cluster.member_count,
        "avg_severity": avg_severity,
        "is_novel_spike": is_novel_spike,
        "signal_type": cluster.signal_type,
    }
    
    # Critical severity -> Create ticket
    if avg_severity >= thresholds.critical_severity:
        return ActionDecision(
            action=ActionType.CREATE_TICKET,
            reason=f"Critical severity ({avg_severity:.2f}) with {cluster.member_count} occurrences",
            priority=1,
            evidence=evidence,
        )
    
    # High severity -> Create ticket
    if avg_severity >= thresholds.high_severity and cluster.member_count >= thresholds.min_frequency * 2:
        return ActionDecision(
            action=ActionType.CREATE_TICKET,
            reason=f"High severity ({avg_severity:.2f}) with significant frequency ({cluster.member_count})",
            priority=2,
            evidence=evidence,
        )
    
    # Novel spike -> Investigate
    if is_novel_spike:
        return ActionDecision(
            action=ActionType.INVESTIGATE,
            reason=f"Novel spike detected: pattern is new within last {thresholds.novelty_window_days} days",
            priority=2,
            evidence=evidence,
        )
    
    # Medium severity with high frequency -> Expand eval coverage
    if avg_severity >= thresholds.medium_severity and cluster.member_count >= thresholds.min_frequency * 3:
        return ActionDecision(
            action=ActionType.EXPAND_EVAL_COVERAGE,
            reason=f"Medium severity ({avg_severity:.2f}) with high frequency ({cluster.member_count})",
            priority=3,
            evidence=evidence,
        )
    
    # Low severity but high frequency -> Monitor
    if cluster.member_count >= thresholds.min_frequency * 5:
        return ActionDecision(
            action=ActionType.MONITOR,
            reason=f"Low severity but very high frequency ({cluster.member_count})",
            priority=4,
            evidence=evidence,
        )
    
    return None


def _is_novel_spike(
    cluster: Cluster,
    db: DBSession,
    window_days: int,
) -> bool:
    """
    Check if this cluster represents a novel pattern (recent spike).
    
    Returns True if most signals are from the recent window.
    """
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    
    # Count recent vs total
    recent_count = db.query(func.count(Signal.id)).filter(
        Signal.cluster_id == cluster.id,
        Signal.detected_at > cutoff,
    ).scalar() or 0
    
    total_count = cluster.member_count or 1
    
    # Novel if >70% of signals are recent
    return recent_count / total_count > 0.7


def prioritize_actions(decisions: list[ActionDecision]) -> list[ActionDecision]:
    """
    Sort action decisions by priority.
    
    Args:
        decisions: List of action decisions
        
    Returns:
        Sorted list (highest priority first)
    """
    return sorted(decisions, key=lambda d: d.priority)


def filter_actionable(
    decisions: list[ActionDecision],
    max_tickets: int = 5,
) -> list[ActionDecision]:
    """
    Filter decisions to only actionable ones, with limits.
    
    Args:
        decisions: All decisions
        max_tickets: Maximum tickets to create per batch
        
    Returns:
        Filtered list
    """
    sorted_decisions = prioritize_actions(decisions)
    
    result = []
    ticket_count = 0
    
    for decision in sorted_decisions:
        if decision.action == ActionType.CREATE_TICKET:
            if ticket_count < max_tickets:
                result.append(decision)
                ticket_count += 1
        elif decision.action in [ActionType.INVESTIGATE, ActionType.EXPAND_EVAL_COVERAGE]:
            result.append(decision)
    
    return result


def get_cluster_trend(
    cluster: Cluster,
    db: DBSession,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """
    Get trend data for a cluster over time.
    
    Args:
        cluster: The cluster to analyze
        db: Database session
        lookback_days: How many days to look back
        
    Returns:
        Trend data with daily counts
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    
    signals = db.query(Signal).filter(
        Signal.cluster_id == cluster.id,
        Signal.detected_at > cutoff,
    ).all()
    
    # Group by day
    daily_counts: dict[str, int] = {}
    for signal in signals:
        if signal.detected_at:
            day = signal.detected_at.strftime("%Y-%m-%d")
            daily_counts[day] = daily_counts.get(day, 0) + 1
    
    # Calculate trend
    counts = list(daily_counts.values())
    if len(counts) >= 2:
        first_half = sum(counts[:len(counts)//2])
        second_half = sum(counts[len(counts)//2:])
        trend = "increasing" if second_half > first_half * 1.2 else (
            "decreasing" if second_half < first_half * 0.8 else "stable"
        )
    else:
        trend = "insufficient_data"
    
    return {
        "daily_counts": daily_counts,
        "total": len(signals),
        "trend": trend,
        "lookback_days": lookback_days,
    }
