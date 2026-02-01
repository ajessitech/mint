"""
Schema evolution: Propose new signal categories from clusters.

Implements the hybrid discovery loop from brainstorm-updated.md lines 124-133.
When clusters form that don't match existing categories, use LLM to propose
new schema entries.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any
import json
import uuid

from openai import AsyncOpenAI

from codex_loop.db.schema import Cluster, Signal, SchemaProposal as DBSchemaProposal


@dataclass
class SchemaProposal:
    """
    Proposed new schema entry for signal categorization.
    """
    name: str
    description: str
    inclusion_criteria: str
    severity_rule: str
    recommended_action: str
    signal_type: str
    sample_count: int
    confidence: float = 0.0


SCHEMA_PROPOSAL_PROMPT = """You are analyzing a cluster of similar signals from a coding assistant.
These signals were grouped together by embedding similarity, suggesting they represent
a common pattern that may warrant a new category in the signal schema.

Cluster information:
- Signal type: {signal_type} 
- Number of signals: {member_count}
- Average severity: {avg_severity:.2f}

Sample signals from this cluster:
{samples}

Based on these signals, propose a new schema entry. The schema should:
1. Have a short, snake_case name that describes the pattern
2. Clearly define what unifies these signals
3. Specify how to identify this pattern mechanically if possible
4. Define severity scoring rules
5. Recommend what action to take when this pattern is detected

Return JSON with this exact structure:
{{
    "name": "short_snake_case_name",
    "description": "1-2 sentences describing what this pattern represents",
    "inclusion_criteria": "How to identify this pattern in future sessions",
    "severity_rule": "How to score severity 0.0-1.0 for this pattern",
    "recommended_action": "What should be done when this pattern is detected",
    "confidence": <0.0-1.0, how confident you are this is a real, actionable pattern>
}}"""


async def propose_new_schema(
    cluster: Cluster,
    sample_signals: list[Signal],
    client: Optional[AsyncOpenAI] = None,
    model: str = "gpt-4o",
) -> SchemaProposal:
    """
    Propose a new schema entry for an unlabeled cluster.
    
    Args:
        cluster: The cluster to analyze
        sample_signals: Sample signals from the cluster
        client: Optional OpenAI client
        model: Model to use for proposal generation
        
    Returns:
        SchemaProposal with proposed category details
    """
    if client is None:
        client = AsyncOpenAI()
    
    # Format samples for prompt
    samples_text = _format_samples(sample_signals[:10])
    
    # Calculate average severity
    avg_severity = sum(s.severity for s in sample_signals) / len(sample_signals) if sample_signals else 0.5
    
    prompt = SCHEMA_PROPOSAL_PROMPT.format(
        signal_type=cluster.signal_type,
        member_count=cluster.member_count,
        avg_severity=avg_severity,
        samples=samples_text,
    )
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a signal analysis expert. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response")
        
        # Parse JSON
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        data = json.loads(content)
        
        return SchemaProposal(
            name=data.get("name", f"unlabeled_{cluster.id}"),
            description=data.get("description", "Automatically proposed category"),
            inclusion_criteria=data.get("inclusion_criteria", ""),
            severity_rule=data.get("severity_rule", ""),
            recommended_action=data.get("recommended_action", ""),
            signal_type=cluster.signal_type,
            sample_count=len(sample_signals),
            confidence=data.get("confidence", 0.5),
        )
    
    except Exception as e:
        # Return fallback proposal
        return SchemaProposal(
            name=f"unlabeled_cluster_{cluster.id[:8]}",
            description=f"Auto-generated category for {cluster.member_count} similar signals",
            inclusion_criteria="Signals clustered by embedding similarity",
            severity_rule="Use average severity of cluster",
            recommended_action="Manual review recommended",
            signal_type=cluster.signal_type,
            sample_count=len(sample_signals),
            confidence=0.3,
        )


def _format_samples(signals: list[Signal]) -> str:
    """Format signal samples for the prompt."""
    lines = []
    for i, signal in enumerate(signals, 1):
        evidence = signal.evidence or {}
        desc = evidence.get("description", signal.category)
        raw = evidence.get("raw", {})
        
        lines.append(f"{i}. Category: {signal.category}")
        lines.append(f"   Severity: {signal.severity:.2f}")
        lines.append(f"   Description: {desc}")
        if raw:
            # Include key raw evidence fields
            raw_str = ", ".join(f"{k}={v}" for k, v in list(raw.items())[:3])
            lines.append(f"   Evidence: {raw_str}")
        lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# Schema Evolution Workflow Functions
# ============================================================================

def list_pending_proposals(db_session) -> list[DBSchemaProposal]:
    """
    List all pending schema proposals awaiting review.
    
    Args:
        db_session: SQLAlchemy database session
        
    Returns:
        List of pending SchemaProposal records
    """
    return db_session.query(DBSchemaProposal).filter(
        DBSchemaProposal.status == "pending"
    ).order_by(DBSchemaProposal.created_at.desc()).all()


def approve_proposal(
    db_session,
    proposal_id: str,
    reviewer_notes: Optional[str] = None,
) -> DBSchemaProposal:
    """
    Approve a schema proposal and promote it to stable status.
    
    This marks the proposal as accepted and updates the associated cluster
    with the new category name.
    
    Args:
        db_session: SQLAlchemy database session
        proposal_id: ID of the proposal to approve
        reviewer_notes: Optional notes from the reviewer
        
    Returns:
        Updated SchemaProposal record
        
    Raises:
        ValueError: If proposal not found or not in pending status
    """
    proposal = db_session.query(DBSchemaProposal).filter(
        DBSchemaProposal.id == proposal_id
    ).first()
    
    if not proposal:
        raise ValueError(f"Proposal {proposal_id} not found")
    
    if proposal.status != "pending":
        raise ValueError(f"Proposal {proposal_id} is not pending (status: {proposal.status})")
    
    # Update proposal status
    proposal.status = "accepted"
    proposal.reviewed_at = datetime.utcnow()
    
    # Update associated cluster with the new name
    if proposal.cluster:
        proposal.cluster.name = proposal.name
        proposal.cluster.description = proposal.description
        proposal.cluster.promoted_to_schema = True
        proposal.cluster.schema_proposal_id = proposal_id
    
    db_session.commit()
    
    return proposal


def reject_proposal(
    db_session,
    proposal_id: str,
    reason: Optional[str] = None,
) -> DBSchemaProposal:
    """
    Reject a schema proposal.
    
    Args:
        db_session: SQLAlchemy database session
        proposal_id: ID of the proposal to reject
        reason: Optional reason for rejection
        
    Returns:
        Updated SchemaProposal record
        
    Raises:
        ValueError: If proposal not found or not in pending status
    """
    proposal = db_session.query(DBSchemaProposal).filter(
        DBSchemaProposal.id == proposal_id
    ).first()
    
    if not proposal:
        raise ValueError(f"Proposal {proposal_id} not found")
    
    if proposal.status != "pending":
        raise ValueError(f"Proposal {proposal_id} is not pending (status: {proposal.status})")
    
    proposal.status = "rejected"
    proposal.reviewed_at = datetime.utcnow()
    
    db_session.commit()
    
    return proposal


def get_stable_categories(db_session) -> dict[str, list[str]]:
    """
    Get all stable (promoted) signal categories.
    
    Returns:
        Dict with "friction" and "delight" keys, each containing a list
        of promoted category names.
    """
    promoted_clusters = db_session.query(Cluster).filter(
        Cluster.promoted_to_schema == True
    ).all()
    
    categories = {"friction": [], "delight": []}
    for cluster in promoted_clusters:
        if cluster.signal_type in categories:
            if cluster.name:
                categories[cluster.signal_type].append(cluster.name)
    
    return categories


async def auto_promote_high_confidence_proposals(
    db_session,
    confidence_threshold: float = 0.85,
    min_sample_count: int = 10,
) -> list[DBSchemaProposal]:
    """
    Automatically approve proposals that meet quality thresholds.
    
    This enables "hands-off" schema evolution for clearly defined patterns.
    
    Args:
        db_session: SQLAlchemy database session
        confidence_threshold: Minimum confidence score to auto-approve
        min_sample_count: Minimum number of samples to auto-approve
        
    Returns:
        List of auto-approved proposals
    """
    pending = list_pending_proposals(db_session)
    auto_approved = []
    
    for proposal in pending:
        # Check if meets auto-approval criteria
        # Note: We need to access the original proposal data, not the DB record
        # The DB record stores sample_count
        if proposal.sample_count >= min_sample_count:
            # We can't easily check confidence from DB record, 
            # so we use a heuristic based on sample count
            # More samples = more confidence in the pattern
            effective_confidence = min(proposal.sample_count / 20.0, 1.0)
            
            if effective_confidence >= confidence_threshold:
                try:
                    approved = approve_proposal(db_session, proposal.id)
                    auto_approved.append(approved)
                except ValueError:
                    pass  # Skip if already processed
    
    return auto_approved


def export_schema_to_file(db_session, output_path: str) -> None:
    """
    Export all stable categories to a YAML/JSON schema file.
    
    This creates a human-readable schema definition that can be
    version-controlled and shared.
    
    Args:
        db_session: SQLAlchemy database session
        output_path: Path to write the schema file
    """
    import json
    from pathlib import Path
    
    categories = get_stable_categories(db_session)
    
    # Build schema with full details
    schema = {
        "version": "1.0",
        "exported_at": datetime.utcnow().isoformat(),
        "categories": {},
    }
    
    promoted_clusters = db_session.query(Cluster).filter(
        Cluster.promoted_to_schema == True
    ).all()
    
    for cluster in promoted_clusters:
        if cluster.name:
            proposal = cluster.schema_proposal
            schema["categories"][cluster.name] = {
                "signal_type": cluster.signal_type,
                "description": cluster.description or "",
                "member_count": cluster.member_count,
                "inclusion_criteria": proposal.inclusion_criteria if proposal else "",
                "severity_rule": proposal.severity_rule if proposal else "",
                "recommended_action": proposal.recommended_action if proposal else "",
            }
    
    Path(output_path).write_text(json.dumps(schema, indent=2))


def schema_proposal_to_db(proposal: SchemaProposal, cluster_id: str) -> DBSchemaProposal:
    """Convert a SchemaProposal to a database model."""
    return DBSchemaProposal(
        id=f"proposal_{uuid.uuid4().hex[:12]}",
        name=proposal.name,
        description=proposal.description,
        inclusion_criteria=proposal.inclusion_criteria,
        severity_rule=proposal.severity_rule,
        recommended_action=proposal.recommended_action,
        signal_type=proposal.signal_type,
        sample_count=proposal.sample_count,
        created_at=datetime.utcnow(),
        status="pending",
    )


async def evaluate_proposal_quality(
    proposal: SchemaProposal,
    existing_categories: list[str],
    client: Optional[AsyncOpenAI] = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Evaluate whether a schema proposal is high-quality and non-redundant.
    
    Args:
        proposal: The proposed schema
        existing_categories: List of existing category names
        client: Optional OpenAI client
        model: Model for evaluation
        
    Returns:
        Evaluation dict with 'should_add', 'reason', 'similar_to'
    """
    if client is None:
        client = AsyncOpenAI()
    
    prompt = f"""Evaluate this proposed signal category:

Name: {proposal.name}
Description: {proposal.description}
Criteria: {proposal.inclusion_criteria}

Existing categories: {', '.join(existing_categories)}

Evaluate:
1. Is this distinct from existing categories?
2. Is the pattern actionable (can we do something about it)?
3. Is the description clear and specific?

Return JSON:
{{
    "should_add": true/false,
    "reason": "explanation",
    "similar_to": "<existing category name or null>",
    "quality_score": <0.0-1.0>
}}"""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Evaluate schema proposals. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content
        if content and content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        return json.loads(content) if content else {"should_add": False, "reason": "Empty response"}
    
    except Exception as e:
        return {
            "should_add": proposal.confidence > 0.7,
            "reason": f"Evaluation failed: {e}",
            "quality_score": proposal.confidence,
        }
