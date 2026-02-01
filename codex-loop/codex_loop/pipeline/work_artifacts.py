"""
Work artifact generation: Create tickets and PRs from clusters.

Implements the close loop stage from brainstorm-updated.md lines 380-386.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import json

from openai import AsyncOpenAI

from codex_loop.db.schema import Cluster, Signal


@dataclass
class TicketContent:
    """Generated ticket content."""
    title: str
    description: str
    reproduction_steps: list[str]
    suggested_fix: str
    evidence_pointers: list[dict[str, Any]]
    harbor_dataset: Optional[Path] = None
    priority: str = "medium"
    labels: list[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"# {self.title}",
            "",
            "## Description",
            "",
            self.description,
            "",
            "## Priority",
            "",
            f"**{self.priority.upper()}**",
            "",
            "## Reproduction Steps",
            "",
        ]
        
        for i, step in enumerate(self.reproduction_steps, 1):
            lines.append(f"{i}. {step}")
        
        lines.extend([
            "",
            "## Evidence",
            "",
        ])
        
        for evidence in self.evidence_pointers:
            session_id = evidence.get("session_id", "unknown")
            turn_id = evidence.get("turn_id", "N/A")
            lines.append(f"- Session: `{session_id}`, Turn: `{turn_id}`")
        
        if self.harbor_dataset:
            lines.extend([
                "",
                "## Validation Dataset",
                "",
                f"Harbor dataset: `{self.harbor_dataset}`",
                "",
                "Run validation:",
                "```bash",
                f"harbor run -p {self.harbor_dataset} -a codex -m gpt-5.2",
                "```",
            ])
        
        lines.extend([
            "",
            "## Suggested Fix",
            "",
            self.suggested_fix,
            "",
            "## Labels",
            "",
            ", ".join(f"`{label}`" for label in self.labels),
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "reproduction_steps": self.reproduction_steps,
            "suggested_fix": self.suggested_fix,
            "evidence_pointers": self.evidence_pointers,
            "harbor_dataset": str(self.harbor_dataset) if self.harbor_dataset else None,
            "priority": self.priority,
            "labels": self.labels,
        }


TICKET_GENERATION_PROMPT = """Generate a bug ticket for this friction pattern cluster.

Cluster Information:
- Name: {cluster_name}
- Description: {cluster_description}
- Signal Type: {signal_type}
- Occurrences: {member_count}
- Average Severity: {avg_severity:.2f}

Sample Evidence:
{evidence_samples}

Harbor Dataset Path: {harbor_dataset}

Generate a complete ticket with:
1. A clear, specific title
2. A description explaining the problem and its impact
3. Concrete reproduction steps
4. A suggested fix approach
5. Priority (critical/high/medium/low)
6. Relevant labels

Return JSON:
{{
    "title": "Clear bug title",
    "description": "Multi-sentence description of the problem",
    "reproduction_steps": ["Step 1", "Step 2", "Step 3"],
    "suggested_fix": "Description of how to fix this",
    "priority": "high",
    "labels": ["friction", "ux", "codex-core"]
}}"""


async def generate_ticket(
    cluster: Cluster,
    signals: list[Signal],
    harbor_dataset: Optional[Path] = None,
    client: Optional[AsyncOpenAI] = None,
    model: str = "gpt-5.2",
) -> TicketContent:
    """
    Generate a ticket from a cluster of signals.
    
    Args:
        cluster: The cluster to generate a ticket for
        signals: Signals in the cluster
        harbor_dataset: Path to Harbor dataset for validation
        client: Optional OpenAI client
        model: Model to use for generation
        
    Returns:
        Generated TicketContent
    """
    if client is None:
        client = AsyncOpenAI()
    
    # Format evidence samples
    evidence_samples = _format_evidence(signals[:5])
    
    # Calculate average severity
    avg_severity = sum(s.severity for s in signals) / len(signals) if signals else 0.5
    
    prompt = TICKET_GENERATION_PROMPT.format(
        cluster_name=cluster.name or cluster.id,
        cluster_description=cluster.description or "No description",
        signal_type=cluster.signal_type,
        member_count=cluster.member_count,
        avg_severity=avg_severity,
        evidence_samples=evidence_samples,
        harbor_dataset=harbor_dataset or "Not available",
    )
    
    try:
        response = await client.responses.create(
            model=model,
            instructions="You are a technical writer creating bug tickets. Return only valid JSON.",
            input=prompt,
        )
        
        content = response.output_text
        if not content:
            raise ValueError("Empty response")
        
        # Parse JSON
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        data = json.loads(content)
        
        # Build evidence pointers
        evidence_pointers = []
        for signal in signals:
            evidence = signal.evidence or {}
            evidence_pointers.append({
                "session_id": signal.session_id,
                "turn_id": signal.turn_id,
                "signal_id": signal.id,
                "category": signal.category,
                "severity": signal.severity,
                **evidence.get("raw", {}),
            })
        
        return TicketContent(
            title=data.get("title", f"Friction: {cluster.name or cluster.id}"),
            description=data.get("description", "Auto-generated from friction cluster"),
            reproduction_steps=data.get("reproduction_steps", ["See evidence pointers"]),
            suggested_fix=data.get("suggested_fix", "Investigation required"),
            evidence_pointers=evidence_pointers,
            harbor_dataset=harbor_dataset,
            priority=data.get("priority", "medium"),
            labels=data.get("labels", ["friction", "auto-generated"]),
        )
    
    except Exception as e:
        # Fallback ticket
        return TicketContent(
            title=f"Friction Pattern: {cluster.name or cluster.id}",
            description=f"Cluster of {cluster.member_count} similar {cluster.signal_type} signals.\n\nError generating details: {e}",
            reproduction_steps=["Review signal evidence in database"],
            suggested_fix="Manual investigation required",
            evidence_pointers=[
                {"session_id": s.session_id, "signal_id": s.id}
                for s in signals[:5]
            ],
            harbor_dataset=harbor_dataset,
            priority="medium",
            labels=["friction", "auto-generated", "needs-review"],
        )


def _format_evidence(signals: list[Signal]) -> str:
    """Format signal evidence for the prompt."""
    lines = []
    
    for i, signal in enumerate(signals, 1):
        evidence = signal.evidence or {}
        lines.append(f"{i}. Category: {signal.category}")
        lines.append(f"   Severity: {signal.severity:.2f}")
        lines.append(f"   Session: {signal.session_id}")
        
        if evidence.get("description"):
            lines.append(f"   Description: {evidence['description']}")
        
        raw = evidence.get("raw", {})
        if raw:
            for key, value in list(raw.items())[:2]:
                lines.append(f"   {key}: {value}")
        
        lines.append("")
    
    return "\n".join(lines)


async def generate_pr_description(
    ticket: TicketContent,
    fix_summary: str,
    client: Optional[AsyncOpenAI] = None,
) -> str:
    """
    Generate a PR description for a fix.
    
    Args:
        ticket: The ticket being fixed
        fix_summary: Summary of the fix
        client: Optional OpenAI client
        
    Returns:
        PR description markdown
    """
    lines = [
        f"## Summary",
        "",
        f"Fixes friction pattern: {ticket.title}",
        "",
        f"## Changes",
        "",
        fix_summary,
        "",
        f"## Testing",
        "",
        "- [ ] Ran local tests",
        "- [ ] Tested manually with reproduction steps",
    ]
    
    if ticket.harbor_dataset:
        lines.extend([
            f"- [ ] Ran Harbor validation:",
            f"  ```",
            f"  harbor run -p {ticket.harbor_dataset} -a codex -m gpt-5.2",
            f"  ```",
        ])
    
    lines.extend([
        "",
        f"## Related",
        "",
        f"Evidence: {len(ticket.evidence_pointers)} signals",
        "",
    ])
    
    return "\n".join(lines)


def write_ticket_to_file(ticket: TicketContent, output_path: Path) -> None:
    """Write a ticket to a markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ticket.to_markdown())


def write_ticket_json(ticket: TicketContent, output_path: Path) -> None:
    """Write a ticket to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ticket.to_dict(), indent=2))
