"""
Harbor task generator: Create Harbor tasks from trace segments.

Generates the Harbor task structure:
- instruction.md: User prompt
- task.toml: Metadata
- environment/Dockerfile: Environment setup
- tests/test.sh: Verifier script
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import json
import os

from codex_loop.db.schema import Session, Signal, Cluster
from codex_loop.ingest.spine_reader import (
    TraceRecord,
    parse_spine,
    get_session_meta,
    get_events_by_type,
)
from codex_loop.distill.segment_selector import (
    select_segment,
    select_segments_for_cluster,
    SegmentWindow,
)
from sqlalchemy.orm import Session as DBSession


@dataclass
class HarborTask:
    """A generated Harbor task."""
    task_id: str
    instruction: str
    dockerfile: str
    verifier_script: str
    metadata: dict[str, Any]
    solution_hint: Optional[str] = None


async def distill_signal_to_task(
    signal: Signal,
    output_dir: Path,
    codex_home: Optional[Path] = None,
) -> HarborTask:
    """
    Distill a signal into a Harbor task.
    
    Args:
        signal: The signal to distill
        output_dir: Directory to write task files
        codex_home: Path to Codex home for artifact lookup
        
    Returns:
        HarborTask object
    """
    # Load trace records
    records = None
    if signal.session and signal.session.spine_path:
        spine_path = Path(signal.session.spine_path)
        if spine_path.exists():
            records = parse_spine(spine_path)
    
    # Select segment
    segment = select_segment(signal, records)
    
    # Generate task
    task = _generate_task_from_segment(signal, segment, records)
    
    # Write to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_task_to_disk(task, output_dir)
    
    return task


async def distill_cluster_to_dataset(
    cluster: Cluster,
    db: DBSession,
    codex_home: Path,
    output_path: Path,
    max_tasks: int = 10,
) -> list[HarborTask]:
    """
    Distill a cluster of signals into a Harbor dataset.
    
    Args:
        cluster: The cluster to distill
        db: Database session
        codex_home: Path to Codex home
        output_path: Directory for the dataset
        max_tasks: Maximum tasks to generate
        
    Returns:
        List of generated HarborTask objects
    """
    # Get signals for this cluster
    signals = db.query(Signal).filter(Signal.cluster_id == cluster.id).all()
    
    if not signals:
        return []
    
    # Select diverse segments
    segments = select_segments_for_cluster(signals, max_segments=max_tasks)
    
    tasks = []
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, segment in enumerate(segments):
        task_dir = output_path / f"task_{i:03d}"
        
        if segment.signal:
            task = _generate_task_from_segment(segment.signal, segment, segment.records)
            _write_task_to_disk(task, task_dir)
            tasks.append(task)
    
    # Write dataset metadata
    dataset_meta = {
        "cluster_id": cluster.id,
        "cluster_name": cluster.name,
        "signal_type": cluster.signal_type,
        "task_count": len(tasks),
        "generated_at": datetime.utcnow().isoformat(),
    }
    (output_path / "dataset.json").write_text(json.dumps(dataset_meta, indent=2))
    
    return tasks


def _generate_task_from_segment(
    signal: Signal,
    segment: Optional[SegmentWindow],
    records: Optional[list[TraceRecord]],
) -> HarborTask:
    """Generate a HarborTask from a signal and segment."""
    
    task_id = f"task_{signal.id[:12]}"
    
    # Generate instruction
    instruction = _generate_instruction(signal, segment)
    
    # Generate Dockerfile
    dockerfile = _generate_dockerfile(signal, records)
    
    # Generate verifier
    verifier_script = _generate_verifier(signal, segment)
    
    # Build metadata
    metadata = {
        "id": task_id,
        "source": "codex_closed_loop",
        "signal_id": signal.id,
        "signal_type": signal.signal_type,
        "signal_category": signal.category,
        "severity": signal.severity,
        "session_id": signal.session_id,
        "generated_at": datetime.utcnow().isoformat(),
    }
    
    # Add provenance
    evidence = signal.evidence or {}
    if evidence.get("turn_id"):
        metadata["turn_id"] = evidence["turn_id"]
    if evidence.get("seq_range"):
        metadata["seq_range"] = evidence["seq_range"]
    
    return HarborTask(
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        verifier_script=verifier_script,
        metadata=metadata,
    )


def _generate_instruction(
    signal: Signal,
    segment: Optional[SegmentWindow],
) -> str:
    """Generate the instruction.md content."""
    
    # Use user message from segment if available
    if segment and segment.user_message:
        user_request = segment.user_message
    else:
        # Generate from signal context
        user_request = _infer_user_request(signal)
    
    # Build instruction
    lines = [
        f"# Task: {signal.category.replace('_', ' ').title()}",
        "",
        "## User Request",
        "",
        user_request,
        "",
    ]
    
    # Add expected outcome
    if segment and segment.expected_outcome:
        lines.extend([
            "## Expected Outcome",
            "",
            segment.expected_outcome,
            "",
        ])
    
    # Add context if friction signal
    if signal.signal_type == "friction":
        lines.extend([
            "## Note",
            "",
            f"This task is derived from a {signal.category} friction signal.",
            "The goal is to handle this scenario without triggering the friction pattern.",
            "",
        ])
    
    return "\n".join(lines)


def _generate_dockerfile(
    signal: Signal,
    records: Optional[list[TraceRecord]],
) -> str:
    """Generate the Dockerfile content."""
    
    # Get git info from session meta
    git_repo = None
    git_commit = None
    cwd = "/workspace"
    
    if records:
        meta = get_session_meta(records)
        if meta:
            git_info = meta.payload.get("git", {})
            git_repo = git_info.get("repository_url")
            git_commit = git_info.get("commit_hash")
            session_meta = meta.payload.get("meta", {})
            cwd = session_meta.get("cwd", "/workspace")
    
    lines = [
        "FROM python:3.11-slim",
        "",
        "# Install system dependencies",
        "RUN apt-get update && apt-get install -y \\",
        "    git \\",
        "    curl \\",
        "    && rm -rf /var/lib/apt/lists/*",
        "",
        "WORKDIR /workspace",
        "",
    ]
    
    if git_repo:
        lines.extend([
            f"# Clone repository",
            f"RUN git clone {git_repo} . || true",
            "",
        ])
        if git_commit:
            lines.extend([
                f"# Checkout specific commit",
                f"RUN git checkout {git_commit} || true",
                "",
            ])
    
    lines.extend([
        "# Install Python dependencies if requirements.txt exists",
        "RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi",
        "",
        "# Create logs directory for verifier",
        "RUN mkdir -p /logs/verifier",
        "",
        "CMD [\"bash\"]",
    ])
    
    return "\n".join(lines)


def _generate_verifier(
    signal: Signal,
    segment: Optional[SegmentWindow],
) -> str:
    """Generate the test.sh verifier script."""
    
    lines = [
        "#!/bin/bash",
        "set -e",
        "",
        "# Harbor verifier script",
        "# Writes reward (0.0-1.0) to /logs/verifier/reward.txt",
        "",
        "REWARD=0.0",
        "",
    ]
    
    # Generate verification based on signal type
    if signal.signal_type == "friction":
        lines.extend(_generate_friction_verifier(signal))
    else:
        lines.extend(_generate_delight_verifier(signal))
    
    lines.extend([
        "",
        "# Write final reward",
        "echo $REWARD > /logs/verifier/reward.txt",
        "echo \"Verification complete. Reward: $REWARD\"",
    ])
    
    return "\n".join(lines)


def _generate_friction_verifier(signal: Signal) -> list[str]:
    """Generate verifier checks for friction signals."""
    
    category = signal.category.lower()
    lines = []
    
    if "exec" in category or "failure" in category:
        lines.extend([
            "# Check for successful command execution",
            "if [ -f /logs/agent/commands.log ]; then",
            "    if grep -q 'exit_code=0' /logs/agent/commands.log; then",
            "        REWARD=$(echo \"$REWARD + 0.5\" | bc)",
            "    fi",
            "fi",
            "",
        ])
    
    if "denial" in category:
        lines.extend([
            "# Check that no commands were denied",
            "if [ -f /logs/agent/approvals.log ]; then",
            "    DENIAL_COUNT=$(grep -c 'denied' /logs/agent/approvals.log || echo 0)",
            "    if [ \"$DENIAL_COUNT\" -eq 0 ]; then",
            "        REWARD=$(echo \"$REWARD + 0.5\" | bc)",
            "    fi",
            "fi",
            "",
        ])
    
    if "error" in category:
        lines.extend([
            "# Check for absence of errors",
            "if [ -f /logs/agent/errors.log ]; then",
            "    ERROR_COUNT=$(wc -l < /logs/agent/errors.log)",
            "    if [ \"$ERROR_COUNT\" -eq 0 ]; then",
            "        REWARD=$(echo \"$REWARD + 0.5\" | bc)",
            "    fi",
            "else",
            "    REWARD=$(echo \"$REWARD + 0.5\" | bc)",
            "fi",
            "",
        ])
    
    # Default: check task completion
    lines.extend([
        "# Check for task completion indicator",
        "if [ -f /workspace/.task_complete ]; then",
        "    REWARD=1.0",
        "fi",
    ])
    
    return lines


def _generate_delight_verifier(signal: Signal) -> list[str]:
    """Generate verifier checks for delight signals."""
    
    return [
        "# Check for successful task completion",
        "if [ -f /logs/agent/trajectory.json ]; then",
        "    # Task was executed",
        "    REWARD=0.5",
        "    ",
        "    # Check for efficiency (few turns)",
        "    TURN_COUNT=$(grep -c 'user_turn' /logs/agent/trajectory.json || echo 10)",
        "    if [ \"$TURN_COUNT\" -le 3 ]; then",
        "        REWARD=$(echo \"$REWARD + 0.3\" | bc)",
        "    fi",
        "    ",
        "    # Check for clean execution (no errors)",
        "    if ! grep -q 'error' /logs/agent/trajectory.json; then",
        "        REWARD=$(echo \"$REWARD + 0.2\" | bc)",
        "    fi",
        "fi",
    ]


def _infer_user_request(signal: Signal) -> str:
    """Infer a user request from signal evidence."""
    
    evidence = signal.evidence or {}
    description = evidence.get("description", "")
    
    if description:
        return f"Complete the following task: {description}"
    
    # Generic fallback based on category
    category = signal.category
    return f"Complete a coding task that previously resulted in a {category.replace('_', ' ')} signal."


def _write_task_to_disk(task: HarborTask, output_dir: Path) -> None:
    """Write a HarborTask to disk in Harbor format."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write instruction.md
    (output_dir / "instruction.md").write_text(task.instruction)
    
    # Write task.toml
    import toml
    (output_dir / "task.toml").write_text(toml.dumps(task.metadata))
    
    # Write environment/Dockerfile
    env_dir = output_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(task.dockerfile)
    
    # Write tests/test.sh
    tests_dir = output_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_script = tests_dir / "test.sh"
    test_script.write_text(task.verifier_script)
    test_script.chmod(0o755)
    
    # Write solution hint if available
    if task.solution_hint:
        solution_dir = output_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(task.solution_hint)
