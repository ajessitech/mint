"""
Harbor runner: Execute Harbor evaluations and parse results.

Implements the measurement stage from brainstorm-updated.md lines 364-370.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import asyncio
import json
import os
import shutil


@dataclass
class TrialResult:
    """Result of a single Harbor trial."""
    task: str
    reward: float
    trajectory: Optional[Path] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarborJobResult:
    """Result of a Harbor job (multiple trials)."""
    job_id: str
    dataset_path: Path
    agent: str
    model: str
    results: list[TrialResult]
    started_at: datetime
    completed_at: Optional[datetime] = None
    job_dir: Optional[Path] = None
    
    @property
    def mean_reward(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reward for r in self.results) / len(self.results)
    
    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.reward > 0) / len(self.results)
    
    @property
    def task_count(self) -> int:
        return len(self.results)


async def run_harbor_evaluation(
    dataset_path: Path,
    agent: str = "codex",
    model: str = "gpt-5.2",
    parallel: int = 4,
    timeout: int = 600,
    job_dir: Optional[Path] = None,
) -> HarborJobResult:
    """
    Run Harbor evaluation on a dataset.
    
    Args:
        dataset_path: Path to Harbor dataset directory
        agent: Agent to evaluate (e.g., "codex")
        model: Model to use (e.g., "gpt-5.2")
        parallel: Number of parallel workers
        timeout: Timeout per task in seconds
        job_dir: Optional custom job directory
        
    Returns:
        HarborJobResult with all trial results
    """
    # Verify harbor is installed
    harbor_path = shutil.which("harbor")
    if not harbor_path:
        raise FileNotFoundError(
            "Harbor not found. Install with: uv tool install harbor"
        )
    
    started_at = datetime.utcnow()
    job_id = f"job_{started_at.strftime('%Y%m%d_%H%M%S')}"
    
    # Build command
    cmd = [
        "harbor", "run",
        "-p", str(dataset_path),
        "-a", agent,
        "-m", model,
        "-n", str(parallel),
    ]
    
    if job_dir:
        cmd.extend(["--job-dir", str(job_dir)])
    
    # Run Harbor
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout * 10,  # Overall job timeout
        )
    except asyncio.TimeoutError:
        process.kill()
        raise TimeoutError(f"Harbor job timed out after {timeout * 10}s")
    
    if process.returncode != 0:
        error = stderr.decode() if stderr else "Unknown error"
        raise RuntimeError(f"Harbor failed: {error}")
    
    # Find job directory from output
    output = stdout.decode()
    actual_job_dir = _parse_job_dir(output, job_dir)
    
    # Parse results
    results = parse_harbor_results(actual_job_dir) if actual_job_dir else []
    
    return HarborJobResult(
        job_id=job_id,
        dataset_path=dataset_path,
        agent=agent,
        model=model,
        results=results,
        started_at=started_at,
        completed_at=datetime.utcnow(),
        job_dir=actual_job_dir,
    )


def parse_harbor_results(job_dir: Path) -> list[TrialResult]:
    """
    Parse Harbor results from a job directory.
    
    Looks for reward.txt files in each trial directory.
    
    Args:
        job_dir: Path to Harbor job directory
        
    Returns:
        List of TrialResult objects
    """
    results = []
    
    if not job_dir.exists():
        return results
    
    # Harbor stores results in subdirectories per task
    for trial_dir in job_dir.iterdir():
        if not trial_dir.is_dir():
            continue
        
        task_name = trial_dir.name
        
        # Look for reward file
        reward_file = trial_dir / "verifier" / "reward.txt"
        if not reward_file.exists():
            # Try alternate location
            reward_file = trial_dir / "logs" / "verifier" / "reward.txt"
        
        reward = 0.0
        if reward_file.exists():
            try:
                reward = float(reward_file.read_text().strip())
            except (ValueError, IOError):
                pass
        
        # Look for trajectory
        trajectory = trial_dir / "agent" / "trajectory.json"
        if not trajectory.exists():
            trajectory = None
        
        # Look for metadata/timing
        metadata = {}
        meta_file = trial_dir / "meta.json"
        if meta_file.exists():
            try:
                metadata = json.loads(meta_file.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        
        results.append(TrialResult(
            task=task_name,
            reward=reward,
            trajectory=trajectory,
            duration_ms=metadata.get("duration_ms"),
            error=metadata.get("error"),
            metadata=metadata,
        ))
    
    return results


def _parse_job_dir(output: str, provided_dir: Optional[Path]) -> Optional[Path]:
    """Parse job directory from Harbor output."""
    if provided_dir and provided_dir.exists():
        return provided_dir
    
    # Look for "Results written to:" in output
    for line in output.split("\n"):
        if "Results" in line and "written to" in line.lower():
            # Extract path
            parts = line.split(":")
            if len(parts) >= 2:
                path_str = parts[-1].strip()
                path = Path(path_str)
                if path.exists():
                    return path
    
    # Try default location
    default_jobs = Path("jobs")
    if default_jobs.exists():
        # Get most recent job
        job_dirs = sorted(default_jobs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if job_dirs:
            return job_dirs[0]
    
    return None


async def run_single_task(
    task_path: Path,
    agent: str = "codex",
    model: str = "gpt-5.2",
    timeout: int = 300,
) -> TrialResult:
    """
    Run Harbor on a single task.
    
    Args:
        task_path: Path to task directory
        agent: Agent to evaluate
        model: Model to use
        timeout: Timeout in seconds
        
    Returns:
        TrialResult for this task
    """
    # Create temporary dataset with just this task
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        task_name = task_path.name
        
        # Copy task to temp dataset
        shutil.copytree(task_path, tmp_path / task_name)
        
        # Run Harbor
        result = await run_harbor_evaluation(
            dataset_path=tmp_path,
            agent=agent,
            model=model,
            parallel=1,
            timeout=timeout,
        )
        
        if result.results:
            return result.results[0]
        else:
            return TrialResult(
                task=task_name,
                reward=0.0,
                error="No results returned",
            )


def summarize_results(results: list[TrialResult]) -> dict[str, Any]:
    """
    Generate summary statistics for trial results.
    
    Args:
        results: List of TrialResult objects
        
    Returns:
        Summary dict with mean, success rate, etc.
    """
    if not results:
        return {
            "count": 0,
            "mean_reward": 0.0,
            "success_rate": 0.0,
            "failed_tasks": [],
        }
    
    rewards = [r.reward for r in results]
    successes = [r for r in results if r.reward > 0]
    failures = [r for r in results if r.reward == 0]
    
    return {
        "count": len(results),
        "mean_reward": sum(rewards) / len(rewards),
        "success_rate": len(successes) / len(results),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "successful_tasks": [r.task for r in successes],
        "failed_tasks": [r.task for r in failures],
    }
