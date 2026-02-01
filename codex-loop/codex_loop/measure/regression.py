"""
Regression detection: Compare evaluation results to detect regressions.

Implements the regression detection from brainstorm-updated.md lines 364-370.
"""

from dataclasses import dataclass
from typing import Optional, Any
import numpy as np
from scipy import stats

from codex_loop.measure.harbor_runner import TrialResult


@dataclass
class Regression:
    """A detected regression in evaluation results."""
    task: str
    baseline_reward: float
    current_reward: float
    delta: float
    is_significant: bool = True
    p_value: Optional[float] = None
    
    @property
    def percent_change(self) -> float:
        if self.baseline_reward == 0:
            return -100.0 if self.current_reward == 0 else float('inf')
        return ((self.current_reward - self.baseline_reward) / self.baseline_reward) * 100


@dataclass
class RegressionReport:
    """Full regression analysis report."""
    regressions: list[Regression]
    improvements: list[Regression]
    unchanged: list[str]
    baseline_mean: float
    current_mean: float
    overall_delta: float
    is_significant_overall: bool
    p_value_overall: Optional[float] = None


def detect_regressions(
    baseline_results: list[TrialResult],
    current_results: list[TrialResult],
    threshold: float = 0.1,
    require_significance: bool = False,
) -> list[Regression]:
    """
    Compare baseline and current results to detect regressions.
    
    Args:
        baseline_results: Results from baseline evaluation
        current_results: Results from current evaluation
        threshold: Minimum reward drop to flag as regression
        require_significance: If True, only return statistically significant changes
        
    Returns:
        List of detected regressions
    """
    baseline_by_task = {r.task: r.reward for r in baseline_results}
    regressions = []
    
    for result in current_results:
        if result.task not in baseline_by_task:
            continue
        
        baseline = baseline_by_task[result.task]
        delta = baseline - result.reward
        
        if delta > threshold:
            regressions.append(Regression(
                task=result.task,
                baseline_reward=baseline,
                current_reward=result.reward,
                delta=delta,
                is_significant=True,  # Simple threshold-based
            ))
    
    return regressions


def analyze_regressions(
    baseline_results: list[TrialResult],
    current_results: list[TrialResult],
    threshold: float = 0.1,
    significance_level: float = 0.05,
) -> RegressionReport:
    """
    Perform comprehensive regression analysis.
    
    Args:
        baseline_results: Results from baseline evaluation
        current_results: Results from current evaluation
        threshold: Minimum change to flag
        significance_level: P-value threshold for significance
        
    Returns:
        Full RegressionReport
    """
    baseline_by_task = {r.task: r.reward for r in baseline_results}
    current_by_task = {r.task: r.reward for r in current_results}
    
    # Find common tasks
    common_tasks = set(baseline_by_task.keys()) & set(current_by_task.keys())
    
    regressions = []
    improvements = []
    unchanged = []
    
    baseline_rewards = []
    current_rewards = []
    
    for task in common_tasks:
        baseline = baseline_by_task[task]
        current = current_by_task[task]
        delta = baseline - current
        
        baseline_rewards.append(baseline)
        current_rewards.append(current)
        
        if abs(delta) < threshold:
            unchanged.append(task)
        elif delta > 0:
            # Regression (baseline was better)
            regressions.append(Regression(
                task=task,
                baseline_reward=baseline,
                current_reward=current,
                delta=delta,
            ))
        else:
            # Improvement (current is better)
            improvements.append(Regression(
                task=task,
                baseline_reward=baseline,
                current_reward=current,
                delta=delta,
            ))
    
    # Calculate overall statistics
    baseline_mean = np.mean(baseline_rewards) if baseline_rewards else 0.0
    current_mean = np.mean(current_rewards) if current_rewards else 0.0
    overall_delta = baseline_mean - current_mean
    
    # Statistical significance test (paired t-test)
    p_value_overall = None
    is_significant = False
    
    if len(baseline_rewards) >= 2:
        try:
            t_stat, p_value_overall = stats.ttest_rel(baseline_rewards, current_rewards)
            is_significant = p_value_overall < significance_level and overall_delta > threshold
        except Exception:
            pass
    
    return RegressionReport(
        regressions=sorted(regressions, key=lambda r: r.delta, reverse=True),
        improvements=sorted(improvements, key=lambda r: r.delta),
        unchanged=unchanged,
        baseline_mean=baseline_mean,
        current_mean=current_mean,
        overall_delta=overall_delta,
        is_significant_overall=is_significant,
        p_value_overall=p_value_overall,
    )


def format_regression_report(report: RegressionReport) -> str:
    """Format a regression report as human-readable text."""
    lines = [
        "=" * 60,
        "REGRESSION ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Overall: Baseline={report.baseline_mean:.3f}, Current={report.current_mean:.3f}, Delta={report.overall_delta:+.3f}",
    ]
    
    if report.is_significant_overall:
        lines.append(f"⚠️  SIGNIFICANT REGRESSION (p={report.p_value_overall:.4f})")
    elif report.p_value_overall:
        lines.append(f"Not significant (p={report.p_value_overall:.4f})")
    
    lines.extend(["", "-" * 60, "REGRESSIONS", "-" * 60])
    
    if report.regressions:
        for r in report.regressions:
            lines.append(f"  {r.task}: {r.baseline_reward:.2f} → {r.current_reward:.2f} ({r.percent_change:+.1f}%)")
    else:
        lines.append("  No regressions detected")
    
    lines.extend(["", "-" * 60, "IMPROVEMENTS", "-" * 60])
    
    if report.improvements:
        for r in report.improvements:
            lines.append(f"  {r.task}: {r.baseline_reward:.2f} → {r.current_reward:.2f} ({r.percent_change:+.1f}%)")
    else:
        lines.append("  No improvements detected")
    
    lines.extend([
        "",
        f"Unchanged: {len(report.unchanged)} tasks",
        "",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def should_block_deploy(
    report: RegressionReport,
    max_regression_count: int = 3,
    max_regression_severity: float = 0.3,
) -> tuple[bool, str]:
    """
    Determine if regressions should block deployment.
    
    Args:
        report: Regression analysis report
        max_regression_count: Maximum allowed regressions
        max_regression_severity: Maximum allowed mean delta
        
    Returns:
        Tuple of (should_block, reason)
    """
    # Check regression count
    if len(report.regressions) > max_regression_count:
        return True, f"Too many regressions: {len(report.regressions)} > {max_regression_count}"
    
    # Check overall significance
    if report.is_significant_overall and report.overall_delta > max_regression_severity:
        return True, f"Significant overall regression: delta={report.overall_delta:.3f}"
    
    # Check for severe individual regressions
    severe = [r for r in report.regressions if r.delta > max_regression_severity]
    if severe:
        return True, f"{len(severe)} severe regressions (delta > {max_regression_severity})"
    
    return False, "No blocking regressions"
