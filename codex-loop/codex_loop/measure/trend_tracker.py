"""
Trend tracker: Track evaluation metrics over time.

Implements trend tracking from brainstorm-updated.md for longitudinal analysis.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Any
import json

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session as DBSession

from codex_loop.db.schema import EvalRun
from codex_loop.measure.harbor_runner import TrialResult, HarborJobResult


class TrendTracker:
    """
    Track evaluation trends over time.
    
    Records eval runs and provides trend analysis.
    """
    
    def __init__(self, db: DBSession):
        """
        Initialize the trend tracker.
        
        Args:
            db: Database session for storing/querying runs
        """
        self.db = db
    
    def record_eval_run(
        self,
        job_result: HarborJobResult,
        dataset_id: Optional[str] = None,
        codex_version: Optional[str] = None,
    ) -> EvalRun:
        """
        Record an evaluation run.
        
        Args:
            job_result: Results from Harbor run
            dataset_id: Optional dataset identifier
            codex_version: Optional Codex version string
            
        Returns:
            Created EvalRun record
        """
        run = EvalRun(
            id=job_result.job_id,
            dataset_id=dataset_id or str(job_result.dataset_path),
            agent=job_result.agent,
            model=job_result.model,
            codex_version=codex_version,
            task_count=job_result.task_count,
            mean_reward=job_result.mean_reward,
            success_rate=job_result.success_rate,
            results_json=[
                {
                    "task": r.task,
                    "reward": r.reward,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                }
                for r in job_result.results
            ],
            started_at=job_result.started_at,
            completed_at=job_result.completed_at,
            job_dir=str(job_result.job_dir) if job_result.job_dir else None,
        )
        
        self.db.add(run)
        self.db.commit()
        
        return run
    
    def get_trend(
        self,
        dataset_id: str,
        lookback_days: int = 30,
        model: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get trend data for a dataset.
        
        Args:
            dataset_id: Dataset to get trends for
            lookback_days: How many days to look back
            model: Optional model filter
            
        Returns:
            DataFrame with trend data
        """
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        
        query = self.db.query(EvalRun).filter(
            EvalRun.dataset_id == dataset_id,
            EvalRun.started_at > cutoff,
        )
        
        if model:
            query = query.filter(EvalRun.model == model)
        
        runs = query.order_by(EvalRun.started_at).all()
        
        if not runs:
            return pd.DataFrame()
        
        data = []
        for run in runs:
            data.append({
                "date": run.started_at.strftime("%Y-%m-%d"),
                "datetime": run.started_at,
                "mean_reward": run.mean_reward,
                "success_rate": run.success_rate,
                "task_count": run.task_count,
                "model": run.model,
                "codex_version": run.codex_version,
            })
        
        return pd.DataFrame(data)
    
    def get_latest_run(
        self,
        dataset_id: str,
        model: Optional[str] = None,
    ) -> Optional[EvalRun]:
        """Get the most recent eval run for a dataset."""
        query = self.db.query(EvalRun).filter(
            EvalRun.dataset_id == dataset_id,
        )
        
        if model:
            query = query.filter(EvalRun.model == model)
        
        return query.order_by(EvalRun.started_at.desc()).first()
    
    def get_baseline(
        self,
        dataset_id: str,
        codex_version: str,
        model: Optional[str] = None,
    ) -> Optional[EvalRun]:
        """Get a specific baseline run."""
        query = self.db.query(EvalRun).filter(
            EvalRun.dataset_id == dataset_id,
            EvalRun.codex_version == codex_version,
        )
        
        if model:
            query = query.filter(EvalRun.model == model)
        
        return query.order_by(EvalRun.started_at.desc()).first()
    
    def compute_trend_stats(
        self,
        dataset_id: str,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """
        Compute trend statistics.
        
        Args:
            dataset_id: Dataset to analyze
            lookback_days: Lookback window
            
        Returns:
            Dict with trend statistics
        """
        df = self.get_trend(dataset_id, lookback_days)
        
        if df.empty:
            return {
                "trend": "insufficient_data",
                "data_points": 0,
            }
        
        rewards = df["mean_reward"].values
        
        # Calculate trend direction
        if len(rewards) >= 2:
            first_half = rewards[:len(rewards)//2].mean()
            second_half = rewards[len(rewards)//2:].mean()
            
            if second_half > first_half * 1.05:
                trend = "improving"
            elif second_half < first_half * 0.95:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "data_points": len(df),
            "latest_reward": float(rewards[-1]) if len(rewards) > 0 else None,
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()) if len(rewards) > 1 else 0.0,
            "min_reward": float(rewards.min()),
            "max_reward": float(rewards.max()),
            "lookback_days": lookback_days,
        }


def format_trend_chart(df: pd.DataFrame, width: int = 50) -> str:
    """
    Format trend data as ASCII chart.
    
    Args:
        df: Trend DataFrame from TrendTracker.get_trend()
        width: Chart width in characters
        
    Returns:
        ASCII chart string
    """
    if df.empty:
        return "No trend data available"
    
    rewards = df["mean_reward"].values
    dates = df["date"].values
    
    min_r = min(rewards)
    max_r = max(rewards)
    range_r = max_r - min_r or 1
    
    lines = [
        "Mean Reward Trend",
        "-" * (width + 10),
    ]
    
    for i, (date, reward) in enumerate(zip(dates, rewards)):
        # Normalize to width
        bar_len = int((reward - min_r) / range_r * width)
        bar = "█" * bar_len + "░" * (width - bar_len)
        lines.append(f"{date} |{bar}| {reward:.2f}")
    
    lines.extend([
        "-" * (width + 10),
        f"Min: {min_r:.2f}  Max: {max_r:.2f}  Points: {len(rewards)}",
    ])
    
    return "\n".join(lines)


def compare_versions(
    tracker: TrendTracker,
    dataset_id: str,
    version_a: str,
    version_b: str,
) -> dict[str, Any]:
    """
    Compare two Codex versions.
    
    Args:
        tracker: TrendTracker instance
        dataset_id: Dataset to compare on
        version_a: First version
        version_b: Second version
        
    Returns:
        Comparison results
    """
    run_a = tracker.get_baseline(dataset_id, version_a)
    run_b = tracker.get_baseline(dataset_id, version_b)
    
    if not run_a or not run_b:
        return {
            "comparison": "insufficient_data",
            "missing": "version_a" if not run_a else "version_b",
        }
    
    delta_reward = run_b.mean_reward - run_a.mean_reward
    delta_success = run_b.success_rate - run_a.success_rate
    
    return {
        "comparison": "complete",
        "version_a": {
            "version": version_a,
            "mean_reward": run_a.mean_reward,
            "success_rate": run_a.success_rate,
            "date": run_a.started_at.isoformat(),
        },
        "version_b": {
            "version": version_b,
            "mean_reward": run_b.mean_reward,
            "success_rate": run_b.success_rate,
            "date": run_b.started_at.isoformat(),
        },
        "delta": {
            "mean_reward": delta_reward,
            "success_rate": delta_success,
        },
        "improved": delta_reward > 0.05,
        "regressed": delta_reward < -0.05,
    }
