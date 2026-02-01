#!/usr/bin/env python3
"""
Display detailed statistics for the demo database.
Shows rich visualizations of the data for impressive demos.
"""

from pathlib import Path
from collections import Counter
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.layout import Layout
from rich.columns import Columns
from rich import box

from codex_loop.db import get_session, Session, Signal

console = Console()


def main():
    db_path = Path("./demo.db")
    if not db_path.exists():
        console.print("[red]No demo.db found. Run the daily pipeline first.[/red]")
        return
    
    db = get_session(db_path)
    
    # Get counts
    total_sessions = db.query(Session).count()
    processed_sessions = db.query(Session).filter(Session.signals_computed == True).count()
    
    all_signals = db.query(Signal).all()
    friction_signals = [s for s in all_signals if s.signal_type == "friction"]
    delight_signals = [s for s in all_signals if s.signal_type == "delight"]
    
    friction_categories = Counter(s.category for s in friction_signals)
    delight_categories = Counter(s.category for s in delight_signals)
    
    # Session sources
    sessions = db.query(Session).all()
    source_counts = Counter(s.source for s in sessions)
    model_counts = Counter(s.model or "unknown" for s in sessions)
    
    # Header
    console.print()
    console.print(Panel.fit(
        "[bold blue]Codex Closed Loop - Demo Statistics[/bold blue]\n\n"
        f"📊 Analyzing [bold]{total_sessions}[/bold] Codex sessions\n"
        f"🔍 Detected [bold]{len(all_signals)}[/bold] signals",
        title="Dashboard",
        border_style="blue",
    ))
    console.print()
    
    # Create summary table
    summary = Table(title="📈 Overview", box=box.ROUNDED)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green", justify="right")
    summary.add_column("Details", style="dim")
    
    summary.add_row("Total Sessions", str(total_sessions), f"{processed_sessions} processed")
    summary.add_row("Friction Signals", str(len(friction_signals)), f"{len(friction_categories)} categories")
    summary.add_row("Delight Signals", str(len(delight_signals)), f"{len(delight_categories)} categories")
    
    # Calculate friction rate
    friction_rate = len(friction_signals) / total_sessions if total_sessions else 0
    delight_rate = len(delight_signals) / total_sessions if total_sessions else 0
    summary.add_row("Friction Rate", f"{friction_rate:.1f}/session", "signals per session")
    summary.add_row("Delight Rate", f"{delight_rate:.1f}/session", "signals per session")
    
    console.print(summary)
    console.print()
    
    # Friction breakdown with bar chart
    console.print("[bold red]🔴 Friction Signals[/bold red]")
    console.print()
    
    max_count = max(friction_categories.values()) if friction_categories else 1
    for category, count in friction_categories.most_common(10):
        bar_width = int((count / max_count) * 30)
        bar = "█" * bar_width + "░" * (30 - bar_width)
        console.print(f"  {category:25} {bar} {count:4}")
    
    console.print()
    
    # Delight breakdown with bar chart
    console.print("[bold green]🟢 Delight Signals[/bold green]")
    console.print()
    
    max_count = max(delight_categories.values()) if delight_categories else 1
    for category, count in delight_categories.most_common(10):
        bar_width = int((count / max_count) * 30)
        bar = "█" * bar_width + "░" * (30 - bar_width)
        console.print(f"  {category:25} {bar} {count:4}")
    
    console.print()
    
    # Session sources
    console.print("[bold]📱 Session Sources[/bold]")
    source_table = Table(box=box.SIMPLE)
    source_table.add_column("Source")
    source_table.add_column("Count", justify="right")
    source_table.add_column("Percentage", justify="right")
    
    for source, count in source_counts.most_common():
        pct = (count / total_sessions) * 100 if total_sessions else 0
        source_table.add_row(source, str(count), f"{pct:.1f}%")
    
    console.print(source_table)
    console.print()
    
    # Top friction patterns to investigate
    console.print("[bold yellow]⚠️  Top Issues to Investigate[/bold yellow]")
    console.print()
    
    high_severity = [s for s in friction_signals if s.severity >= 0.7]
    high_severity_categories = Counter(s.category for s in high_severity)
    
    for i, (category, count) in enumerate(high_severity_categories.most_common(5), 1):
        severity_avg = sum(s.severity for s in friction_signals if s.category == category) / count
        console.print(f"  {i}. [bold]{category}[/bold]")
        console.print(f"     {count} occurrences, avg severity: {severity_avg:.2f}")
        console.print()
    
    # Summary quote
    console.print(Panel(
        "[italic]Every session becomes data. Every friction becomes an eval.\n"
        "Every fix gets measured. That's the closed loop.[/italic]",
        title="The Vision",
        border_style="cyan",
    ))


if __name__ == "__main__":
    main()
