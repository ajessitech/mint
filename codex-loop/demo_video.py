#!/usr/bin/env python3
"""
Codex Closed Loop - Demo Video Script

A single script that walks through the entire demo.
Just run: python demo_video.py
Press ENTER to advance through each stage.

Perfect for recording a 2-minute demo video.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from collections import Counter

# Rich for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box
from rich.text import Text
from rich.align import Align
from rich.live import Live

console = Console()

# Colors
BLUE = "blue"
GREEN = "green"
RED = "red"
YELLOW = "yellow"
CYAN = "cyan"
MAGENTA = "magenta"

PROJECT_DIR = Path(__file__).parent
DEMO_CODEX_HOME = PROJECT_DIR / "demo_codex_home"
DEMO_DB = PROJECT_DIR / "demo.db"


def clear_screen():
    """Clear terminal screen."""
    console.clear()


def wait_for_enter(message: str = "Press ENTER to continue..."):
    """Wait for user to press enter."""
    console.print()
    console.print(f"[dim]{message}[/dim]")
    input()


def show_title():
    """Show the opening title screen."""
    clear_screen()
    
    title = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║      ██████╗ ██████╗ ██████╗ ███████╗██╗  ██╗                                 ║
║     ██╔════╝██╔═══██╗██╔══██╗██╔════╝╚██╗██╔╝                                 ║
║     ██║     ██║   ██║██║  ██║█████╗   ╚███╔╝                                  ║
║     ██║     ██║   ██║██║  ██║██╔══╝   ██╔██╗                                  ║
║     ╚██████╗╚██████╔╝██████╔╝███████╗██╔╝ ██╗                                 ║
║      ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝                                 ║
║                                                                               ║
║               ██████╗██╗      ██████╗ ███████╗███████╗██████╗                 ║
║              ██╔════╝██║     ██╔═══██╗██╔════╝██╔════╝██╔══██╗                ║
║              ██║     ██║     ██║   ██║███████╗█████╗  ██║  ██║                ║
║              ██║     ██║     ██║   ██║╚════██║██╔══╝  ██║  ██║                ║
║              ╚██████╗███████╗╚██████╔╝███████║███████╗██████╔╝                ║
║               ╚═════╝╚══════╝ ╚═════╝ ╚══════╝╚══════╝╚═════╝                 ║
║                                                                               ║
║                          ██╗      ██████╗  ██████╗ ██████╗                    ║
║                          ██║     ██╔═══██╗██╔═══██╗██╔══██╗                   ║
║                          ██║     ██║   ██║██║   ██║██████╔╝                   ║
║                          ██║     ██║   ██║██║   ██║██╔═══╝                    ║
║                          ███████╗╚██████╔╝╚██████╔╝██║                        ║
║                          ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝                        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    console.print(title, style="bold cyan")
    console.print()
    console.print(Align.center("[bold white]A Self-Improving AI Coding Agent[/bold white]"))
    console.print()
    console.print(Align.center("[dim]OpenAI Hackathon 2026 - Agentic Software Engineering Track[/dim]"))
    
    wait_for_enter()


def show_problem():
    """Stage 1: Show the problem we're solving."""
    clear_screen()
    
    console.print(Panel(
        "[bold]THE PROBLEM[/bold]",
        style="red",
        padding=(0, 2),
    ))
    console.print()
    
    problems = [
        ("❌", "AI coding agents fail in predictable ways", "Repeated errors, confused by platform, ignored instructions"),
        ("❌", "No systematic learning from failures", "Same mistakes happen over and over"),
        ("❌", "No way to measure improvements", "How do you know if a fix actually helped?"),
        ("❌", "Feedback is lost", "User frustration disappears into the void"),
    ]
    
    for emoji, title, desc in problems:
        time.sleep(0.5)
        console.print(f"  {emoji}  [bold]{title}[/bold]")
        console.print(f"      [dim]{desc}[/dim]")
        console.print()
    
    wait_for_enter()


def show_solution():
    """Stage 2: Show the solution."""
    clear_screen()
    
    console.print(Panel(
        "[bold]THE SOLUTION: CLOSED LOOP[/bold]",
        style="green",
        padding=(0, 2),
    ))
    console.print()
    
    # ASCII diagram
    diagram = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │     ┌──────────────┐                                            │
    │     │    CODEX     │  Every session is captured                 │
    │     │   SESSION    │                                            │
    │     └──────┬───────┘                                            │
    │            │                                                    │
    │            ▼                                                    │
    │     ┌──────────────┐                                            │
    │     │ TRACE SPINE  │  Structured logs of all actions            │
    │     └──────┬───────┘                                            │
    │            │                                                    │
    │            ▼                                                    │
    │     ┌──────────────┐                                            │
    │     │   ANALYSIS   │  Detect friction & delight signals         │
    │     │   PIPELINE   │  using rules + LLM                         │
    │     └──────┬───────┘                                            │
    │            │                                                    │
    │            ▼                                                    │
    │     ┌──────────────┐                                            │
    │     │    HARBOR    │  Reproducible evaluation tasks             │
    │     │    TASKS     │                                            │
    │     └──────┬───────┘                                            │
    │            │                                                    │
    │            ▼                                                    │
    │     ┌──────────────┐                                            │
    │     │   MEASURED   │  Track improvements over time              │
    │     │     FIX      │◄─────────────────────────────────────┐     │
    │     └──────────────┘                                      │     │
    │                                                           │     │
    │     ════════════════════════════════════════════════════  │     │
    │                    THE LOOP CLOSES                        │     │
    │     ════════════════════════════════════════════════════──┘     │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
"""
    console.print(diagram, style="cyan")
    
    wait_for_enter()


def show_data_collection():
    """Stage 3: Show the data we've collected."""
    clear_screen()
    
    console.print(Panel(
        "[bold]STAGE 1: DATA COLLECTION[/bold]",
        style="blue",
        padding=(0, 2),
    ))
    console.print()
    
    # Check if data exists
    if not DEMO_CODEX_HOME.exists():
        console.print("[yellow]Generating synthetic Codex sessions...[/yellow]")
        console.print()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Generating 500 sessions...", total=100)
            
            # Run the generator
            proc = subprocess.Popen(
                [sys.executable, "scripts/generate_demo_data.py", 
                 "--output", str(DEMO_CODEX_HOME / "trace_spine"),
                 "--count", "500"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=PROJECT_DIR,
            )
            
            while proc.poll() is None:
                progress.update(task, advance=2)
                time.sleep(0.1)
            
            progress.update(task, completed=100)
        
        console.print()
    
    # Count sessions
    trace_dir = DEMO_CODEX_HOME / "trace_spine"
    session_count = len([d for d in trace_dir.iterdir() if d.is_dir()]) if trace_dir.exists() else 0
    
    console.print(f"  📁 [bold]Trace Spine Directory[/bold]: {trace_dir}")
    console.print()
    
    # Animated counter
    console.print("  📊 Sessions captured: ", end="")
    for i in range(0, session_count + 1, max(1, session_count // 20)):
        console.print(f"\r  📊 Sessions captured: [bold green]{i}[/bold green]", end="")
        time.sleep(0.05)
    console.print(f"\r  📊 Sessions captured: [bold green]{session_count}[/bold green]")
    
    console.print()
    console.print("  [dim]Each session contains:[/dim]")
    console.print("    • User requests and instructions")
    console.print("    • Tool calls (shell commands, file edits)")
    console.print("    • Approval decisions")
    console.print("    • Errors and warnings")
    console.print("    • Timing information")
    
    wait_for_enter()


def show_signal_detection():
    """Stage 4: Show signal detection."""
    clear_screen()
    
    console.print(Panel(
        "[bold]STAGE 2: SIGNAL DETECTION[/bold]",
        style="magenta",
        padding=(0, 2),
    ))
    console.print()
    
    # Run pipeline if needed
    if not DEMO_DB.exists():
        console.print("[yellow]Running analysis pipeline...[/yellow]")
        console.print()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("Analyzing sessions...", total=None)
            
            subprocess.run(
                [sys.executable, "-m", "codex_loop.cli", "daily",
                 "--codex-home", str(DEMO_CODEX_HOME),
                 "--db", str(DEMO_DB),
                 "--lookback-hours", "1000",
                 "--skip-semantic", "--skip-clustering"],
                capture_output=True,
                cwd=PROJECT_DIR,
            )
        
        console.print()
    
    # Load data from database
    from codex_loop.db import get_session, Session, Signal
    
    db = get_session(DEMO_DB)
    total_sessions = db.query(Session).count()
    all_signals = db.query(Signal).all()
    friction_signals = [s for s in all_signals if s.signal_type == "friction"]
    delight_signals = [s for s in all_signals if s.signal_type == "delight"]
    
    friction_categories = Counter(s.category for s in friction_signals)
    delight_categories = Counter(s.category for s in delight_signals)
    
    # Summary panel
    summary_text = f"""
[bold]Sessions Analyzed:[/bold] {total_sessions}
[bold]Signals Detected:[/bold] {len(all_signals)}

[red]Friction Signals:[/red] {len(friction_signals)}
[green]Delight Signals:[/green] {len(delight_signals)}
"""
    console.print(Panel(summary_text.strip(), title="Analysis Results", border_style="cyan"))
    console.print()
    
    # Friction bar chart
    console.print("[bold red]🔴 FRICTION SIGNALS[/bold red] [dim](patterns that hurt UX)[/dim]")
    console.print()
    
    max_count = max(friction_categories.values()) if friction_categories else 1
    for category, count in friction_categories.most_common():
        bar_width = int((count / max_count) * 35)
        bar = "█" * bar_width
        console.print(f"  {category:22} [red]{bar}[/red] {count}")
        time.sleep(0.1)
    
    console.print()
    
    # Delight bar chart
    console.print("[bold green]🟢 DELIGHT SIGNALS[/bold green] [dim](patterns that work well)[/dim]")
    console.print()
    
    max_count = max(delight_categories.values()) if delight_categories else 1
    for category, count in delight_categories.most_common():
        bar_width = int((count / max_count) * 35)
        bar = "█" * bar_width
        console.print(f"  {category:22} [green]{bar}[/green] {count}")
        time.sleep(0.1)
    
    wait_for_enter()


def show_top_issues():
    """Stage 5: Show top issues."""
    clear_screen()
    
    console.print(Panel(
        "[bold]STAGE 3: PRIORITIZED ISSUES[/bold]",
        style="yellow",
        padding=(0, 2),
    ))
    console.print()
    
    from codex_loop.db import get_session, Signal
    
    db = get_session(DEMO_DB)
    friction_signals = db.query(Signal).filter(Signal.signal_type == "friction").all()
    
    # Group by category with severity
    category_stats = {}
    for s in friction_signals:
        if s.category not in category_stats:
            category_stats[s.category] = {"count": 0, "total_severity": 0}
        category_stats[s.category]["count"] += 1
        category_stats[s.category]["total_severity"] += s.severity
    
    # Sort by impact (count * avg_severity)
    sorted_categories = sorted(
        category_stats.items(),
        key=lambda x: x[1]["count"] * (x[1]["total_severity"] / x[1]["count"]),
        reverse=True
    )
    
    console.print("  [bold]Top issues to fix, ranked by impact:[/bold]")
    console.print()
    
    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Issue Category", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Avg Severity", justify="right")
    table.add_column("Impact Score", justify="right", style="red")
    
    for i, (category, stats) in enumerate(sorted_categories[:7], 1):
        count = stats["count"]
        avg_sev = stats["total_severity"] / count
        impact = count * avg_sev
        
        table.add_row(
            str(i),
            category.replace("_", " ").title(),
            str(count),
            f"{avg_sev:.2f}",
            f"{impact:.1f}"
        )
        time.sleep(0.2)
    
    console.print(table)
    
    console.print()
    console.print("  [dim]Impact Score = Count × Average Severity[/dim]")
    console.print("  [dim]Fix high-impact issues first for maximum improvement[/dim]")
    
    wait_for_enter()


def show_harbor_task():
    """Stage 6: Show Harbor task generation."""
    clear_screen()
    
    console.print(Panel(
        "[bold]STAGE 4: GENERATE EVALUATION TASK[/bold]",
        style="cyan",
        padding=(0, 2),
    ))
    console.print()
    
    from codex_loop.db import get_session, Signal
    
    db = get_session(DEMO_DB)
    
    # Get a high-severity signal
    signal = db.query(Signal).filter(
        Signal.signal_type == "friction",
        Signal.category == "exec_failure"
    ).first()
    
    if not signal:
        console.print("[red]No signals found[/red]")
        return
    
    console.print(f"  [bold]Converting friction signal to Harbor task...[/bold]")
    console.print()
    console.print(f"  Signal ID: [cyan]{signal.id}[/cyan]")
    console.print(f"  Category: [yellow]{signal.category}[/yellow]")
    console.print(f"  Severity: [red]{signal.severity:.2f}[/red]")
    console.print()
    
    # Generate task
    task_dir = PROJECT_DIR / "demo_harbor_task"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task("Generating Harbor task...", total=None)
        
        subprocess.run(
            [sys.executable, "-m", "codex_loop.cli", "distill",
             "--db", str(DEMO_DB),
             "--signal-id", signal.id,
             "--output", str(task_dir)],
            capture_output=True,
            cwd=PROJECT_DIR,
        )
    
    console.print()
    console.print("  [green]✓ Harbor task generated![/green]")
    console.print()
    
    # Show task structure
    console.print("  [bold]📦 Task Structure:[/bold]")
    console.print()
    
    tree = """    demo_harbor_task/
    ├── instruction.md   ← What the agent should do
    ├── task.toml        ← Metadata and config
    ├── environment/     ← Container setup
    │   └── Dockerfile
    └── tests/           ← Verification scripts
        └── verify.sh"""
    console.print(tree, style="dim")
    
    console.print()
    
    # Show instruction content
    instruction_file = task_dir / "instruction.md"
    if instruction_file.exists():
        content = instruction_file.read_text()
        console.print(Panel(content, title="instruction.md", border_style="green"))
    
    wait_for_enter()


def show_closing():
    """Final slide: The value proposition."""
    clear_screen()
    
    console.print()
    console.print()
    
    closing = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                        [bold cyan]THE CLOSED LOOP IN ACTION[/bold cyan]                           ║
║                                                                               ║
║   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐ ║
║   │   CAPTURE   │────▶│   DETECT    │────▶│   DISTILL   │────▶│   MEASURE   │ ║
║   │  Sessions   │     │  Signals    │     │   Tasks     │     │    Fixes    │ ║
║   └─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘ ║
║         ▲                                                           │        ║
║         │                                                           │        ║
║         └───────────────────────────────────────────────────────────┘        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    console.print(closing)
    
    console.print()
    
    value_props = [
        ("Every session becomes data", "Nothing is wasted"),
        ("Every friction becomes an eval", "Problems become tests"),
        ("Every fix gets measured", "Improvements are proven"),
    ]
    
    for title, subtitle in value_props:
        console.print(f"    ✨ [bold]{title}[/bold]")
        console.print(f"       [dim]{subtitle}[/dim]")
        console.print()
        time.sleep(0.5)
    
    console.print()
    console.print(Align.center("[bold green]Codex learns from itself.[/bold green]"))
    console.print(Align.center("[bold green]That's the closed loop.[/bold green]"))
    console.print()
    
    # Tech stack
    console.print(Panel(
        "[bold]Built with:[/bold]\n"
        "• OpenAI GPT-4o for semantic signal detection\n"
        "• OpenAI text-embedding-3-small for clustering\n"
        "• Codex trace spine format for data capture\n"
        "• Harbor framework for reproducible evals\n"
        "• SQLite + ChromaDB for local analysis",
        title="Tech Stack",
        border_style="dim",
    ))
    
    console.print()
    console.print(Align.center("[dim]github.com/your-repo/codex-loop[/dim]"))
    console.print()


def main():
    """Run the full demo."""
    os.chdir(PROJECT_DIR)
    
    try:
        show_title()
        show_problem()
        show_solution()
        show_data_collection()
        show_signal_detection()
        show_top_issues()
        show_harbor_task()
        show_closing()
        
        console.print()
        console.print("[bold green]Demo complete![/bold green]")
        console.print()
        
    except KeyboardInterrupt:
        console.print("\n[dim]Demo interrupted[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
