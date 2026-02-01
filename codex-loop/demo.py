#!/usr/bin/env python3
"""
Demo: Codex-Native Closed Loop

This script demonstrates the complete closed loop from brainstorm-updated.md:
1. Ingest trace spines from ~/.codex/trace_spine/
2. Detect friction and delight signals
3. Cluster signals to discover patterns
4. Distill a friction cluster into a Harbor task
5. Run Harbor evaluation (optional)
6. Generate work artifacts

Usage:
    python demo.py [--codex-home ~/.codex] [--output ./demo_output]
    python demo.py --use-fixtures  # Use bundled test fixtures
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import shutil
import tempfile

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.command()
@click.option("--codex-home", type=Path, default=Path.home() / ".codex",
              help="Path to Codex home directory")
@click.option("--output", type=Path, default=Path("./demo_output"),
              help="Output directory for demo artifacts")
@click.option("--use-fixtures", is_flag=True,
              help="Use bundled test fixtures instead of real traces")
@click.option("--skip-harbor", is_flag=True,
              help="Skip Harbor evaluation")
@click.option("--skip-openai", is_flag=True,
              help="Skip OpenAI API calls (embeddings/semantic)")
def main(codex_home: Path, output: Path, use_fixtures: bool, skip_harbor: bool, skip_openai: bool):
    """Run the Codex Closed Loop demo."""
    asyncio.run(_run_demo(codex_home, output, use_fixtures, skip_harbor, skip_openai))


async def _run_demo(
    codex_home: Path,
    output: Path,
    use_fixtures: bool,
    skip_harbor: bool,
    skip_openai: bool,
):
    """Run the demo."""
    console.print(Panel.fit(
        "[bold blue]Codex-Native Closed Loop Demo[/bold blue]\n\n"
        "Demonstrating the 7-stage system:\n"
        "1. Capture (already done by Codex)\n"
        "2-3. Ship/Canonicalize (ingest to SQLite)\n"
        "4. Analyze (signals + clustering)\n"
        "5. Distill (Harbor tasks)\n"
        "6. Measure (Harbor eval)\n"
        "7. Close Loop (work artifacts)",
        title="Welcome"
    ))
    
    # Set up demo environment
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / "codex_loop.db"
    vector_path = output / "vectors"
    
    # If using fixtures, copy them to a temp codex_home
    if use_fixtures:
        console.print("\n[yellow]Using bundled test fixtures[/yellow]")
        fixtures_dir = Path(__file__).parent / "tests" / "fixtures"
        
        if not fixtures_dir.exists():
            console.print("[red]Fixtures not found. Run from codex-loop directory.[/red]")
            sys.exit(1)
        
        # Create temp codex_home with fixtures
        temp_codex_home = output / "temp_codex_home"
        trace_dir = temp_codex_home / "trace_spine"
        
        # Copy each fixture as a separate session
        for fixture in fixtures_dir.glob("*.jsonl"):
            session_dir = trace_dir / fixture.stem
            session_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(fixture, session_dir / "segment-000.jsonl")
        
        codex_home = temp_codex_home
    
    # Import after setup to avoid slow startup
    from codex_loop.db import init_db, get_session, Session, Signal, Cluster
    from codex_loop.db.vector_store import ChromaVectorStore
    from codex_loop.ingest import find_all_spines, parse_spine, canonicalize_session
    from codex_loop.signals import detect_mechanical_signals
    from codex_loop.signals.types import SignalType
    
    # === Stage 2-3: Initialize and Ingest ===
    console.print("\n[bold]Stage 2-3: Initialize canonical store & ingest[/bold]")
    
    init_db(db_path)
    db = get_session(db_path)
    
    console.print(f"  SQLite: {db_path}")
    console.print(f"  Vector store: {vector_path}")
    
    # Find spines
    spines = find_all_spines(codex_home)
    console.print(f"  Found {len(spines)} trace spine(s)")
    
    if not spines:
        console.print("[yellow]No trace spines found. Run Codex to generate some, or use --use-fixtures[/yellow]")
        return
    
    # Ingest sessions
    sessions_added = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting sessions...", total=len(spines))
        
        for spine_path in spines:
            try:
                records = parse_spine(spine_path)
                if not records:
                    progress.advance(task)
                    continue
                
                session = canonicalize_session(records, spine_path)
                
                # Check if exists
                existing = db.query(Session).filter(Session.id == session.id).first()
                if not existing:
                    db.add(session)
                    sessions_added += 1
                
            except Exception as e:
                console.print(f"[red]Error ingesting {spine_path.name}: {e}[/red]")
            
            progress.advance(task)
    
    db.commit()
    console.print(f"  Ingested {sessions_added} new session(s)")
    
    # === Stage 4: Detect Signals ===
    console.print("\n[bold]Stage 4: Detect friction/delight signals[/bold]")
    
    sessions_to_process = db.query(Session).filter(Session.signals_computed == False).all()
    
    total_signals = 0
    friction_count = 0
    delight_count = 0
    
    for session in sessions_to_process:
        try:
            records = []
            if session.spine_path:
                records = parse_spine(Path(session.spine_path))
            
            signals = detect_mechanical_signals(session, records)
            
            for sig in signals:
                signal_obj = Signal(
                    id=f"sig_{session.id}_{total_signals}",
                    session_id=session.id,
                    turn_id=sig.turn_id,
                    signal_type=sig.signal_type.value,
                    category=sig.category.value if hasattr(sig.category, 'value') else str(sig.category),
                    severity=sig.severity,
                    evidence=sig.to_evidence_dict(),
                    detector="mechanical",
                )
                db.add(signal_obj)
                total_signals += 1
                
                if sig.signal_type == SignalType.FRICTION:
                    friction_count += 1
                else:
                    delight_count += 1
            
            session.signals_computed = True
            
        except Exception as e:
            console.print(f"[yellow]Error processing {session.id}: {e}[/yellow]")
    
    db.commit()
    
    # Display signals table
    table = Table(title="Detected Signals")
    table.add_column("Type", style="bold")
    table.add_column("Category")
    table.add_column("Count")
    
    from collections import Counter
    all_signals = db.query(Signal).all()
    friction_cats = Counter(s.category for s in all_signals if s.signal_type == "friction")
    delight_cats = Counter(s.category for s in all_signals if s.signal_type == "delight")
    
    for cat, count in friction_cats.most_common(5):
        table.add_row("[red]friction[/red]", cat, str(count))
    for cat, count in delight_cats.most_common(5):
        table.add_row("[green]delight[/green]", cat, str(count))
    
    console.print(table)
    console.print(f"  Total: {len(all_signals)} signals ({friction_count} friction, {delight_count} delight)")
    
    # === Stage 4 continued: Clustering (if OpenAI available) ===
    if not skip_openai:
        console.print("\n[bold]Stage 4 (continued): Cluster signals for discovery[/bold]")
        
        try:
            from codex_loop.embeddings import SignalEncoder, SignalClusterer
            
            vector_store = ChromaVectorStore(vector_path)
            encoder = SignalEncoder()
            
            # Generate embeddings for a sample
            signals_to_embed = db.query(Signal).filter(Signal.embedding_id == None).limit(20).all()
            
            console.print(f"  Generating embeddings for {len(signals_to_embed)} signal(s)...")
            
            for signal in signals_to_embed:
                try:
                    embedding = await encoder.encode_signal(signal)
                    embedding_id = vector_store.add_signal_embedding(
                        embedding,
                        signal.id,
                        metadata={"signal_type": signal.signal_type, "category": signal.category},
                    )
                    signal.embedding_id = embedding_id
                except Exception as e:
                    console.print(f"[yellow]Embedding error: {e}[/yellow]")
                    break
            
            db.commit()
            
            # Cluster
            clusterer = SignalClusterer(vector_store)
            clusters_created = 0
            
            for signal_type in ["friction", "delight"]:
                try:
                    result = clusterer.cluster_signals(signal_type, min_cluster_size=2)
                    for cluster in result.clusters:
                        db.add(cluster)
                        clusters_created += 1
                except Exception:
                    pass
            
            db.commit()
            console.print(f"  Created {clusters_created} cluster(s)")
            
        except Exception as e:
            console.print(f"[yellow]Clustering skipped: {e}[/yellow]")
    else:
        console.print("\n[yellow]Skipping clustering (--skip-openai)[/yellow]")
    
    # === Stage 5: Distill ===
    console.print("\n[bold]Stage 5: Distill to Harbor task[/bold]")
    
    # Pick a friction signal to distill
    friction_signal = db.query(Signal).filter(Signal.signal_type == "friction").first()
    
    if friction_signal:
        from codex_loop.distill import distill_signal_to_task
        
        task_dir = output / "harbor_task"
        
        try:
            task = await distill_signal_to_task(friction_signal, task_dir)
            console.print(f"  Created Harbor task at: {task_dir}")
            console.print(f"  Task ID: {task.task_id}")
            console.print(f"  Files: instruction.md, task.toml, environment/Dockerfile, tests/test.sh")
        except Exception as e:
            console.print(f"[yellow]Distillation error: {e}[/yellow]")
    else:
        console.print("[yellow]No friction signals to distill[/yellow]")
    
    # === Stage 6: Measure (optional) ===
    if not skip_harbor and (output / "harbor_task").exists():
        console.print("\n[bold]Stage 6: Run Harbor evaluation[/bold]")
        
        from codex_loop.measure import run_harbor_evaluation
        
        try:
            console.print("  Running Harbor...")
            result = await run_harbor_evaluation(
                dataset_path=output / "harbor_task",
                agent="codex",
                model="gpt-4",
                parallel=1,
            )
            
            console.print(f"  Tasks evaluated: {result.task_count}")
            console.print(f"  Mean reward: {result.mean_reward:.2f}")
            console.print(f"  Success rate: {result.success_rate:.1%}")
            
        except FileNotFoundError:
            console.print("[yellow]Harbor not installed. Install with: uv tool install harbor[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Harbor evaluation skipped: {e}[/yellow]")
    else:
        console.print("\n[yellow]Skipping Harbor evaluation (--skip-harbor)[/yellow]")
    
    # === Stage 7: Work Artifacts ===
    console.print("\n[bold]Stage 7: Generate work artifact[/bold]")
    
    clusters = db.query(Cluster).all()
    
    if clusters and not skip_openai:
        from codex_loop.pipeline.work_artifacts import generate_ticket
        
        cluster = clusters[0]
        signals = db.query(Signal).filter(Signal.cluster_id == cluster.id).all()
        
        if not signals:
            signals = db.query(Signal).filter(Signal.signal_type == cluster.signal_type).limit(5).all()
        
        try:
            ticket = await generate_ticket(cluster, signals, output / "harbor_task")
            
            ticket_path = output / "generated_ticket.md"
            ticket_path.write_text(ticket.to_markdown())
            
            console.print(f"  Generated ticket: {ticket_path}")
            console.print(f"  Title: {ticket.title}")
            console.print(f"  Priority: {ticket.priority}")
            
        except Exception as e:
            console.print(f"[yellow]Ticket generation error: {e}[/yellow]")
    else:
        console.print("[yellow]No clusters for ticket generation[/yellow]")
    
    # === Summary ===
    final_sessions = db.query(Session).count()
    final_signals = db.query(Signal).count()
    final_clusters = db.query(Cluster).count()
    
    console.print(Panel.fit(
        f"[bold green]Demo Complete![/bold green]\n\n"
        f"Sessions analyzed: {final_sessions}\n"
        f"Signals detected: {final_signals}\n"
        f"Clusters formed: {final_clusters}\n\n"
        f"Output directory: {output}\n\n"
        f"[dim]Next steps:\n"
        f"1. Review generated_ticket.md\n"
        f"2. Run: harbor run -p {output}/harbor_task -a codex -m gpt-4\n"
        f"3. Set up daily pipeline: codex-loop daily --codex-home ~/.codex[/dim]",
        title="Summary"
    ))


if __name__ == "__main__":
    main()
