"""
CLI entry point for codex-loop.

Provides commands for:
- daily: Run daily analysis pipeline
- ingest: Ingest trace spines
- signals: Detect signals
- cluster: Run clustering
- distill: Distill to Harbor tasks
- measure: Run Harbor evaluation
- report: Generate analysis report
- ticket: Generate work artifacts
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Codex Closed Loop: Analysis pipeline for Codex sessions."""
    pass


@main.command()
@click.option("--codex-home", type=Path, default=Path.home() / ".codex",
              help="Path to Codex home directory")
@click.option("--db", type=Path, default=Path("./codex_loop.db"),
              help="Path to SQLite database")
@click.option("--lookback-hours", type=int, default=24,
              help="Hours to look back for new sessions")
@click.option("--skip-semantic", is_flag=True,
              help="Skip LLM-based semantic signal detection")
@click.option("--skip-clustering", is_flag=True,
              help="Skip clustering step")
@click.option("--verbose", "-v", is_flag=True,
              help="Print progress")
def daily(codex_home, db, lookback_hours, skip_semantic, skip_clustering, verbose):
    """Run daily analysis pipeline."""
    from codex_loop.pipeline.daily_batch import run_daily_pipeline, format_daily_report
    
    async def _run():
        report = await run_daily_pipeline(
            codex_home=codex_home,
            db_path=db,
            lookback_hours=lookback_hours,
            skip_semantic=skip_semantic,
            skip_clustering=skip_clustering,
            verbose=verbose,
        )
        console.print(format_daily_report(report))
    
    asyncio.run(_run())


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
def init(db):
    """Initialize database."""
    from codex_loop.db import init_db
    
    init_db(db)
    console.print(f"[green]Database initialized at {db}[/green]")


@main.command()
@click.option("--codex-home", type=Path, default=Path.home() / ".codex",
              help="Path to Codex home directory")
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
@click.option("--lookback-hours", type=int, default=24 * 365,
              help="Hours to look back")
def ingest(codex_home, db, lookback_hours):
    """Ingest trace spines into database."""
    from codex_loop.db import init_db, get_session, Session
    from codex_loop.ingest import find_spines_since, parse_spine, canonicalize_session
    
    init_db(db)
    db_session = get_session(db)
    
    spines = find_spines_since(codex_home, hours=lookback_hours)
    console.print(f"Found {len(spines)} spine(s)")
    
    ingested = 0
    for spine_path in spines:
        try:
            records = parse_spine(spine_path)
            if not records:
                continue
            
            session = canonicalize_session(records, spine_path)
            
            existing = db_session.query(Session).filter(Session.id == session.id).first()
            if existing:
                continue
            
            db_session.add(session)
            ingested += 1
            
        except Exception as e:
            console.print(f"[yellow]Error ingesting {spine_path}: {e}[/yellow]")
    
    db_session.commit()
    console.print(f"[green]Ingested {ingested} new session(s)[/green]")


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
def signals(db):
    """Detect signals in sessions."""
    from codex_loop.db import get_session, Session, Signal
    from codex_loop.ingest import parse_spine
    from codex_loop.signals import detect_mechanical_signals
    
    db_session = get_session(db)
    
    sessions = db_session.query(Session).filter(Session.signals_computed == False).all()
    console.print(f"Processing {len(sessions)} session(s)")
    
    total_signals = 0
    for session in sessions:
        records = []
        if session.spine_path:
            records = parse_spine(Path(session.spine_path))
        
        mechanical = detect_mechanical_signals(session, records)
        
        for sig in mechanical:
            signal = Signal(
                id=f"sig_{session.id}_{total_signals}",
                session_id=session.id,
                turn_id=sig.turn_id,
                signal_type=sig.signal_type.value,
                category=sig.category.value if hasattr(sig.category, 'value') else str(sig.category),
                severity=sig.severity,
                evidence=sig.to_evidence_dict(),
                detector="mechanical",
            )
            db_session.add(signal)
            total_signals += 1
        
        session.signals_computed = True
    
    db_session.commit()
    console.print(f"[green]Detected {total_signals} signal(s)[/green]")


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
@click.option("--vectors", type=Path, default=None,
              help="Path to vector store")
def cluster(db, vectors):
    """Run signal clustering."""
    from codex_loop.db import get_session, Signal, Cluster
    from codex_loop.db.vector_store import ChromaVectorStore
    from codex_loop.embeddings import SignalEncoder, SignalClusterer
    
    if vectors is None:
        vectors = db.parent / "vectors"
    
    db_session = get_session(db)
    vector_store = ChromaVectorStore(vectors)
    
    console.print("Generating embeddings...")
    
    async def _run():
        encoder = SignalEncoder()
        signals = db_session.query(Signal).filter(Signal.embedding_id == None).limit(100).all()
        
        for signal in signals:
            try:
                embedding = await encoder.encode_signal(signal)
                embedding_id = vector_store.add_signal_embedding(
                    embedding, signal.id,
                    metadata={"signal_type": signal.signal_type, "category": signal.category},
                )
                signal.embedding_id = embedding_id
            except Exception as e:
                console.print(f"[yellow]Embedding error: {e}[/yellow]")
                break
        
        db_session.commit()
        console.print(f"Generated embeddings for {len(signals)} signal(s)")
        
        # Cluster
        clusterer = SignalClusterer(vector_store)
        total_clusters = 0
        
        for signal_type in ["friction", "delight"]:
            result = clusterer.cluster_signals(signal_type, min_cluster_size=3)
            for clust in result.clusters:
                db_session.add(clust)
                total_clusters += 1
        
        db_session.commit()
        console.print(f"[green]Created {total_clusters} cluster(s)[/green]")
    
    asyncio.run(_run())


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
@click.option("--signal-id", type=str, default=None,
              help="Signal ID to distill")
@click.option("--cluster-id", type=str, default=None,
              help="Cluster ID to distill")
@click.option("--output", type=Path, required=True,
              help="Output directory")
def distill(db, signal_id, cluster_id, output):
    """Distill signals/clusters into Harbor tasks."""
    from codex_loop.db import get_session, Signal, Cluster
    from codex_loop.distill import distill_signal_to_task, distill_cluster_to_dataset
    
    db_session = get_session(db)
    
    async def _run():
        if signal_id:
            signal = db_session.query(Signal).filter(Signal.id == signal_id).first()
            if not signal:
                console.print(f"[red]Signal {signal_id} not found[/red]")
                return
            
            task = await distill_signal_to_task(signal, output)
            console.print(f"[green]Created task at {output}[/green]")
            
        elif cluster_id:
            clust = db_session.query(Cluster).filter(Cluster.id == cluster_id).first()
            if not clust:
                console.print(f"[red]Cluster {cluster_id} not found[/red]")
                return
            
            tasks = await distill_cluster_to_dataset(
                clust, db_session, Path.home() / ".codex", output
            )
            console.print(f"[green]Created {len(tasks)} task(s) at {output}[/green]")
        else:
            console.print("[red]Specify --signal-id or --cluster-id[/red]")
    
    asyncio.run(_run())


@main.command()
@click.option("--dataset", type=Path, required=True,
              help="Path to Harbor dataset")
@click.option("--agent", type=str, default="codex",
              help="Agent to evaluate")
@click.option("--model", type=str, default="gpt-4",
              help="Model to use")
@click.option("--baseline", type=Path, default=None,
              help="Previous results for regression check")
def measure(dataset, agent, model, baseline):
    """Run Harbor evaluation."""
    from codex_loop.measure import run_harbor_evaluation, detect_regressions, summarize_results
    
    async def _run():
        console.print(f"Running Harbor evaluation on {dataset}...")
        
        try:
            result = await run_harbor_evaluation(
                dataset_path=dataset,
                agent=agent,
                model=model,
            )
            
            summary = summarize_results(result.results)
            
            table = Table(title="Evaluation Results")
            table.add_column("Metric")
            table.add_column("Value")
            
            table.add_row("Tasks", str(summary["count"]))
            table.add_row("Mean Reward", f"{summary['mean_reward']:.3f}")
            table.add_row("Success Rate", f"{summary['success_rate']:.1%}")
            
            console.print(table)
            
            if baseline:
                baseline_results = json.loads(baseline.read_text())
                from codex_loop.measure.harbor_runner import TrialResult
                baseline_trials = [TrialResult(**r) for r in baseline_results]
                
                regressions = detect_regressions(baseline_trials, result.results)
                if regressions:
                    console.print(f"[red]Detected {len(regressions)} regression(s)[/red]")
                else:
                    console.print("[green]No regressions detected[/green]")
                    
        except FileNotFoundError:
            console.print("[red]Harbor not installed. Install with: uv tool install harbor[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(_run())


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
def report(db):
    """Generate analysis report."""
    from codex_loop.db import get_session, Session, Signal, Cluster
    from collections import Counter
    
    db_session = get_session(db)
    
    # Session stats
    total_sessions = db_session.query(Session).count()
    processed_sessions = db_session.query(Session).filter(Session.signals_computed == True).count()
    
    # Signal stats
    friction_signals = db_session.query(Signal).filter(Signal.signal_type == "friction").all()
    delight_signals = db_session.query(Signal).filter(Signal.signal_type == "delight").all()
    
    friction_categories = Counter(s.category for s in friction_signals)
    delight_categories = Counter(s.category for s in delight_signals)
    
    # Cluster stats
    clusters = db_session.query(Cluster).all()
    
    # Display
    console.print(Panel.fit(
        f"[bold]Sessions[/bold]: {total_sessions} total, {processed_sessions} processed\n"
        f"[bold]Signals[/bold]: {len(friction_signals)} friction, {len(delight_signals)} delight\n"
        f"[bold]Clusters[/bold]: {len(clusters)}",
        title="Analysis Summary"
    ))
    
    if friction_categories:
        table = Table(title="Friction Signals by Category")
        table.add_column("Category")
        table.add_column("Count")
        
        for category, count in friction_categories.most_common(10):
            table.add_row(category, str(count))
        
        console.print(table)
    
    if delight_categories:
        table = Table(title="Delight Signals by Category")
        table.add_column("Category")
        table.add_column("Count")
        
        for category, count in delight_categories.most_common(10):
            table.add_row(category, str(count))
        
        console.print(table)


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
@click.option("--cluster-id", type=str, required=True,
              help="Cluster ID to generate ticket for")
@click.option("--output", type=Path, default=None,
              help="Output file")
def ticket(db, cluster_id, output):
    """Generate a ticket for a cluster."""
    from codex_loop.db import get_session, Signal, Cluster
    from codex_loop.pipeline.work_artifacts import generate_ticket
    
    db_session = get_session(db)
    
    async def _run():
        clust = db_session.query(Cluster).filter(Cluster.id == cluster_id).first()
        if not clust:
            console.print(f"[red]Cluster {cluster_id} not found[/red]")
            return
        
        signals = db_session.query(Signal).filter(Signal.cluster_id == cluster_id).all()
        
        ticket_content = await generate_ticket(clust, signals)
        
        if output:
            output.write_text(ticket_content.to_markdown())
            console.print(f"[green]Ticket written to {output}[/green]")
        else:
            console.print(ticket_content.to_markdown())
    
    asyncio.run(_run())


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
@click.option("--dataset", type=str, required=True,
              help="Dataset ID to show trends for")
@click.option("--days", type=int, default=30,
              help="Lookback days")
def trends(db, dataset, days):
    """Show evaluation trends."""
    from codex_loop.db import get_session
    from codex_loop.measure.trend_tracker import TrendTracker, format_trend_chart
    
    db_session = get_session(db)
    tracker = TrendTracker(db_session)
    
    df = tracker.get_trend(dataset, lookback_days=days)
    
    if df.empty:
        console.print("[yellow]No trend data available[/yellow]")
        return
    
    console.print(format_trend_chart(df))
    
    stats = tracker.compute_trend_stats(dataset, lookback_days=days)
    console.print(f"\nTrend: [bold]{stats['trend']}[/bold]")


# ============================================================================
# Schema Evolution Commands
# ============================================================================

@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
def schema_proposals(db):
    """List pending schema proposals."""
    from codex_loop.db import get_session
    from codex_loop.discovery import list_pending_proposals
    
    db_session = get_session(db)
    proposals = list_pending_proposals(db_session)
    
    if not proposals:
        console.print("[yellow]No pending schema proposals[/yellow]")
        return
    
    table = Table(title="Pending Schema Proposals")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Samples")
    table.add_column("Created")
    
    for p in proposals:
        table.add_row(
            p.id[:12],
            p.name,
            p.signal_type,
            str(p.sample_count),
            p.created_at.strftime("%Y-%m-%d") if p.created_at else "N/A",
        )
    
    console.print(table)


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
@click.option("--proposal-id", type=str, required=True,
              help="Proposal ID to approve")
def schema_approve(db, proposal_id):
    """Approve a schema proposal and promote it to stable category."""
    from codex_loop.db import get_session
    from codex_loop.discovery import approve_proposal
    
    db_session = get_session(db)
    
    try:
        proposal = approve_proposal(db_session, proposal_id)
        console.print(f"[green]Approved proposal: {proposal.name}[/green]")
        console.print(f"Category '{proposal.name}' is now a stable {proposal.signal_type} signal.")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
@click.option("--proposal-id", type=str, required=True,
              help="Proposal ID to reject")
@click.option("--reason", type=str, default=None,
              help="Reason for rejection")
def schema_reject(db, proposal_id, reason):
    """Reject a schema proposal."""
    from codex_loop.db import get_session
    from codex_loop.discovery import reject_proposal
    
    db_session = get_session(db)
    
    try:
        proposal = reject_proposal(db_session, proposal_id, reason)
        console.print(f"[yellow]Rejected proposal: {proposal.name}[/yellow]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
def schema_categories(db):
    """List all stable (promoted) signal categories."""
    from codex_loop.db import get_session
    from codex_loop.discovery import get_stable_categories
    
    db_session = get_session(db)
    categories = get_stable_categories(db_session)
    
    console.print(Panel.fit(
        f"[bold]Friction Categories[/bold]: {len(categories['friction'])}\n"
        f"[bold]Delight Categories[/bold]: {len(categories['delight'])}",
        title="Stable Schema Categories"
    ))
    
    if categories['friction']:
        table = Table(title="Friction Categories")
        table.add_column("Category")
        for cat in categories['friction']:
            table.add_row(cat)
        console.print(table)
    
    if categories['delight']:
        table = Table(title="Delight Categories")
        table.add_column("Category")
        for cat in categories['delight']:
            table.add_row(cat)
        console.print(table)


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
@click.option("--output", type=Path, required=True,
              help="Output path for schema JSON file")
def schema_export(db, output):
    """Export stable categories to a JSON schema file."""
    from codex_loop.db import get_session
    from codex_loop.discovery import export_schema_to_file
    
    db_session = get_session(db)
    export_schema_to_file(db_session, str(output))
    console.print(f"[green]Schema exported to {output}[/green]")


@main.command()
@click.option("--db", type=Path, required=True,
              help="Path to SQLite database")
@click.option("--confidence", type=float, default=0.85,
              help="Minimum confidence threshold for auto-approval")
@click.option("--min-samples", type=int, default=10,
              help="Minimum sample count for auto-approval")
def schema_auto_promote(db, confidence, min_samples):
    """Auto-approve high-confidence schema proposals."""
    from codex_loop.db import get_session
    from codex_loop.discovery import auto_promote_high_confidence_proposals
    
    db_session = get_session(db)
    
    async def _run():
        approved = await auto_promote_high_confidence_proposals(
            db_session,
            confidence_threshold=confidence,
            min_sample_count=min_samples,
        )
        
        if approved:
            console.print(f"[green]Auto-approved {len(approved)} proposal(s):[/green]")
            for p in approved:
                console.print(f"  - {p.name}")
        else:
            console.print("[yellow]No proposals met auto-approval criteria[/yellow]")
    
    asyncio.run(_run())


if __name__ == "__main__":
    main()
