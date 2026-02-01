"""
Daily batch pipeline: Orchestrate the closed loop analysis.

Implements the daily pipeline from brainstorm-updated.md lines 372-377:
1. Find new trace spines
2. Ingest into SQLite
3. Compute signals (mechanical + semantic)
4. Generate embeddings
5. Update clusters
6. Check for schema proposals
7. Generate report
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import asyncio

from openai import AsyncOpenAI
from sqlalchemy.orm import Session as DBSession

from codex_loop.db import init_db, get_session, Session, Signal, Cluster
from codex_loop.db.vector_store import ChromaVectorStore
from codex_loop.ingest import find_spines_since, parse_spine, canonicalize_session
from codex_loop.signals import detect_mechanical_signals, detect_semantic_signals
from codex_loop.embeddings import SignalEncoder, SignalClusterer
from codex_loop.discovery import propose_new_schema


@dataclass
class DailyReport:
    """Summary of daily pipeline run."""
    run_date: datetime
    spines_found: int
    sessions_ingested: int
    signals_detected: int
    friction_signals: int
    delight_signals: int
    clusters_updated: int
    schema_proposals: int
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_date": self.run_date.isoformat(),
            "spines_found": self.spines_found,
            "sessions_ingested": self.sessions_ingested,
            "signals_detected": self.signals_detected,
            "friction_signals": self.friction_signals,
            "delight_signals": self.delight_signals,
            "clusters_updated": self.clusters_updated,
            "schema_proposals": self.schema_proposals,
            "errors": self.errors,
        }


async def run_daily_pipeline(
    codex_home: Path,
    db_path: Path,
    vector_store_path: Optional[Path] = None,
    lookback_hours: int = 24,
    skip_semantic: bool = False,
    skip_clustering: bool = False,
    verbose: bool = False,
) -> DailyReport:
    """
    Run the daily analysis pipeline.
    
    Args:
        codex_home: Path to ~/.codex
        db_path: Path to SQLite database
        vector_store_path: Path to ChromaDB storage (defaults to db_path.parent/vectors)
        lookback_hours: Hours to look back for new spines
        skip_semantic: If True, skip LLM-based semantic signals
        skip_clustering: If True, skip clustering step
        verbose: Print progress
        
    Returns:
        DailyReport summarizing the run
    """
    report = DailyReport(
        run_date=datetime.utcnow(),
        spines_found=0,
        sessions_ingested=0,
        signals_detected=0,
        friction_signals=0,
        delight_signals=0,
        clusters_updated=0,
        schema_proposals=0,
    )
    
    # Initialize storage
    if verbose:
        print(f"Initializing database at {db_path}")
    init_db(db_path)
    db = get_session(db_path)
    
    if vector_store_path is None:
        vector_store_path = db_path.parent / "vectors"
    
    vector_store = ChromaVectorStore(vector_store_path)
    
    # Step 1: Find new spines
    if verbose:
        print(f"Finding spines from last {lookback_hours} hours...")
    
    new_spines = find_spines_since(codex_home, hours=lookback_hours)
    report.spines_found = len(new_spines)
    
    if verbose:
        print(f"Found {len(new_spines)} spine(s)")
    
    # Step 2: Ingest into SQLite
    if verbose:
        print("Ingesting sessions...")
    
    for spine_path in new_spines:
        try:
            records = parse_spine(spine_path)
            if not records:
                continue
            
            session = canonicalize_session(records, spine_path)
            
            # Check if already exists
            existing = db.query(Session).filter(Session.id == session.id).first()
            if existing:
                continue
            
            db.add(session)
            report.sessions_ingested += 1
            
        except Exception as e:
            report.errors.append(f"Ingest error for {spine_path}: {e}")
    
    db.commit()
    
    if verbose:
        print(f"Ingested {report.sessions_ingested} new session(s)")
    
    # Step 3-4: Compute signals
    if verbose:
        print("Computing signals...")
    
    sessions_to_process = db.query(Session).filter(
        Session.signals_computed == False
    ).all()
    
    openai_client = None
    if not skip_semantic:
        try:
            openai_client = AsyncOpenAI()
        except Exception:
            if verbose:
                print("Warning: OpenAI client not available, skipping semantic signals")
            skip_semantic = True
    
    for session in sessions_to_process:
        try:
            # Load records
            if session.spine_path:
                records = parse_spine(Path(session.spine_path))
            else:
                records = []
            
            # Mechanical signals
            mechanical = detect_mechanical_signals(session, records)
            for sig_result in mechanical:
                signal = Signal(
                    id=f"sig_{session.id}_{len(db.query(Signal).filter(Signal.session_id == session.id).all())}",
                    session_id=session.id,
                    turn_id=sig_result.turn_id,
                    signal_type=sig_result.signal_type.value,
                    category=sig_result.category.value if hasattr(sig_result.category, 'value') else str(sig_result.category),
                    severity=sig_result.severity,
                    evidence=sig_result.to_evidence_dict(),
                    detector="mechanical",
                )
                db.add(signal)
                report.signals_detected += 1
                if sig_result.signal_type.value == "friction":
                    report.friction_signals += 1
                else:
                    report.delight_signals += 1
            
            # Semantic signals
            if not skip_semantic and openai_client:
                try:
                    semantic = await detect_semantic_signals(session, records, openai_client)
                    for sig_result in semantic:
                        signal = Signal(
                            id=f"sig_{session.id}_sem_{len(db.query(Signal).filter(Signal.session_id == session.id).all())}",
                            session_id=session.id,
                            turn_id=sig_result.turn_id,
                            signal_type=sig_result.signal_type.value,
                            category=sig_result.category.value if hasattr(sig_result.category, 'value') else str(sig_result.category),
                            severity=sig_result.severity,
                            evidence=sig_result.to_evidence_dict(),
                            detector="semantic",
                        )
                        db.add(signal)
                        report.signals_detected += 1
                        if sig_result.signal_type.value == "friction":
                            report.friction_signals += 1
                        else:
                            report.delight_signals += 1
                except Exception as e:
                    report.errors.append(f"Semantic signal error for {session.id}: {e}")
            
            session.signals_computed = True
            
        except Exception as e:
            report.errors.append(f"Signal detection error for {session.id}: {e}")
    
    db.commit()
    
    if verbose:
        print(f"Detected {report.signals_detected} signal(s) ({report.friction_signals} friction, {report.delight_signals} delight)")
    
    # Step 5: Generate embeddings
    if not skip_clustering:
        if verbose:
            print("Generating embeddings...")
        
        try:
            encoder = SignalEncoder()
            signals_without_embeddings = db.query(Signal).filter(
                Signal.embedding_id == None
            ).limit(100).all()  # Limit for cost control
            
            for signal in signals_without_embeddings:
                try:
                    embedding = await encoder.encode_signal(signal)
                    embedding_id = vector_store.add_signal_embedding(
                        embedding,
                        signal.id,
                        metadata={
                            "signal_type": signal.signal_type,
                            "category": signal.category,
                        },
                    )
                    signal.embedding_id = embedding_id
                except Exception as e:
                    report.errors.append(f"Embedding error for {signal.id}: {e}")
                    break  # Stop on API errors
            
            db.commit()
            
        except Exception as e:
            report.errors.append(f"Embedding initialization error: {e}")
    
    # Step 6: Update clusters
    if not skip_clustering:
        if verbose:
            print("Updating clusters...")
        
        try:
            clusterer = SignalClusterer(vector_store)
            
            for signal_type in ["friction", "delight"]:
                result = clusterer.cluster_signals(
                    signal_type=signal_type,
                    min_cluster_size=3,
                )
                
                for cluster in result.clusters:
                    db.add(cluster)
                    report.clusters_updated += 1
            
            db.commit()
            
        except Exception as e:
            report.errors.append(f"Clustering error: {e}")
    
    if verbose:
        print(f"Updated {report.clusters_updated} cluster(s)")
    
    # Step 7: Check for schema proposals
    if not skip_clustering and openai_client:
        if verbose:
            print("Checking for schema proposals...")
        
        unlabeled_clusters = db.query(Cluster).filter(
            Cluster.name == None,
            Cluster.member_count >= 5,
        ).all()
        
        for cluster in unlabeled_clusters:
            try:
                samples = db.query(Signal).filter(
                    Signal.cluster_id == cluster.id
                ).limit(10).all()
                
                if samples:
                    proposal = await propose_new_schema(cluster, samples, openai_client)
                    if proposal.confidence > 0.6:
                        cluster.name = proposal.name
                        cluster.description = proposal.description
                        report.schema_proposals += 1
                        
            except Exception as e:
                report.errors.append(f"Schema proposal error for {cluster.id}: {e}")
        
        db.commit()
    
    if verbose:
        print(f"Generated {report.schema_proposals} schema proposal(s)")
        if report.errors:
            print(f"Errors: {len(report.errors)}")
    
    return report


def format_daily_report(report: DailyReport) -> str:
    """Format a daily report as human-readable text."""
    lines = [
        "=" * 60,
        "DAILY PIPELINE REPORT",
        f"Run Date: {report.run_date.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "INGESTION",
        f"  Spines found: {report.spines_found}",
        f"  Sessions ingested: {report.sessions_ingested}",
        "",
        "SIGNALS",
        f"  Total detected: {report.signals_detected}",
        f"  Friction: {report.friction_signals}",
        f"  Delight: {report.delight_signals}",
        "",
        "DISCOVERY",
        f"  Clusters updated: {report.clusters_updated}",
        f"  Schema proposals: {report.schema_proposals}",
        "",
    ]
    
    if report.errors:
        lines.extend([
            "ERRORS",
            *[f"  - {e}" for e in report.errors[:5]],
        ])
        if len(report.errors) > 5:
            lines.append(f"  ... and {len(report.errors) - 5} more")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
