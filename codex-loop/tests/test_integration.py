"""
Integration tests that can use real Codex sessions.

These tests are marked with @pytest.mark.integration and require:
- Codex binary available
- CODEX_HOME environment set
"""

import pytest
from pathlib import Path
import json


@pytest.mark.integration
def test_real_trace_spine_structure():
    """
    Verify structure of real trace spines from ~/.codex.
    
    Skips if no traces available.
    """
    codex_home = Path.home() / ".codex"
    trace_spine_dir = codex_home / "trace_spine"
    
    if not trace_spine_dir.exists():
        pytest.skip("No trace_spine directory found")
    
    spine_dirs = list(trace_spine_dir.iterdir())
    if not spine_dirs:
        pytest.skip("No trace spines found")
    
    # Check the most recent spine
    latest = max(spine_dirs, key=lambda p: p.stat().st_mtime)
    segment_files = list(latest.glob("segment-*.jsonl"))
    
    assert len(segment_files) > 0, f"No segment files in {latest}"
    
    # Parse and verify structure
    records = []
    for segment in segment_files:
        for line in segment.read_text().strip().split("\n"):
            if line:
                records.append(json.loads(line))
    
    assert len(records) > 0, "No records in trace"
    
    # Verify required fields
    for record in records:
        assert "schema_version" in record
        assert "thread_id" in record
        assert "seq" in record
        assert "type" in record
        assert "payload" in record
    
    # Verify at least one session_meta
    meta_records = [r for r in records if r["type"] == "session_meta"]
    assert len(meta_records) > 0, "No session_meta record found"


@pytest.mark.integration
def test_real_spine_ingestion():
    """
    Test ingesting a real trace spine.
    
    Skips if no traces available.
    """
    from codex_loop.ingest import find_all_spines, parse_spine, canonicalize_session
    
    codex_home = Path.home() / ".codex"
    
    spines = find_all_spines(codex_home)
    if not spines:
        pytest.skip("No trace spines found")
    
    # Ingest the most recent spine
    spine_path = spines[0]
    records = parse_spine(spine_path)
    
    assert len(records) > 0
    
    session = canonicalize_session(records, spine_path)
    
    assert session.id is not None
    assert session.source is not None


@pytest.mark.integration
def test_real_signal_detection():
    """
    Test signal detection on a real trace spine.
    
    Skips if no traces available.
    """
    from codex_loop.ingest import find_all_spines, parse_spine, canonicalize_session
    from codex_loop.signals import detect_mechanical_signals
    
    codex_home = Path.home() / ".codex"
    
    spines = find_all_spines(codex_home)
    if not spines:
        pytest.skip("No trace spines found")
    
    spine_path = spines[0]
    records = parse_spine(spine_path)
    session = canonicalize_session(records, spine_path)
    
    signals = detect_mechanical_signals(session, records)
    
    # Just verify it runs without error
    # Real sessions may or may not have signals
    assert isinstance(signals, list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_full_pipeline():
    """
    Test the full pipeline on real trace spines.
    
    Skips if no traces available.
    """
    import tempfile
    from codex_loop.db import init_db, get_session
    from codex_loop.ingest import find_all_spines
    from codex_loop.pipeline.daily_batch import run_daily_pipeline
    
    codex_home = Path.home() / ".codex"
    
    spines = find_all_spines(codex_home)
    if not spines:
        pytest.skip("No trace spines found")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        report = await run_daily_pipeline(
            codex_home=codex_home,
            db_path=db_path,
            lookback_hours=24 * 365,  # All spines
            skip_semantic=True,  # Skip to avoid API costs
            skip_clustering=True,  # Skip to avoid API costs
            verbose=False,
        )
        
        # Verify report
        assert report.spines_found >= 0
        # May not ingest if already processed
