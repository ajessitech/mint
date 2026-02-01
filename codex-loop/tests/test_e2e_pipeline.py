"""
End-to-end tests for the codex-loop pipeline.

Tests the complete flow: trace spine -> ingest -> signals -> clustering -> distill
"""

import pytest
from pathlib import Path
import shutil
import tempfile

from codex_loop.db import init_db, get_session, Session, Signal, Cluster
from codex_loop.db.vector_store import ChromaVectorStore
from codex_loop.ingest import parse_spine, canonicalize_session
from codex_loop.signals import detect_mechanical_signals
from codex_loop.signals.types import SignalType


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def test_codex_home(tmp_path):
    """Set up a fake CODEX_HOME with test fixtures."""
    codex_home = tmp_path / ".codex"
    trace_dir = codex_home / "trace_spine"
    
    # Copy friction fixtures
    friction_dir = trace_dir / "test-friction-001"
    friction_dir.mkdir(parents=True)
    shutil.copy(
        FIXTURES_DIR / "friction_repeated_denial.jsonl",
        friction_dir / "segment-000.jsonl"
    )
    
    # Copy delight fixtures
    delight_dir = trace_dir / "test-delight-001"
    delight_dir.mkdir(parents=True)
    shutil.copy(
        FIXTURES_DIR / "delight_fast_completion.jsonl",
        delight_dir / "segment-000.jsonl"
    )
    
    return codex_home


@pytest.fixture
def db_session(tmp_path):
    """Create a test database."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    return get_session(db_path)


class TestSpineReader:
    """Tests for trace spine reading."""
    
    def test_parse_friction_spine(self):
        """Test parsing friction spine fixture."""
        spine_path = FIXTURES_DIR / "friction_repeated_denial.jsonl"
        
        # Create a temp directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            spine_dir = Path(tmpdir) / "test-session"
            spine_dir.mkdir()
            shutil.copy(spine_path, spine_dir / "segment-000.jsonl")
            
            records = parse_spine(spine_dir)
            
            assert len(records) > 0
            assert records[0].type == "session_meta"
            assert records[0].thread_id == "test-friction-001"
    
    def test_parse_delight_spine(self):
        """Test parsing delight spine fixture."""
        spine_path = FIXTURES_DIR / "delight_fast_completion.jsonl"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            spine_dir = Path(tmpdir) / "test-session"
            spine_dir.mkdir()
            shutil.copy(spine_path, spine_dir / "segment-000.jsonl")
            
            records = parse_spine(spine_dir)
            
            assert len(records) > 0
            assert records[0].thread_id == "test-delight-001"


class TestCanonicalizer:
    """Tests for session canonicalization."""
    
    def test_canonicalize_friction_session(self):
        """Test canonicalizing a friction session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spine_dir = Path(tmpdir) / "test-session"
            spine_dir.mkdir()
            shutil.copy(
                FIXTURES_DIR / "friction_repeated_denial.jsonl",
                spine_dir / "segment-000.jsonl"
            )
            
            records = parse_spine(spine_dir)
            session = canonicalize_session(records, spine_dir)
            
            assert session.id == "test-friction-001"
            assert session.source == "CLI"
            assert session.turn_count >= 1
            assert session.facets.get("denial_count", 0) >= 3
    
    def test_canonicalize_delight_session(self):
        """Test canonicalizing a delight session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spine_dir = Path(tmpdir) / "test-session"
            spine_dir.mkdir()
            shutil.copy(
                FIXTURES_DIR / "delight_fast_completion.jsonl",
                spine_dir / "segment-000.jsonl"
            )
            
            records = parse_spine(spine_dir)
            session = canonicalize_session(records, spine_dir)
            
            assert session.id == "test-delight-001"
            assert session.facets.get("denial_count", 0) == 0


class TestMechanicalSignals:
    """Tests for mechanical signal detection."""
    
    def test_detect_repeated_denial(self):
        """Test detection of repeated denial pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spine_dir = Path(tmpdir) / "test-session"
            spine_dir.mkdir()
            shutil.copy(
                FIXTURES_DIR / "friction_repeated_denial.jsonl",
                spine_dir / "segment-000.jsonl"
            )
            
            records = parse_spine(spine_dir)
            session = canonicalize_session(records, spine_dir)
            
            signals = detect_mechanical_signals(session, records)
            
            assert len(signals) > 0
            
            friction_signals = [s for s in signals if s.signal_type == SignalType.FRICTION]
            assert len(friction_signals) > 0
            
            denial_signals = [s for s in friction_signals if "denial" in str(s.category).lower()]
            assert len(denial_signals) > 0
    
    def test_detect_exec_failure(self):
        """Test detection of exec failure pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spine_dir = Path(tmpdir) / "test-session"
            spine_dir.mkdir()
            shutil.copy(
                FIXTURES_DIR / "friction_exec_failure.jsonl",
                spine_dir / "segment-000.jsonl"
            )
            
            records = parse_spine(spine_dir)
            session = canonicalize_session(records, spine_dir)
            
            signals = detect_mechanical_signals(session, records)
            
            friction_signals = [s for s in signals if s.signal_type == SignalType.FRICTION]
            
            # Should detect exec failure
            failure_signals = [s for s in friction_signals if "failure" in str(s.category).lower()]
            assert len(failure_signals) > 0
    
    def test_detect_compaction_churn(self):
        """Test detection of compaction churn pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spine_dir = Path(tmpdir) / "test-session"
            spine_dir.mkdir()
            shutil.copy(
                FIXTURES_DIR / "friction_compaction_churn.jsonl",
                spine_dir / "segment-000.jsonl"
            )
            
            records = parse_spine(spine_dir)
            session = canonicalize_session(records, spine_dir)
            
            signals = detect_mechanical_signals(session, records)
            
            friction_signals = [s for s in signals if s.signal_type == SignalType.FRICTION]
            
            # Should detect compaction signals
            compaction_signals = [s for s in friction_signals if "compaction" in str(s.category).lower()]
            assert len(compaction_signals) > 0
    
    def test_detect_fast_completion_delight(self):
        """Test detection of fast completion delight pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spine_dir = Path(tmpdir) / "test-session"
            spine_dir.mkdir()
            shutil.copy(
                FIXTURES_DIR / "delight_fast_completion.jsonl",
                spine_dir / "segment-000.jsonl"
            )
            
            records = parse_spine(spine_dir)
            session = canonicalize_session(records, spine_dir)
            
            signals = detect_mechanical_signals(session, records)
            
            delight_signals = [s for s in signals if s.signal_type == SignalType.DELIGHT]
            assert len(delight_signals) > 0
    
    def test_detect_zero_denial_delight(self):
        """Test detection of zero denial delight pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spine_dir = Path(tmpdir) / "test-session"
            spine_dir.mkdir()
            shutil.copy(
                FIXTURES_DIR / "delight_zero_denial.jsonl",
                spine_dir / "segment-000.jsonl"
            )
            
            records = parse_spine(spine_dir)
            session = canonicalize_session(records, spine_dir)
            
            signals = detect_mechanical_signals(session, records)
            
            delight_signals = [s for s in signals if s.signal_type == SignalType.DELIGHT]
            
            # Should have zero denial and/or first attempt success
            assert len(delight_signals) > 0


class TestRegression:
    """Tests for regression detection."""
    
    def test_detect_regressions(self):
        """Test regression detection between two result sets."""
        from codex_loop.measure.regression import detect_regressions
        from codex_loop.measure.harbor_runner import TrialResult
        
        baseline = [
            TrialResult(task="task-1", reward=1.0, trajectory=None),
            TrialResult(task="task-2", reward=1.0, trajectory=None),
            TrialResult(task="task-3", reward=0.5, trajectory=None),
        ]
        
        current = [
            TrialResult(task="task-1", reward=1.0, trajectory=None),  # No change
            TrialResult(task="task-2", reward=0.0, trajectory=None),  # Regression!
            TrialResult(task="task-3", reward=0.8, trajectory=None),  # Improvement
        ]
        
        regressions = detect_regressions(baseline, current, threshold=0.1)
        
        assert len(regressions) == 1
        assert regressions[0].task == "task-2"
        assert regressions[0].delta == 1.0
    
    def test_no_regressions(self):
        """Test when there are no regressions."""
        from codex_loop.measure.regression import detect_regressions
        from codex_loop.measure.harbor_runner import TrialResult
        
        baseline = [
            TrialResult(task="task-1", reward=0.5, trajectory=None),
        ]
        
        current = [
            TrialResult(task="task-1", reward=0.6, trajectory=None),  # Improvement
        ]
        
        regressions = detect_regressions(baseline, current, threshold=0.1)
        
        assert len(regressions) == 0


class TestSignalCategories:
    """Test that signal categories match brainstorm-updated.md."""
    
    def test_friction_categories_exist(self):
        """Verify expected friction categories are defined."""
        from codex_loop.signals.types import SignalCategory
        
        expected_friction = [
            "error_event",
            "exec_failure",
            "denial_exec",
            "denial_patch",
            "repeated_denial",
            "undo",
            "rollback",
            "compaction",
            "repeated_compaction",
        ]
        
        for category in expected_friction:
            # Check category exists (case-insensitive)
            found = any(
                category.lower() in c.value.lower()
                for c in SignalCategory
            )
            assert found, f"Missing friction category: {category}"
    
    def test_delight_categories_exist(self):
        """Verify expected delight categories are defined."""
        from codex_loop.signals.types import SignalCategory
        
        expected_delight = [
            "fast_completion",
            "first_attempt_success",
            "zero_denial",
        ]
        
        for category in expected_delight:
            found = any(
                category.lower() in c.value.lower()
                for c in SignalCategory
            )
            assert found, f"Missing delight category: {category}"


@pytest.mark.asyncio
async def test_full_pipeline_integration(tmp_path):
    """
    Integration test for the full pipeline.
    
    Tests: ingest -> signals -> database storage
    """
    # Set up fake CODEX_HOME
    codex_home = tmp_path / ".codex"
    trace_dir = codex_home / "trace_spine" / "test-friction-001"
    trace_dir.mkdir(parents=True)
    shutil.copy(
        FIXTURES_DIR / "friction_repeated_denial.jsonl",
        trace_dir / "segment-000.jsonl"
    )
    
    # Set up database
    db_path = tmp_path / "test.db"
    init_db(db_path)
    db = get_session(db_path)
    
    # Ingest
    records = parse_spine(trace_dir)
    session = canonicalize_session(records, trace_dir)
    db.add(session)
    db.commit()
    
    # Verify session
    stored_session = db.query(Session).filter(Session.id == "test-friction-001").first()
    assert stored_session is not None
    assert stored_session.source == "CLI"
    
    # Detect signals
    signals = detect_mechanical_signals(stored_session, records)
    
    for sig in signals:
        signal_obj = Signal(
            id=f"sig_{stored_session.id}_{len(db.query(Signal).all())}",
            session_id=stored_session.id,
            turn_id=sig.turn_id,
            signal_type=sig.signal_type.value,
            category=sig.category.value if hasattr(sig.category, 'value') else str(sig.category),
            severity=sig.severity,
            evidence=sig.to_evidence_dict(),
            detector="mechanical",
        )
        db.add(signal_obj)
    
    stored_session.signals_computed = True
    db.commit()
    
    # Verify signals
    stored_signals = db.query(Signal).all()
    assert len(stored_signals) > 0
    
    friction_signals = [s for s in stored_signals if s.signal_type == "friction"]
    assert len(friction_signals) > 0
