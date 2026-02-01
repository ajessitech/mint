"""
SQLAlchemy models for the Codex Closed Loop canonical store.

Implements the schema from brainstorm-updated.md for sessions, turns,
tool calls, signals, clusters, and eval runs.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import json

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    JSON,
    Index,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
    Session as DBSession,
)
from sqlalchemy.engine import Engine

Base = declarative_base()


class Session(Base):
    """
    Canonical session record from trace spine.
    
    Maps to: brainstorm-updated.md lines 711-729 (session-level fields)
    """
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)  # thread_id
    source = Column(String, nullable=False)  # CLI, VSCode, SubAgent, etc.
    model_provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    cwd = Column(String, nullable=True)
    git_repo_url = Column(String, nullable=True)
    git_commit = Column(String, nullable=True)
    git_branch = Column(String, nullable=True)
    cli_version = Column(String, nullable=True)
    originator = Column(String, nullable=True)
    
    # Computed metrics
    compaction_count = Column(Integer, default=0)
    turn_count = Column(Integer, default=0)
    tool_call_count = Column(Integer, default=0)
    
    # Facets stored as JSON (brainstorm-updated.md lines 93-99)
    facets = Column(JSON, default=dict)
    
    # Processing state
    signals_computed = Column(Boolean, default=False)
    embeddings_computed = Column(Boolean, default=False)
    
    # Raw spine path for reference
    spine_path = Column(String, nullable=True)
    
    # Relationships
    turns = relationship("Turn", back_populates="session", cascade="all, delete-orphan")
    signals = relationship("Signal", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_sessions_started_at", "started_at"),
        Index("ix_sessions_source", "source"),
        Index("ix_sessions_signals_computed", "signals_computed"),
    )


class Turn(Base):
    """
    Per-turn context record.
    
    Maps to: brainstorm-updated.md lines 731-758 (turn-level fields)
    """
    __tablename__ = "turns"

    id = Column(String, primary_key=True)  # turn_id
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    seq_start = Column(Integer, nullable=False)
    seq_end = Column(Integer, nullable=True)
    
    # Policies (critical for facet analysis)
    approval_policy = Column(String, nullable=True)
    sandbox_policy = Column(String, nullable=True)
    model = Column(String, nullable=True)
    
    # User input
    user_message = Column(Text, nullable=True)
    
    # Timestamps
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    
    # Relationships
    session = relationship("Session", back_populates="turns")
    tool_calls = relationship("ToolCall", back_populates="turn", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_turns_session_id", "session_id"),
    )


class ToolCall(Base):
    """
    Tool call record with approval and execution details.
    
    Maps to: brainstorm-updated.md lines 760-791 (tool-call fields)
    """
    __tablename__ = "tool_calls"

    call_id = Column(String, primary_key=True)
    turn_id = Column(String, ForeignKey("turns.id"), nullable=False)
    tool_name = Column(String, nullable=False)
    arguments = Column(Text, nullable=True)
    
    # Execution results
    exit_code = Column(Integer, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    stdout = Column(Text, nullable=True)
    stderr = Column(Text, nullable=True)
    
    # Approval state
    approval_requested = Column(Boolean, default=False)
    approved = Column(Boolean, nullable=True)
    approval_decision = Column(String, nullable=True)  # approved, denied, etc.
    
    # Relationships
    turn = relationship("Turn", back_populates="tool_calls")

    __table_args__ = (
        Index("ix_tool_calls_turn_id", "turn_id"),
        Index("ix_tool_calls_tool_name", "tool_name"),
    )


class Signal(Base):
    """
    Friction or delight signal detected in a session.
    
    Maps to: brainstorm-updated.md lines 109-118 (signal patterns)
    """
    __tablename__ = "signals"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    turn_id = Column(String, nullable=True)
    
    # Signal classification
    signal_type = Column(String, nullable=False)  # "friction" or "delight"
    category = Column(String, nullable=False)  # e.g., "repeated_denial", "fast_completion"
    severity = Column(Float, nullable=False)  # 0.0 to 1.0
    
    # Evidence pointers (brainstorm-updated.md lines 897-901)
    evidence = Column(JSON, default=dict)  # thread_id, turn_ids, call_ids, seq_range
    
    # Embedding link
    embedding_id = Column(String, nullable=True)
    
    # Cluster assignment
    cluster_id = Column(String, ForeignKey("clusters.id"), nullable=True)
    
    # Detection metadata
    detected_at = Column(DateTime, default=datetime.utcnow)
    detector = Column(String, nullable=True)  # "mechanical" or "semantic"
    
    # Relationships
    session = relationship("Session", back_populates="signals")
    cluster = relationship("Cluster", back_populates="signals")

    __table_args__ = (
        Index("ix_signals_session_id", "session_id"),
        Index("ix_signals_signal_type", "signal_type"),
        Index("ix_signals_category", "category"),
        Index("ix_signals_cluster_id", "cluster_id"),
        Index("ix_signals_embedding_id", "embedding_id"),
    )


class Cluster(Base):
    """
    Cluster of similar signals for discovery.
    
    Maps to: brainstorm-updated.md lines 124-133 (hybrid discovery)
    """
    __tablename__ = "clusters"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    signal_type = Column(String, nullable=False)  # "friction" or "delight"
    
    # Embedding reference
    centroid_embedding_id = Column(String, nullable=True)
    
    # Cluster metrics
    member_count = Column(Integer, default=0)
    avg_severity = Column(Float, nullable=True)
    
    # Lifecycle
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Schema evolution
    promoted_to_schema = Column(Boolean, default=False)
    schema_proposal_id = Column(String, ForeignKey("schema_proposals.id"), nullable=True)
    
    # Relationships
    signals = relationship("Signal", back_populates="cluster")
    schema_proposal = relationship("SchemaProposal", back_populates="cluster")

    __table_args__ = (
        Index("ix_clusters_signal_type", "signal_type"),
        Index("ix_clusters_promoted", "promoted_to_schema"),
    )


class SchemaProposal(Base):
    """
    Proposed new schema entry from cluster discovery.
    
    Maps to: brainstorm-updated.md lines 124-133 (schema evolution)
    """
    __tablename__ = "schema_proposals"

    id = Column(String, primary_key=True)
    
    # Proposal content
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    inclusion_criteria = Column(Text, nullable=True)
    severity_rule = Column(Text, nullable=True)
    recommended_action = Column(Text, nullable=True)
    
    # Source
    signal_type = Column(String, nullable=False)
    sample_count = Column(Integer, default=0)
    
    # Lifecycle
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")  # pending, accepted, rejected
    reviewed_at = Column(DateTime, nullable=True)
    
    # Relationships
    cluster = relationship("Cluster", back_populates="schema_proposal", uselist=False)

    __table_args__ = (
        Index("ix_schema_proposals_status", "status"),
    )


class EvalRun(Base):
    """
    Harbor evaluation run record.
    
    Maps to: brainstorm-updated.md lines 364-370 (measurement)
    """
    __tablename__ = "eval_runs"

    id = Column(String, primary_key=True)
    dataset_id = Column(String, nullable=False)
    
    # Eval configuration
    agent = Column(String, nullable=False)
    model = Column(String, nullable=False)
    codex_version = Column(String, nullable=True)
    
    # Results
    task_count = Column(Integer, default=0)
    mean_reward = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=True)
    results_json = Column(JSON, default=list)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Artifacts
    job_dir = Column(String, nullable=True)

    __table_args__ = (
        Index("ix_eval_runs_dataset_id", "dataset_id"),
        Index("ix_eval_runs_started_at", "started_at"),
    )


# Database initialization functions

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine(db_path: Path | str) -> Engine:
    """Get or create database engine."""
    global _engine
    if _engine is None or str(db_path) not in str(_engine.url):
        _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    return _engine


def init_db(db_path: Path | str) -> Engine:
    """Initialize database with all tables."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def get_session(db_path: Path | str) -> DBSession:
    """Get a new database session."""
    engine = get_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session()


def SessionLocal(db_path: Path | str) -> DBSession:
    """Alias for get_session for compatibility."""
    return get_session(db_path)
