# End-to-End Guide: Codex Closed Loop System

This guide walks through the complete closed loop workflow from capturing Codex sessions to generating improvement tickets.

## System Overview

The closed loop system consists of two packages working together:

```
┌──────────────────────────────────────────────────────────────┐
│                    codex-rs (Rust)                           │
│  Captures all Codex session activity into trace spines       │
│  Location: ~/.codex/trace_spine/<session-id>/                │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   codex-loop (Python)                        │
│  Analyzes traces → Detects signals → Generates Harbor tasks  │
│  → Measures improvements → Creates tickets                   │
└──────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Codex CLI** (generates trace spines)
2. **Python 3.11+** with codex-loop installed
3. **OpenAI API key** (for semantic signals and embeddings)
4. **Harbor** (optional, for evaluation)

### Install codex-loop

```bash
cd codex-loop
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Set API key
export OPENAI_API_KEY="sk-..."
```

## Stage 1: Generate Trace Spines (Automatic)

When you use Codex, trace spines are automatically captured:

```bash
# Normal Codex usage generates traces
codex "Create a Python hello world script"
codex "Fix the bug in my authentication code"

# Traces are stored in:
ls ~/.codex/trace_spine/
```

Each session creates a directory with `segment-*.jsonl` files containing:
- Session metadata
- User submissions (SQ)
- Agent events (EQ)
- Compaction boundaries
- Artifact references (git info, diffs)

## Stage 2: Initialize Database

```bash
# Initialize SQLite database
codex-loop init --db ./codex_loop.db
```

## Stage 3: Ingest Trace Spines

```bash
# Ingest all traces from the last 24 hours
codex-loop ingest --codex-home ~/.codex --db ./codex_loop.db

# Or ingest everything
codex-loop ingest --codex-home ~/.codex --db ./codex_loop.db --lookback-hours 8760
```

## Stage 4: Detect Signals

```bash
# Detect mechanical signals (rule-based)
codex-loop signals --db ./codex_loop.db

# Or run the full daily pipeline (includes semantic signals)
codex-loop daily --codex-home ~/.codex --db ./codex_loop.db
```

### Signal Types Detected

**Friction signals:**
- `repeated_denial` - User denied approval 3+ times
- `exec_failure` - Commands exited with non-zero code
- `compaction` - Context hit limits and was compacted
- `undo`, `rollback` - User backtracked
- `escalation_tone` - User showed frustration (semantic)

**Delight signals:**
- `fast_completion` - Task done in ≤3 turns
- `zero_denial` - All approvals granted
- `positive_feedback` - User expressed satisfaction (semantic)

## Stage 5: View Analysis Report

```bash
codex-loop report --db ./codex_loop.db
```

Output:
```
┌────────────────────────────────┐
│ Analysis Summary               │
├────────────────────────────────┤
│ Sessions: 42 total, 42 processed│
│ Signals: 15 friction, 8 delight │
│ Clusters: 3                     │
└────────────────────────────────┘

Friction Signals by Category
┌─────────────────┬───────┐
│ Category        │ Count │
├─────────────────┼───────┤
│ repeated_denial │ 5     │
│ exec_failure    │ 4     │
│ compaction      │ 3     │
└─────────────────┴───────┘
```

## Stage 6: Cluster Signals

```bash
# Generate embeddings and cluster similar signals
codex-loop cluster --db ./codex_loop.db --vectors ./vectors
```

This uses HDBSCAN to find patterns across friction/delight signals.

## Stage 7: Schema Evolution

When clusters form, the system can propose new signal categories:

```bash
# List pending proposals
codex-loop schema-proposals --db ./codex_loop.db

# Approve a proposal (promotes to stable category)
codex-loop schema-approve --db ./codex_loop.db --proposal-id proposal_abc123

# Reject a proposal
codex-loop schema-reject --db ./codex_loop.db --proposal-id proposal_abc123 --reason "Too vague"

# Auto-approve high-confidence proposals
codex-loop schema-auto-promote --db ./codex_loop.db --confidence 0.85 --min-samples 10

# Export stable categories to JSON
codex-loop schema-export --db ./codex_loop.db --output schema.json
```

## Stage 8: Distill to Harbor Tasks

```bash
# Distill a specific signal to a Harbor task
codex-loop distill --db ./codex_loop.db --signal-id sig_123 --output ./harbor_task

# Or distill a whole cluster to a dataset
codex-loop distill --db ./codex_loop.db --cluster-id cluster_456 --output ./harbor_dataset
```

The generated task includes:
- `instruction.md` - User request
- `task.toml` - Metadata (signal_type, cluster_id, etc.)
- `environment/Dockerfile` - Environment setup
- `tests/test.sh` - Verifier script

## Stage 9: Run Harbor Evaluation

```bash
# Evaluate with Harbor
codex-loop measure --dataset ./harbor_dataset --agent codex --model gpt-5.2

# Check for regressions against baseline
codex-loop measure --dataset ./harbor_dataset --agent codex --model gpt-5.2 --baseline ./baseline.json
```

## Stage 10: Generate Tickets

```bash
# Generate a ticket from a cluster
codex-loop ticket --db ./codex_loop.db --cluster-id cluster_456 --output ticket.md
```

Output includes:
- Problem description
- Reproduction steps
- Evidence pointers (session IDs, turn IDs)
- Harbor dataset link for validation
- Suggested fix approach

## Stage 11: Track Trends

```bash
# View evaluation trends over time
codex-loop trends --db ./codex_loop.db --dataset harbor_dataset_001 --days 30
```

## Complete Daily Workflow

For production use, run the daily pipeline:

```bash
codex-loop daily \
  --codex-home ~/.codex \
  --db ./codex_loop.db \
  --lookback-hours 24 \
  --verbose
```

This runs all stages automatically:
1. Find new spines
2. Ingest into SQLite
3. Detect mechanical signals
4. Detect semantic signals (with OpenAI)
5. Generate embeddings
6. Update clusters
7. Generate schema proposals
8. Print report

## Rust CLI Integration

The codex-rs CLI also provides trace-related commands:

```bash
# Export a rollout+ bundle
codex trace bundle --session-id <uuid> --output ./bundle

# Distill to Harbor task (with optional signal metadata)
codex trace distill \
  --bundle ./bundle \
  --output ./task \
  --start-seq 10 \
  --end-seq 50 \
  --signal-type friction \
  --cluster-id cluster_123 \
  --signal-category repeated_denial

# Package into Harbor dataset
codex trace dataset --task ./task1 --task ./task2 --output ./dataset

# Export to ATIF v1.4 format
codex trace atif --bundle ./bundle --output ./atif
```

## Testing the System

### Run Unit Tests

```bash
cd codex-loop
pytest tests/test_e2e_pipeline.py -v
```

### Run with Fixtures

```bash
# Use bundled test fixtures
python demo.py --use-fixtures --skip-openai --output ./demo_output
```

### Run with Real Data

```bash
# Use your actual Codex traces
python demo.py --codex-home ~/.codex --output ./demo_output
```

## Troubleshooting

### No traces found
- Check `ls ~/.codex/trace_spine/` for session directories
- Each session needs `segment-*.jsonl` files

### Signal detection fails
- Ensure spine records have correct schema_version
- Check that session_meta record exists

### Clustering fails
- Need at least 5 signals of each type
- Check HDBSCAN is installed: `pip install hdbscan`

### Harbor not found
- Install with: `uv tool install harbor`

## Data Model

### SQLite Tables

| Table | Description |
|-------|-------------|
| sessions | Session metadata and facets |
| turns | Per-turn context with policies |
| tool_calls | Tool calls with approval state |
| signals | Friction/delight signals |
| clusters | Signal clusters |
| schema_proposals | Proposed new categories |
| eval_runs | Harbor evaluation results |

### ChromaDB Collections

| Collection | Description |
|------------|-------------|
| signal_embeddings | Signal embedding vectors |
| session_embeddings | Session summary embeddings |
| cluster_centroids | Cluster center vectors |

## Best Practices

1. **Run daily pipeline** - Keeps analysis current
2. **Review schema proposals** - Human oversight improves quality
3. **Use clusters for datasets** - Better than individual signals
4. **Track baseline regressions** - Catch issues early
5. **Export stable schema** - Version control your signal definitions
