# Codex Closed Loop

**Captures Codex sessions, detects friction patterns, and converts them into reproducible evaluation tasks—enabling recursive self-improvement for AI coding agents.**

## How to Run

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/codex-closed-loop.git
cd codex-closed-loop
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Set OpenAI API key (optional, for semantic analysis)
export OPENAI_API_KEY="sk-..."
```

## Demo Steps

```bash
# 1. Run analysis on your Codex sessions
codex-loop daily --codex-home ~/.codex --db analysis.db --skip-semantic

# 2. View the friction/delight report
codex-loop report --db analysis.db

# 3. Distill a friction pattern into a Harbor evaluation task
codex-loop distill --db analysis.db --signal-id <signal-id> --output ./harbor_task
```

**What you'll see:**
- Sessions ingested from `~/.codex/trace_spine/`
- Friction signals: exec failures, repeated denials, context truncation
- Delight signals: fast completion, zero denials, first-attempt success
- Harbor task with `instruction.md`, `Dockerfile`, and `verify.sh`

## Architecture

```
Codex Session → Trace Spine → Analysis Pipeline → Signals → Harbor Task
       ↑                                                         ↓
       └─────────────── Measured Fix ◄── Evaluation ◄───────────┘
```

**The loop:** Every session becomes data. Every friction becomes an eval. Every fix gets measured.

## Signal Types

| Friction | Delight |
|----------|---------|
| exec_failure | fast_completion |
| repeated_denial | zero_denial |
| compaction | first_attempt_success |
| patch_failure | positive_feedback |
| timeout | |

## Tech Stack

- Python 3.11+ / SQLite / ChromaDB
- OpenAI GPT-5.2 via Responses API (semantic signal detection)
- OpenAI text-embedding-3-small (clustering)
- Harbor (reproducible agent evaluation)

---

*OpenAI Hackathon 2026 — Agentic Software Engineering with Codex*
