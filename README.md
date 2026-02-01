# Mint: Codex Closed-Loop System

A self-improving feedback loop for AI coding agents that automatically captures sessions, detects friction patterns, and converts them into reproducible evaluation tasks.

## Repository Structure

- **`codex-loop/`** - Python implementation of the closed-loop pipeline (signal detection, Harbor task generation, trend tracking)
- **`codex/`** - Submodule reference to the Codex CLI fork with trace spine integration
- **`brainstorm-updated.md`** - System design document (source of truth for architecture)

## Codex Changes

The Codex-related changes (trace spine capture, session recording) can be found in the forked repository:

**https://github.com/ajessitech/codex**

Key additions:
- `codex-rs/core/src/trace_spine.rs` - Trace spine implementation
- `codex-rs/cli/src/trace_cmd.rs` - CLI command for trace operations
- `codex-rs/core/src/atif_export.rs` - ATIF export functionality

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/ajessitech/mint.git
cd mint

# Install codex-loop
cd codex-loop
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run the demo
export OPENAI_API_KEY="sk-..."
python demo_video.py
```

## How It Works

```
Codex Session → Trace Spine → SQLite + ChromaDB → Signal Detection → Harbor Task
                                                         ↓
                                                  Measured Improvement
                                                         ↓
                                                   Loop Closes ←──────┘
```

1. **Capture** - Codex sessions are recorded to trace spine format
2. **Ingest** - Sessions are canonicalized and stored in SQLite + vector DB
3. **Detect** - Mechanical and semantic signals identify friction/delight patterns
4. **Distill** - High-impact friction is converted to Harbor evaluation tasks
5. **Measure** - Tasks are run to establish baselines and track improvements

---

*Built for OpenAI Hackathon 2026 — Track: Agentic Software Engineering with Codex*
