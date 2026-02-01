# Codex Closed Loop

**Automatically captures Codex sessions, detects friction patterns (failures, denials, confusion), and converts them into reproducible evaluation tasks—creating a self-improving feedback loop for AI coding agents.**

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/codex-closed-loop.git
cd codex-closed-loop
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Set OpenAI API key (for semantic analysis)
export OPENAI_API_KEY="sk-..."

# 3. Run the demo
python demo_video.py
```

## Demo Steps

Run `python demo_video.py` and press **ENTER** to advance through each stage:

1. **Problem** → Shows why AI agents need systematic improvement
2. **Solution** → Architecture diagram of the closed loop
3. **Data** → 500 synthetic Codex sessions captured
4. **Signals** → 1,736 friction/delight patterns detected (bar charts)
5. **Issues** → Top problems ranked by impact score
6. **Harbor Task** → Friction converted to reproducible eval task
7. **Closing** → The value: every session → data → eval → measured fix

Total runtime: ~2 minutes with narration pauses.

## What Gets Detected

| Friction (Problems) | Delight (Success) |
|---------------------|-------------------|
| Command failures | Fast completion (≤3 turns) |
| Repeated denials | Zero denials |
| Context truncation | First-attempt success |
| Patch failures | Positive feedback |
| Timeouts & errors | |

## Architecture

```
Codex Session → Trace Spine → SQLite + ChromaDB → Signal Detection → Harbor Task
                                                         ↓
                                                  Measured Improvement
                                                         ↓
                                                   Loop Closes ←──────┘
```

## Tech Stack

- **Python 3.11+** with SQLite + ChromaDB for local analysis
- **OpenAI GPT-4o** for semantic signal detection
- **OpenAI text-embedding-3-small** for clustering similar issues
- **Harbor** framework for reproducible agent evaluations

---

*Built for OpenAI Hackathon 2026 — Track: Agentic Software Engineering with Codex*
