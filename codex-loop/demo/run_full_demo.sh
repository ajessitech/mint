#!/bin/bash
# Full Demo Script for Codex Closed Loop
# Run this script for the hackathon demo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Codex Closed Loop - Self-Improving AI Coding Agent     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Check if demo data exists
if [ ! -d "./demo_codex_home/trace_spine" ]; then
    echo -e "${YELLOW}Generating 500 synthetic sessions...${NC}"
    python scripts/generate_demo_data.py --output ./demo_codex_home/trace_spine --count 500
    echo ""
fi

# Check if database exists
if [ ! -f "./demo.db" ]; then
    echo -e "${YELLOW}Running analysis pipeline...${NC}"
    python -m codex_loop.cli daily \
        --codex-home ./demo_codex_home \
        --db ./demo.db \
        --lookback-hours 1000 \
        --skip-semantic \
        --skip-clustering \
        --verbose
    echo ""
fi

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STAGE 1-3: Trace Spine Capture & Ingestion${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Every Codex session is automatically captured as a trace spine."
echo "Trace spines contain: user requests, tool calls, approvals, errors, etc."
echo ""
echo "📁 Trace spine directory:"
echo "   $(ls -d ./demo_codex_home/trace_spine/*/ | wc -l | tr -d ' ') sessions captured"
echo ""

sleep 2

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STAGE 4: Signal Detection & Analysis${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python -m codex_loop.cli report --db ./demo.db

sleep 2

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STAGE 5: Distill Friction into Harbor Task${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Get a friction signal to distill
SIGNAL_ID=$(python -c "
from codex_loop.db import get_session, Signal
db = get_session('./demo.db')
sig = db.query(Signal).filter(Signal.signal_type == 'friction', Signal.category == 'exec_failure').first()
print(sig.id if sig else '')
")

if [ -n "$SIGNAL_ID" ]; then
    echo "Distilling signal: $SIGNAL_ID"
    rm -rf ./demo_harbor_task
    python -m codex_loop.cli distill \
        --db ./demo.db \
        --signal-id "$SIGNAL_ID" \
        --output ./demo_harbor_task
    
    echo ""
    echo "📦 Generated Harbor task structure:"
    ls -la ./demo_harbor_task/
    echo ""
    echo "📄 Task instruction:"
    echo "────────────────────"
    cat ./demo_harbor_task/instruction.md
    echo ""
else
    echo "No friction signals found"
fi

sleep 2

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}SUMMARY: The Closed Loop${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  ┌─────────────────┐"
echo "  │  Codex Session  │"
echo "  └────────┬────────┘"
echo "           │"
echo "           ▼"
echo "  ┌─────────────────┐"
echo "  │  Trace Spine    │ ◄── Automatic capture"
echo "  └────────┬────────┘"
echo "           │"
echo "           ▼"
echo "  ┌─────────────────┐"
echo "  │ SQLite+ChromaDB │ ◄── Canonical store"
echo "  └────────┬────────┘"
echo "           │"
echo "           ▼"
echo "  ┌─────────────────┐"
echo "  │ Signal Detection│ ◄── Friction & Delight"
echo "  └────────┬────────┘"
echo "           │"
echo "           ▼"
echo "  ┌─────────────────┐"
echo "  │  Harbor Tasks   │ ◄── Reproducible evals"
echo "  └────────┬────────┘"
echo "           │"
echo "           ▼"
echo "  ┌─────────────────┐"
echo "  │  Measured Fix   │ ◄── Close the loop!"
echo "  └─────────────────┘"
echo ""
echo -e "${BLUE}Every session becomes data. Every friction becomes an eval.${NC}"
echo -e "${BLUE}Every fix gets measured. That's the closed loop.${NC}"
echo ""
