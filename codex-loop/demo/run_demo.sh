#!/bin/bash
# Codex Closed Loop Demo Script
# 
# This script demonstrates the complete closed loop system.
# Run from the codex-loop directory.

set -euo pipefail

echo "=========================================="
echo "Codex Closed Loop Demo"
echo "=========================================="
echo

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Run this script from the codex-loop directory"
    exit 1
fi

# Create output directory
OUTPUT_DIR="./demo_output_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo

# Option 1: Use bundled fixtures (no API key needed for basic demo)
echo "=== Demo 1: Using Test Fixtures (No API Key Needed) ==="
echo

python demo.py --use-fixtures --skip-openai --output "$OUTPUT_DIR/fixtures_demo"

echo
echo "Demo 1 complete. Check $OUTPUT_DIR/fixtures_demo/"
echo

# Option 2: If OPENAI_API_KEY is set, run full demo
if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "=== Demo 2: Full Demo with OpenAI (Embeddings + Semantic Signals) ==="
    echo
    
    python demo.py --use-fixtures --output "$OUTPUT_DIR/full_demo"
    
    echo
    echo "Demo 2 complete. Check $OUTPUT_DIR/full_demo/"
    echo
else
    echo "Skipping Demo 2 (set OPENAI_API_KEY for full demo)"
fi

# Option 3: If real traces exist, use them
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
if [ -d "$CODEX_HOME/trace_spine" ] && [ -n "$(ls -A "$CODEX_HOME/trace_spine" 2>/dev/null)" ]; then
    echo "=== Demo 3: Using Real Codex Traces ==="
    echo "Found traces in $CODEX_HOME/trace_spine"
    echo
    
    python demo.py --codex-home "$CODEX_HOME" --skip-openai --output "$OUTPUT_DIR/real_traces_demo"
    
    echo
    echo "Demo 3 complete. Check $OUTPUT_DIR/real_traces_demo/"
else
    echo "Skipping Demo 3 (no real traces found in $CODEX_HOME/trace_spine)"
fi

echo
echo "=========================================="
echo "All demos complete!"
echo "=========================================="
echo
echo "Generated artifacts in: $OUTPUT_DIR"
echo
echo "Next steps:"
echo "  1. Review generated_ticket.md files"
echo "  2. Check harbor_task/ directories"
echo "  3. Run: harbor run -p $OUTPUT_DIR/*/harbor_task -a codex -m gpt-4"
echo
