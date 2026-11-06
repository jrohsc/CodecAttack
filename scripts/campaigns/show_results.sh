#!/bin/bash
# Show attack results summary for results_qa and results_transcribe
# Usage: bash show_results.sh [--category]  (add --category for per-category breakdown)

cd "$(dirname "$0")"
python3 summarize_results.py "$@"
