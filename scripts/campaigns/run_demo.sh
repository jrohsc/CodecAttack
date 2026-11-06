#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

PORT="${1:-3300}"

echo "Building demo data..."
python build_demo.py

echo ""
echo "Serving demo at http://$(hostname):${PORT}"
echo "Press Ctrl+C to stop."
python -m http.server "$PORT" --directory demo/
