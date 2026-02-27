#!/usr/bin/env bash
# run_tests.sh — run the full test suite
# Usage: ./run_tests.sh [optional unittest args]
# When pytest is installed: pytest tests/ -v
set -e
cd "$(dirname "$0")"

if command -v pytest &>/dev/null; then
    pytest tests/ -v "$@"
else
    echo "Running with stdlib unittest (install pytest for richer output)"
    python3 -m unittest discover -s tests/unit -v "$@"
fi
