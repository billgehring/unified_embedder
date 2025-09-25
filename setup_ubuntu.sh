#!/usr/bin/env bash
set -euo pipefail

echo "🔎 Cleaning old venv and caches..."
rm -rf .venv __pycache__ .pytest_cache .mypy_cache .coverage htmlcov

echo "🐍 Ensuring correct Python version..."
# Install the pinned Python version if .python-version exists
uv python install || true

echo "📦 Creating fresh virtual environment..."
uv venv

echo "🔄 Syncing dependencies from lockfile..."
uv sync

echo "✅ Done! You can now run 'uv run <cmd>' to start your app."
