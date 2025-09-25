# unified_embedder

## Quickstart
- Install dependencies: `uv sync`
- Download spaCy models (minimum): `uv run -m spacy download en_core_web_sm en_core_web_trf`
- Start Qdrant (local): `./start_qdrant_docker.sh`
- Run the embedder (example):
  ```bash
  uv run unified_embedder.py \
    --docs_dir ./sample_documents \
    --qdrant --qdrant_collection sample \
    --qdrant_url http://localhost:6333 \
    --enable_colbert_tokens \
    --debug
  ```

## Key Documents
- `PROJECT_ROADMAP.md` – Strategic direction, initiatives, and operational reference.
- `TASK_TRACKER.md` – Canonical list of active/queued work items.
- `docs/DECISION_LOG.md` – Historical record of major process/architecture decisions.
- `PROJECT_OVERVIEW.md` – Detailed capabilities, architecture, and dependency catalog.
- `docs/DOCS_INDEX.md` – Full documentation index.

See `AGENTS.md` for repository conventions and agent-specific guidance.
