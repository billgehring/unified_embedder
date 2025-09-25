# Project Roadmap

## Vision
- Deliver a platform-agnostic embedding pipeline that ingests full course websites (text, slides, audio/video) and serves ultra-fast retrieval for the Socrating tutor across macOS laptops and Ubuntu/NVIDIA workstations.
- Provide multi-modal context (text + visuals) so the tutor can reason about lecture videos, slide imagery, and supplemental documents and respond with Socratic guidance.
- Maintain production reliability with transparent monitoring, reproducible runs, and tight alignment with the `shared-rag-service` consumer.

## Baseline (2025-03)
- Dense + sparse + ColBERT token retrieval with adaptive weighting and reranking (`unified_embedder.py`, `hybrid_qdrant_store.py`, `reranking.py`).
- Enhanced Docling converter with OCR quality checks (macOS Vision API, Tesseract, RapidOCR, EasyOCR) and 1024-token hybrid chunking.
- Performance tooling (`performance_benchmark.py`, `performance_monitor.py`, `monitored_retrieval.py`) targeting <50 ms voice responses.
- Qdrant-first deployment with scripts for Docker startup (`start_qdrant_docker.sh`), connectivity diagnostics, and GPU readiness tests.
- Documentation spread across `PROJECT_OVERVIEW.md`, `CLAUDE.md`, and ad-hoc notes.

## Strategic Themes

### 1. Multi-Modal Ingestion & Curation
- Build automated lecture video transcription and alignment pipeline (LMS exports, YouTube/Vimeo links, Panopto/Leccap) with speaker diarization and slide timestamping.
- Capture slide imagery, diagrams, and assignment screenshots; embed with CLIP/vision transformers and link to source chunks.
- Normalize Canvas exports, web archives, and scraped HTML into a consistent metadata schema with provenance for cross-course analytics.

### 2. Retrieval Quality & Evaluation
- Extend ColBERT and reranker stages to blend textual and visual embeddings; evaluate cross-modal query coverage.
- Introduce curriculum-aware scoring (unit/week, learning objectives) and difficulty metadata for Socratic prompting.
- Standardize evaluation harness: scripted benchmark suites, regression dashboards, and human-in-the-loop review flows for new collections.

### 3. Platform & Performance Readiness (macOS ↔ Ubuntu/NVIDIA)
- Package dependency stacks with correct platform markers (MPS vs CUDA builds) and automate hardware detection in setup scripts.
- Harden shell entrypoints (`run_embedder.sh`, `start_qdrant_docker.sh`, `test_retrieval.sh`) for `/usr/bin/env bash`, `.env` overrides, and non-Interactive CI runs.
- Expand performance optimizer to profile multi-GPU, MIG, and CPU-only fallbacks; expose recommended settings in generated reports.

### 4. Tutoring System Integration
- Encode collection naming/versioning conventions shared with `../shared-rag-service` and expose health checks before publishing.
- Provide schema exporters and dataset manifests consumable by downstream RAG services and analytics pipelines.
- Deliver prompt-ready context packages (text + image references) tuned for Socrating tutor personas (instructor, TA, student).

### 5. Operations & Governance
- Consolidate roadmap, task tracker, decision log, and docs index to replace CLAUDE-specific guidance.
- Instrument logging for privacy redaction, audit trails, and automated alerts routed to tutor ops dashboards.
- Document rollback, reprocessing, and sampling procedures for high-stakes tutoring domains (psychology, neuroscience).

## Initiative Timeline

| Horizon | Focus | Key Deliverables | Linked Tasks |
|---------|-------|------------------|--------------|
| **Now** (≤1 month) | Platform hygiene & documentation | Cross-platform shell fixes, dependency markers, roadmap/task tracker rollout, doc index | TT-001, TT-002, TT-003, TT-004, TT-011 |
| **Next** (1–3 months) | Multi-modal ingestion groundwork | Video transcript ingestion MVP, slide/image embedding pipeline, metadata normalization spec | TT-005, TT-006, TT-008, TT-009 |
| **Later** (3–6 months) | Retrieval excellence & tutor alignment | Multi-modal reranking, curriculum-aware scoring, evaluation dashboards, automated collection publishing | TT-007, TT-010, TT-012 |
| **Explore** (6+ months) | Advanced tutoring experiences | Real-time lecture syncing, generative augmentation, adaptive Socratic dialog planning | TT-013, TT-014 |

## Operational Reference (from legacy CLAUDE.md)
- **Environment setup**: `uv sync` then `uv run -m spacy download en_core_web_sm en_core_web_trf` (add `en_core_web_lg` on macOS if Vision OCR is enabled).
- **Embed course content**: `uv run unified_embedder.py --docs_dir <path> --qdrant --qdrant_collection <name> --qdrant_url http://localhost:6333 --enable_colbert_tokens --colbert_model colbert-ir/colbertv2.0` plus `--force_ocr`, `--rerank`, and `--enable_dedup` as needed.
- **Retrieval diagnostics**: `uv run test_retrieval.py --collection <name> --multi_vector --adaptive_weights --show_timing` and `./test_retrieval.sh <name>`.
- **Performance checks**: `uv run performance_benchmark.py --collection <name> --concurrent_users 50 --queries_per_user 10`; monitor via `uv run monitored_retrieval.py`.
- **GPU readiness (Ubuntu)**: `uv run test_gpu_setup.py` to validate drivers, CUDA, NVLink, and optimizer recommendations.
- **Qdrant**: start via `docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant`; validate with `uv run test_qdrant_connectivity.py`.

## Cross-Project Alignment
- `shared-rag-service` consumes dense/sparse/ColBERT collections generated here—coordinate `QDRANT_URL`, embedding models, and collection naming.
- Publish collection manifests with version + feature flags so downstream services can detect reranker, ColBERT, and multi-modal availability.
- Mirror retrieval regression suites between repositories (`test_retrieval.py` ↔ `test_retrieval_service.py`).

## Maintenance Notes
- Update this roadmap whenever major initiatives shift, new tasks are added to `TASK_TRACKER.md`, or architecture capabilities change.
- Deprecate CLAUDE-specific instructions by copying relevant updates here first, then leaving a forward link in `CLAUDE.md`.
