# Task Tracker

## Usage Guidelines
- Maintain this tracker as the single source of truth for in-flight and planned work (replaces CLAUDE.md task sections).
- Update status after each working session; keep entries in sync with `TodoWrite_Log.md` when using the TodoWrite tool.
- Status values: `Backlog`, `In Progress`, `Blocked`, `Done`.
- Priority values: `P0` (urgent safeguard), `P1` (short-term), `P2` (medium-term), `P3` (research / nice-to-have).
- Include links or references (logs, PRs, docs) in the Notes column when available.

## Active Tasks
| ID | Title | Owner | Status | Priority | Notes |
|----|-------|-------|--------|----------|-------|
| TT-001 | Harden shell entrypoints for macOS/Linux parity (`start_qdrant_docker.sh`, `test_retrieval.sh`, `run_embedder.sh`) | Team | Backlog | P1 | Use `/usr/bin/env bash`, support non-Dropbox paths, add `set -euo pipefail` where safe. |
| TT-002 | Add platform markers / optional installs for macOS-only deps (`ocrmac`, vision extras) | Team | Backlog | P0 | Prevent `uv sync` failure on Ubuntu/NVIDIA nodes; document optional extras in `pyproject.toml`. |
| TT-003 | Retire unused `perform_ocr()` stub from `unified_embedder.py` | Team | Backlog | P2 | Remove confusing placeholder; point developers to `enhanced_docling_converter.py`. |
| TT-004 | Refresh `run_embedder.sh` defaults to BGE-M3 + config profiles | Team | Backlog | P1 | Provide sample configs for Mac (MPS) vs Ubuntu (CUDA); parameterize skip patterns. |
| TT-005 | Lecture video ingestion MVP (transcripts + audio extraction) | Team | Backlog | P1 | Target Leccap/Panopto exports first; align metadata schema with chunk payloads. |
| TT-006 | Slide & image embedding pipeline (CLIP/vision encoder integration) | Team | Backlog | P1 | Generate linked visual embeddings; store pointers in Qdrant payload for tutor rendering. |
| TT-007 | Multi-modal retrieval evaluation harness | Team | Backlog | P2 | Extend `test_retrieval.py` to score text+image queries; produce regression reports. |
| TT-008 | Metadata normalization spec for web + LMS exports | Team | Backlog | P2 | Define canonical fields (unit/week/topic, modality, difficulty); update `hybrid_qdrant_store.py`. |
| TT-009 | GPU/MPS optimization self-check (report generator) | Team | Backlog | P2 | Expand `performance_optimizer` to emit recommended env settings per run. |
| TT-010 | shared-rag-service publishing contract | Team | Backlog | P1 | Automate manifest export + validation before collections go live. |
| TT-011 | Documentation index & consolidation sweep | Team | In Progress | P1 | Move legacy guidance from `CLAUDE.md` into roadmap, tracker, decision log; update `docs/`. |
| TT-012 | Curriculum-aware ranking signals | Research | Backlog | P3 | Model topic sequencing and difficulty to drive Socratic questioning. |
| TT-013 | Real-time lecture sync prototype | Research | Backlog | P3 | Align questions with video timestamps; evaluate streaming ingestion. |
| TT-014 | Adaptive Socratic dialog planning | Research | Backlog | P3 | Explore multi-agent prompt chaining using richer metadata. |

## Recently Completed
| ID | Title | Completed | Notes |
|----|-------|-----------|-------|
| TW-20250109-01 | OCR architecture assessment | 2025-01-09 | See `TodoWrite_Log.md` for detailed findings; confirms enhanced Docling + OCR quality pipeline. |

## Backlog Intake Checklist
1. Confirm task aligns with roadmap theme(s).
2. Capture expected deliverables, owners, and dependencies.
3. Tag related collections / course IDs if applicable.
4. When work finishes, move the row to **Recently Completed** with completion date and artifact links.

