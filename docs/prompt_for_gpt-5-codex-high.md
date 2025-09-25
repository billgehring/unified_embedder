# Prompt
We want to optimize and enhance this project so that it serves as the ultimate embedder for creating a RAG-based tutoring system for college courses. The system will be used to embed undergraduate course websites in psychology (which include lecture videos) for the in-progress socrating tutoring system with visual as well as verbal input and output. The tutor can present visual images and text to a student and ask for explanations, and the student can present images and text to the tutor to request socratic tutoring. 1. please inspect the current project for issues and things that could be further optmized for use on Ubuntu/Nvidia as well as MacOS. 2.  Please identify potential project roadmap goals that would make this system better for embedding in the fashion describe above.  3. please create a PROJECT_ROADMAP.md and TASK_TRACKER.md files and add instructions to keep them updated to your AGENTS.md file. We are migrating away from Claude Code, so please incorporate any useful information from CLAUDE.md into these other files. 4. If there are any other project management tools you favor, please add and integrate them in this project. 5. Please organize the project's documents, scripts, and folders to be most accessible for you.

# result

> Created a consolidated planning stack (roadmap, tracker, decision log) and relocated legacy docs so planning now lives in agent-friendly files instead
of CLAUDE.md.

Updates

- Added PROJECT_ROADMAP.md (lines 1-66) capturing vision, strategic themes, timelines, and operational commands distilled from CLAUDE.md.
- Added TASK_TRACKER.md (lines 1-38) with usage rules plus an initial backlog of TT-001…TT-014 tailored to the psychology tutor scenario.
- Introduced docs/DECISION_LOG.md (lines 1-17) and docs/DOCS_INDEX.md (lines 1-19) to manage decisions and document discoverability; moved the
historical Claude evaluation report to docs/legacy/claude_emedding_evaluation.md.
- Updated AGENTS.md:50-53 to require keeping the roadmap, task tracker, decision log, and docs index current; CLAUDE.md:1-4 now points readers to the
new artifacts.
- Refreshed README.md:1-22 with a quickstart and pointers to the new documentation stack.

Key Issues Found

- start_qdrant_docker.sh:1-3 and test_retrieval.sh:1-40 hard-code the Homebrew bash path and assume interactive use; they need /usr/bin/env bash,
safety flags, and portable defaults.
- pyproject.toml:17 forces installation of ocrmac, which breaks uv sync on Ubuntu/NVIDIA nodes; should be wrapped in macOS-only extras.
- setup_ubuntu.sh:4-5 deletes .venv and caches on every run—dangerous on shared machines; consider prompting before destructive cleanup.
- run_embedder.sh:17-188 still anchors to Dropbox paths and legacy BAAI/bge-large-en-v1.5 defaults, creating drift from the new BGE-M3 baseline.
- unified_embedder.py:386-401 retains an unused perform_ocr() stub with a TODO, despite the real OCR flow living in enhanced_docling_converter.py.

New Roadmap & Task Artifacts

- Strategic goals and multi-modal tutoring focus are in PROJECT_ROADMAP.md; task IDs there map directly to TASK_TRACKER.md.
- Decision history begins in docs/DECISION_LOG.md; doc locations are summarized in docs/DOCS_INDEX.md.

Next Steps

1. Tackle TT-001/TT-002 to harden shell entrypoints and platform-specific dependencies.
2. Align run_embedder.sh with the new defaults (TT-004) and remove the OCR stub (TT-003).
3. Start planning TT-005/TT-006 for lecture video and visual embedding ingestion.

No tests were run; docs-only changes. Let me know when you want to dive into any of the backlog items.
