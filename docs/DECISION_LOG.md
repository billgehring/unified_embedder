# Decision Log

## Usage
- Record significant architectural, tooling, or process decisions with enough context for future contributors.
- Each entry should include date, decision, rationale, impacted components, and follow-up tasks (link to `TASK_TRACKER.md`).
- Keep entries in reverse chronological order.

## Entries

### 2025-02-14 — Documentation Consolidation & Planning Stack Refresh
- **Context:** Migrating away from Claude Code instructions; need single-source docs for Codex agents and human maintainers.
- **Decision:** Introduced `PROJECT_ROADMAP.md`, `TASK_TRACKER.md`, and this decision log; moved `claude_emedding_evaluation.md` to `docs/legacy/`; added `docs/DOCS_INDEX.md` for discoverability.
- **Rationale:** Centralize planning artifacts, reduce duplication, and make documentation agent-friendly.
- **Impacted Components:** Documentation tree (`AGENTS.md`, `CLAUDE.md`, `docs/`), task management process.
- **Follow-Up:**
  - TT-001/TT-004/TT-011 to continue documentation consolidation and shell hardening.
  - Monitor for lingering references to CLAUDE-specific workflows during future work.

