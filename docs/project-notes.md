# Project notes

Evergreen, project-level notes and scratch space. **Not a decision record** — anything here is non-authoritative; only `DECISIONS.md` is binding. (This file absorbs the role of the former root-level `NOTES.md`.)

## Documentation conventions — where information lives

| Kind of information | Home |
|---|---|
| Architectural decisions | `DECISIONS.md` (current value) + `CHANGELOG.md` (amendment history), same commit |
| Magic numbers / thresholds / formulas | `constants.py` (frozen dataclasses) |
| Task → which files to load | `INDEX.md` |
| Per-domain detail for a coding task | the matching context card (`*.md` at repo root) |
| v2 failure analysis, rationale, build-order/planning record | `docs/post-mortem.md` (frozen; never loaded in coding sessions) |
| Per-session activity | `docs/sessions/` |
| Open questions, empirical observations not yet earning a `DECISIONS.md` line, reminders, half-formed alternatives | this file |

Things that **do not** go here: architectural decisions (→ `DECISIONS.md` + `CHANGELOG.md`), constants (→ `constants.py`), per-session activity (→ `docs/sessions/`).

## Open

(empty — populate as questions arise)
