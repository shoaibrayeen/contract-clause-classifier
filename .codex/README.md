# Codex

OpenAI Codex reads the repository rules from `AGENTS.md` at the project root automatically.
There is no separate Codex configuration; edit `AGENTS.md` to change the rules.

Rules that matter most for Codex sessions:

- Run everything with `uv run`; never start full training (`make train`) unless asked, use `make sanity`.
- Keep the API OpenAI-compatible and the output shape `[{name, confidence}]`.
- `make test` must pass before you finish.
- Always update `MEMORY.md` (decisions, measurements, gotchas, dated) and `CHANGELOG.md` before reporting a task complete.
