# CLAUDE.md

@AGENTS.md

## Claude Code specifics

- Shared agent rules, commands and layout are in AGENTS.md (imported above). Project decisions and gotchas are in @MEMORY.md.
- Long-running processes the user should watch (training, the API server) belong in a Terminal-panel tab, not a background Bash call. Training is the user's to start; give them the command (`make train` or `make all`) unless they ask you to run it.
- `make` runs every step in Docker by default. If Docker is not running, use `RUNNER=local` rather than starting the owner's Colima VM. To verify a training change, run `make sanity RUNNER=local` (about 5 minutes) and inspect `artifacts/runs/sanity/telemetry/`.
- To verify serving without the GPU, use `DEVICE=cpu uv run python scripts/05_serve.py --port 8011` and open `/playground` in the built-in browser. Point `RUNS_DIR` at a scratch folder so tests never touch the user's real `artifacts/runs`.
- Scratch files go in the session scratchpad, not the repo.
- Follow the **Model rules** and **Changelog rules** sections of AGENTS.md: never overwrite or delete a model version, bump `model_version` only through `make bump`, back accuracy claims with `make evaluate` numbers, and keep `CHANGELOG.md` and `changelog.html` in sync.
- Before ending a task, update `MEMORY.md` in the repo (decisions, measurements, gotchas, dated) per AGENTS.md rule 12, and add user-visible changes to `CHANGELOG.md`. Also save durable user preferences to your own persistent memory.
