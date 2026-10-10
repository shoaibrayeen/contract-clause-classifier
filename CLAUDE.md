# CLAUDE.md

@AGENTS.md

## Claude Code specifics

- Shared agent rules, commands and layout are in AGENTS.md (imported above). Project decisions and gotchas are in @MEMORY.md.
- Long-running processes the user should watch (training, the API server) belong in a Terminal-panel tab, not a background Bash call. Training is the user's to start; give them the command (`make train` or `make all`) unless they ask you to run it.
- To verify a training change, run `make sanity` (about 3 minutes) and inspect `artifacts/runs/sanity/telemetry/`.
- To verify serving without the GPU, use `DEVICE=cpu uv run python scripts/05_serve.py --port 8011`.
- Scratch files go in the session scratchpad, not the repo.
