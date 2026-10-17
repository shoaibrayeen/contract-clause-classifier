# Contract clause classifier - common commands. Run `make help`.
#
# Model versions: every training run creates <train.run_name>-v<model_version> (from the config) and never
# overwrites an existing model. `make all` trains, evaluates, renders the GIF, then bumps model_version.
#
# Where commands run: RUNNER=docker (default) runs every step in the `tools` container from docker-compose.yml, with
# this folder mounted into it, so outputs land here as usual. RUNNER=local runs them with `uv run` on this machine.
# Docker on macOS cannot use the Apple GPU: transformer training (train, resume, restart, sanity, all) in Docker needs
# an NVIDIA GPU (GPU=1) and otherwise stops with a hint; on a Mac use `make all RUNNER=local`.

CONFIG ?= configs/ledgar_qwen3_0.6b.yaml
PORT   ?= 8000
IMAGE  ?= contract-clause-classifier:1.0.0
TOOLS_IMAGE ?= contract-clause-classifier-tools:1.0.0
MODEL  ?= latest
UV     ?= uv
CAFFEINATE := $(shell command -v caffeinate >/dev/null 2>&1 && echo "caffeinate -i")

RUNNER ?= docker
GPU    ?= 0
ALLOW_CPU_TRAIN ?= 0
COMPOSE ?= docker compose

# The tools container runs as you, so files it writes into the project are yours.
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)
export PROJECT_DIR := $(notdir $(CURDIR))
export PORT IMAGE TOOLS_IMAGE

ifeq ($(RUNNER),local)
PY       := $(UV) run python
PYTEST   := $(UV) run pytest
PY_TRAIN := $(PY)
else ifeq ($(RUNNER),docker)
DOCKER_RUN := $(COMPOSE) --progress quiet run --rm
PY       := $(DOCKER_RUN) tools python
PYTEST   := $(DOCKER_RUN) tools pytest
PY_TRAIN := $(DOCKER_RUN) $(if $(filter 1,$(GPU)),tools-gpu,tools) python
else
$(error RUNNER must be docker or local, not "$(RUNNER)")
endif

.DEFAULT_GOAL := help
.PHONY: help setup data train resume restart finish evaluate gif terminal-gif bump models next adopt sanity predict serve playground-gif test rules changelog-html \
        classical all-classical classical-train classical-evaluate classical-bump model-pages compare \
        docker-build docker-up docker-down shell train-check all clean-runs

help: ## Show this help
	@echo "Commands run in Docker by default; add RUNNER=local to run them with uv on this machine (runner now: $(RUNNER))."
	@awk 'BEGIN{FS=":.*## "} /^[a-zA-Z_-]+:.*## /{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

ifeq ($(RUNNER),local)
setup: ## Prepare the runner: build the Docker images, or with RUNNER=local install Python 3.12 + pinned deps into .venv
	$(UV) sync
else
setup: docker-build
endif

# ---------------------------------------------------------------- pipeline

FAMILY ?= transformer

ifeq ($(FAMILY),classical)
all: classical ## Full pipeline. Transformer by default; `make all FAMILY=classical` runs the classical flow instead
	@$(PY) scripts/07_models.py list --config $(CONFIG)
else
all: train-check data train finish
	@$(PY) scripts/07_models.py list --config $(CONFIG)
endif

all-classical: classical ## Same as `make classical`: train -> evaluate -> model pages -> bump the classical version

data: ## Step 1: download LEDGAR and tokenize (idempotent)
	$(PY) scripts/01_prepare_data.py --config $(CONFIG)

train: train-check ## Step 2: train the next model version with live dashboard + telemetry (keeps the Mac awake)
	$(CAFFEINATE) $(PY_TRAIN) scripts/02_train.py --config $(CONFIG)

finish: evaluate gif terminal-gif bump ## Steps 3-5 for the newest model: evaluate + calibrate, GIFs, bump model_version

evaluate: ## Step 3: calibrate on validation + full test report (MODEL=latest or a version id)
	$(PY) scripts/03_evaluate.py --config $(CONFIG) --model $(MODEL)

gif: ## Step 4: render a model's telemetry to a GIF and copy it to demo/ (MODEL=latest or a version id)
	$(PY) scripts/06_render_gif.py --run $(MODEL) --copy-to demo

terminal-gif: ## Step 4b: replay the training dashboard epoch by epoch as demo/<run>_terminal.gif (MODEL=latest)
	$(PY) scripts/09_render_terminal_gif.py --run $(MODEL)

bump: ## Step 5: raise model_version in the config after the current version completed
	$(PY) scripts/07_models.py bump --config $(CONFIG)

resume: train-check ## Continue the unfinished current version, then evaluate, GIF and bump
	$(CAFFEINATE) $(PY_TRAIN) scripts/02_train.py --config $(CONFIG) --resume
	$(MAKE) finish CONFIG=$(CONFIG)

restart: train-check ## Discard the unfinished current version, train it from scratch, then evaluate, GIF and bump
	$(CAFFEINATE) $(PY_TRAIN) scripts/02_train.py --config $(CONFIG) --restart
	$(MAKE) finish CONFIG=$(CONFIG)

# ---------------------------------------------------------------- classical model (TF-IDF + logistic regression)

CLASSICAL_CONFIG ?= configs/classical/ledgar_tfidf_lr.yaml

classical: data classical-train classical-evaluate model-pages classical-bump ## Classical flow: data -> train -> evaluate -> model pages -> bump its version (CPU, ~30 min)

classical-train: ## Train the next classical version -> artifacts/classical/runs/ledgar-tfidf-lr-v<N>
	$(PY) scripts/classical/01_train.py --config $(CLASSICAL_CONFIG)

classical-evaluate: ## Calibrate + full test report for the latest classical model
	$(PY) scripts/classical/02_evaluate.py --config $(CLASSICAL_CONFIG)

classical-bump: ## Raise model_version in the classical config after its current version completed
	$(PY) scripts/07_models.py bump --config $(CLASSICAL_CONFIG)

model-pages: ## Generate docs/models/{compare,transformer,classical}.html from the saved models
	$(PY) scripts/10_model_pages.py

compare: ## Both models side by side on a file: make compare FILE=contract.pdf
	$(PY) scripts/11_compare.py $(or $(FILE),demo/sample_contract.txt)

# ---------------------------------------------------------------- versions

models: ## List all trained model versions (latest marked) with their metrics
	$(PY) scripts/07_models.py list --config $(CONFIG)

next: ## Show the name and version the next training run will use
	$(PY) scripts/07_models.py next --config $(CONFIG)

adopt: ## Turn an unversioned finished run into a version: make adopt RUN=<run-dir-name>
	$(PY) scripts/07_models.py adopt $(RUN) --config $(CONFIG)

# ---------------------------------------------------------------- use the model

sanity: train-check ## 5-minute smoke run on 256 examples (unversioned run "sanity"; never becomes latest over a real model)
	$(PY_TRAIN) scripts/02_train.py --config $(CONFIG) --run-name sanity --limit 256 --epochs 2 --eval-steps 32 --eval-limit 256 --grad-accum 1

predict: ## Classify a file: make predict FILE=contract.pdf [MODEL=ledgar-qwen3-0.6b-v1]
	$(PY) scripts/04_predict.py $(or $(FILE),demo/sample_contract.txt) --config $(CONFIG) --model $(MODEL)

serve: ## API + playground on http://localhost:$(PORT)/playground (serves every version; Ctrl+C stops it)
	@echo "Playground: http://localhost:$(PORT)/playground   API docs: http://localhost:$(PORT)/docs"
ifeq ($(RUNNER),local)
	$(UV) run python scripts/05_serve.py --port $(PORT)
else
	@echo "Docker runs the models on CPU; for the Apple GPU use: make serve RUNNER=local"
	$(COMPOSE) up --build api
endif

playground-gif: ## Record demo/playground_usage.gif with the latest models (always on this machine: needs Chrome and `uv sync`)
	$(UV) run python scripts/08_record_playground.py --port $(PORT)

test: ## Run the test suite (no trained model needed)
	$(PYTEST) -q

changelog-html: ## Regenerate changelog.html from CHANGELOG.md
	$(PY) scripts/sync_changelog_html.py

rules: ## Regenerate every AI assistant's rules file from AGENTS.md
	$(PY) scripts/sync_agent_rules.py

# ---------------------------------------------------------------- docker

docker-build: ## Build both images: the API (serving) and tools (every make step); rerun after changing deps
	$(COMPOSE) --profile tools build api tools

docker-up: ## Run the API in Docker in the background (mounts ./artifacts, ./data and ./docs/models)
	$(COMPOSE) up --build -d api && $(COMPOSE) ps

docker-down: ## Stop the Docker API
	$(COMPOSE) down

shell: ## Open a shell in the tools container (project mounted at /app)
	$(DOCKER_RUN) tools bash

# Transformer training needs a GPU. Docker on macOS has none (no Apple GPU passthrough), so in Docker it would run on
# the CPU for days and need more memory than the Docker VM has. Stop early with a hint instead.
train-check:
ifeq ($(RUNNER),docker)
ifneq ($(GPU),1)
ifneq ($(ALLOW_CPU_TRAIN),1)
	@echo "Transformer training in Docker needs an NVIDIA GPU, and Docker on macOS cannot use the Apple GPU."
	@echo "  On this Mac, train natively:   make $(MAKECMDGOALS) RUNNER=local"
	@echo "  Linux with an NVIDIA GPU:      make $(MAKECMDGOALS) GPU=1"
	@echo "  Force CPU training in Docker:  make $(MAKECMDGOALS) ALLOW_CPU_TRAIN=1   (days, and 25+ GB of memory)"
	@echo "The classical model trains fine in Docker:  make classical"
	@exit 1
endif
endif
endif
	@:

clean-runs: ## Delete ALL training runs and the artifacts/best link (asks first)
	@read -p "Delete artifacts/runs/* and artifacts/best? This removes every model version. [y/N] " ans && [ "$$ans" = y ] && rm -rf artifacts/runs/* artifacts/best || echo "Kept."
