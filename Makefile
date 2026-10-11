# Contract clause classifier - common commands. Run `make help`.
#
# Model versions: every training run creates <train.run_name>-v<model_version> (from the config) and never
# overwrites an existing model. `make all` trains, evaluates, renders the GIF, then bumps model_version.

CONFIG ?= configs/ledgar_qwen3_0.6b.yaml
PORT   ?= 8000
IMAGE  ?= contract-clause-classifier:1.0.0
MODEL  ?= latest
UV     ?= uv
CAFFEINATE := $(shell command -v caffeinate >/dev/null 2>&1 && echo "caffeinate -i")

.DEFAULT_GOAL := help
.PHONY: help setup data train resume restart finish evaluate gif terminal-gif bump models next adopt sanity predict serve playground-gif test rules changelog-html \
        classical all-classical classical-train classical-evaluate classical-bump model-pages compare \
        docker-build docker-up docker-down all clean-runs

help: ## Show this help
	@awk 'BEGIN{FS=":.*## "} /^[a-zA-Z_-]+:.*## /{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install Python 3.12 + pinned dependencies into .venv
	$(UV) sync

# ---------------------------------------------------------------- pipeline

FAMILY ?= transformer

ifeq ($(FAMILY),classical)
all: classical ## Full pipeline. Transformer by default; `make all FAMILY=classical` runs the classical flow instead
	@$(UV) run python scripts/07_models.py list --config $(CONFIG)
else
all: data train finish
	@$(UV) run python scripts/07_models.py list --config $(CONFIG)
endif

all-classical: classical ## Same as `make classical`: train -> evaluate -> model pages -> bump the classical version

data: ## Step 1: download LEDGAR and tokenize (idempotent)
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG)

train: ## Step 2: train the next model version with live dashboard + telemetry (keeps the Mac awake)
	$(CAFFEINATE) $(UV) run python scripts/02_train.py --config $(CONFIG)

finish: evaluate gif terminal-gif bump ## Steps 3-5 for the newest model: evaluate + calibrate, GIFs, bump model_version

evaluate: ## Step 3: calibrate on validation + full test report (MODEL=latest or a version id)
	$(UV) run python scripts/03_evaluate.py --config $(CONFIG) --model $(MODEL)

gif: ## Step 4: render a model's telemetry to a GIF and copy it to demo/ (MODEL=latest or a version id)
	$(UV) run python scripts/06_render_gif.py --run $(MODEL) --copy-to demo

terminal-gif: ## Step 4b: replay the training dashboard epoch by epoch as demo/<run>_terminal.gif (MODEL=latest)
	$(UV) run python scripts/09_render_terminal_gif.py --run $(MODEL)

bump: ## Step 5: raise model_version in the config after the current version completed
	$(UV) run python scripts/07_models.py bump --config $(CONFIG)

resume: ## Continue the unfinished current version, then evaluate, GIF and bump
	$(CAFFEINATE) $(UV) run python scripts/02_train.py --config $(CONFIG) --resume
	$(MAKE) finish CONFIG=$(CONFIG)

restart: ## Discard the unfinished current version, train it from scratch, then evaluate, GIF and bump
	$(CAFFEINATE) $(UV) run python scripts/02_train.py --config $(CONFIG) --restart
	$(MAKE) finish CONFIG=$(CONFIG)

# ---------------------------------------------------------------- classical model (TF-IDF + logistic regression)

CLASSICAL_CONFIG ?= configs/classical/ledgar_tfidf_lr.yaml

classical: data classical-train classical-evaluate model-pages classical-bump ## Classical flow: data -> train -> evaluate -> model pages -> bump its version (CPU, ~30 min)

classical-train: ## Train the next classical version -> artifacts/classical/runs/ledgar-tfidf-lr-v<N>
	$(UV) run python scripts/classical/01_train.py --config $(CLASSICAL_CONFIG)

classical-evaluate: ## Calibrate + full test report for the latest classical model
	$(UV) run python scripts/classical/02_evaluate.py --config $(CLASSICAL_CONFIG)

classical-bump: ## Raise model_version in the classical config after its current version completed
	$(UV) run python scripts/07_models.py bump --config $(CLASSICAL_CONFIG)

model-pages: ## Generate docs/models/{compare,transformer,classical}.html from the saved models
	$(UV) run python scripts/10_model_pages.py

compare: ## Both models side by side on a file: make compare FILE=contract.pdf
	$(UV) run python scripts/11_compare.py $(or $(FILE),demo/sample_contract.txt)

# ---------------------------------------------------------------- versions

models: ## List all trained model versions (latest marked) with their metrics
	$(UV) run python scripts/07_models.py list --config $(CONFIG)

next: ## Show the name and version the next training run will use
	$(UV) run python scripts/07_models.py next --config $(CONFIG)

adopt: ## Turn an unversioned finished run into a version: make adopt RUN=<run-dir-name>
	$(UV) run python scripts/07_models.py adopt $(RUN) --config $(CONFIG)

# ---------------------------------------------------------------- use the model

sanity: ## 5-minute smoke run on 256 examples (unversioned run "sanity"; never becomes latest over a real model)
	$(UV) run python scripts/02_train.py --config $(CONFIG) --run-name sanity --limit 256 --epochs 2 --eval-steps 32 --eval-limit 256 --grad-accum 1

predict: ## Classify a file: make predict FILE=contract.pdf [MODEL=ledgar-qwen3-0.6b-v1]
	$(UV) run python scripts/04_predict.py $(or $(FILE),demo/sample_contract.txt) --config $(CONFIG) --model $(MODEL)

serve: ## API + playground on http://localhost:$(PORT)/playground (serves every version; default = latest)
	@echo "Playground: http://localhost:$(PORT)/playground   API docs: http://localhost:$(PORT)/docs"
	$(UV) run python scripts/05_serve.py --port $(PORT)

playground-gif: ## Record demo/playground_usage.gif of the playground with the latest model (needs Chrome)
	$(UV) run python scripts/08_record_playground.py --port $(PORT)

test: ## Run the test suite (no trained model needed)
	$(UV) run pytest -q

changelog-html: ## Regenerate changelog.html from CHANGELOG.md
	$(UV) run python scripts/sync_changelog_html.py

rules: ## Regenerate every AI assistant's rules file from AGENTS.md
	$(UV) run python scripts/sync_agent_rules.py

# ---------------------------------------------------------------- docker

docker-build: ## Build the CPU serving image
	docker build -t $(IMAGE) .

docker-up: ## Run the API in Docker (mounts ./artifacts and ./data)
	docker compose up --build -d && docker compose ps

docker-down: ## Stop the Docker API
	docker compose down

clean-runs: ## Delete ALL training runs and the artifacts/best link (asks first)
	@read -p "Delete artifacts/runs/* and artifacts/best? This removes every model version. [y/N] " ans && [ "$$ans" = y ] && rm -rf artifacts/runs/* artifacts/best || echo "Kept."
