# Contract clause classifier - common commands. Run `make help`.
# Every target is idempotent; `make all` runs the full pipeline end to end.

CONFIG ?= configs/ledgar_qwen3_0.6b.yaml
RUN    ?= ledgar-qwen3-0.6b-lora
PORT   ?= 8000
IMAGE  ?= contract-clause-classifier:0.1.0
UV     ?= uv
CAFFEINATE := $(shell command -v caffeinate >/dev/null 2>&1 && echo "caffeinate -i")

.DEFAULT_GOAL := help
.PHONY: help setup data train resume sanity evaluate gif predict serve test docker-build docker-up docker-down all clean-runs

help: ## Show this help
	@awk 'BEGIN{FS=":.*## "} /^[a-zA-Z_-]+:.*## /{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install Python 3.12 + pinned dependencies into .venv
	$(UV) sync

data: ## Step 1: download LEDGAR and tokenize (idempotent)
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG)

train: ## Step 2: fine-tune with live console dashboard + full telemetry (keeps the Mac awake)
	$(CAFFEINATE) $(UV) run python scripts/02_train.py --config $(CONFIG) --run-name $(RUN)

resume: ## Continue an interrupted training run from its last checkpoint
	$(CAFFEINATE) $(UV) run python scripts/02_train.py --config $(CONFIG) --run-name $(RUN) --resume

sanity: ## 3-minute smoke run on 256 examples (proves the loop learns; run name "sanity")
	$(UV) run python scripts/02_train.py --config $(CONFIG) --run-name sanity --limit 256 --epochs 2 --eval-steps 32 --eval-limit 256 --grad-accum 1

evaluate: ## Step 3: calibrate on validation + full test-set report for artifacts/best
	$(UV) run python scripts/03_evaluate.py --config $(CONFIG) --model-dir artifacts/best

gif: ## Render the run's telemetry into <run>/visuals/training.gif and copy it to demo/
	$(UV) run python scripts/06_render_gif.py --run artifacts/runs/$(RUN)
	mkdir -p demo && cp artifacts/runs/$(RUN)/visuals/training.gif demo/$(RUN)_training.gif && cp artifacts/runs/$(RUN)/visuals/training.png demo/$(RUN)_training_final_frame.png

predict: ## Step 4: classify demo/sample_contract.txt (override with FILE=path)
	$(UV) run python scripts/04_predict.py $(or $(FILE),demo/sample_contract.txt)

serve: ## Step 5: OpenAI-compatible API on http://localhost:$(PORT) (uses MPS/CUDA if available)
	$(UV) run python scripts/05_serve.py --port $(PORT)

test: ## Run the test suite (no trained model needed)
	$(UV) run pytest -q

docker-build: ## Build the CPU serving image
	docker build -t $(IMAGE) .

docker-up: ## Run the API in Docker (mounts ./artifacts and ./data)
	docker compose up --build -d && docker compose ps

docker-down: ## Stop the Docker API
	docker compose down

all: data train evaluate gif ## Full pipeline: data -> train -> evaluate/calibrate -> GIF
	@echo "Done. Model: artifacts/best -> $$(readlink artifacts/best). Serve it with: make serve"

clean-runs: ## Delete all training runs and the artifacts/best link (asks first)
	@read -p "Delete artifacts/runs/* and artifacts/best? [y/N] " ans && [ "$$ans" = y ] && rm -rf artifacts/runs/* artifacts/best || echo "Kept."
