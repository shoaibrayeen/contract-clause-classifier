"""Model construction (training) and loading (inference).

Backbone: any HF causal LM with a `*ForSequenceClassification` class (Qwen3, Qwen2.5,
SmolLM2/Llama ...). The LM head is replaced by a linear `score` head over the last
non-pad token. LoRA adapters (PEFT) are added for training; for serving the adapters
are merged so inference needs plain `transformers` only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

from .config import Config
from .labels import id2label, label2id, save_labels

log = logging.getLogger(__name__)

CALIBRATION_FILE = "calibration.json"
LABELS_FILE = "labels.json"


def load_tokenizer(name_or_path: str | Path) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(str(name_or_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_model(cfg: Config, label_names: list[str], tokenizer: PreTrainedTokenizerBase):
    """Create the trainable classifier. Weights are fp32; mixed precision via autocast."""
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label_names),
        id2label=id2label(label_names),
        label2id=label2id(label_names),
        dtype=torch.float32,
        # A base that already has a classification head with another label count gets a fresh head.
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    if cfg.full_finetune or not cfg.lora.enabled:
        return model

    from peft import LoraConfig, TaskType, get_peft_model  # training-only dependency

    if cfg.train.gradient_checkpointing:
        model.enable_input_require_grads()
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        modules_to_save=["score"],
    )
    return get_peft_model(model, peft_cfg)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def is_peft(model) -> bool:
    return hasattr(model, "peft_config")


def save_checkpoint(model, tokenizer, out_dir: Path, label_names: list[str]) -> None:
    """Save adapter (LoRA) or full weights + tokenizer + labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    save_labels(label_names, out_dir / LABELS_FILE)


def load_trained(checkpoint_dir: str | Path, device: torch.device | str = "cpu"):
    """Load a training checkpoint (LoRA adapter dir or full model dir)."""
    checkpoint_dir = Path(checkpoint_dir)
    tok = load_tokenizer(checkpoint_dir)
    if (checkpoint_dir / "adapter_config.json").exists():
        from peft import PeftModel

        from .labels import load_labels

        # The adapter dir has no base config, so rebuild the base with the right label count first.
        names = load_labels(checkpoint_dir / LABELS_FILE)
        base_name = json.loads((checkpoint_dir / "adapter_config.json").read_text())["base_model_name_or_path"]
        base = AutoModelForSequenceClassification.from_pretrained(
            base_name, num_labels=len(names), id2label=id2label(names), label2id=label2id(names), dtype=torch.float32
        )
        base.config.pad_token_id = tok.pad_token_id
        model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir), dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    return model.to(device).eval(), tok


def export_merged(checkpoint_dir: str | Path, out_dir: str | Path, dtype: torch.dtype = torch.bfloat16) -> Path:
    """Merge LoRA into the backbone and save a plain HF model for serving."""
    checkpoint_dir, out_dir = Path(checkpoint_dir), Path(out_dir)
    model, tok = load_trained(checkpoint_dir, "cpu")
    if is_peft(model):
        model = model.merge_and_unload()
    model = model.to(dtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    for extra in (LABELS_FILE, CALIBRATION_FILE):
        if (checkpoint_dir / extra).exists():
            (out_dir / extra).write_bytes((checkpoint_dir / extra).read_bytes())
    make_readable(out_dir)
    log.info("Exported merged model to %s", out_dir)
    return out_dir


def make_readable(directory: str | Path) -> None:
    """safetensors are written via a 0600 temp file; make them readable by other users (e.g. Docker's app user)."""
    for f in Path(directory).rglob("*"):
        if f.is_file():
            f.chmod(0o644)
        elif f.is_dir():
            f.chmod(0o755)


def save_calibration(model_dir: str | Path, temperature: float, **extra) -> None:
    Path(model_dir, CALIBRATION_FILE).write_text(json.dumps({"temperature": temperature, **extra}, indent=2))


def load_temperature(model_dir: str | Path) -> float:
    p = Path(model_dir, CALIBRATION_FILE)
    if p.exists():
        return float(json.loads(p.read_text()).get("temperature", 1.0))
    return 1.0
