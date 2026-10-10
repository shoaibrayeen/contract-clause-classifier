"""LEDGAR data: download, label names, tokenisation, batching.

LEDGAR (LexGLUE) = 80k contract provisions from SEC EDGAR filings, 100 clause types.
Splits: train 60k (2016-17), validation 10k (2018), test 10k (2019).
We read the auto-converted parquet directly (no `datasets` loading script needed).
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from .config import DataConfig, resolve_path
from .labels import load_labels, save_labels

log = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
FALLBACK_REPO = "lighteval/lexglue"
DATASETS_SERVER = "https://datasets-server.huggingface.co/parquet?dataset={repo}"


# --------------------------------------------------------------------------- download

def raw_dir(cfg: DataConfig) -> Path:
    return resolve_path(cfg.data_dir) / "raw" / cfg.dataset_config


def split_path(cfg: DataConfig, split: str) -> Path:
    return raw_dir(cfg) / f"{split}.parquet"


def labels_path(cfg: DataConfig) -> Path:
    return raw_dir(cfg) / "labels.json"


def _download_via_hub(cfg: DataConfig, split: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=cfg.dataset_repo,
        repo_type="dataset",
        revision=cfg.dataset_revision,
        filename=f"{cfg.dataset_config}/{split}/0000.parquet",
    )
    shutil.copyfile(cached, dest)


def _download_via_datasets_server(repo: str, config: str, split: str, dest: Path) -> None:
    with urllib.request.urlopen(DATASETS_SERVER.format(repo=repo), timeout=60) as resp:
        files = json.load(resp)["parquet_files"]
    urls = [f["url"] for f in files if f["config"] == config and f["split"] == split]
    if not urls:
        raise FileNotFoundError(f"No parquet for {repo}/{config}/{split}")
    tables = []
    for url in urls:
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tables.append(pq.read_table(tmp))
        tmp.unlink()
    pq.write_table(pa.concat_tables(tables), dest)


def label_names_from_parquet(path: Path) -> list[str]:
    """HF parquet exports store ClassLabel names in the schema metadata."""
    meta = pq.read_schema(path).metadata or {}
    info = json.loads(meta[b"huggingface"])
    return list(info["info"]["features"]["label"]["names"])


def download_ledgar(cfg: DataConfig, force: bool = False) -> dict[str, Path]:
    """Download all three splits (idempotent). Returns split -> parquet path."""
    out = raw_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in SPLITS:
        dest = split_path(cfg, split)
        if dest.exists() and not force:
            paths[split] = dest
            continue
        log.info("Downloading LEDGAR %s split ...", split)
        try:
            _download_via_hub(cfg, split, dest)
        except Exception as exc:  # network / revision issues -> mirror
            log.warning("Hub download failed (%s); trying %s via datasets-server", exc, FALLBACK_REPO)
            _download_via_datasets_server(FALLBACK_REPO, cfg.dataset_config, split, dest)
        paths[split] = dest
    if not labels_path(cfg).exists() or force:
        save_labels(label_names_from_parquet(paths["train"]), labels_path(cfg))
    return paths


def ensure_data(cfg: DataConfig) -> None:
    if not all(split_path(cfg, s).exists() for s in SPLITS) or not labels_path(cfg).exists():
        download_ledgar(cfg)


def load_split(cfg: DataConfig, split: str) -> tuple[list[str], list[int]]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    ensure_data(cfg)
    table = pq.read_table(split_path(cfg, split), columns=["text", "label"])
    return table.column("text").to_pylist(), table.column("label").to_pylist()


def get_label_names(cfg: DataConfig) -> list[str]:
    ensure_data(cfg)
    return load_labels(labels_path(cfg))


# --------------------------------------------------------------------------- tokenisation

def tokenize_texts(tokenizer, texts: Sequence[str], max_len: int, batch: int = 2048) -> list[list[int]]:
    ids: list[list[int]] = []
    for i in range(0, len(texts), batch):
        enc = tokenizer(
            list(texts[i : i + batch]),
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        )
        ids.extend(enc["input_ids"])
    return ids


def cache_path(cfg: DataConfig, split: str, tokenizer_name: str) -> Path:
    tag = tokenizer_name.replace("/", "__")
    return resolve_path(cfg.data_dir) / "cache" / f"{cfg.dataset_config}-{tag}-{cfg.max_len}" / f"{split}.parquet"


def load_tokenized(cfg: DataConfig, split: str, tokenizer, tokenizer_name: str) -> tuple[list[list[int]], list[int]]:
    """Tokenise a split once and cache it as parquet (list<int32> column)."""
    path = cache_path(cfg, split, tokenizer_name)
    if path.exists():
        t = pq.read_table(path)
        return t.column("input_ids").to_pylist(), t.column("label").to_pylist()
    texts, labels = load_split(cfg, split)
    ids = tokenize_texts(tokenizer, texts, cfg.max_len)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"input_ids": pa.array(ids, type=pa.list_(pa.int32())), "label": pa.array(labels, pa.int32())}),
        path,
    )
    return ids, labels


def length_stats(ids: Sequence[Sequence[int]], max_len: int) -> dict[str, float]:
    lens = np.array([len(x) for x in ids])
    return {
        "n": int(len(lens)),
        "mean": float(lens.mean()),
        "p50": float(np.percentile(lens, 50)),
        "p90": float(np.percentile(lens, 90)),
        "p99": float(np.percentile(lens, 99)),
        "max": int(lens.max()),
        "truncated_frac": float((lens >= max_len).mean()),
    }


def class_weights(labels: Sequence[int], num_labels: int) -> torch.Tensor:
    """Inverse-sqrt-frequency weights, normalised to mean 1."""
    counts = np.bincount(np.asarray(labels), minlength=num_labels).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# --------------------------------------------------------------------------- torch dataset

class TokenizedDataset(Dataset):
    """Items are (input_ids, label, source_index); source_index = row id in the original split."""

    def __init__(self, input_ids: list[list[int]], labels: list[int], indices: list[int] | None = None):
        assert len(input_ids) == len(labels)
        self.input_ids = input_ids
        self.labels = labels
        self.indices = indices if indices is not None else list(range(len(labels)))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[list[int], int, int]:
        return self.input_ids[i], self.labels[i], self.indices[i]

    def lengths(self) -> list[int]:
        return [len(x) for x in self.input_ids]


class LengthGroupedBatchSampler(Sampler[list[int]]):
    """Shuffle, group into mega-batches, sort each by length, cut into batches.

    Cuts padding waste ~3x versus random batching while keeping batches varied.
    Deterministic per (seed, epoch) so a resumed run sees the same order.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int = 0, mega: int = 50, shuffle: bool = True):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.mega = mega
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        idx = list(range(len(self.lengths)))
        if not self.shuffle:
            idx.sort(key=lambda i: self.lengths[i])
            return [idx[i : i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(idx)
        size = self.batch_size * self.mega
        batches: list[list[int]] = []
        for m in range(0, len(idx), size):
            chunk = sorted(idx[m : m + size], key=lambda i: self.lengths[i], reverse=True)
            batches.extend(chunk[i : i + self.batch_size] for i in range(0, len(chunk), self.batch_size))
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def make_collate(pad_id: int):
    """Right-pad to the longest sequence in the batch."""

    def collate(batch: list[tuple[list[int], int, int]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item[0]) for item in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row, item in enumerate(batch):
            ids[row, : len(item[0])] = torch.tensor(item[0], dtype=torch.long)
            mask[row, : len(item[0])] = 1
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        index = torch.tensor([item[2] for item in batch], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels, "index": index}

    return collate


# --------------------------------------------------------------------------- API store

class DatasetStore:
    """Lazy, cached, read-only access to raw splits for the paginated dataset API."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self._tables: dict[str, pa.Table] = {}
        self._labels: list[str] | None = None

    def labels(self) -> list[str]:
        if self._labels is None:
            self._labels = get_label_names(self.cfg)
        return self._labels

    def table(self, split: str) -> pa.Table:
        if split not in SPLITS:
            raise KeyError(split)
        if split not in self._tables:
            ensure_data(self.cfg)
            self._tables[split] = pq.read_table(split_path(self.cfg, split), memory_map=True)
        return self._tables[split]

    def page(self, split: str, offset: int, limit: int) -> dict:
        table = self.table(split)
        total = table.num_rows
        names = self.labels()
        rows = table.slice(offset, limit).to_pylist() if offset < total else []
        items = [
            {"id": f"{split}-{offset + i}", "text": r["text"], "label_id": int(r["label"]), "label": names[int(r["label"])]}
            for i, r in enumerate(rows)
        ]
        nxt = offset + len(items)
        return {
            "split": split,
            "offset": offset,
            "limit": limit,
            "total": total,
            "next_offset": nxt if nxt < total else None,
            "items": items,
        }
