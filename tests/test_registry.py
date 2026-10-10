"""Model versioning: names, discovery, latest selection, training-run planning, config bumps."""

import os
import time

import pytest

from classification.registry import (
    ModelNotFoundError,
    RegistryError,
    UnfinishedRunError,
    adopt_run,
    bump_config_version,
    latest_model,
    list_models,
    plan_training_run,
    read_config_version,
    refresh_best_link,
    resolve_model,
    resolve_model_dir,
    split_name,
    update_model_info,
    versioned_name,
    write_config_version,
)
from conftest import make_run

BASE = "ledgar-qwen3-0.6b-lora"

CONFIG = """# comment stays
model_name: Qwen/Qwen3-0.6B-Base
model_version: 1            # next model version to train
full_finetune: false
train:
  run_name: ledgar-qwen3-0.6b-lora   # base name
"""


@pytest.fixture
def runs(tmp_path):
    d = tmp_path / "artifacts" / "runs"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(CONFIG)
    return p


def test_names():
    assert split_name(f"{BASE}-v12") == (BASE, 12)
    assert split_name("sanity") == ("sanity", None)
    assert split_name("my-v2-model") == ("my-v2-model", None)
    assert versioned_name(BASE, 3) == f"{BASE}-v3"


def test_list_models_only_complete_runs(runs):
    make_run(runs, f"{BASE}-v1", 100)
    make_run(runs, f"{BASE}-v2", 200, complete=False)
    make_run(runs, "_scratch", 50)
    ids = [m.id for m in list_models(runs)]
    assert ids == [f"{BASE}-v1"]


def test_latest_prefers_versioned_over_unversioned(runs):
    make_run(runs, f"{BASE}-v1", 100)
    make_run(runs, f"{BASE}-v2", 200)
    make_run(runs, "sanity", 300)  # newer, but unversioned
    assert latest_model(list_models(runs)).id == f"{BASE}-v2"


def test_latest_falls_back_to_unversioned(runs):
    make_run(runs, "sanity", 300)
    make_run(runs, "ledgar-qwen3-0.6b-lora", 200)  # legacy run trained before versioning
    assert latest_model(list_models(runs)).id == "sanity"


def test_resolve_model_rules(runs):
    make_run(runs, f"{BASE}-v1", 100)
    make_run(runs, f"{BASE}-v2", 200)
    make_run(runs, "other-v7", 50)
    models = list_models(runs)
    assert resolve_model(models, None).id == f"{BASE}-v2"
    assert resolve_model(models, "").id == f"{BASE}-v2"
    assert resolve_model(models, "latest").id == f"{BASE}-v2"
    assert resolve_model(models, "my-alias", aliases=("latest", "my-alias")).id == f"{BASE}-v2"
    assert resolve_model(models, f"{BASE}-v1").id == f"{BASE}-v1"
    assert resolve_model(models, "other").id == "other-v7"  # base name -> latest version of that base
    with pytest.raises(ModelNotFoundError, match="does not exist"):
        resolve_model(models, f"{BASE}-v9")
    with pytest.raises(ModelNotFoundError, match="No trained models"):
        resolve_model([], None)


def test_resolve_model_dir_accepts_paths_and_names(runs):
    run = make_run(runs, f"{BASE}-v1", 100)
    assert resolve_model_dir(runs, "latest") == run / "merged"
    assert resolve_model_dir(runs, str(run)) == run / "merged"
    assert resolve_model_dir(runs, str(run / "merged")) == run / "merged"


def test_plan_fresh_run_is_v1(runs):
    assert plan_training_run(runs, BASE, 1)[:2] == (f"{BASE}-v1", 1)


def test_plan_never_overwrites_a_completed_version(runs):
    make_run(runs, f"{BASE}-v1", 100)
    make_run(runs, f"{BASE}-v2", 200)
    name, version, notes = plan_training_run(runs, BASE, 1)
    assert (name, version) == (f"{BASE}-v3", 3)
    assert any("never overwritten" in n for n in notes)


def test_plan_unfinished_run_requires_resume_or_restart(runs):
    make_run(runs, f"{BASE}-v1", 100, complete=False)
    with pytest.raises(UnfinishedRunError, match="make resume"):
        plan_training_run(runs, BASE, 1)
    assert plan_training_run(runs, BASE, 1, resume=True)[:2] == (f"{BASE}-v1", 1)
    name, _, notes = plan_training_run(runs, BASE, 1, restart=True)
    assert name == f"{BASE}-v1" and not (runs / name).exists() and notes


def test_plan_resume_errors(runs):
    with pytest.raises(RegistryError, match="No unfinished run"):
        plan_training_run(runs, BASE, 1, resume=True)
    make_run(runs, f"{BASE}-v1", 100)
    with pytest.raises(RegistryError, match="already complete"):
        plan_training_run(runs, BASE, 1, resume=True)


def test_plan_strips_version_from_base_name(runs):
    name, _, notes = plan_training_run(runs, f"{BASE}-v4", 2)
    assert name == f"{BASE}-v2" and notes


def test_config_version_read_write_preserves_comments(config):
    assert read_config_version(config) == 1
    write_config_version(config, 2)
    text = config.read_text()
    assert "model_version: 2            # next model version to train" in text
    assert text.startswith("# comment stays") and "run_name: ledgar-qwen3-0.6b-lora   # base name" in text
    assert read_config_version(config) == 2


def test_config_version_inserted_when_missing(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model_name: x\ntrain:\n  run_name: r\n")
    assert read_config_version(p) == 1
    write_config_version(p, 5)
    assert read_config_version(p) == 5 and p.read_text().index("model_version") > p.read_text().index("model_name")


def test_bump_only_after_completion(runs, config):
    assert bump_config_version(config, runs, BASE) == (1, 1)  # nothing finished: stay on v1
    make_run(runs, f"{BASE}-v1", 100, complete=False)
    assert bump_config_version(config, runs, BASE) == (1, 1)  # unfinished: retry v1 next time
    make_run(runs, f"{BASE}-v1", 100)
    assert bump_config_version(config, runs, BASE) == (1, 2)
    assert bump_config_version(config, runs, BASE) == (2, 2)  # idempotent
    assert read_config_version(config) == 2


def test_full_versioning_cycle(runs, config):
    """make all twice: v1 then v2, both kept, latest is v2."""
    for expected in (1, 2):
        name, version, _ = plan_training_run(runs, BASE, read_config_version(config))
        assert version == expected
        make_run(runs, name, time.time() + expected)  # "training"
        bump_config_version(config, runs, BASE)
    assert [m.id for m in list_models(runs)] == [f"{BASE}-v1", f"{BASE}-v2"]
    assert read_config_version(config) == 3
    assert latest_model(list_models(runs)).id == f"{BASE}-v2"


def test_refresh_best_link_points_at_latest(runs):
    assert refresh_best_link(runs) is None
    make_run(runs, f"{BASE}-v1", 100)
    make_run(runs, f"{BASE}-v2", 200)
    link = refresh_best_link(runs)
    assert link == runs.parent / "best" and os.readlink(link) == f"runs/{BASE}-v2/merged"


def test_adopt_legacy_run(runs):
    make_run(runs, BASE, 100, version=None)
    dst = adopt_run(runs, BASE, BASE, 1)
    assert dst.name == f"{BASE}-v1" and not (runs / BASE).exists()
    m = list_models(runs)[0]
    assert (m.id, m.version, m.info["adopted_from"]) == (f"{BASE}-v1", 1, BASE)
    with pytest.raises(RegistryError):
        adopt_run(runs, "missing", BASE, 2)


def test_update_model_info_creates_file_for_legacy_runs(runs):
    run = make_run(runs, "old-run", 100)
    (run / "merged" / "model_info.json").unlink()
    info = update_model_info(run / "merged", test={"accuracy": 0.9})
    assert info["id"] == "old-run" and info["version"] is None and info["test"]["accuracy"] == 0.9


def test_resume_needs_a_checkpoint(runs):
    run = make_run(runs, f"{BASE}-v1", 100, complete=False)
    (run / "checkpoint-last" / "trainer_state.json").unlink()  # died before the first evaluation
    with pytest.raises(RegistryError, match="make restart"):
        plan_training_run(runs, BASE, 1, resume=True)
    assert plan_training_run(runs, BASE, 1, restart=True)[0] == f"{BASE}-v1"
