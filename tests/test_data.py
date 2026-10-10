import torch

from classification.data import DatasetStore, LengthGroupedBatchSampler, TokenizedDataset, make_collate


def test_dataset_page_defaults(tiny_data):
    store = DatasetStore(tiny_data)
    page = store.page("train", 0, 10)
    assert page["total"] == 25 and len(page["items"]) == 10 and page["next_offset"] == 10
    item = page["items"][0]
    assert set(item) == {"id", "text", "label_id", "label", "original_label"}
    assert item["label"] == store.labels()[item["label_id"]]


def test_dataset_last_page(tiny_data):
    page = DatasetStore(tiny_data).page("train", 20, 10)
    assert len(page["items"]) == 5 and page["next_offset"] is None
    assert DatasetStore(tiny_data).page("train", 100, 10)["items"] == []


def test_length_grouped_sampler_covers_all():
    lengths = [5, 50, 7, 300, 12, 2, 90, 33, 1, 64]
    s = LengthGroupedBatchSampler(lengths, batch_size=3, seed=1, mega=2)
    batches = list(s)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    s.set_epoch(1)
    assert list(s) != batches or len(lengths) < 4  # reshuffled per epoch


def test_collate_pads_right_and_keeps_index():
    ds = TokenizedDataset([[1, 2, 3], [4]], [0, 1], indices=[10, 20])
    batch = make_collate(pad_id=0)([ds[0], ds[1]])
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert torch.equal(batch["index"], torch.tensor([10, 20]))


def test_recorder_sessions(tmp_path):
    from classification.telemetry import RunRecorder

    rec = RunRecorder(tmp_path)
    rec.write_meta({"a": 1}, new_session=True)
    rec.write_meta({"summary": {"x": 1}})
    rec2 = RunRecorder(tmp_path)  # a resumed process
    rec2.write_meta({"a": 2}, new_session=True)
    import json

    meta = json.loads((tmp_path / "telemetry" / "meta.json").read_text())
    assert [s["resumed"] for s in meta["sessions"]] == [False, True]
    assert meta["a"] == 2 and meta["summary"] == {"x": 1}


def test_label_groups_merge_and_validate(tiny_data, tmp_path):
    import pytest

    from classification.data import build_label_space, get_label_space, load_split

    raw = ["Governing Laws", "Confidentiality", "Applicable Laws", "Notices"]
    space = build_label_space(raw, {"Governing Laws": ["Governing Laws", "Applicable Laws"]})
    assert space.names == ["Governing Laws", "Confidentiality", "Notices"]
    assert space.map([0, 1, 2, 3]) == [0, 1, 0, 2]
    with pytest.raises(ValueError, match="unknown label"):
        build_label_space(raw, {"X": ["Nope"]})
    with pytest.raises(ValueError, match="two groups"):
        build_label_space(raw, {"A": ["Notices"], "B": ["Notices"]})

    # end to end through a groups file: tiny_data labels are 0..4 of conftest.LABELS
    groups = tmp_path / "groups.yaml"
    groups.write_text("groups:\n  Legal: [Governing Laws, Notices]\n")
    tiny_data.label_groups = str(groups)
    names = get_label_space(tiny_data).names
    assert names == ["Legal", "Confidentiality", "Terminations", "Assignments"]
    _, labels = load_split(tiny_data, "train")
    assert set(labels) == {0, 1, 2, 3}
    item = DatasetStore(tiny_data).page("train", 3, 1)["items"][0]  # raw label 3 = Notices
    assert (item["label"], item["original_label"], item["label_id"]) == ("Legal", "Notices", 0)


def test_label_space_roundtrip_and_model_lookup(tiny_data, tmp_path):
    import pytest

    from classification.data import build_label_space, get_label_space, get_raw_label_names
    from classification.model import LABEL_SPACE_FILE, label_space_for_model

    raw = get_raw_label_names(tiny_data)
    old = build_label_space(raw, {"Legal": ["Governing Laws", "Notices"]}, source="old.yaml")
    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    old.save(model_dir / LABEL_SPACE_FILE)
    # A model trained with the old grouping is scored with it even though the config now has no groups.
    space = label_space_for_model(model_dir, tiny_data, old.names)
    assert space.names == old.names and space.map([0, 3]) == [0, 0] and space.source == "old.yaml"
    with pytest.raises(ValueError, match="does not match"):
        label_space_for_model(model_dir, tiny_data, raw)
    # Without the file: current space, original labels, or a clear error.
    (model_dir / LABEL_SPACE_FILE).unlink()
    assert label_space_for_model(model_dir, tiny_data, get_label_space(tiny_data).names).names == raw
    assert label_space_for_model(model_dir, tiny_data, raw).names == raw
    with pytest.raises(ValueError, match="match neither"):
        label_space_for_model(model_dir, tiny_data, ["Something", "Else"])
