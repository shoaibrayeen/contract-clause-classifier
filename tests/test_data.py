import torch

from classification.data import DatasetStore, LengthGroupedBatchSampler, TokenizedDataset, make_collate


def test_dataset_page_defaults(tiny_data):
    store = DatasetStore(tiny_data)
    page = store.page("train", 0, 10)
    assert page["total"] == 25 and len(page["items"]) == 10 and page["next_offset"] == 10
    item = page["items"][0]
    assert set(item) == {"id", "text", "label_id", "label"}
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
