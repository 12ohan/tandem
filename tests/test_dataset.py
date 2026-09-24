from __future__ import annotations

import json
from pathlib import Path
import pytest
from tandem.rl.dataset import PromptDataset, PromptItem


def test_prompt_item_parsing():
    # Test standard keys
    item1 = PromptItem.from_dict({"prompt": "Hello", "ground_truth": "World"})
    assert item1.prompt == "Hello"
    assert item1.ground_truth == "World"
    assert item1.metadata == {}

    # Test alternative instruction / answer keys
    item2 = PromptItem.from_dict({"instruction": "Calculate 2+2", "answer": "4", "difficulty": "easy"})
    assert item2.prompt == "Calculate 2+2"
    assert item2.ground_truth == "4"
    assert item2.metadata == {"difficulty": "easy"}

    # Test missing prompt raises KeyError
    with pytest.raises(KeyError):
        PromptItem.from_dict({"data": "No prompt here"})


def test_prompt_dataset_from_list():
    raw = [
        "Simple string prompt",
        {"prompt": "Dict prompt", "ground_truth": "A"},
        PromptItem(prompt="Item prompt", ground_truth="B", metadata={"source": "test"}),
    ]
    ds = PromptDataset.from_list(raw)
    assert len(ds) == 3
    assert ds[0].prompt == "Simple string prompt"
    assert ds[1].prompt == "Dict prompt"
    assert ds[1].ground_truth == "A"
    assert ds[2].prompt == "Item prompt"
    assert ds[2].metadata["source"] == "test"

    # Slice indexing
    sub_ds = ds[:2]
    assert isinstance(sub_ds, PromptDataset)
    assert len(sub_ds) == 2


def test_prompt_dataset_jsonl_roundtrip(tmp_path: Path):
    jsonl_file = tmp_path / "test_prompts.jsonl"
    lines = [
        {"prompt": "What is the capital of France?", "ground_truth": "Paris"},
        {"prompt": "What is the capital of Germany?", "ground_truth": "Berlin", "extra": 123},
    ]
    with open(jsonl_file, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    ds = PromptDataset.from_jsonl(jsonl_file)
    assert len(ds) == 2
    assert ds[0].prompt == "What is the capital of France?"
    assert ds[0].ground_truth == "Paris"
    assert ds[1].metadata == {"extra": 123}

    # Export to new jsonl
    out_file = tmp_path / "exported.jsonl"
    ds.to_jsonl(out_file)

    ds_reloaded = PromptDataset.from_jsonl(out_file)
    assert len(ds_reloaded) == 2
    assert ds_reloaded[1].prompt == "What is the capital of Germany?"


def test_prompt_dataset_batches():
    items = [PromptItem(prompt=f"Prompt {i}") for i in range(10)]
    ds = PromptDataset(items)

    batches = list(ds.iter_batches(batch_size=3, shuffle=False))
    assert len(batches) == 4
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    assert len(batches[3]) == 1
    assert batches[0][0].prompt == "Prompt 0"

    # Shuffled iteration
    shuffled_batches = list(ds.iter_batches(batch_size=5, shuffle=True, seed=42))
    assert len(shuffled_batches) == 2
    assert len(shuffled_batches[0]) == 5
