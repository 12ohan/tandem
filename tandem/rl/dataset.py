from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union


@dataclass
class PromptItem:
    """Represents a single prompt instance for RL training."""

    prompt: str
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {"prompt": self.prompt}
        if self.ground_truth is not None:
            result["ground_truth"] = self.ground_truth
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptItem:
        """Parse PromptItem from dictionary with flexible key resolution."""
        # Find prompt key
        prompt_val = None
        for key in ("prompt", "instruction", "question", "input", "problem"):
            if key in data and data[key] is not None:
                prompt_val = str(data[key])
                break

        if prompt_val is None:
            raise KeyError(
                f"Could not find prompt in data keys: {list(data.keys())}. "
                "Expected one of: 'prompt', 'instruction', 'question', 'input', 'problem'."
            )

        # Find target/ground_truth key
        target_val = None
        for key in ("ground_truth", "answer", "target", "solution", "label", "output"):
            if key in data and data[key] is not None:
                target_val = str(data[key])
                break

        # Collect residual metadata
        reserved_keys = {
            "prompt", "instruction", "question", "input", "problem",
            "ground_truth", "answer", "target", "solution", "label", "output", "metadata"
        }
        metadata = {}
        if "metadata" in data and isinstance(data["metadata"], dict):
            metadata.update(data["metadata"])
        for k, v in data.items():
            if k not in reserved_keys:
                metadata[k] = v

        return cls(prompt=prompt_val, ground_truth=target_val, metadata=metadata)


class PromptDataset(Sequence[PromptItem]):
    """Extensible dataset container for DiffuGRPO prompts and targets."""

    def __init__(self, items: Optional[List[PromptItem]] = None):
        self._items: List[PromptItem] = list(items) if items is not None else []

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: Union[int, slice]) -> Union[PromptItem, PromptDataset]:
        if isinstance(index, slice):
            return PromptDataset(self._items[index])
        return self._items[index]

    def __iter__(self) -> Iterator[PromptItem]:
        return iter(self._items)

    def append(self, item: Union[PromptItem, str, Dict[str, Any]]) -> None:
        if isinstance(item, PromptItem):
            self._items.append(item)
        elif isinstance(item, str):
            self._items.append(PromptItem(prompt=item))
        elif isinstance(item, dict):
            self._items.append(PromptItem.from_dict(item))
        else:
            raise TypeError(f"Unsupported item type: {type(item)}")

    def extend(self, items: Sequence[Union[PromptItem, str, Dict[str, Any]]]) -> None:
        for item in items:
            self.append(item)

    @classmethod
    def from_list(
        cls, items: Sequence[Union[str, Dict[str, Any], PromptItem]]
    ) -> PromptDataset:
        """Create a dataset from a Python list of strings, dicts, or PromptItems."""
        dataset = cls()
        dataset.extend(items)
        return dataset

    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> PromptDataset:
        """Load prompts from a JSON Lines (.jsonl) file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        items: List[PromptItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        items.append(PromptItem.from_dict(data))
                    elif isinstance(data, str):
                        items.append(PromptItem(prompt=data))
                    else:
                        raise ValueError(f"Line {line_idx + 1} is neither dict nor string: {data}")
                except Exception as e:
                    raise ValueError(f"Failed to parse line {line_idx + 1} in {path}: {e}") from e

        return cls(items)

    def to_jsonl(self, path: Union[str, Path]) -> None:
        """Export prompt dataset to a JSON Lines file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in self._items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> PromptDataset:
        """Load prompts from a JSON file (array of strings or dicts)."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected top-level JSON array in {path}, got {type(data)}")

        return cls.from_list(data)

    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """Export prompt dataset to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([item.to_dict() for item in self._items], f, indent=indent, ensure_ascii=False)

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        prompt_col: str = "prompt",
        target_col: Optional[str] = None,
    ) -> PromptDataset:
        """Load prompts from a CSV or TSV file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

        items: List[PromptItem] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if prompt_col not in row:
                    raise KeyError(f"Column '{prompt_col}' not found in CSV. Found: {list(row.keys())}")
                prompt = row[prompt_col]
                target = row.get(target_col, None) if target_col else None
                metadata = {k: v for k, v in row.items() if k not in (prompt_col, target_col)}
                items.append(PromptItem(prompt=prompt, ground_truth=target, metadata=metadata))

        return cls(items)

    def iter_batches(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> Iterator[List[PromptItem]]:
        """Yield batches of PromptItems for training steps."""
        indices = list(range(len(self._items)))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            yield [self._items[idx] for idx in batch_indices]
