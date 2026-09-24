#!/usr/bin/env python3
"""DiffuGRPO Training Demonstration Script for Tandem.

Shows how to:
1. Load or feed custom prompts and ground truths via PromptDataset.
2. Define domain-specific or composite reward rubrics.
3. Configure and run DiffuGRPOTrainer with self-speculative rollouts on Apple Silicon.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure tandem package is importable when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from tandem import (
    BaseReward,
    CompositeReward,
    DiffuGRPOTrainer,
    FormatReward,
    GRPOTrainerConfig,
    PromptDataset,
    PromptItem,
    RegexMatchReward,
    TandemEngine,
)


class DomainKeywordReward(BaseReward):
    """Custom project-specific reward: checks if key clinical / conceptual terms appear."""

    def __init__(self, key_terms_weight: float = 0.5):
        self.key_terms_weight = key_terms_weight

    def compute_reward(self, prompt: str, completion: str, **kwargs) -> float:
        target = kwargs.get("ground_truth", None)
        if not target:
            return 0.0

        # Case-insensitive substring match
        if str(target).strip().lower() in completion.lower():
            return self.key_terms_weight
        return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="DiffuGRPO Training with Tandem")
    parser.add_argument(
        "--dataset",
        type=str,
        default="examples/sample_prompts.jsonl",
        help="Path to JSONL prompt file.",
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps.")
    parser.add_argument("--group-size", type=int, default=4, help="Completions per prompt (G).")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max new tokens per rollout.")
    parser.add_argument("--block-size", type=int, default=4, help="Speculative draft size (K).")
    parser.add_argument("--head-only", action="store_true", help="Train diffusion head only.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset from: {args.dataset}")
    dataset_path = Path(args.dataset)
    if dataset_path.is_file():
        dataset = PromptDataset.from_jsonl(dataset_path)
    else:
        # Fallback inline dataset
        dataset = PromptDataset.from_list([
            {"prompt": "Calculate 15 * 12.", "ground_truth": "180"},
            {"prompt": "What is the capital of Japan?", "ground_truth": "Tokyo"},
        ])

    print(f"Loaded {len(dataset)} prompt items.")

    # 1. Compose Reward Evaluator
    # Format reward (0.5 for <think>...</think><answer>...</answer>) + Domain match (0.5 for correct answer)
    reward_fn = CompositeReward([
        (FormatReward(think_reward=0.25, answer_reward=0.25), 1.0),
        (DomainKeywordReward(key_terms_weight=0.5), 1.0),
    ])

    # 2. Configure GRPO Trainer
    config = GRPOTrainerConfig(
        group_size=args.group_size,
        learning_rate=args.lr,
        max_new_tokens=args.max_tokens,
        block_size=args.block_size,
        train_head_only=args.head_only,
        log_interval=1,
    )

    print("Initializing TandemEngine (Nemotron-Labs-Diffusion-3B)...")
    engine = TandemEngine()

    print("Initializing DiffuGRPOTrainer...")
    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config)

    print(f"Starting DiffuGRPO training for {args.steps} steps...")
    history = trainer.train(dataset, steps=args.steps)
    print(f"Completed {len(history)} training steps.")


if __name__ == "__main__":
    main()
