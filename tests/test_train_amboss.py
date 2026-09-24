from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tandem.rl.amboss import AmbossDifferentialReward
from tandem.rl.dataset import PromptItem
from tandem.rl.trainer import DiffuGRPOTrainer, GRPOTrainerConfig
from tandem.rl.trajectory import TrajectoryCollector
from tandem.engine.candidate_scorer import CandidateDifferentialScorer
from tandem.engine.spec import GenerationOutput
from tandem.train_amboss import parse_args, run_eval_probe, run_preflight_checks


class MockTokenizer:
    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<answer>": 3,
            "acute": 10,
            "cholecystitis": 11,
            "appendicitis": 12,
            "pneumonia": 13,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = [1] if add_special_tokens else []
        clean = text.replace("<answer>", " <answer> ").replace("\n", " ").lower()
        for w in clean.split():
            tok_id = self.vocab.get(w, (abs(hash(w)) % 40) + 15)
            tokens.append(tok_id)
        return tokens or [1]

    def decode(self, token_ids: List[int]) -> str:
        return " ".join([self.inv_vocab.get(t, f"tok_{t}") for t in token_ids])


class MockDynamicCache:
    def __init__(self, initial_len: int = 0):
        self._seq_len = initial_len

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_len

    def crop(self, max_length: int) -> int:
        self._seq_len = max_length
        return self._seq_len


class MockTrainModel(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.Linear(hidden_dim, hidden_dim)
        self.diffusion_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        h = self.encoder(h)
        return self.diffusion_head(h)


class MockTrainRunner:
    def __init__(self, model: nn.Module):
        self.model = model
        self.tokenizer = MockTokenizer()
        self.device = torch.device("cpu")
        self.mask_token_id = 63
        self.eos_token_ids = {0, 2}

    def forward_causal(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[MockDynamicCache] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[MockDynamicCache]]:
        B, seq_len = input_ids.shape
        logits = self.model(input_ids)

        # Ensure stable finite values for logprobs
        logits = logits.clamp(-15.0, 15.0)

        if not use_cache:
            return logits, None

        if past_key_values is None:
            cache = MockDynamicCache(initial_len=seq_len)
        else:
            cache = past_key_values
            cache._seq_len += seq_len

        return logits, cache


class MockTrainEngine:
    def __init__(self, runner: MockTrainRunner):
        self.runner = runner

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 8,
        block_size: int = 2,
        temperature: float = 1.0,
        return_logprob: bool = True,
    ) -> GenerationOutput:
        prompt_tokens = self.runner.tokenizer.encode(prompt, add_special_tokens=True)
        # Generate answer tags for testing reward
        comp_text = "<think>Patient has acute cholecystitis.</think><answer>Acute cholecystitis</answer>"
        comp_tokens = self.runner.tokenizer.encode(comp_text, add_special_tokens=False)

        collector = TrajectoryCollector(prompt_tokens, temperature=temperature)
        for tok in comp_tokens:
            collector.append_step(token_id=tok, logprob=-1.20, entropy=0.45)

        return GenerationOutput(
            text=comp_text,
            token_ids=comp_tokens,
            num_generated_tokens=len(comp_tokens),
            num_forward_passes=2,
            wall_time=0.01,
            tok_per_sec=400.0,
            acceptance_rate=0.80,
            trajectory=collector.to_trajectory(),
        )


def test_train_amboss_parse_args_defaults():
    """Verify default CLI argument parsing."""
    args = parse_args([])
    assert args.steps == 50
    assert args.group_size == 4
    assert args.batch_size == 1
    assert args.lr == 1e-5
    assert args.kl_beta == 0.04
    assert args.curriculum is True
    assert args.format_primed is False
    assert args.skip_preflight is False


def test_train_amboss_parse_args_custom():
    """Verify customized CLI argument parsing."""
    custom_argv = [
        "--steps", "25",
        "--group-size", "8",
        "--batch-size", "2",
        "--lr", "2e-5",
        "--kl-beta", "0.08",
        "--no-curriculum",
        "--format-primed",
        "--skip-preflight",
        "--save-dir", "checkpoints/test_run",
    ]
    args = parse_args(custom_argv)
    assert args.steps == 25
    assert args.group_size == 8
    assert args.batch_size == 2
    assert args.lr == 2e-5
    assert args.kl_beta == 0.08
    assert args.curriculum is False
    assert args.format_primed is True
    assert args.skip_preflight is True
    assert args.save_dir == "checkpoints/test_run"


def test_train_amboss_preflight_checks():
    """Verify run_preflight_checks passes with mock model and engine."""
    model = MockTrainModel()
    runner = MockTrainRunner(model)
    engine = MockTrainEngine(runner)

    sample_item = PromptItem(
        prompt="Patient presents with acute RUQ pain.",
        ground_truth="Acute cholecystitis",
        metadata={
            "all_candidates": ["Acute cholecystitis", "Appendicitis", "Pneumonia"],
        },
    )

    # Should run without AssertionError
    run_preflight_checks(engine, sample_item)


def test_train_amboss_eval_probe():
    """Verify run_eval_probe computes Top-k, gold probability mass, and entropy."""
    model = MockTrainModel()
    runner = MockTrainRunner(model)
    engine = MockTrainEngine(runner)
    scorer = CandidateDifferentialScorer(engine)

    eval_slice = [
        PromptItem(
            prompt="Vignette 1",
            ground_truth="Acute cholecystitis",
            metadata={"all_candidates": ["Acute cholecystitis", "Appendicitis"]},
        ),
        PromptItem(
            prompt="Vignette 2",
            ground_truth="Appendicitis",
            metadata={"all_candidates": ["Appendicitis", "Pneumonia"]},
        ),
    ]

    metrics = run_eval_probe(scorer, eval_slice, step=10)
    assert "eval_top1" in metrics
    assert "eval_top3" in metrics
    assert "eval_gold_mass" in metrics
    assert "eval_gold_rank" in metrics
    assert "eval_entropy" in metrics
    assert metrics["eval_cases"] == 2.0
    assert 0.0 <= metrics["eval_top1"] <= 1.0
    assert 0.0 <= metrics["eval_gold_mass"] <= 1.0
    assert metrics["eval_entropy"] >= 0.0


def test_train_amboss_simulated_step():
    """Verify a complete simulated RL training step with DiffuGRPO and AmbossDifferentialReward."""
    model = MockTrainModel()
    runner = MockTrainRunner(model)
    engine = MockTrainEngine(runner)

    reward_fn = AmbossDifferentialReward(
        gold_reward=2.0,
        ruleout_credit_per_candidate=0.20,
        max_ruleout_credit=0.80,
        gate_on_gold=False,
    )

    config = GRPOTrainerConfig(
        group_size=2,
        batch_size=1,
        learning_rate=1e-4,
        max_new_tokens=8,
        block_size=2,
        train_head_only=True,
    )

    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config)

    batch_prompts = [
        PromptItem(
            prompt="Clinical vignette",
            ground_truth="Acute cholecystitis",
            metadata={
                "differential_candidates": ["Appendicitis"],
                "all_candidates": ["Acute cholecystitis", "Appendicitis"],
                "distractor_buts": {"Appendicitis": "presents with RLQ pain"},
            },
        )
    ]

    metrics = trainer.step(batch_prompts, global_step=1)

    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "kl_loss" in metrics
    assert "mean_reward" in metrics
    assert metrics["mean_reward"] >= 2.0
    assert "acceptance_rate" in metrics
