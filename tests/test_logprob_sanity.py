from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import pytest
import torch
import torch.nn.functional as F

# Ensure tandem package is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tandem import TandemEngine, EngineConfig, load_amboss_questions

AMBOSS_QUESTIONS_DIR = Path(
    "/Users/rohanmaster/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Resources/data/raw/amboss_qbank/questions"
)


@pytest.mark.integration
def test_amboss_teacher_forced_gold_logprob_sanity():
    """Teacher-force a gold Amboss completion through NLD-3B on MPS
    and assert finite per-token logprobs in [-20, 0] with no NaNs.
    """
    if not AMBOSS_QUESTIONS_DIR.exists():
        pytest.skip("Amboss questions directory not found on disk")

    questions = load_amboss_questions(AMBOSS_QUESTIONS_DIR, max_questions=1)
    assert len(questions) >= 1, "Must load at least one Amboss question"
    item = questions[0]

    # Initialize TandemEngine on default device (MPS on Apple Silicon)
    config = EngineConfig(temperature=1.0, return_logprob=True)
    engine = TandemEngine(config)
    tokenizer = engine.runner.tokenizer
    device = engine.runner.device

    # Format teacher-forced sequence: prompt + gold answer tag
    prompt_text = item.prompt
    completion_text = f"<answer>{item.ground_truth}</answer>"

    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    comp_tokens = tokenizer.encode(completion_text, add_special_tokens=False)

    P = len(prompt_tokens)
    T = len(comp_tokens)
    assert P > 0, "Prompt tokens must be non-empty"
    assert T > 0, "Completion tokens must be non-empty"

    full_tokens = prompt_tokens + comp_tokens
    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)

    # 1. Batched causal forward pass
    with torch.no_grad():
        logits, _ = engine.runner.forward_causal(input_ids, use_cache=False)

    assert logits.shape == (1, P + T, engine.runner.model.config.vocab_size)

    # Completion logits predict comp_tokens[0..T-1] at positions [P-1 .. P+T-2]
    comp_logits = logits[0, P - 1 : P + T - 1, :].float()
    target_ids = torch.tensor(comp_tokens, dtype=torch.long, device=device)

    # 2. Compute log probabilities
    log_probs = F.log_softmax(comp_logits, dim=-1)
    token_logprobs = log_probs[torch.arange(T, device=device), target_ids]

    # 3. Assertions requested by frontier review:
    # (i) All logprobs must be finite (no NaNs, no infinities)
    assert torch.isfinite(token_logprobs).all(), "All per-token logprobs must be finite"
    assert not torch.isnan(token_logprobs).any(), "Per-token logprobs must contain no NaNs"

    # (ii) All logprobs must be <= 0.0 (probability <= 1.0)
    assert (token_logprobs <= 0.0).all(), "Log-probabilities must be <= 0.0"

    # (iii) Strict bounds check: in [-20.0, 0.0]
    min_logp = token_logprobs.min().item()
    max_logp = token_logprobs.max().item()
    mean_logp = token_logprobs.mean().item()

    assert min_logp >= -20.0, (
        f"Minimum logprob {min_logp:.4f} dropped below numerical floor -20.0"
    )
    assert max_logp <= 0.0, f"Maximum logprob {max_logp:.4f} exceeded 0.0"
    assert -15.0 <= mean_logp <= 0.0, (
        f"Mean logprob {mean_logp:.4f} outside expected range [-15.0, 0.0]"
    )

    # (iv) Verification printout for receipts
    print(f"Amboss Question ID: {item.metadata.get('qid', 'unknown')}")
    print(f"Gold Diagnosis:     {item.ground_truth}")
    print(f"Prompt Length:      {P} tokens")
    print(f"Completion Length:  {T} tokens")
    print(f"Min Token Logprob:  {min_logp:.4f}")
    print(f"Max Token Logprob:  {max_logp:.4f}")
    print(f"Mean Token Logprob: {mean_logp:.4f}")


def run_sanity_standalone():
    print("=" * 80)
    print("AMBOSS TEACHER-FORCED GOLD LOGPROB SANITY CHECK (MPS)")
    print("=" * 80)
    test_amboss_teacher_forced_gold_logprob_sanity()
    print("=" * 80)
    print("RECEIPT: ALL AMBOSS TEACHER-FORCED LOGPROBS FINITE IN [-20, 0], NO NANS.")
    print("=" * 80)


if __name__ == "__main__":
    run_sanity_standalone()
