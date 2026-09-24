from __future__ import annotations

import pytest
import torch
from tandem.engine.matcher import match_speculative_tokens
from tandem.engine.sampler import sample_gumbel_max, sample_residual


def test_matcher_full_acceptance():
    draft = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    target = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64)
    
    result = match_speculative_tokens(draft, target)
    assert result.num_accepted == 4
    assert result.accepted_tokens == [10, 20, 30, 40]
    assert result.bonus_token == 50
    assert result.total_emitted == 5


def test_matcher_partial_acceptance():
    draft = torch.tensor([10, 20, 999, 40], dtype=torch.int64)
    target = torch.tensor([10, 20, 35, 45, 55], dtype=torch.int64)
    
    result = match_speculative_tokens(draft, target)
    assert result.num_accepted == 2
    assert result.accepted_tokens == [10, 20]
    assert result.bonus_token == 35  # target[2]
    assert result.total_emitted == 3


def test_matcher_zero_acceptance():
    draft = torch.tensor([999, 20, 30, 40], dtype=torch.int64)
    target = torch.tensor([12, 22, 32, 42, 52], dtype=torch.int64)
    
    result = match_speculative_tokens(draft, target)
    assert result.num_accepted == 0
    assert result.accepted_tokens == []
    assert result.bonus_token == 12  # target[0]
    assert result.total_emitted == 1


def test_sampler_gumbel_no_mutation():
    logits = torch.randn(4, 100)
    original = logits.clone()
    
    sampled = sample_gumbel_max(logits, temperature=0.7, mask_id=50)
    
    # Assert logits was never modified in place
    assert torch.equal(logits, original)
    assert sampled.shape == (4,)
    # Verify masked position is never sampled
    assert not (sampled == 50).any()


def test_sampler_residual():
    target = torch.tensor([[0.1, 0.2, 0.4, 0.3]], dtype=torch.float32)
    draft = torch.tensor([[0.2, 0.2, 0.2, 0.4]], dtype=torch.float32)
    
    sampled = sample_residual(target, draft)
    assert sampled.item() in (2,)  # only index 2 has target > draft (0.4 > 0.2)
