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


def test_sampler_residual_canary_distribution():
    """Verify Leviathan canary: q = [0.9, 0.1], p = [0.1, 0.9].
    
    Residual r(y) = max(0, p(y) - q(y)) / sum(...) must evaluate to [0.0, 1.0].
    Simulated speculative acceptance-rejection must recover p = [0.1, 0.9].
    """
    p = torch.tensor([[0.1, 0.9]], dtype=torch.float32)
    q = torch.tensor([[0.9, 0.1]], dtype=torch.float32)

    # 1. Analytic residual check
    residual = torch.clamp(p - q, min=0.0)
    norm = torch.sum(residual, dim=-1, keepdim=True)
    r = residual / norm
    assert torch.allclose(r, torch.tensor([[0.0, 1.0]]), atol=1e-6)

    # 2. End-to-end Monte Carlo simulation of speculative acceptance/rejection
    torch.manual_seed(42)
    num_samples = 20_000
    p_flat = p[0]
    q_flat = q[0]
    
    outputs = []
    for _ in range(num_samples):
        # Propose draft token from q
        draft_tok = torch.multinomial(q_flat, num_samples=1).item()
        
        # Speculative acceptance probability: min(1.0, p(x) / q(x))
        accept_prob = min(1.0, (p_flat[draft_tok] / q_flat[draft_tok]).item())
        if torch.rand(1).item() < accept_prob:
            outputs.append(draft_tok)
        else:
            # Rejection: sample from residual
            res_tok = sample_residual(p, q).item()
            outputs.append(res_tok)

    counts = torch.bincount(torch.tensor(outputs), minlength=2).float()
    empirical_p = counts / num_samples

    # Must match p = [0.1, 0.9] within tight Monte Carlo bounds (3 sigma ~ 0.006)
    assert abs(empirical_p[0].item() - 0.1) < 0.01
    assert abs(empirical_p[1].item() - 0.9) < 0.01

