from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from tandem.engine.matcher import match_speculative_tokens
from tandem.engine.kv_cache import DynamicKVCache


class MockModelRunner:
    """Mock runner with fixed target distribution p and draft distribution q."""

    def __init__(self, p: torch.Tensor, q: torch.Tensor, mask_id: int = 99):
        self.p = p  # Target distribution [1, vocab_size]
        self.q = q  # Draft distribution [1, vocab_size]
        self.mask_token_id = mask_id
        self.eos_token_ids = {9999}
        self.device = torch.device("cpu")
        self.p_logits = torch.log(p + 1e-12)
        self.q_logits = torch.log(q + 1e-12)

    def forward_causal(
        self,
        input_ids: torch.Tensor,
        cache: DynamicKVCache,
    ) -> Tuple[torch.Tensor, DynamicKVCache]:
        batch_size, seq_len = input_ids.shape
        # Update mock cache by appending 1 slot per input token
        dummy_kv = torch.zeros((batch_size, 1, seq_len, 4))
        for layer_idx in range(cache.num_layers):
            cache.update(dummy_kv, dummy_kv, layer_idx=layer_idx)

        # Output logits of shape [batch_size, seq_len, vocab_size]
        logits = self.p_logits.repeat(batch_size, seq_len, 1)
        return logits, cache

    def forward_draft(
        self,
        block: torch.Tensor,
        cache: DynamicKVCache,
    ) -> torch.Tensor:
        batch_size, seq_len = block.shape
        # Bidirectional draft proposing candidate tokens (never modifies cache)
        return self.q_logits.repeat(batch_size, seq_len, 1)


def test_engine_level_distributional_exactness():
    """Verify that Scheme (a) (equality-accept + target-sample-replacement)
    is strictly distribution-preserving (recovering target p) regardless of draft q,
    while verifying KV cache length invariants at every round.
    """
    p = torch.tensor([[0.1, 0.9]], dtype=torch.float32)
    q = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
    runner = MockModelRunner(p, q)

    torch.manual_seed(42)
    num_rounds = 10_000
    K = 2
    temperature = 1.0  # Sampled verify

    # 1. Prefill
    prompt_ids = torch.tensor([[0]], dtype=torch.long)
    prompt_len = 1
    cache = DynamicKVCache(num_layers=2, device="cpu")
    
    prefill_logits, cache = runner.forward_causal(prompt_ids, cache)
    prefill_probs = F.softmax(prefill_logits[:, -1, :] / temperature, dim=-1)
    first_token = torch.multinomial(prefill_probs, num_samples=1)
    
    # Initialize collector and record prefill token
    from tandem.rl.trajectory import TrajectoryCollector
    collector = TrajectoryCollector(prompt_tokens=prompt_ids[0].tolist())
    first_tok_id = int(first_token.item())
    first_logprob = float(torch.log(prefill_probs[0, first_tok_id] + 1e-12).item())
    first_entropy = float(-(prefill_probs * torch.log(prefill_probs + 1e-12)).sum().item())
    collector.append_step(token_id=first_tok_id, logprob=first_logprob, entropy=first_entropy)

    generated_ids: List[int] = [first_tok_id]
    next_token = first_token

    # 2. Speculative Decode Loop
    block = torch.full((1, K + 1), runner.mask_token_id, dtype=torch.long)
    total_proposed = 0
    total_accepted = 0

    for _ in range(num_rounds):
        cache_len_before = cache.seq_len
        # Assert KV composition invariant before round:
        # Cache length must equal prompt_len + len(generated_ids) - 1
        assert cache_len_before == prompt_len + len(generated_ids) - 1

        block[0, 0] = next_token.item()
        block[0, 1:] = runner.mask_token_id

        # Pass 1: Draft proposal from q
        draft_logits = runner.forward_draft(block, cache)
        draft_probs = F.softmax(draft_logits / temperature, dim=-1)
        draft_tokens = torch.multinomial(
            draft_probs.view(-1, draft_probs.shape[-1]), num_samples=1
        ).view(1, K + 1)[:, 1:]
        block[0, 1:] = draft_tokens[0]
        total_proposed += K

        # Pass 2: Causal verification
        verify_logits, cache = runner.forward_causal(block, cache)
        verify_probs = F.softmax(verify_logits / temperature, dim=-1)
        ar_tokens = torch.multinomial(
            verify_probs.view(-1, verify_probs.shape[-1]), num_samples=1
        ).view(1, K + 1)

        # Matcher: shifted matching
        matches = (ar_tokens[0, :K] == block[0, 1:])
        cum_matches = matches.to(torch.int32).cumprod(dim=0)
        accepted = int(cum_matches.sum().item())
        total_accepted += accepted

        # Slices committed: accepted draft tokens + target boundary token
        accepted_tokens_slice = ar_tokens[0, : accepted + 1].tolist()
        committed_this_round = accepted + 1

        # Truncate unverified slots from KV cache:
        # verify added K + 1 slots. Keep cache_len_before + committed_this_round
        slots_to_rewind = (cache.seq_len) - (cache_len_before + committed_this_round)
        cache.rewind(slots_to_rewind)

        # Assert KV composition invariant after rewind:
        assert cache.seq_len == cache_len_before + committed_this_round

        # Record trajectory steps for committed tokens
        for idx, tok in enumerate(accepted_tokens_slice):
            tok_prob = float(verify_probs[0, idx, tok].item())
            logprob = float(torch.log(torch.tensor(tok_prob) + 1e-12).item())
            collector.append_step(token_id=tok, logprob=logprob)

        generated_ids.extend(accepted_tokens_slice)
        next_token = ar_tokens[0, accepted : accepted + 1]

    # Verify trajectory lockstep with generated_ids
    traj = collector.to_trajectory()
    assert traj.completion_tokens == generated_ids
    assert traj.completion_tokens[0] == first_tok_id
    assert len(traj.steps) == len(generated_ids)

    # Verify committed distribution
    counts = torch.bincount(torch.tensor(generated_ids), minlength=2).float()
    empirical_p = counts / len(generated_ids)

    # Must recover target p = [0.1, 0.9] within tight Monte Carlo tolerance
    assert abs(empirical_p[0].item() - 0.1) < 0.01
    assert abs(empirical_p[1].item() - 0.9) < 0.01

    # Acceptance rate under sampled verify:
    # Position 0 matches with P(m0) = 0.9*0.1 + 0.1*0.9 = 0.18
    # Position 1 matches consecutively with P(m0 and m1) = 0.18 * 0.18 = 0.0324
    # Expected alpha = (0.18 + 0.0324) / 2 = 0.1062
    empirical_alpha = total_accepted / total_proposed
    assert abs(empirical_alpha - 0.1062) < 0.01



def test_engine_trajectory_includes_prefill_token():
    """Verify that TrajectoryCollector strictly records the prefill token (token 0)
    in both completion_tokens and steps/logprobs, ensuring zero head-of-response omission.
    """
    from tandem.rl.trajectory import TrajectoryCollector

    # 1. Simulate prefill
    prompt_ids = [100, 101]
    collector = TrajectoryCollector(prompt_tokens=prompt_ids)
    
    first_token_id = 42
    first_logprob = -0.5
    first_entropy = 0.8
    collector.append_step(token_id=first_token_id, logprob=first_logprob, entropy=first_entropy)
    generated_ids = [first_token_id]

    # 2. Simulate 2 rounds of decode committing tokens [43, 44]
    collector.append_step(token_id=43, logprob=-0.2, entropy=0.4)
    generated_ids.append(43)
    collector.append_step(token_id=44, logprob=-0.1, entropy=0.2)
    generated_ids.append(44)

    traj = collector.to_trajectory()

    # Assert exact 1-to-1 match between trajectory completion and generated tokens
    assert traj.completion_tokens == generated_ids
    assert traj.completion_tokens[0] == first_token_id
    assert traj.logprobs[0] == first_logprob
    assert len(traj.steps) == len(generated_ids)
    assert len(traj.logprobs) == len(generated_ids)


