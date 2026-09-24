from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import torch


@dataclass
class MatchResult:
    accepted_tokens: List[int]
    num_accepted: int
    bonus_token: Optional[int]
    total_emitted: int


def match_speculative_tokens(
    draft_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> MatchResult:
    """Evaluate consecutive speculative token matches using shifted causal alignment.
    
    Args:
        draft_tokens: Tensor of shape [K] containing proposed candidate tokens.
        target_tokens: Tensor of shape [K+1] containing causal AR predictions,
                       where target_tokens[i] is the target distribution output
                       conditioned on [seed, draft_tokens[:i]].
                       
    Returns:
        MatchResult containing the accepted token list, counts, and boundary token.
    """
    K = draft_tokens.shape[0]
    assert target_tokens.shape[0] >= K, f"target_tokens ({target_tokens.shape[0]}) must be at least K ({K})"

    # Elementwise comparison between draft[i] and causal prediction for position i
    matches = (draft_tokens == target_tokens[:K])

    # Invariant: Must cast bool to int32 before cumprod to prevent PyTorch MPS/CPU runtime error
    cum_matches = matches.to(torch.int32).cumprod(dim=0)
    num_accepted = int(cum_matches.sum().item())

    accepted_list: List[int] = []
    bonus_token: Optional[int] = None

    if num_accepted > 0:
        accepted_list = draft_tokens[:num_accepted].tolist()

    if num_accepted < K:
        # First rejected token is replaced by the causal target prediction at position num_accepted
        # Cloned to prevent buffer clobbering
        bonus_token = int(target_tokens[num_accepted].item())
    else:
        # All K tokens accepted; attach the bonus token from position K if available
        if target_tokens.shape[0] > K:
            bonus_token = int(target_tokens[K].item())

    total_emitted = len(accepted_list) + (1 if bonus_token is not None else 0)

    return MatchResult(
        accepted_tokens=accepted_list,
        num_accepted=num_accepted,
        bonus_token=bonus_token,
        total_emitted=total_emitted,
    )
