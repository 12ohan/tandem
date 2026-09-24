from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple


def sample_gumbel_max(
    logits: torch.Tensor,
    temperature: float = 1.0,
    mask_id: int | None = None,
) -> torch.Tensor:
    """Sample tokens via log-space Gumbel-max trick without float64 exp allocations.
    
    Preserves logit immutability by masking on a cloned tensor copy.
    """
    if temperature <= 0.0:
        if mask_id is not None:
            # Clone before masking to keep original logits uncorrupted
            cloned = logits.clone()
            cloned[..., mask_id] = -float("inf")
            return torch.argmax(cloned, dim=-1)
        return torch.argmax(logits, dim=-1)

    # Clone before masking to avoid mutating caller's tensor
    work_logits = logits.clone()
    if mask_id is not None:
        work_logits[..., mask_id] = -float("inf")

    # Log-space Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
    # Using float32 for stable uniform random sampling
    uniform = torch.rand(work_logits.shape, device=work_logits.device, dtype=torch.float32)
    uniform = torch.clamp(uniform, min=1e-10, max=1.0 - 1e-7)
    gumbel_noise = -torch.log(-torch.log(uniform)).to(work_logits.dtype)

    scaled_logits = work_logits / max(temperature, 1e-5)
    return torch.argmax(scaled_logits + gumbel_noise, dim=-1)


def sample_residual(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    mask_id: int | None = None,
) -> torch.Tensor:
    """Sample from the true residual speculative distribution on draft rejection:
    
        P(y) = max(0, p(y) - q(y)) / sum(max(0, p(y') - q(y')))
        
    Guarantees that the combined proposal + verification process exactly recovers
    the target distribution p without substitution bias under Leviathan ratio-test
    speculative decoding (Scheme b).
    
    Args:
        target_probs: Target distribution p(y), shape [..., vocab_size]
        draft_probs: Proposal distribution q(y), shape [..., vocab_size]
        mask_id: Optional token ID to explicitly exclude from sampling support (S-2)
    """
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    
    if mask_id is not None:
        residual = residual.clone()
        residual[..., mask_id] = 0.0

    norm = torch.sum(residual, dim=-1, keepdim=True)
    
    # In case residual sums to near zero due to numerical precision, fallback to target
    fallback_mask = norm <= 1e-8
    
    fallback_target = target_probs
    if mask_id is not None:
        fallback_target = target_probs.clone()
        fallback_target[..., mask_id] = 0.0
        fallback_norm = torch.sum(fallback_target, dim=-1, keepdim=True)
        fallback_target = fallback_target / torch.clamp(fallback_norm, min=1e-8)

    safe_residual = torch.where(fallback_mask, fallback_target, residual / torch.clamp(norm, min=1e-8))
    
    return torch.multinomial(safe_residual, num_samples=1).squeeze(-1)

