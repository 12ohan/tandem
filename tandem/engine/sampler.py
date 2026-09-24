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
) -> torch.Tensor:
    """Sample from the true residual speculative distribution on draft rejection:
    
        P(y) = max(0, p(y) - q(y)) / sum(max(0, p(y') - q(y')))
        
    Guarantees that the combined proposal + verification process exactly recovers
    the target distribution p without substitution bias.
    """
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    norm = torch.sum(residual, dim=-1, keepdim=True)
    
    # In case residual sums to near zero due to numerical precision, fallback to target
    fallback_mask = norm <= 1e-8
    safe_residual = torch.where(fallback_mask, target_probs, residual / torch.clamp(norm, min=1e-8))
    
    return torch.multinomial(safe_residual, num_samples=1).squeeze(-1)
