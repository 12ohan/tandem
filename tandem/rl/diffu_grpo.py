from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Literal
import torch
import torch.nn.functional as F

from tandem.rl.trajectory import Trajectory


@dataclass
class GRPOMetrics:
    total_loss: float
    policy_loss: float
    kl_loss: float
    mean_advantage: float
    clip_fraction: float


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-6,
    normalize_by_std: bool = True,
) -> torch.Tensor:
    r"""Compute within-group normalized advantages for GRPO.
    
    A_i = (R_i - mean(R)) / (std(R) + eps) if normalize_by_std else (R_i - mean(R))
    
    Args:
        rewards: Tensor of shape [G] representing scalar rewards for G rollouts of a single prompt.
        eps: Small constant for numerical stability.
        normalize_by_std: If True (default), divides by empirical standard deviation.
                          If False, mean-centers only (ablation against low-variance overweighting).
        
    Returns:
        advantages: Tensor of shape [G]. If all rewards are identical or G <= 1, returns zeros (no NaN).
    """
    assert rewards.dim() == 1, f"Expected 1D rewards tensor [G], got shape {rewards.shape}"
    G = rewards.shape[0]
    if G <= 1:
        return torch.zeros_like(rewards)

    mean_r = torch.mean(rewards)
    diff = rewards - mean_r

    if not normalize_by_std:
        return diff

    std_r = torch.std(rewards, unbiased=False)
    if std_r < 1e-8:
        # All completions scored identically: 0 / eps = 0.0, zero advantage gradient
        return torch.zeros_like(rewards)

    advantages = diff / (std_r + eps)
    return advantages


def compute_grpo_loss(
    policy_logprobs: List[torch.Tensor],
    old_logprobs: List[torch.Tensor],
    advantages: torch.Tensor,
    ref_logprobs: Optional[List[torch.Tensor]] = None,
    clip_eps: float = 0.2,
    beta_kl: float = 0.04,
    divisor_mode: Literal["token_mean", "group_mean"] = "token_mean",
) -> Tuple[torch.Tensor, GRPOMetrics]:
    r"""Compute Group Relative Policy Optimization (GRPO) loss with PPO clipping and additive reference KL.
    
    Loss Formulation:
        L = - (1 / N) \sum_{i} \sum_{t} \min(r_{i,t} A_i, clip(r_{i,t}, 1-\epsilon, 1+\epsilon) A_i)
            + \beta * (1 / N) \sum_{i} \sum_{t} D_KL( \pi_\theta || \pi_{ref} )
            
        where N = \sum_i |o_i| (total completion tokens under token_mean normalization),
        and D_KL uses Schulman's non-negative k3 estimator:
            k3 = \exp(\log \pi_{ref} - \log \pi_\theta) - (\log \pi_{ref} - \log \pi_\theta) - 1 >= 0
    
    Args:
        policy_logprobs: List of G tensors, each shape [T_i], representing log \pi_\theta(y_t)
        old_logprobs: List of G tensors, each shape [T_i], representing log \pi_old(y_t)
        advantages: Tensor of shape [G] containing normalized group advantages
        ref_logprobs: Optional list of G tensors for reference model KL penalty
        clip_eps: Clipping range (1 - clip_eps, 1 + clip_eps)
        beta_kl: KL divergence penalty weight (additive outside clip)
        divisor_mode: "token_mean" (standard GRPO token-level average) or
                      "group_mean" (completion-level average, Dr. GRPO critique ablation)
        
    Returns:
        loss: Scalar differentiable loss tensor
        metrics: GRPOMetrics summary dataclass
    """

    G = len(policy_logprobs)
    assert G == len(old_logprobs) == advantages.shape[0], (
        f"Group size mismatch: policy ({G}), old ({len(old_logprobs)}), advantages ({advantages.shape[0]})"
    )

    token_surrogate_losses = []
    token_kl_losses = []
    completion_policy_losses = []
    completion_kl_losses = []
    total_tokens = 0
    clipped_tokens = 0

    device = advantages.device

    for i in range(G):
        pi_logp = policy_logprobs[i]
        old_logp = old_logprobs[i]
        adv_i = advantages[i]
        T_i = pi_logp.shape[0]

        if T_i == 0:
            continue

        # Importance ratio: r_t = \exp(log \pi_\theta - log \pi_old)
        # Clamped in log-space to prevent exp() overflow
        log_ratio = torch.clamp(pi_logp - old_logp.detach(), min=-10.0, max=10.0)
        ratio = torch.exp(log_ratio)

        # Surrogate objectives
        surr1 = ratio * adv_i
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_i
        token_surr = -torch.min(surr1, surr2)  # shape [T_i]
        token_surrogate_losses.append(token_surr)
        completion_policy_losses.append(token_surr.mean())

        # Track clipping statistics
        with torch.no_grad():
            is_clipped = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
            clipped_tokens += int(is_clipped.sum().item())
            total_tokens += T_i

        # Reference KL divergence: D_KL(\pi_\theta || \pi_ref)
        # Using Schulman's unbiased non-negative estimator:
        # k3 = \exp(\log \pi_{ref} - \log \pi_\theta) - (\log \pi_{ref} - \log \pi_\theta) - 1 >= 0
        if ref_logprobs is not None:
            ref_logp = ref_logprobs[i].detach()
            log_ref_ratio = ref_logp - pi_logp
            token_kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1.0
            token_kl_losses.append(token_kl)
            completion_kl_losses.append(token_kl.mean())
        else:
            token_kl_losses.append(torch.zeros_like(pi_logp))
            completion_kl_losses.append(torch.tensor(0.0, device=device))

    if total_tokens == 0 or not token_surrogate_losses:
        zero = torch.tensor(0.0, requires_grad=True, device=device)
        return zero, GRPOMetrics(0.0, 0.0, 0.0, float(advantages.mean().item()), 0.0)

    if divisor_mode == "token_mean":
        all_surrogates = torch.cat(token_surrogate_losses)
        all_kl = torch.cat(token_kl_losses)
        mean_policy_loss = all_surrogates.sum() / total_tokens
        mean_kl_loss = all_kl.sum() / total_tokens
    else:  # "group_mean"
        mean_policy_loss = torch.stack(completion_policy_losses).mean()
        mean_kl_loss = torch.stack(completion_kl_losses).mean()

    # KL is additive outside the min/clip so its gradient survives token clipping
    total_loss = mean_policy_loss + beta_kl * mean_kl_loss
    clip_fraction = float(clipped_tokens / max(total_tokens, 1))

    metrics = GRPOMetrics(
        total_loss=float(total_loss.item()),
        policy_loss=float(mean_policy_loss.item()),
        kl_loss=float(mean_kl_loss.item()),
        mean_advantage=float(advantages.mean().item()),
        clip_fraction=clip_fraction,
    )

    return total_loss, metrics
