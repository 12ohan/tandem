from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
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
) -> torch.Tensor:
    """Compute within-group normalized advantages for GRPO.
    
    Args:
        rewards: Tensor of shape [G] representing scalar rewards for G rollouts of a single prompt.
        eps: Small constant for numerical stability.
        
    Returns:
        advantages: Tensor of shape [G] normalized as (R - mean) / (std + eps).
                    If all rewards are identical, returns zeros.
    """
    assert rewards.dim() == 1, f"Expected 1D rewards tensor [G], got shape {rewards.shape}"
    G = rewards.shape[0]
    if G <= 1:
        return torch.zeros_like(rewards)

    mean_r = torch.mean(rewards)
    std_r = torch.std(rewards, unbiased=False)

    if std_r < 1e-8:
        # All completions scored identically: zero advantage gradient
        return torch.zeros_like(rewards)

    advantages = (rewards - mean_r) / (std_r + eps)
    return advantages


def compute_grpo_loss(
    policy_logprobs: List[torch.Tensor],
    old_logprobs: List[torch.Tensor],
    advantages: torch.Tensor,
    ref_logprobs: Optional[List[torch.Tensor]] = None,
    clip_eps: float = 0.2,
    beta_kl: float = 0.04,
) -> Tuple[torch.Tensor, GRPOMetrics]:
    r"""Compute Group Relative Policy Optimization (GRPO) loss with PPO clipping and reference KL.
    
    Args:
        policy_logprobs: List of G tensors, each shape [T_i], representing log \pi_\theta(y_t)
        old_logprobs: List of G tensors, each shape [T_i], representing log \pi_old(y_t)
        advantages: Tensor of shape [G] containing normalized group advantages
        ref_logprobs: Optional list of G tensors for reference model KL penalty
        clip_eps: Clipping range (1 - clip_eps, 1 + clip_eps)
        beta_kl: KL divergence penalty weight
        
    Returns:
        loss: Scalar differentiable loss tensor
        metrics: GRPOMetrics summary dataclass
    """

    G = len(policy_logprobs)
    assert G == len(old_logprobs) == advantages.shape[0], (
        f"Group size mismatch: policy ({G}), old ({len(old_logprobs)}), advantages ({advantages.shape[0]})"
    )

    group_policy_losses = []
    group_kl_losses = []
    total_tokens = 0
    clipped_tokens = 0

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
        policy_loss_i = -torch.min(surr1, surr2).mean()
        group_policy_losses.append(policy_loss_i)

        # Track clipping statistics
        with torch.no_grad():
            is_clipped = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
            clipped_tokens += int(is_clipped.sum().item())
            total_tokens += T_i

        # Reference KL divergence: D_KL(\pi_\theta || \pi_ref)
        # Using Schulman's unbiased non-negative estimator:
        # k3 = \exp(log_ref - log_\pi) - (log_ref - log_\pi) - 1
        if ref_logprobs is not None:
            ref_logp = ref_logprobs[i].detach()
            log_ref_ratio = ref_logp - pi_logp
            kl_i = torch.exp(log_ref_ratio) - log_ref_ratio - 1.0
            group_kl_losses.append(kl_i.mean())
        else:
            group_kl_losses.append(torch.tensor(0.0, device=pi_logp.device))

    if not group_policy_losses:
        zero = torch.tensor(0.0, requires_grad=True)
        return zero, GRPOMetrics(0.0, 0.0, 0.0, float(advantages.mean().item()), 0.0)

    mean_policy_loss = torch.stack(group_policy_losses).mean()
    mean_kl_loss = torch.stack(group_kl_losses).mean()

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
