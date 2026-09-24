from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F
from tandem.rl.diffu_grpo import compute_group_advantages, compute_grpo_loss


def test_group_advantages_normalization():
    # Rewards: [1.0, 0.0, 0.5, 0.5]
    rewards = torch.tensor([1.0, 0.0, 0.5, 0.5], dtype=torch.float32)
    adv = compute_group_advantages(rewards)

    assert adv.shape == (4,)
    # Mean advantage must be 0
    assert abs(adv.mean().item()) < 1e-6
    # Highest reward must have highest positive advantage
    assert adv[0] > 0
    # Lowest reward must have lowest negative advantage
    assert adv[1] < 0
    # Equal rewards must have equal advantages
    assert adv[2] == adv[3]


def test_group_advantages_identical_rewards():
    # If all completions scored identically, advantage must be exactly zero (no NaN)
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    adv = compute_group_advantages(rewards)

    assert torch.all(adv == 0.0)
    assert not torch.isnan(adv).any()


def test_degenerate_groups():
    # G = 1 degenerate group
    rewards_g1 = torch.tensor([0.75], dtype=torch.float32)
    adv_g1 = compute_group_advantages(rewards_g1)
    assert adv_g1.shape == (1,)
    assert adv_g1.item() == 0.0

    # Ablation flag: normalize_by_std=False
    rewards = torch.tensor([10.0, 20.0], dtype=torch.float32)
    adv_unnorm = compute_group_advantages(rewards, normalize_by_std=False)
    # mean = 15.0 -> diff = [-5.0, 5.0]
    assert torch.allclose(adv_unnorm, torch.tensor([-5.0, 5.0]))


def test_grpo_loss_differentiable_and_gradients():
    G = 3
    T = 4
    # Policy logprobs require grad
    policy_logprobs = [
        torch.randn(T, requires_grad=True),
        torch.randn(T, requires_grad=True),
        torch.randn(T, requires_grad=True),
    ]
    old_logprobs = [p.detach().clone() for p in policy_logprobs]
    advantages = torch.tensor([1.0, -1.0, 0.0])

    loss, metrics = compute_grpo_loss(
        policy_logprobs=policy_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        clip_eps=0.2,
        beta_kl=0.01,
    )

    assert loss.requires_grad
    loss.backward()

    # High advantage branch (completion 0 with adv=1.0) must have negative gradient
    # to encourage higher log probability: d(-adv * pi)/d(pi) = -adv < 0
    assert policy_logprobs[0].grad is not None
    assert not torch.isnan(policy_logprobs[0].grad).any()
    assert policy_logprobs[0].grad.mean().item() < 0.0

    # Low advantage branch (completion 1 with adv=-1.0) must have positive gradient
    assert policy_logprobs[1].grad is not None
    assert policy_logprobs[1].grad.mean().item() > 0.0


def test_clip_asymmetry_gradients():
    """Verify PPO/GRPO clip asymmetry:
    - If A > 0 and r > 1 + eps: loss is clipped to -(1+eps)*A -> gradient with respect to pi is 0.
    - If A < 0 and r > 1 + eps: loss is unclipped -r*A -> gradient with respect to pi is non-zero.
    - If A > 0 and r < 1 - eps: loss is unclipped -r*A -> gradient with respect to pi is non-zero.
    - If A < 0 and r < 1 - eps: loss is clipped to -(1-eps)*A -> gradient with respect to pi is 0.
    """
    clip_eps = 0.2

    # Case 1: A > 0, r > 1 + eps (ratio = exp(0.5) ~ 1.65 > 1.2) -> Clipped, grad = 0
    pi_1 = torch.tensor([0.5], requires_grad=True)
    old_1 = torch.tensor([0.0])
    loss_1, _ = compute_grpo_loss([pi_1], [old_1], torch.tensor([1.0]), clip_eps=clip_eps, beta_kl=0.0)
    loss_1.backward()
    assert pi_1.grad.item() == 0.0

    # Case 2: A < 0, r > 1 + eps -> Unclipped, grad != 0
    pi_2 = torch.tensor([0.5], requires_grad=True)
    old_2 = torch.tensor([0.0])
    loss_2, _ = compute_grpo_loss([pi_2], [old_2], torch.tensor([-1.0]), clip_eps=clip_eps, beta_kl=0.0)
    loss_2.backward()
    assert abs(pi_2.grad.item()) > 0.0

    # Case 3: A > 0, r < 1 - eps (ratio = exp(-0.5) ~ 0.606 < 0.8) -> Unclipped, grad != 0
    pi_3 = torch.tensor([-0.5], requires_grad=True)
    old_3 = torch.tensor([0.0])
    loss_3, _ = compute_grpo_loss([pi_3], [old_3], torch.tensor([1.0]), clip_eps=clip_eps, beta_kl=0.0)
    loss_3.backward()
    assert abs(pi_3.grad.item()) > 0.0

    # Case 4: A < 0, r < 1 - eps -> Clipped, grad = 0
    pi_4 = torch.tensor([-0.5], requires_grad=True)
    old_4 = torch.tensor([0.0])
    loss_4, _ = compute_grpo_loss([pi_4], [old_4], torch.tensor([-1.0]), clip_eps=clip_eps, beta_kl=0.0)
    loss_4.backward()
    assert pi_4.grad.item() == 0.0


def test_k3_pointwise_nonnegativity_and_agreement():
    """Verify that Schulman k3 estimator:
    1. Is strictly non-negative pointwise: exp(u) - u - 1 >= 0 for all u.
    2. Agrees with direct KL divergence on toy distributions up to second order.
    """
    # 1. Pointwise non-negativity across wide range of log ratios
    u = torch.linspace(-10.0, 10.0, 1000)
    k3 = torch.exp(u) - u - 1.0
    assert (k3 >= -1e-6).all(), "k3 violated pointwise non-negativity"
    # Minimum is at u = 0, where k3 = 0.0
    u_zero = torch.tensor([0.0])
    assert torch.allclose(torch.exp(u_zero) - u_zero - 1.0, torch.tensor([0.0]))

    # 2. Agreement with direct KL on toy Bernoulli distribution
    # Let P = [0.6, 0.4], Q = [0.55, 0.45]
    p = torch.tensor([0.6, 0.4])
    q = torch.tensor([0.55, 0.45])
    direct_kl = (p * torch.log(p / q)).sum().item()

    # In sample space, expectation of k3 under P:
    # E_P[ exp(log q - log p) - (log q - log p) - 1 ]
    # = sum_x p(x) * [ q(x)/p(x) - log(q/p) - 1 ] = sum q(x) - 1 + sum p(x) log(p/q) = direct_kl
    u_samples = torch.log(q) - torch.log(p)
    k3_samples = torch.exp(u_samples) - u_samples - 1.0
    expected_k3 = (p * k3_samples).sum().item()

    assert abs(expected_k3 - direct_kl) < 1e-6
