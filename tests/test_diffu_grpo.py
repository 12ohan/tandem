from __future__ import annotations

import pytest
import torch
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
    # If all completions scored identically, advantage must be exactly zero
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    adv = compute_group_advantages(rewards)

    assert torch.all(adv == 0.0)


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


def test_grpo_loss_clipping():
    # Force policy to be much higher than old (ratio > 1 + clip_eps)
    T = 2
    policy_logprobs = [torch.tensor([0.0, 0.0], requires_grad=True)]
    old_logprobs = [torch.tensor([-2.0, -2.0])]  # ratio = exp(2.0) ~ 7.39 >> 1.2
    advantages = torch.tensor([1.0])

    loss, metrics = compute_grpo_loss(
        policy_logprobs=policy_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        clip_eps=0.2,
        beta_kl=0.0,
    )

    # When clipped at 1 + 0.2 = 1.2, surrogate loss is -1.2 * 1.0 = -1.2
    assert loss.item() == pytest.approx(-1.2)
    assert metrics.clip_fraction == 1.0


def test_grpo_kl_penalty():
    T = 3
    # Policy diverges from reference
    policy_logprobs = [torch.tensor([-0.5, -0.5, -0.5])]
    ref_logprobs = [torch.tensor([-0.1, -0.1, -0.1])]
    old_logprobs = [torch.tensor([-0.5, -0.5, -0.5])]
    advantages = torch.tensor([0.0])  # zero policy advantage to isolate KL

    loss_low_kl, metrics_low = compute_grpo_loss(
        policy_logprobs=policy_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        ref_logprobs=ref_logprobs,
        beta_kl=0.01,
    )

    loss_high_kl, metrics_high = compute_grpo_loss(
        policy_logprobs=policy_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        ref_logprobs=ref_logprobs,
        beta_kl=0.1,
    )

    # Higher beta_kl must increase total penalty loss
    assert loss_high_kl.item() > loss_low_kl.item()
    assert metrics_high.kl_loss > 0.0
