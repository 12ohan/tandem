from __future__ import annotations

import pytest
from tandem.rl.reward import (
    CompositeReward,
    FormatReward,
    MathCorrectnessReward,
    RegexMatchReward,
    TokenValidityReward,
)


def test_token_validity_reward():
    eos_ids = {2, 131070}
    mask_id = 131071
    reward_fn = TokenValidityReward(mask_token_id=mask_id, eos_token_ids=eos_ids, require_terminal_eos=True)

    # 1. Valid clean termination
    valid_tokens = [10, 20, 30, 2]
    assert reward_fn("prompt", "text", token_ids=valid_tokens) == 1.0

    # 2. Reject if mask_token_id is present
    masked_tokens = [10, mask_id, 30, 2]
    assert reward_fn("prompt", "text", token_ids=masked_tokens) == 0.0

    # 3. Reject if EOS appears mid-sequence
    mid_eos_tokens = [10, 2, 30, 40]
    assert reward_fn("prompt", "text", token_ids=mid_eos_tokens) == 0.0

    # 4. Reject if terminal EOS is missing when require_terminal_eos=True
    truncated_tokens = [10, 20, 30, 40]
    assert reward_fn("prompt", "text", token_ids=truncated_tokens) == 0.0


def test_math_correctness_reward():
    reward_fn = MathCorrectnessReward()

    # 1. Answer in <answer> tags
    comp1 = "<think>Compute 2+2</think><answer>4</answer>"
    assert reward_fn("What is 2+2?", comp1, ground_truth="4") == 1.0
    assert reward_fn("What is 2+2?", comp1, ground_truth="5") == 0.0

    # 2. Answer in \boxed{}
    comp2 = "Therefore, the result is \\boxed{ 42 }."
    assert reward_fn("Find x", comp2, ground_truth="42") == 1.0

    # 3. Answer with ####
    comp3 = "Step by step solution... #### 100"
    assert reward_fn("Find value", comp3, ground_truth="100") == 1.0

    # 4. Float comparison equivalence
    comp4 = "<answer>3.14159</answer>"
    assert reward_fn("pi", comp4, ground_truth="3.141590") == 1.0


def test_format_reward_correct():
    reward = FormatReward()
    completion = "<think>Let me analyze the problem.</think><answer>42</answer>"
    score = reward.compute_reward("prompt", completion)
    assert score == 1.0  # 0.5 think + 0.5 answer


def test_format_reward_missing_think():
    reward = FormatReward()
    completion = "<answer>42</answer>"
    score = reward.compute_reward("prompt", completion)
    assert score == 0.5  # only answer


def test_format_reward_inverted_tags():
    reward = FormatReward()
    completion = "<answer>42</answer><think>Wait let me think</think>"
    score = reward.compute_reward("prompt", completion)
    assert score == 0.5  # think follows answer, so answer bonus not awarded


def test_regex_match_reward():
    reward = RegexMatchReward(pattern=r"\\boxed\{([^}]+)\}")
    comp_match = "The solution is \\boxed{42}."
    assert reward.compute_reward("prompt", comp_match, ground_truth="42") == 1.0
    assert reward.compute_reward("prompt", comp_match, ground_truth="99") == 0.0


def test_composite_reward():
    comp_reward = CompositeReward([
        (FormatReward(), 0.5),
        (MathCorrectnessReward(), 0.5),
    ])
    full_pass = "<think>Reasoning</think><answer>42</answer>"
    score = comp_reward.compute_reward("prompt", full_pass, ground_truth="42")
    assert score == 1.0
