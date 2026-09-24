from __future__ import annotations

import pytest
from tandem.rl.reward import (
    FormatReward,
    RegexMatchReward,
    LengthPenaltyReward,
    CompositeReward,
)


def test_format_reward_correct():
    reward_fn = FormatReward()
    completion = "<think>Let me analyze the problem step by step.</think><answer>42</answer>"
    score = reward_fn("Prompt", completion)
    assert score == 1.0


def test_format_reward_missing_think():
    reward_fn = FormatReward()
    completion = "The answer is <answer>42</answer>"
    score = reward_fn("Prompt", completion)
    assert score == 0.5  # only answer reward


def test_format_reward_inverted_tags():
    reward_fn = FormatReward()
    completion = "<answer>42</answer><think>Wait I solved it first</think>"
    score = reward_fn("Prompt", completion)
    # Think was after answer: think gets +0.5, but answer check requires think <= answer -> no answer reward
    assert score == 0.5


def test_regex_match_reward():
    reward_fn = RegexMatchReward()
    completion = "<think>Calculating 2+2</think> The result is \\boxed{4}."
    score = reward_fn("What is 2+2?", completion, target="4")
    assert score == 1.0

    score_wrong = reward_fn("What is 2+2?", completion, target="5")
    assert score_wrong == 0.0


def test_length_penalty_reward():
    reward_fn = LengthPenaltyReward(max_length=50, penalty_weight=0.1)
    short_text = "Short answer"
    assert reward_fn("Prompt", short_text) == 0.0

    long_text = "A" * 70  # 20 chars over budget
    assert reward_fn("Prompt", long_text) == pytest.approx(-2.0)


def test_composite_reward():
    fmt = FormatReward(think_reward=0.5, answer_reward=0.5)
    regex = RegexMatchReward(match_reward=1.0)
    composite = CompositeReward([(fmt, 0.4), (regex, 0.6)])

    # Perfect format + correct answer: 0.4 * 1.0 + 0.6 * 1.0 = 1.0
    good_completion = "<think>logic</think><answer>\\boxed{42}</answer>"
    assert composite("Prompt", good_completion, target="42") == pytest.approx(1.0)

    # Format ok, but wrong answer: 0.4 * 1.0 + 0.6 * 0.0 = 0.4
    wrong_completion = "<think>logic</think><answer>\\boxed{99}</answer>"
    assert composite("Prompt", wrong_completion, target="42") == pytest.approx(0.4)
