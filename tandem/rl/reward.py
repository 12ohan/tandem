from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional


class BaseReward(ABC):
    """Abstract base class for reward evaluators in DiffuGRPO."""

    @abstractmethod
    def compute_reward(self, prompt: str, completion: str, **kwargs) -> float:
        """Compute and return a scalar reward for a prompt-completion pair."""
        raise NotImplementedError

    def __call__(self, prompt: str, completion: str, **kwargs) -> float:
        return self.compute_reward(prompt, completion, **kwargs)


class FormatReward(BaseReward):
    """Reward for adhering to structured chain-of-thought XML tags.
    
    Default: checks for `<think>...</think>` reasoning block followed by
    `<answer>...</answer>` solution block.
    """

    def __init__(
        self,
        think_start: str = "<think>",
        think_end: str = "</think>",
        answer_start: str = "<answer>",
        answer_end: str = "</answer>",
        think_reward: float = 0.5,
        answer_reward: float = 0.5,
    ):
        self.think_start = think_start
        self.think_end = think_end
        self.answer_start = answer_start
        self.answer_end = answer_end
        self.think_reward = think_reward
        self.answer_reward = answer_reward

    def compute_reward(self, prompt: str, completion: str, **kwargs) -> float:
        reward = 0.0

        # Check for matching think tags
        has_think = (
            self.think_start in completion
            and self.think_end in completion
            and completion.index(self.think_start) < completion.index(self.think_end)
        )
        if has_think:
            reward += self.think_reward

        # Check for matching answer tags
        has_answer = (
            self.answer_start in completion
            and self.answer_end in completion
            and completion.index(self.answer_start) < completion.index(self.answer_end)
        )
        if has_answer:
            # Enforce that thinking precedes answering
            if has_think:
                if completion.index(self.think_end) <= completion.index(self.answer_start):
                    reward += self.answer_reward
            else:
                reward += self.answer_reward

        return reward


class RegexMatchReward(BaseReward):
    """Reward for exact regex matches against ground truth (e.g. \\boxed{ans} or #### ans)."""

    def __init__(
        self,
        pattern: str = r"\\boxed\{([^}]+)\}",
        match_reward: float = 1.0,
        mismatch_reward: float = 0.0,
    ):
        self.pattern = re.compile(pattern)
        self.match_reward = match_reward
        self.mismatch_reward = mismatch_reward

    def extract_answer(self, text: str) -> Optional[str]:
        matches = self.pattern.findall(text)
        if matches:
            return matches[-1].strip()
        return None

    def compute_reward(self, prompt: str, completion: str, target: Optional[str] = None, **kwargs) -> float:
        if target is None:
            target = kwargs.get("ground_truth", None)
        if target is None:
            return self.mismatch_reward

        extracted = self.extract_answer(completion)
        if extracted is not None and extracted.strip().lower() == str(target).strip().lower():
            return self.match_reward
        return self.mismatch_reward


class LengthPenaltyReward(BaseReward):
    """Soft penalty for exceeding a desired token or character budget."""

    def __init__(self, max_length: int = 1024, penalty_weight: float = 0.001):
        self.max_length = max_length
        self.penalty_weight = penalty_weight

    def compute_reward(self, prompt: str, completion: str, **kwargs) -> float:
        char_len = len(completion)
        if char_len > self.max_length:
            return -self.penalty_weight * (char_len - self.max_length)
        return 0.0


class CompositeReward(BaseReward):
    """Combines multiple reward components with user-specified weights."""

    def __init__(self, reward_components: List[Tuple[BaseReward, float]]):
        self.reward_components = reward_components

    def compute_reward(self, prompt: str, completion: str, **kwargs) -> float:
        total = 0.0
        for reward_fn, weight in self.reward_components:
            score = reward_fn.compute_reward(prompt, completion, **kwargs)
            total += weight * score
        return total
