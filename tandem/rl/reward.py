from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


class BaseReward(ABC):
    """Abstract base class for reward evaluators in DiffuGRPO."""

    @abstractmethod
    def compute_reward(
        self,
        prompt: str,
        completion: str,
        token_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> float:
        """Compute and return a scalar reward for a prompt-completion pair."""
        raise NotImplementedError

    def __call__(
        self,
        prompt: str,
        completion: str,
        token_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> float:
        return self.compute_reward(prompt, completion, token_ids=token_ids, **kwargs)


class TokenValidityReward(BaseReward):
    """Token-level structural validity evaluator.
    
    Verifies:
    1. No mask_token_id in committed tokens.
    2. EOS is terminal (if present, only at final index).
    3. Proper termination (not quota-truncated without EOS, if require_terminal_eos=True).
    """

    def __init__(
        self,
        mask_token_id: int = 131071,
        eos_token_ids: Optional[Set[int]] = None,
        require_terminal_eos: bool = False,
        valid_reward: float = 1.0,
        invalid_penalty: float = 0.0,
    ):
        self.mask_token_id = mask_token_id
        self.eos_token_ids = eos_token_ids if eos_token_ids is not None else {2, 131070}
        self.require_terminal_eos = require_terminal_eos
        self.valid_reward = valid_reward
        self.invalid_penalty = invalid_penalty

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        token_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> float:
        if token_ids is None or len(token_ids) == 0:
            return self.invalid_penalty

        # Check 1: mask_token_id must never appear in committed tokens
        if self.mask_token_id in token_ids:
            return self.invalid_penalty

        # Check 2: EOS must not appear before the terminal token
        for tok in token_ids[:-1]:
            if tok in self.eos_token_ids:
                return self.invalid_penalty

        # Check 3: If terminal EOS required, final token must be an EOS token
        if self.require_terminal_eos and token_ids[-1] not in self.eos_token_ids:
            return self.invalid_penalty

        return self.valid_reward


class FormatReward(BaseReward):
    """Reward for adhering to structured chain-of-thought XML tags: <think>...</think><answer>...</answer>."""

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

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        token_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> float:
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


class MathCorrectnessReward(BaseReward):
    """Extracts answer from XML tags (<answer>...</answer>), \\boxed{...}, or #### and evaluates correctness."""

    def __init__(
        self,
        match_reward: float = 1.0,
        mismatch_reward: float = 0.0,
    ):
        self.match_reward = match_reward
        self.mismatch_reward = mismatch_reward
        self.xml_answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        self.boxed_pattern = re.compile(r"\\boxed\{([^}]+)\}")
        self.hash_pattern = re.compile(r"####\s*([^\n]+)")

    def extract_answer(self, text: str) -> Optional[str]:
        # Priority 1: <answer>...</answer>
        xml_matches = self.xml_answer_pattern.findall(text)
        if xml_matches:
            return xml_matches[-1].strip()

        # Priority 2: \boxed{...}
        boxed_matches = self.boxed_pattern.findall(text)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # Priority 3: #### ...
        hash_matches = self.hash_pattern.findall(text)
        if hash_matches:
            return hash_matches[-1].strip()

        return None

    def _normalize(self, val: str) -> str:
        s = val.strip().lower()
        # Remove common latex spacing and formatting
        s = s.replace(" ", "").replace("$", "").replace("\\text{", "").replace("}", "")
        return s

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        token_ids: Optional[List[int]] = None,
        target: Optional[str] = None,
        **kwargs,
    ) -> float:
        if target is None:
            target = kwargs.get("ground_truth", None)
        if target is None:
            return self.mismatch_reward

        extracted = self.extract_answer(completion)
        if extracted is None:
            return self.mismatch_reward

        norm_extracted = self._normalize(extracted)
        norm_target = self._normalize(str(target))

        if norm_extracted == norm_target:
            return self.match_reward

        # Try numeric comparison if both parse as floats
        try:
            val_ext = float(norm_extracted)
            val_tgt = float(norm_target)
            if abs(val_ext - val_tgt) < 1e-5:
                return self.match_reward
        except ValueError:
            pass

        return self.mismatch_reward


class RegexMatchReward(BaseReward):
    """Reward for exact regex matches against ground truth."""

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

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        token_ids: Optional[List[int]] = None,
        target: Optional[str] = None,
        **kwargs,
    ) -> float:
        if target is None:
            target = kwargs.get("ground_truth", None)
        if target is None:
            return self.mismatch_reward

        extracted = self.extract_answer(completion)
        if extracted is not None and extracted.strip().lower() == str(target).strip().lower():
            return self.match_reward
        return self.mismatch_reward


class CompositeReward(BaseReward):
    """Combines multiple reward components with user-specified weights."""

    def __init__(self, reward_components: Sequence[Tuple[BaseReward, float]]):
        self.reward_components = list(reward_components)

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        token_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> float:
        total = 0.0
        for reward_fn, weight in self.reward_components:
            score = reward_fn.compute_reward(prompt, completion, token_ids=token_ids, **kwargs)
            total += weight * score
        return total
