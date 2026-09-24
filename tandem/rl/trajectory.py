from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class TrajectoryStep:
    token_id: int
    logprob: float
    entropy: Optional[float] = None
    reveal_step: Optional[int] = None


@dataclass
class Trajectory:
    prompt_tokens: List[int]
    steps: List[TrajectoryStep] = field(default_factory=list)
    completion_tokens: List[int] = field(default_factory=list)
    
    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.completion_tokens)
        
    @property
    def logprobs(self) -> List[float]:
        return [s.logprob for s in self.steps]
        
    @property
    def entropies(self) -> List[Optional[float]]:
        return [s.entropy for s in self.steps]


class TrajectoryCollector:
    """Collects and validates DiffuGRPO RL trajectory channels in strict lockstep."""

    def __init__(self, prompt_tokens: List[int]):
        self.prompt_tokens = list(prompt_tokens)
        self.steps: List[TrajectoryStep] = []
        self.completion_tokens: List[int] = []

    def append_step(
        self,
        token_id: int,
        logprob: float,
        entropy: Optional[float] = None,
        reveal_step: Optional[int] = None,
    ) -> None:
        """Append a committed token step with strict validation."""
        # Sanity-check numerical integrity (poison defense for GRPO)
        if math.isnan(logprob) or math.isinf(logprob):
            raise ValueError(f"Poisoned logprob detected: {logprob} for token_id={token_id}")

        step = TrajectoryStep(
            token_id=token_id,
            logprob=logprob,
            entropy=entropy,
            reveal_step=reveal_step,
        )
        self.steps.append(step)
        self.completion_tokens.append(token_id)

    def truncate_at_eos(self, eos_token_ids: List[int]) -> int:
        """Find the earliest EOS token and truncate all parallel channels in lockstep."""
        first_eos_idx = None
        for i, token in enumerate(self.completion_tokens):
            if token in eos_token_ids:
                first_eos_idx = i
                break

        if first_eos_idx is not None:
            cutoff = first_eos_idx + 1
            self.completion_tokens = self.completion_tokens[:cutoff]
            self.steps = self.steps[:cutoff]
            return cutoff

        return len(self.completion_tokens)

    def to_trajectory(self) -> Trajectory:
        assert len(self.steps) == len(self.completion_tokens), (
            f"Channel desync: steps ({len(self.steps)}) != tokens ({len(self.completion_tokens)})"
        )
        return Trajectory(
            prompt_tokens=list(self.prompt_tokens),
            steps=list(self.steps),
            completion_tokens=list(self.completion_tokens),
        )
