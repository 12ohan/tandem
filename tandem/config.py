from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, List
import torch


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_dtype(device: str) -> torch.dtype:
    if device in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


@dataclass
class EngineConfig:
    model_id: str = field(
        default_factory=lambda: os.environ.get("MODEL_ID", "nvidia/Nemotron-Labs-Diffusion-3B")
    )
    block_size: int = 4
    device: str = field(default_factory=get_default_device)
    dtype: Optional[torch.dtype] = None
    mask_id: int = 131071
    eos_token_ids: List[int] = field(default_factory=lambda: [2, 131070])
    
    # Speculative decoding parameters
    max_context_len: int = 16384
    temperature: float = 0.0
    top_p: float = 1.0
    
    # Diffusion generation settings
    nfe: int = 1
    
    # RL & Trajectory Settings
    return_logprob: bool = False
    return_entropy: bool = False
    rl_safe: bool = True
    manages_own_kv: bool = False

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = get_default_dtype(self.device)
            
        assert self.block_size > 0, "block_size must be >= 1"
        assert self.max_context_len > self.block_size, "max_context_len must exceed block_size"
        
        # RL safety gates
        if self.rl_safe and self.temperature == 0.0 and self.return_logprob:
            # Note: Deterministic greedy rollouts produce zero within-group advantage in GRPO
            pass
