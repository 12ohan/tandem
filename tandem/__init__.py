from tandem.config import EngineConfig
from tandem.engine.kv_cache import DynamicKVCache
from tandem.engine.matcher import MatchResult, match_speculative_tokens
from tandem.engine.model_runner import TandemModelRunner
from tandem.engine.sampler import sample_gumbel_max, sample_residual
from tandem.engine.spec import GenerationOutput, TandemEngine
from tandem.rl.dataset import PromptDataset, PromptItem
from tandem.rl.diffu_grpo import GRPOMetrics, compute_group_advantages, compute_grpo_loss
from tandem.rl.reward import (
    BaseReward,
    CompositeReward,
    FormatReward,
    MathCorrectnessReward,
    RegexMatchReward,
    TokenValidityReward,
)
from tandem.rl.trainer import DiffuGRPOTrainer, GRPORollout, GRPOTrainerConfig
from tandem.rl.trajectory import Trajectory, TrajectoryCollector, TrajectoryStep

__all__ = [
    "EngineConfig",
    "TandemEngine",
    "GenerationOutput",
    "TandemModelRunner",
    "match_speculative_tokens",
    "MatchResult",
    "sample_gumbel_max",
    "sample_residual",
    "DynamicKVCache",
    "TrajectoryCollector",
    "Trajectory",
    "TrajectoryStep",
    "PromptItem",
    "PromptDataset",
    "BaseReward",
    "FormatReward",
    "MathCorrectnessReward",
    "RegexMatchReward",
    "TokenValidityReward",
    "CompositeReward",
    "compute_group_advantages",
    "compute_grpo_loss",
    "GRPOMetrics",
    "DiffuGRPOTrainer",
    "GRPOTrainerConfig",
    "GRPORollout",
]
