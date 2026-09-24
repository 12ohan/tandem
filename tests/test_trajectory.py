from __future__ import annotations

import pytest
from tandem.rl.trajectory import TrajectoryCollector


def test_trajectory_collector_lockstep():
    collector = TrajectoryCollector(prompt_tokens=[1, 2, 3])
    
    collector.append_step(token_id=100, logprob=-0.5, entropy=1.2, reveal_step=1)
    collector.append_step(token_id=101, logprob=-0.3, entropy=0.8, reveal_step=1)
    collector.append_step(token_id=2, logprob=-0.01, entropy=0.1, reveal_step=2)  # EOS = 2
    collector.append_step(token_id=103, logprob=-1.2, entropy=2.0, reveal_step=2)  # post-EOS garbage
    
    assert len(collector.completion_tokens) == 4
    assert len(collector.steps) == 4
    
    # Truncate at EOS (token 2)
    cutoff = collector.truncate_at_eos(eos_token_ids=[2, 131070])
    assert cutoff == 3
    assert collector.completion_tokens == [100, 101, 2]
    assert len(collector.steps) == 3
    
    traj = collector.to_trajectory()
    assert traj.total_tokens == 6  # 3 prompt + 3 completion
    assert traj.logprobs == [-0.5, -0.3, -0.01]


def test_trajectory_collector_rejects_nan():
    collector = TrajectoryCollector(prompt_tokens=[1])
    with pytest.raises(ValueError, match="Poisoned logprob"):
        collector.append_step(token_id=99, logprob=float("nan"))

    with pytest.raises(ValueError, match="Poisoned logprob"):
        collector.append_step(token_id=99, logprob=float("-inf"))
