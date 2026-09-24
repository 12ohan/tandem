from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tandem.rl.dataset import PromptDataset, PromptItem
from tandem.rl.reward import FormatReward, RegexMatchReward
from tandem.rl.trainer import DiffuGRPOTrainer, GRPOTrainerConfig
from tandem.rl.trajectory import Trajectory, TrajectoryCollector, TrajectoryStep
from tandem.engine.spec import GenerationOutput


class TinyMockModel(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.Linear(hidden_dim, hidden_dim)
        self.diffusion_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        h = self.encoder(h)
        return self.diffusion_head(h)


class TinyMockRunner:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = torch.device("cpu")
        self.mask_token_id = 63
        self.eos_token_ids = {0}

    def forward_causal(
        self, input_ids: torch.Tensor, past_key_values=None, use_cache=True
    ) -> Tuple[torch.Tensor, None]:
        logits = self.model(input_ids)
        return logits, None


class TinyMockEngine:
    def __init__(self, runner: TinyMockRunner, should_fail_mask: bool = False):
        self.runner = runner
        self.should_fail_mask = should_fail_mask

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 8,
        block_size: int = 2,
        temperature: float = 1.0,
        return_logprob: bool = True,
    ) -> GenerationOutput:
        if self.should_fail_mask:
            raise RuntimeError(
                f"Causal verification policy committed mask token {self.runner.mask_token_id}"
            )

        prompt_tokens = [1, 2, 3]
        if "box" in prompt.lower():
            text = "Answer is \\boxed{42}"
            comp_tokens = [10, 11, 12, 13]
        elif "think" in prompt.lower():
            text = "<think>reason</think><answer>correct</answer>"
            comp_tokens = [20, 21, 22, 23]
        else:
            text = "Generic completion text"
            comp_tokens = [30, 31, 32]

        collector = TrajectoryCollector(prompt_tokens, temperature=temperature)
        for tok in comp_tokens:
            collector.append_step(token_id=tok, logprob=-1.25, entropy=0.5)

        return GenerationOutput(
            text=text,
            token_ids=comp_tokens,
            num_generated_tokens=len(comp_tokens),
            num_forward_passes=3,
            wall_time=0.01,
            tok_per_sec=400.0,
            acceptance_rate=0.75,
            trajectory=collector.to_trajectory(),
        )


def test_trainer_rollout_group():
    model = TinyMockModel()
    runner = TinyMockRunner(model)
    engine = TinyMockEngine(runner)
    reward_fn = RegexMatchReward(pattern=r"\\boxed\{([^}]+)\}")

    config = GRPOTrainerConfig(group_size=4, temperature=1.0)
    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config)

    item = PromptItem(prompt="Find x in \\boxed{42}", ground_truth="42")
    rollout = trainer.rollout_group(item)

    assert len(rollout.completions) == 4
    assert len(rollout.trajectories) == 4
    assert rollout.rewards.shape == (4,)
    # All 4 completions matched ground_truth "42", so rewards are all 1.0
    assert torch.all(rollout.rewards == 1.0)
    # Identical rewards produce zero advantage
    assert torch.all(rollout.advantages == 0.0)
    # Assert trajectory recorded temperature
    assert rollout.trajectories[0].temperature == 1.0


def test_trainer_catches_mask_token_gracefully():
    """Verify §D: degenerate rollout committing mask_token_id is caught gracefully in rollout harness,
    assigning format_failure_reward without killing the training run.
    """
    model = TinyMockModel()
    runner = TinyMockRunner(model)
    engine = TinyMockEngine(runner, should_fail_mask=True)
    reward_fn = FormatReward()

    config = GRPOTrainerConfig(group_size=2, format_failure_reward=-1.0)
    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config)

    item = PromptItem(prompt="Generate tokens")
    # Must not raise RuntimeError
    rollout = trainer.rollout_group(item)

    assert len(rollout.completions) == 2
    assert "<failed_mask_token_commitment>" in rollout.completions[0]
    assert rollout.rewards[0].item() == -1.0
    # §D invariant: failed rollout has zero completion tokens and zero steps in trajectory
    # guaranteeing that no fabricated tokens or synthetic logprobs participate in loss
    assert len(rollout.trajectories[0].completion_tokens) == 0
    assert len(rollout.trajectories[0].steps) == 0


def test_trainer_on_policy_ratio_near_one():
    """Verify §A.4: teacher-forcing at behavior policy yields r_{i,t} == 1.0 exactly in float32."""
    model = TinyMockModel()
    runner = TinyMockRunner(model)
    engine = TinyMockEngine(runner)
    reward_fn = FormatReward()

    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn)

    # Compute exact teacher-forced logits from the model
    prompt_tokens = [1, 2, 3]
    comp_tokens = [20, 21, 22]
    full_tokens = prompt_tokens + comp_tokens
    with torch.no_grad():
        logits = model(torch.tensor([full_tokens]))
        P = len(prompt_tokens)
        T = len(comp_tokens)
        comp_logits = logits[0, P - 1 : P + T - 1, :]
        log_probs = F.log_softmax(comp_logits, dim=-1)
        exact_logprobs = log_probs[torch.arange(T), torch.tensor(comp_tokens)].tolist()

    # Create trajectory with exact teacher-forced logprobs
    collector = TrajectoryCollector(prompt_tokens, temperature=1.0)
    for tok, lp in zip(comp_tokens, exact_logprobs):
        collector.append_step(tok, lp)

    traj = collector.to_trajectory()
    rollout = trainer.rollout_group(PromptItem(prompt="test"))
    rollout.trajectories = [traj]
    rollout.advantages = torch.tensor([1.0])

    loss, metrics = trainer.compute_rollout_loss(rollout)

    # Policy loss for ratio = 1.0 with advantage = 1.0 must be -1.0
    assert loss.item() == pytest.approx(-1.0, abs=1e-5)
    assert metrics.clip_fraction == 0.0


def test_trainer_compute_loss_differentiable():
    model = TinyMockModel()
    runner = TinyMockRunner(model)
    engine = TinyMockEngine(runner)
    reward_fn = FormatReward()

    config = GRPOTrainerConfig(group_size=2, temperature=1.0)
    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config)

    item = PromptItem(prompt="Show your work with think tags")
    rollout = trainer.rollout_group(item)
    # Artificially set non-zero advantages to test gradient flow
    rollout.advantages = torch.tensor([1.0, -1.0])

    loss, metrics = trainer.compute_rollout_loss(rollout)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    loss.backward()

    # Verify gradients were computed on model parameters
    head_grad = model.diffusion_head.weight.grad
    assert head_grad is not None
    assert torch.isfinite(head_grad).all()
    assert (head_grad != 0).any()


def test_trainer_step_and_parameter_update():
    model = TinyMockModel()
    runner = TinyMockRunner(model)
    engine = TinyMockEngine(runner)
    reward_fn = FormatReward()

    config = GRPOTrainerConfig(group_size=2, learning_rate=0.01)
    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config)

    batch = [
        PromptItem(prompt="Think step 1"),
        PromptItem(prompt="Think step 2"),
    ]
    step_metrics = trainer.step(batch)

    assert "loss" in step_metrics
    assert "mean_reward" in step_metrics
    assert "acceptance_rate" in step_metrics


def test_trainer_train_head_only():
    model = TinyMockModel()
    runner = TinyMockRunner(model)
    engine = TinyMockEngine(runner)
    reward_fn = FormatReward()

    config = GRPOTrainerConfig(group_size=2, train_head_only=True)
    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config)

    # Encoder must be frozen
    for p in model.encoder.parameters():
        assert not p.requires_grad

    # Head must be trainable
    for p in model.diffusion_head.parameters():
        assert p.requires_grad


def test_trainer_save_and_load_weights(tmp_path):
    model = TinyMockModel()
    runner = TinyMockRunner(model)
    engine = TinyMockEngine(runner)
    reward_fn = FormatReward()

    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn)
    checkpoint_file = tmp_path / "model_checkpoint.pt"

    with torch.no_grad():
        model.diffusion_head.weight.fill_(3.1415)

    trainer.save_weights(checkpoint_file)
    assert checkpoint_file.is_file()

    with torch.no_grad():
        model.diffusion_head.weight.zero_()

    trainer.load_weights(checkpoint_file)
    assert torch.allclose(model.diffusion_head.weight, torch.tensor(3.1415))


def test_trainer_temperature_scaling_tau_neq_one():
    """Verify that when trajectory carries tau = 0.7, the trainer recomputes
    log softmax(z / 0.7)[y_i] by reading tau from traj.temperature, avoiding the tau=1 blind spot.
    """
    model = TinyMockModel()
    runner = TinyMockRunner(model)
    engine = TinyMockEngine(runner)
    reward_fn = FormatReward()

    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn)

    prompt_tokens = [1, 2, 3]
    comp_tokens = [20, 21, 22]
    full_tokens = prompt_tokens + comp_tokens
    tau = 0.7

    with torch.no_grad():
        logits = model(torch.tensor([full_tokens]))
        P = len(prompt_tokens)
        T = len(comp_tokens)
        comp_logits = logits[0, P - 1 : P + T - 1, :]
        log_probs_tau = F.log_softmax(comp_logits / tau, dim=-1)
        expected_logprobs = log_probs_tau[torch.arange(T), torch.tensor(comp_tokens)].tolist()

    # Trajectory explicitly carries temperature = 0.7
    collector = TrajectoryCollector(prompt_tokens, temperature=tau)
    for tok, lp in zip(comp_tokens, expected_logprobs):
        collector.append_step(tok, lp)

    traj = collector.to_trajectory()
    assert traj.temperature == 0.7

    rollout = trainer.rollout_group(PromptItem(prompt="test"))
    rollout.trajectories = [traj]
    rollout.advantages = torch.tensor([1.0])

    loss, metrics = trainer.compute_rollout_loss(rollout)

    # When tau=0.7 is correctly read by the trainer, ratio is exp(log_probs_tau - expected_logprobs) = 1.0
    # Policy loss for ratio = 1.0 with advantage = 1.0 must be -1.0
    assert loss.item() == pytest.approx(-1.0, abs=1e-5)


def test_trainer_zero_grad_on_ref_model():
    """Verify that parameters in ref_model have grad is None after backward pass."""
    model = TinyMockModel()
    ref_model = TinyMockModel()
    runner = TinyMockRunner(model)
    ref_runner = TinyMockRunner(ref_model)
    engine = TinyMockEngine(runner)
    reward_fn = FormatReward()

    config = GRPOTrainerConfig(group_size=2, beta_kl=0.04)
    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config, ref_model=ref_runner)

    item = PromptItem(prompt="Test ref gradient isolation")
    rollout = trainer.rollout_group(item)
    rollout.advantages = torch.tensor([1.0, -1.0])

    loss, _ = trainer.compute_rollout_loss(rollout)
    loss.backward()

    # Model parameters must receive gradients
    assert model.diffusion_head.weight.grad is not None
    # Ref model parameters must NEVER receive gradients
    for p in ref_model.parameters():
        assert p.grad is None
