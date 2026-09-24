from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tandem.rl.dataset import PromptDataset, PromptItem
from tandem.rl.reward import BaseReward, FormatReward, RegexMatchReward
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


def test_trainer_curriculum_hint_injection():
    """Verify that when curriculum_hint_on_zero=True and all rollouts score 0,
    the trainer re-rolls with clinical guidance injected into the prompt.
    """
    model = TinyMockModel()
    runner = TinyMockRunner(model)

    class HintResponsiveEngine(TinyMockEngine):
        def generate(self, prompt, **kwargs):
            # If prompt has guidance, output <think> text which scores on FormatReward
            if "Clinical Guidance" in prompt:
                text = "<think>guided reasoning</think><answer>correct</answer>"
                comp_tokens = [20, 21, 22, 23]
            else:
                text = "completely wrong format"
                comp_tokens = [50, 51]

            collector = TrajectoryCollector([1, 2, 3], temperature=kwargs.get("temperature", 1.0))
            for tok in comp_tokens:
                collector.append_step(token_id=tok, logprob=-1.0)

            return GenerationOutput(
                text=text,
                token_ids=comp_tokens,
                num_generated_tokens=len(comp_tokens),
                num_forward_passes=1,
                wall_time=0.01,
                tok_per_sec=100.0,
                acceptance_rate=1.0,
                trajectory=collector.to_trajectory(),
            )

    engine = HintResponsiveEngine(runner)
    reward_fn = FormatReward()  # scores 1.0 if <think> and <answer> present, else 0.0

    # 1. Without curriculum guidance: all score 0.0, advantages 0.0
    config_noguide = GRPOTrainerConfig(group_size=2, curriculum_hint_on_zero=False)
    trainer_noguide = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config_noguide)
    item_raw = PromptItem(
        prompt="Hard clinical vignette",
        metadata={"learning_objective": "Recognize Streptococcus pneumoniae"},
    )
    rollout_noguide = trainer_noguide.rollout_group(item_raw)
    assert rollout_noguide.mean_reward == 0.0
    assert torch.equal(rollout_noguide.advantages, torch.zeros(2))
    assert "curriculum_hint_applied" not in rollout_noguide.prompt_item.metadata

    # 2. With curriculum guidance: triggers retry with prompt guidance, scores 1.0
    config_guide = GRPOTrainerConfig(
        group_size=2, curriculum_hint_on_zero=True, max_hint_retries=1
    )
    trainer_guide = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config_guide)
    rollout_guided = trainer_guide.rollout_group(item_raw)

    assert rollout_guided.prompt_item.metadata.get("curriculum_hint_applied") is True
    assert "Clinical Guidance: Recognize Streptococcus pneumoniae" in rollout_guided.prompt_item.prompt
    assert rollout_guided.mean_reward == 1.0
    assert "<think>guided reasoning</think>" in rollout_guided.completions[0]


def test_trainer_hint_masks_gold_answer_leak():
    """Verify that curriculum hints screen out and mask direct occurrences of the gold diagnosis."""
    model = TinyMockModel()
    runner = TinyMockRunner(model)
    engine = TinyMockEngine(runner)
    reward_fn = FormatReward()

    config = GRPOTrainerConfig(mask_gold_in_hint=True)
    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config)

    # 1. Test gold_why containing the exact gold diagnosis
    item = PromptItem(
        prompt="Patient vignette",
        ground_truth="Streptococcus pneumoniae",
        metadata={
            "gold_why": "Streptococcus pneumoniae is the most common cause of sepsis in patients with sickle cell disease.",
        },
    )
    hint = trainer._extract_prompt_hint(item)
    assert hint is not None
    # Gold answer and distinctive keywords must NOT appear unmasked
    assert "Streptococcus pneumoniae" not in hint
    assert "pneumoniae" not in hint.lower()
    assert "[CONDITION]" in hint or "[...]" in hint

    # 2. Test learning_objective containing gold diagnosis
    item_lo = PromptItem(
        prompt="Patient vignette",
        ground_truth="Acute appendicitis",
        metadata={
            "learning_objective": "Recognize the clinical presentation of Acute appendicitis in elderly patients.",
        },
    )
    hint_lo = trainer._extract_prompt_hint(item_lo)
    assert hint_lo is not None
    assert "Acute appendicitis" not in hint_lo
    assert "appendicitis" not in hint_lo.lower()
    assert "[CONDITION]" in hint_lo or "[...]" in hint_lo


def test_trainer_dead_group_split_semantics():
    """Verify dead-group split semantics:
    1. All-failed group (all 0.80 < 2.0): triggers hint rescue re-roll.
    2. All-correct group (all 2.0 >= 2.0): bypassed without hint re-roll.
    """
    model = TinyMockModel()
    runner = TinyMockRunner(model)

    class MockFixedReward(BaseReward):
        def __init__(self, score: float):
            self.score = score

        def compute_reward(self, prompt, completion, **kwargs):
            return self.score

    class FixedEngine:
        def __init__(self, runner):
            self.runner = runner

        def generate(self, **kwargs):
            collector = TrajectoryCollector([1, 2, 3])
            collector.append_step(10, -0.5)
            return GenerationOutput(
                text="<think>reasoning</think><answer>output</answer>",
                token_ids=[10],
                num_generated_tokens=1,
                num_forward_passes=1,
                wall_time=0.01,
                tok_per_sec=100.0,
                acceptance_rate=1.0,
                trajectory=collector.to_trajectory(),
            )

    engine = FixedEngine(runner)
    config = GRPOTrainerConfig(group_size=2, curriculum_hint_on_zero=True, max_hint_retries=1)

    item = PromptItem(
        prompt="Clinical vignette",
        ground_truth="Pneumonia",
        metadata={"learning_objective": "Identify lower respiratory infection"},
    )

    # 1. All-failed group (all score 0.80 < 2.0) -> triggers hint rescue
    trainer_failed = DiffuGRPOTrainer(engine=engine, reward_fn=MockFixedReward(0.80), config=config)
    rollout_failed = trainer_failed.rollout_group(item)
    assert rollout_failed.prompt_item.metadata.get("curriculum_hint_applied") is True
    assert "Clinical Guidance:" in rollout_failed.prompt_item.prompt

    # 2. All-correct group (all score 2.0 >= 2.0) -> bypassed without re-roll
    trainer_correct = DiffuGRPOTrainer(engine=engine, reward_fn=MockFixedReward(2.0), config=config)
    rollout_correct = trainer_correct.rollout_group(item)
    assert rollout_correct.prompt_item.metadata.get("curriculum_hint_applied") is None
    assert "Clinical Guidance:" not in rollout_correct.prompt_item.prompt



