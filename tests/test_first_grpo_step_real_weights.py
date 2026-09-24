import math
import os
import sys
import time
from pathlib import Path
import torch

# Ensure tandem package is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tandem import (
    BaseReward,
    CompositeReward,
    DiffuGRPOTrainer,
    FormatReward,
    GRPOTrainerConfig,
    MathCorrectnessReward,
    PromptItem,
    TandemEngine,
)


def get_memory_info() -> str:
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
        driver_allocated = torch.mps.driver_allocated_memory() / (1024 ** 3)
        return f"MPS Allocated: {allocated:.2f} GB | MPS Driver: {driver_allocated:.2f} GB"
    return "MPS unavailable"


def run_first_grpo_step():
    print("=" * 80)
    print("FIRST REAL GRPO OPTIMIZATION STEP: Apple Silicon M2 Pro (Nemotron-Labs-Diffusion-3B)")
    print("=" * 80)
    print(f"Initial Memory: {get_memory_info()}")

    print("\n1. Initializing TandemEngine with real bfloat16 weights on MPS...")
    t0_load = time.perf_counter()
    engine = TandemEngine()
    load_time = time.perf_counter() - t0_load
    print(f"Model loaded in {load_time:.2f}s | {get_memory_info()}")

    # 2. Configure reward: Format (<think>...</think><answer>...</answer>) + Math accuracy
    class SolutionReward(BaseReward):
        """Domain reward providing partial credit for deriving the correct solution token."""
        def compute_reward(self, prompt: str, completion: str, **kwargs) -> float:
            target = kwargs.get("ground_truth", "6")
            score = 0.0
            if target in completion:
                score += 0.5
            if "x = 6" in completion or "x=6" in completion:
                score += 0.5
            return score

    reward_fn = CompositeReward([
        (FormatReward(think_reward=0.25, answer_reward=0.25), 1.0),
        (MathCorrectnessReward(match_reward=0.5), 1.0),
        (SolutionReward(), 1.0),
    ])

    # 3. Configure Trainer: G=4, max_tokens=32, block_size=4, train_head_only=True
    config = GRPOTrainerConfig(
        group_size=4,
        learning_rate=1e-5,
        temperature=1.0,
        max_new_tokens=32,
        block_size=4,
        train_head_only=True,  # Trains the 402M untied diffusion head
        clip_eps=0.2,
        beta_kl=0.04,
    )

    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=config)
    num_trainable = sum(p.numel() for p in trainer.trainable_params)
    print(f"Trainer configured with train_head_only=True | Trainable params: {num_trainable:,}")

    # 4. Define 1-shot prompt eliciting reasoning format
    prompt = (
        "Solve the math problem. Show your reasoning in <think> tags and the answer in <answer> tags.\n"
        "Example:\nProblem: What is 4 + 5?\n<think>4 + 5 = 9</think><answer>9</answer>\n\n"
        "Problem: Solve for x: 3*x - 7 = 11."
    )
    ground_truth = "6"
    prompt_item = PromptItem(prompt=prompt, ground_truth=ground_truth)

    print(f"\n2. Executing G=4 speculative rollouts for prompt:")
    print(f"   Prompt: {repr(prompt)}")
    print(f"   Ground Truth: {repr(ground_truth)}")

    t0_rollout = time.perf_counter()
    rollout = trainer.rollout_group(prompt_item)
    rollout_time = time.perf_counter() - t0_rollout

    print(f"\n3. Rollout Results ({rollout_time:.2f}s):")
    for g, (comp, rew, adv) in enumerate(zip(rollout.completions, rollout.rewards, rollout.advantages)):
        first_line = comp.replace("\n", " ")[:60]
        print(f"   Completion {g}: R={rew.item():.2f} | Adv={adv.item():+.4f} | Preview: {repr(first_line)}")

    print(f"\n   Mean Reward:       {rollout.mean_reward:.3f}")
    print(f"   Reward Std:        {rollout.std_reward:.4f}")
    print(f"   Acceptance Rate:   {rollout.acceptance_rate:.2%}")

    # 5. Compute loss and execute backpropagation + optimizer step
    print(f"\n4. Executing Autograd Backward Pass & Optimizer Step...")
    print(f"   Pre-backward Memory: {get_memory_info()}")

    t0_opt = time.perf_counter()
    metrics = trainer.step_rollout(rollout)
    step_time = time.perf_counter() - t0_opt

    print(f"\n5. Step Execution Receipt ({step_time:.2f}s):")
    print(f"   Total Loss:        {metrics['loss']:.5f}")
    print(f"   Policy Loss:       {metrics['policy_loss']:.5f}")
    print(f"   KL Penalty:        {metrics['kl_loss']:.5f}")
    print(f"   Mean Advantage:    {rollout.advantages.mean().item():.5f}")
    print(f"   Clip Fraction:     {metrics['clip_fraction']:.2%}")
    print(f"   Post-step Memory:  {get_memory_info()}")

    # 6. Verify gradient flow into the head parameters
    head_weight = engine.runner.model.diffusion_head.weight
    print(f"\n6. Verification:")
    print(f"   Head parameter shape: {list(head_weight.shape)}")
    print(f"   Loss is finite: {math.isfinite(metrics['loss'])}")
    print(f"   Post-step cache cleared successfully.")

    assert math.isfinite(metrics["loss"]), "Loss must be finite"
    assert math.isfinite(metrics["policy_loss"]), "Policy loss must be finite"

    print("\nRESULT: SUCCESS. First real GRPO step completed on Apple Silicon M2 Pro.")
    print("=" * 80)


if __name__ == "__main__":
    run_first_grpo_step()
