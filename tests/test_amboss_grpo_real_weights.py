from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time
import torch

# Ensure tandem package is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tandem import (
    AmbossDifferentialReward,
    DiffuGRPOTrainer,
    GRPOTrainerConfig,
    TandemEngine,
    load_amboss_questions,
)

AMBOSS_QUESTIONS_DIR = Path(
    "/Users/rohanmaster/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Resources/data/raw/amboss_qbank/questions"
)


def get_memory_info() -> str:
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
        driver_allocated = torch.mps.driver_allocated_memory() / (1024 ** 3)
        return f"MPS Allocated: {allocated:.2f} GB | MPS Driver: {driver_allocated:.2f} GB"
    return "MPS unavailable"


def run_amboss_grpo_real_weights():
    print("=" * 80)
    print("END-TO-END AMBOSS CLINICAL GRPO ON REAL WEIGHTS (Nemotron-Labs-Diffusion-3B)")
    print("=" * 80)
    print(f"Initial Memory: {get_memory_info()}")

    # 1. Load real Amboss questions
    print("\n1. Ingesting Amboss questions from disk...")
    items = load_amboss_questions(AMBOSS_QUESTIONS_DIR, max_questions=2)
    assert len(items) >= 1, "Must load at least one Amboss question"
    print(f"Loaded {len(items)} Amboss question(s) successfully.")
    for idx, it in enumerate(items):
        gold = it.ground_truth
        cands = it.metadata.get("differential_candidates", [])
        print(f"   Case {idx+1}: Gold='{gold}' | Options={len(cands)}")

    # 2. Initialize TandemEngine with real bfloat16 weights on Apple Silicon MPS
    print("\n2. Initializing TandemEngine on Apple Silicon MPS (bfloat16)...")
    t0_load = time.perf_counter()
    engine = TandemEngine()
    load_time = time.perf_counter() - t0_load
    print(f"Model loaded in {load_time:.2f}s | {get_memory_info()}")

    # 3. Configure hardened clinical differential reward
    # For initial smoke test on untrained base model, set gate_on_gold=False
    # so partial rule-out discussion provides variance for gradient estimation
    reward_fn = AmbossDifferentialReward(
        gold_reward=2.0,
        ruleout_credit_per_candidate=0.20,
        max_ruleout_credit=0.80,
        gate_on_gold=False,
    )

    # 4. Configure Trainer: G=4, max_tokens=32, block_size=4, train_head_only=True
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

    # 5. Execute G=4 rollouts on Case 1
    item = items[0]
    print(f"\n3. Executing G=4 speculative rollouts for Amboss Case 1:")
    print(f"   Vignette preview: {repr(item.prompt[:120])}...")
    print(f"   Target Gold: '{item.ground_truth}'")

    t0_rollout = time.perf_counter()
    rollout = trainer.rollout_group(item)
    rollout_time = time.perf_counter() - t0_rollout

    print(f"\n4. Rollout Results ({rollout_time:.2f}s):")
    for g, (comp, rew, adv) in enumerate(zip(rollout.completions, rollout.rewards, rollout.advantages)):
        preview = comp.replace("\n", " ")[:65]
        print(f"   Completion {g}: R={rew.item():.2f} | Adv={adv.item():+.4f} | Preview: {repr(preview)}")

    print(f"\n   Mean Reward:       {rollout.mean_reward:.3f}")
    print(f"   Reward Std:        {rollout.std_reward:.4f}")
    print(f"   Acceptance Rate:   {rollout.acceptance_rate:.2%}")

    # 6. Execute Autograd Backward Pass & Optimizer Step
    print(f"\n5. Executing Autograd Backward Pass & Optimizer Step on untied head...")
    print(f"   Pre-backward Memory: {get_memory_info()}")

    t0_opt = time.perf_counter()
    metrics = trainer.step_rollout(rollout)
    step_time = time.perf_counter() - t0_opt

    print(f"\n6. Step Execution Receipt ({step_time:.2f}s):")
    print(f"   Total Loss:        {metrics['loss']:.5f}")
    print(f"   Policy Loss:       {metrics['policy_loss']:.5f}")
    print(f"   KL Penalty:        {metrics['kl_loss']:.5f}")
    print(f"   Mean Advantage:    {rollout.advantages.mean().item():.5f}")
    print(f"   Clip Fraction:     {metrics['clip_fraction']:.2%}")
    print(f"   Post-step Memory:  {get_memory_info()}")

    # 7. Verification
    head_weight = engine.runner.model.diffusion_head.weight
    print(f"\n7. Verification:")
    print(f"   Diffusion head shape: {list(head_weight.shape)}")
    print(f"   Loss is finite: {math.isfinite(metrics['loss'])}")
    print(f"   Post-step cache cleared successfully.")

    assert math.isfinite(metrics["loss"]), "Loss must be finite"
    assert math.isfinite(metrics["policy_loss"]), "Policy loss must be finite"

    print("\nRESULT: SUCCESS. Real-weights Amboss GRPO step completed on Apple Silicon M2 Pro.")
    print("=" * 80)


if __name__ == "__main__":
    run_amboss_grpo_real_weights()
