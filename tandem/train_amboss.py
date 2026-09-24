"""Production training entrypoint for DiffuGRPO clinical differential reasoning on Amboss.

Features:
- Deterministic 85% train / 15% held-out SHA-256 partition on Amboss QIDs.
- In-process pre-flight sanity gates (causal logprob finite check + candidate scorer check).
- Group Relative Policy Optimization (GRPO) on the 402M untied diffusion head.
- Screened hint-in-prompt curriculum for dead groups with on-policy importance ratio preservation.
- Full resumable checkpointing (model weights, optimizer moments, step, dataset cursor, RNG states).
- Live dashboard tracking policy loss, KL penalty, mean reward, variance, clip fraction,
  useful acceptance rate (alpha), and periodic evaluation probe (Gold Top-k recall,
  softmax mass vs 10.08% baseline, and Shannon diagnostic entropy).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn.functional as F

from tandem.config import EngineConfig
from tandem.engine.candidate_scorer import CandidateDifferentialScorer
from tandem.engine.spec import TandemEngine
from tandem.rl.amboss import AmbossDifferentialReward, load_amboss_questions
from tandem.rl.dataset import PromptDataset, PromptItem
from tandem.rl.trainer import DiffuGRPOTrainer, GRPOTrainerConfig

AMBOSS_DEFAULT_DIR = Path(
    "/Users/rohanmaster/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Resources/data/raw/amboss_qbank/questions"
)
DEFAULT_PREFIX_TEMPLATE = "\n<answer>"


def get_memory_info() -> Dict[str, float]:
    """Retrieve allocated and driver unified memory metrics on Apple Silicon."""
    info = {"allocated_gb": 0.0, "driver_gb": 0.0}
    if torch.backends.mps.is_available():
        info["allocated_gb"] = torch.mps.current_allocated_memory() / (1024 ** 3)
        info["driver_gb"] = torch.mps.driver_allocated_memory() / (1024 ** 3)
    return info


def _candidate_pool(item: PromptItem, k: Optional[int] = 5) -> List[str]:
    """Build a gold-preserving candidate pool for scoring and evaluation."""
    pool = list(item.metadata.get("all_candidates", []) or [])
    if not pool:
        pool = [item.ground_truth, "Alternative Condition"]

    gold = item.ground_truth
    assert gold, "PromptItem must carry a ground_truth diagnosis"
    assert any(str(c).strip().lower() == str(gold).strip().lower() for c in pool), (
        f"Gold diagnosis {gold!r} must be in candidate pool"
    )

    cleaned: List[str] = []
    seen = set()
    for c in pool:
        c_clean = str(c).strip()
        key = re.sub(r"\s+", " ", c_clean.lower())
        if c_clean and key not in seen:
            seen.add(key)
            cleaned.append(c_clean)

    ordered = [gold] + [c for c in cleaned if c.lower() != str(gold).lower()]
    if k is not None:
        ordered = ordered[:k]
    assert gold in ordered, "gold truncated out of pool"
    return ordered


def run_preflight_checks(engine: TandemEngine, sample_item: PromptItem) -> None:
    """Execute in-session pre-flight integration gates before step 1."""
    print("\n" + "=" * 80)
    print("RUNNING IN-PROCESS PRE-FLIGHT INTEGRATION GATES")
    print("=" * 80)

    # Gate 1: Teacher-forced gold logprob sanity gate
    print("[Pre-flight Gate 1/2] Verifying teacher-forced gold logprob sanity...")
    tokenizer = engine.runner.tokenizer
    device = engine.runner.device
    test_text = sample_item.prompt[:250]
    toks = tokenizer.encode(test_text, add_special_tokens=True)
    tensor_toks = torch.tensor([toks], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = engine.runner.forward_causal(tensor_toks, use_cache=False)
        logprobs = F.log_softmax(logits[0, :-1, :].float(), dim=-1)
        next_toks = torch.tensor(toks[1:], device=device)
        chosen_lps = logprobs[torch.arange(len(next_toks)), next_toks]

    assert not torch.isnan(chosen_lps).any(), "Pre-flight failure: NaNs detected in causal logprobs"
    assert torch.all(chosen_lps <= 0.0), "Pre-flight failure: Logprobs exceed 0.0"
    assert torch.all(chosen_lps >= -30.0), "Pre-flight failure: Pathological underflow in logprobs"
    print(f"   Gate 1 PASSED: Mean causal logprob = {chosen_lps.mean().item():.4f}, zero NaNs.")

    # Gate 2: Candidate Differential Scorer KV-Cache Reuse
    print("[Pre-flight Gate 2/2] Verifying candidate differential scorer on real weights...")
    scorer = CandidateDifferentialScorer(engine)
    candidates = _candidate_pool(sample_item, k=5)
    res = scorer.score_candidates(
        sample_item.prompt,
        candidates,
        prefix_template=DEFAULT_PREFIX_TEMPLATE,
    )
    assert len(res.candidates) > 0, "Pre-flight failure: No candidates scored"
    total_prob = sum(c.candidate_probability for c in res.candidates)
    assert abs(total_prob - 1.0) < 1e-3, f"Pre-flight failure: Probabilities sum to {total_prob}"
    assert res.entropy_bits >= 0.0, "Pre-flight failure: Negative entropy"
    print(f"   Gate 2 PASSED: Scored {len(res.candidates)} candidates, H = {res.entropy_bits:.3f} bits.")
    print("=" * 80 + "\n")


def run_eval_probe(
    scorer: CandidateDifferentialScorer,
    eval_slice: List[PromptItem],
    step: int,
) -> Dict[str, float]:
    """Evaluate gold ranking, probability mass, and entropy on held-out slice."""
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0
    gold_masses: List[float] = []
    gold_ranks: List[int] = []
    entropies: List[float] = []

    for item in eval_slice:
        if not item.ground_truth:
            continue
        try:
            candidates = _candidate_pool(item, k=None)
            res = scorer.score_candidates(
                item.prompt,
                candidates,
                prefix_template=DEFAULT_PREFIX_TEMPLATE,
            )
            total += 1
            entropies.append(res.entropy_bits)

            # Find gold rank and probability
            gold_str = item.ground_truth.strip().lower()
            matched_candidate = None
            for c in res.candidates:
                if c.candidate.strip().lower() == gold_str:
                    matched_candidate = c
                    break

            if matched_candidate is not None:
                gold_masses.append(matched_candidate.candidate_probability)
                gold_ranks.append(matched_candidate.rank)
                if matched_candidate.rank == 1:
                    top1_correct += 1
                if matched_candidate.rank <= 3:
                    top3_correct += 1
                if matched_candidate.rank <= 5:
                    top5_correct += 1
            else:
                gold_masses.append(0.0)
                gold_ranks.append(len(candidates) + 1)
        except Exception:
            continue

    if total == 0:
        return {}

    metrics = {
        "eval_top1": top1_correct / total,
        "eval_top3": top3_correct / total,
        "eval_top5": top5_correct / total,
        "eval_gold_mass": sum(gold_masses) / len(gold_masses) if gold_masses else 0.0,
        "eval_gold_rank": sum(gold_ranks) / len(gold_ranks) if gold_ranks else 0.0,
        "eval_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "eval_cases": float(total),
    }
    print(
        f"[EVAL @ Step {step:03d}] "
        f"Top-1: {metrics['eval_top1']:.1%} | "
        f"Top-3: {metrics['eval_top3']:.1%} | "
        f"Gold Mass: {metrics['eval_gold_mass']:.2%} (base 10.08%) | "
        f"Gold Rank: {metrics['eval_gold_rank']:.2f} (base #4.00) | "
        f"Entropy: {metrics['eval_entropy']:.2f} bits | "
        f"Cases: {total}"
    )
    return metrics


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiffuGRPO Clinical RL Training on Amboss")
    parser.add_argument("--steps", type=int, default=50, help="Number of RL optimization steps")
    parser.add_argument("--group-size", type=int, default=4, help="Number of rollouts per prompt (G)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts per optimization step")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens per rollout")
    parser.add_argument("--block-size", type=int, default=4, help="Diffusion block size (K)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Rollout sampling temperature")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--kl-beta", type=float, default=0.04, help="KL divergence penalty coefficient")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO surrogate clipping epsilon")
    parser.add_argument("--dual-clip-c", type=float, default=3.0, help="NeMo dual-clipping threshold")
    parser.add_argument(
        "--normalize-by-std",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Divide group advantages by std (disable for Dr. GRPO mean-centered advantage)",
    )
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable hint-in-prompt curriculum for zero-variance groups",
    )
    parser.add_argument("--hint-fraction", type=float, default=0.25, help="Fraction of rationale used in hint")
    parser.add_argument("--hint-anneal-steps", type=int, default=100, help="Hint probability decay steps")
    parser.add_argument(
        "--format-primed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable in-context clinical format priming exemplar (default False)",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints/amboss_v1", help="Checkpoint save directory")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=1, help="Log step metrics every N steps")
    parser.add_argument("--eval-every", type=int, default=10, help="Run held-out eval probe every N steps")
    parser.add_argument("--eval-slice-size", type=int, default=50, help="Number of held-out questions to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--amboss-dir",
        type=str,
        default=str(AMBOSS_DEFAULT_DIR),
        help="Directory containing Amboss question JSON files",
    )
    parser.add_argument("--skip-preflight", action="store_true", help="Skip in-process pre-flight sanity gates")
    return parser.parse_args(args)


def train(args: argparse.Namespace) -> None:
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("TANDEM DIFFUGRPO CLINICAL REINFORCEMENT LEARNING")
    print(f"Steps: {args.steps} | Group Size: {args.group_size} | LR: {args.lr} | KL Beta: {args.kl_beta}")
    print(f"Curriculum: {args.curriculum} | Format Primed: {args.format_primed} | Seed: {args.seed}")
    print("=" * 80)

    # 1. Load Amboss Questions (Deterministic 85% train / 15% eval split)
    print("\n[1/4] Loading Amboss Questions with SHA-256 holdout partition...")
    train_items = load_amboss_questions(
        args.amboss_dir,
        split="train",
        eval_split_ratio=0.15,
        eval_seed=args.seed,
        format_primed=args.format_primed,
    )
    eval_items = load_amboss_questions(
        args.amboss_dir,
        split="eval",
        eval_split_ratio=0.15,
        eval_seed=args.seed,
        format_primed=args.format_primed,
    )
    print(f"Loaded {len(train_items)} train questions and {len(eval_items)} held-out eval questions.")

    eval_slice = eval_items[: args.eval_slice_size]

    # 2. Initialize Model and Engine
    print("\n[2/4] Initializing TandemEngine on Apple Silicon MPS (bfloat16)...")
    t0_engine = time.perf_counter()
    engine_config = EngineConfig(block_size=args.block_size, temperature=args.temperature)
    engine = TandemEngine(config=engine_config)
    print(f"Engine initialized in {time.perf_counter() - t0_engine:.2f}s | {get_memory_info()}")

    # 3. Pre-Flight Integration Gates
    if not args.skip_preflight:
        run_preflight_checks(engine, train_items[0])

    # 4. Configure Hardened Reward and Trainer
    print("\n[3/4] Configuring Hardened Amboss Differential Reward & Trainer...")
    reward_fn = AmbossDifferentialReward(
        gold_reward=2.0,
        ruleout_credit_per_candidate=0.20,
        max_ruleout_credit=0.80,
        gate_on_gold=False,  # Magnitude dominance preserves variance across all groups
    )

    trainer_config = GRPOTrainerConfig(
        group_size=args.group_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        clip_eps=args.clip_eps,
        dual_clip_c=args.dual_clip_c,
        normalize_by_std=args.normalize_by_std,
        beta_kl=getattr(args, "kl_beta", 0.04),
        train_head_only=True,  # 402M untied diffusion head
        curriculum_hint_on_zero=args.curriculum,
        hint_fraction=args.hint_fraction,
        hint_anneal_steps=args.hint_anneal_steps,
        save_dir=args.save_dir,
        log_interval=args.log_every,
    )

    trainer = DiffuGRPOTrainer(engine=engine, reward_fn=reward_fn, config=trainer_config)
    scorer = CandidateDifferentialScorer(engine)

    # 5. Checkpoint Resume
    start_step = 0
    cursor = 0
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        ckpt = trainer.load_checkpoint(args.resume_from)
        start_step = trainer.global_step
        cursor = trainer.last_prompt_index
        print(f"Restored to global_step={start_step}, dataset_cursor={cursor}")

    # Baseline pre-training evaluation probe
    if start_step == 0:
        print("\n[Pre-Training Baseline Evaluation Probe]")
        run_eval_probe(scorer, eval_slice, step=0)

    # 6. Training Loop
    print("\n[4/4] Starting DiffuGRPO Training Loop...")
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    total_steps = args.steps
    for step_idx in range(start_step, total_steps):
        current_step = step_idx + 1

        # Fetch batch with dataset cursor
        batch_prompts: List[PromptItem] = []
        for _ in range(args.batch_size):
            batch_prompts.append(train_items[cursor])
            cursor = (cursor + 1) % len(train_items)
        trainer.last_prompt_index = cursor

        t0_step = time.perf_counter()
        metrics = trainer.step(batch_prompts, global_step=current_step)
        dt = time.perf_counter() - t0_step

        mem = get_memory_info()

        # Step logging
        if current_step % args.log_every == 0:
            print(
                f"Step {current_step:03d}/{total_steps:03d} | "
                f"Loss: {metrics['loss']:.4f} (pol: {metrics['policy_loss']:.4f}, kl: {metrics['kl_loss']:.4f}) | "
                f"Reward: {metrics['mean_reward']:.2f} | "
                f"Alpha: {metrics['acceptance_rate']:.1%} | "
                f"Clip: {metrics['clip_fraction']:.1%} | "
                f"Driver: {mem['driver_gb']:.2f} GB | "
                f"Time: {dt:.1f}s"
            )

        # Periodic Eval Probe
        if current_step % args.eval_every == 0:
            run_eval_probe(scorer, eval_slice, step=current_step)

        # Periodic Checkpoint
        if current_step % args.checkpoint_every == 0 or current_step == total_steps:
            ckpt_file = save_path / f"step_{current_step:03d}.pt"
            trainer.save_checkpoint(ckpt_file)
            print(f"   [Checkpoint saved: {ckpt_file}]")

    print("\n" + "=" * 80)
    print("DIFFUGRPO TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
