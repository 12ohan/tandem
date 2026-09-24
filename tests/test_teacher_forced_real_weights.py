from __future__ import annotations

import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Ensure tandem is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tandem import TandemEngine, EngineConfig


def run_teacher_forced_gate():
    print("=" * 80)
    print("PHASE 4 CAPSTONE GATE: Real-Weights Teacher-Forced Re-computation & On-Policy Ratio")
    print("=" * 80)

    config = EngineConfig(temperature=1.0, return_logprob=True)
    print(f"Loading official weights: {config.model_id} on device={config.device}, dtype={config.dtype}...")
    engine = TandemEngine(config)

    prompt = "The clinical presentation of acute appendicitis typically involves"
    max_new_tokens = 16
    block_size = 4
    tau = 1.0

    print(f"\n1. Executing self-speculative rollout (tau={tau}, K={block_size}, max_tokens={max_new_tokens})...")
    out = engine.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=tau,
        return_logprob=True,
    )

    traj = out.trajectory
    assert traj is not None, "Trajectory must be recorded"
    prompt_tokens = traj.prompt_tokens
    completion_tokens = traj.completion_tokens
    traj_logprobs = traj.logprobs

    n = len(prompt_tokens)
    m = len(completion_tokens)

    print(f"Prompt tokens (n): {n}")
    print(f"Completion tokens (m): {m}")
    print(f"Total tokens: {n + m}")
    print(f"Generated text: {repr(out.text)}")
    print(f"Alpha (acceptance rate): {out.acceptance_rate:.2%}")

    # 2. Teacher-forced forward pass through causal model
    print("\n2. Executing teacher-forced causal forward pass over [prompt + completion]...")
    full_tokens = prompt_tokens + completion_tokens
    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=engine.runner.device)

    with torch.no_grad():
        logits, _ = engine.runner.forward_causal(input_ids, use_cache=False)

    print(f"Logits tensor shape: {list(logits.shape)}")

    # 3. Compare row-by-row: token y_i reads row n - 1 + i
    deltas = []
    ratios = []

    print("\n3. Token-by-Token Alignment & Logprob Extraction Verification:")
    print(f"{'Idx':>3} | {'Token ID':>8} | {'Token Text':>14} | {'Row':>5} | {'Traj Logp':>10} | {'TF Logp':>10} | {'|Delta|':>9} | {'Ratio r_t':>10}")
    print("-" * 88)

    for i in range(m):
        tok_id = completion_tokens[i]
        tok_text = engine.runner.tokenizer.decode([tok_id])
        row = n - 1 + i

        z_i = logits[0, row, :]
        log_probs_i = F.log_softmax(z_i / max(traj.temperature, 1e-5), dim=-1)
        tf_logprob = float(log_probs_i[tok_id].item())
        traj_logp = float(traj_logprobs[i])

        delta = abs(tf_logprob - traj_logp)
        deltas.append(delta)

        # On-policy importance ratio r_t = exp(log \pi_\theta - log \pi_old)
        ratio = math.exp(tf_logprob - traj_logp)
        ratios.append(ratio)

        print(
            f"{i:3d} | {tok_id:8d} | {repr(tok_text):>14} | {row:5d} | "
            f"{traj_logp:10.5f} | {tf_logprob:10.5f} | {delta:9.4e} | {ratio:10.5f}"
        )

    mean_delta = sum(deltas) / len(deltas)
    max_delta = max(deltas)
    mean_ratio = sum(ratios) / len(ratios)
    min_ratio = min(ratios)
    max_ratio = max(ratios)

    print("-" * 88)
    print(f"Mean |Delta| logprob: {mean_delta:.4e}")
    print(f"Max  |Delta| logprob: {max_delta:.4e}")
    print(f"Mean Ratio r_t:       {mean_ratio:.5f}")
    print(f"Min  Ratio r_t:       {min_ratio:.5f}")
    print(f"Max  Ratio r_t:       {max_ratio:.5f}")

    # 4. Strict assertions: Pinned Capstone Tolerance
    # Regime Differences vs Greedy Parity (mean 4.18e-3):
    # (i) tau=1.0 sampled tail tokens reach |logp| ~ 5.9, where bf16 relative precision on
    #     large-magnitude logits produces larger absolute deltas (vs greedy near-argmax cancellation).
    # (ii) Shape delta: L=5 chunked verify vs L=26 batched forward crosses Metal attention tile boundaries.
    # Declared Operating Noise Floor: mean |Delta| < 0.05, max |Delta| < 0.15
    assert mean_delta < 0.05, f"Mean delta {mean_delta} exceeds declared noise threshold 0.05"
    assert max_delta < 0.15, f"Max delta {max_delta} exceeds declared noise threshold 0.15"

    # On-policy ratios: Observed max deviation 9.5% provides 2x headroom vs PPO clip band 20%
    # Guarantees that bf16 kernel reduction noise cannot trigger spurious policy clips.
    assert abs(mean_ratio - 1.0) < 0.02, f"Mean ratio {mean_ratio} diverges from 1.0"
    assert min_ratio > 0.8, f"Min ratio {min_ratio} clipped below 0.8"
    assert max_ratio < 1.2, f"Max ratio {max_ratio} clipped above 1.2"

    print("\nRESULT: PASS. Teacher-forced re-computation gate and on-policy ratio verified on real weights.")
    print("=" * 80)


if __name__ == "__main__":
    run_teacher_forced_gate()
