from __future__ import annotations

import argparse
import time
import torch
from tandem import TandemEngine, EngineConfig


def run_ar_baseline(engine: TandemEngine, prompt: str, max_new_tokens: int):
    """Run pure sequential autoregressive decoding (1 token per pass)."""
    device = engine.runner.device
    eos_ids = engine.runner.eos_token_ids
    prompt_ids = engine.runner.tokenizer.encode(prompt, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    logits, past_key_values = engine.runner.forward_causal(prompt_ids, use_cache=True)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated = [int(next_token.item())]
    nfe = 1

    while len(generated) < max_new_tokens:
        if generated[-1] in eos_ids:
            break
        logits, past_key_values = engine.runner.forward_causal(
            next_token, past_key_values=past_key_values, use_cache=True
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        nfe += 1

    wall_time = time.perf_counter() - t0
    tok_per_sec = len(generated) / max(wall_time, 1e-6)
    return len(generated), wall_time, tok_per_sec, nfe


def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark Tandem speculative decoding vs AR baseline on Apple Silicon.")
    parser.add_argument("--model-id", type=str, default="nvidia/Nemotron-Labs-Diffusion-3B")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--prompt", type=str, default="Explain the theory of general relativity in three sentences:")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = EngineConfig(
        model_id=args.model_id,
        device=args.device if args.device else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    print(f"Loading TandemEngine with model: {config.model_id} on {config.device}...")
    engine = TandemEngine(config)

    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Target Generation Length: {args.max_tokens} tokens")
    print("=" * 60)

    # 1. Warm-up
    print("\nWarming up...")
    _ = engine.generate(args.prompt, max_new_tokens=16, block_size=2)

    # 2. Pure AR Baseline
    print("\nRunning Pure Autoregressive Baseline (K=0, 1 token/pass)...")
    ar_tokens, ar_time, ar_tok_s, ar_nfe = run_ar_baseline(engine, args.prompt, args.max_tokens)
    print(f"  Generated {ar_tokens} tokens in {ar_time:.2f}s ({ar_tok_s:.2f} tok/s, NFE={ar_nfe})")

    # 3. Evaluate across block sizes
    results = {}
    for K in args.block_sizes:
        print(f"\nRunning Tandem Speculative Decoding (Block Size K={K})...")
        out = engine.generate(args.prompt, max_new_tokens=args.max_tokens, block_size=K)
        speedup = out.tok_per_sec / max(ar_tok_s, 1e-6)
        print(f"  Generated {out.num_generated_tokens} tokens in {out.wall_time:.2f}s ({out.tok_per_sec:.2f} tok/s, {speedup:.2f}x speedup)")
        print(f"  Forward Passes (NFE): {out.num_forward_passes}")
        print(f"  Draft Acceptance Rate (alpha): {out.acceptance_rate:.2%}")
        results[K] = (out, speedup)

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY (Apple Silicon M2 Pro)")
    print("=" * 65)
    print(f"{'Mode':<18}{'Tok/s':<12}{'Speedup':<12}{'NFE':<8}{'Alpha':<10}")
    print("-" * 65)
    print(f"{'Pure AR (K=0)':<18}{ar_tok_s:<12.2f}{'1.00x':<12}{ar_nfe:<8}{'-':<10}")
    for K, (out, speedup) in results.items():
        mode_str = f"Tandem (K={K})"
        speedup_str = f"{speedup:.2f}x"
        print(f"{mode_str:<18}{out.tok_per_sec:<12.2f}{speedup_str:<12}{out.num_forward_passes:<8}{out.acceptance_rate:<10.2%}")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()

