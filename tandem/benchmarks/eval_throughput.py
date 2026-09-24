from __future__ import annotations

import argparse
import time
import torch
from tandem import TandemEngine, EngineConfig


def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark Tandem speculative decoding vs AR baseline on Apple Silicon.")
    parser.add_argument("--model-id", type=str, default="nvidia/Nemotron-Labs-Diffusion-3B")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--max-tokens", type=int, default=64)
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

    # 2. Evaluate across block sizes
    results = {}
    for K in args.block_sizes:
        print(f"\nRunning Tandem Speculative Decoding (Block Size K={K})...")
        out = engine.generate(args.prompt, max_new_tokens=args.max_tokens, block_size=K)
        print(f"  Generated {out.num_generated_tokens} tokens in {out.wall_time:.2f}s ({out.tok_per_sec:.2f} tok/s)")
        print(f"  Forward Passes (NFE): {out.num_forward_passes}")
        print(f"  Draft Acceptance Rate (alpha): {out.acceptance_rate:.2%}")
        results[K] = out

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Block Size (K)':<15}{'Tok/s':<12}{'NFE':<8}{'Alpha':<10}")
    print("-" * 45)
    for K, out in results.items():
        print(f"{K:<15}{out.tok_per_sec:<12.2f}{out.num_forward_passes:<8}{out.acceptance_rate:<10.2%}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
