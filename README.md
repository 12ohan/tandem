# Tandem

Tandem is a high-performance, standalone self-speculative inference and DiffuGRPO reinforcement learning engine built for NVIDIA's Nemotron-Labs-Diffusion (NLD) tri-mode models, optimized natively for Apple Silicon (PyTorch MPS / MLX).

## Architecture

Tandem executes a 2-pass self-speculative cycle per iteration:
1. **Pass 1 (Bidirectional Draft):** Given prefix context and $K$ trailing mask tokens `[seed, MASK_1, ..., MASK_K]`, the diffusion head proposes $K$ candidate draft tokens in parallel.
2. **Pass 2 (Causal Verification):** Candidate tokens `[seed, draft_1, ..., draft_K]` are evaluated in a single causal autoregressive forward pass with KV cache.
3. **Shifted Matching & KV Rewind:** Accepted tokens satisfy $\text{draft}[i] == \text{ar}[i-1]$. If $c$ tokens match, the remaining $K - c$ unverified KV slots are immediately rewound, and the first rejected token is corrected by the true causal distribution.

## Core Features
- **Native Apple Silicon Acceleration:** First-class execution on Apple M-series chips via PyTorch MPS and Apple MLX with zero CUDA or Triton dependencies.
- **DiffuGRPO RL Integrity:** Unbiased per-token logprob and entropy logging for reinforcement learning policy optimization, eliminating NaN gradients and distribution contamination.
- **Exact Residual Speculative Sampling:** True residual sampling $\max(0, p(y) - q(y)\min(1, p(y)/q(y)))$ on draft rejection, preserving exact target policy distributions.
- **Log-Space Gumbel Sampling:** Zero transient tensor allocation optimization for high-throughput block diffusion.

## Quickstart

```python
from tandem import TandemEngine, EngineConfig

config = EngineConfig(
    model_id="nvidia/Nemotron-Labs-Diffusion-3B",
    block_size=4,
    device="mps",
)
engine = TandemEngine(config)

output = engine.generate("Explain quantum entanglement in two sentences:")
print(output.text)
print(f"Tokens/s: {output.tok_per_sec:.2f}, Acceptance Rate: {output.acceptance_rate:.2%}")
```

## License
Apache-2.0
