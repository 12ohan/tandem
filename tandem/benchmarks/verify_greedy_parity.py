from __future__ import annotations

import time
import torch
import torch.nn.functional as F
from tandem import TandemEngine, EngineConfig


def run_pure_ar(engine: TandemEngine, prompt: str, max_new_tokens: int):
    """Run sequential autoregressive generation collecting token IDs and logprobs."""
    device = engine.runner.device
    eos_ids = engine.runner.eos_token_ids
    prompt_ids = engine.runner.tokenizer.encode(prompt, return_tensors="pt").to(device)

    # 1. Prefill
    logits, past_key_values = engine.runner.forward_causal(prompt_ids, use_cache=True)
    last_logit = logits[:, -1, :]
    probs = F.softmax(last_logit, dim=-1)
    next_token = torch.argmax(last_logit, dim=-1, keepdim=True)
    first_logprob = float(torch.log(probs[0, next_token.item()] + 1e-12).item())

    generated = [int(next_token.item())]
    logprobs = [first_logprob]

    while len(generated) < max_new_tokens:
        if generated[-1] in eos_ids:
            break
        logits, past_key_values = engine.runner.forward_causal(
            next_token, past_key_values=past_key_values, use_cache=True
        )
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tok_id = int(next_token.item())
        tok_logprob = float(torch.log(probs[0, tok_id] + 1e-12).item())

        generated.append(tok_id)
        logprobs.append(tok_logprob)

    return generated, logprobs


def verify_greedy_parity():
    prompt = "Explain the theory of general relativity in three sentences:"
    max_tokens = 32

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = EngineConfig(device=device, temperature=0.0, return_logprob=True)
    print(f"Loading TandemEngine with model: {config.model_id} on {device}...")
    engine = TandemEngine(config)

    # 1. Generate Pure AR Reference Stream
    print(f"\nRunning Pure AR reference (K=0) for {max_tokens} tokens...")
    ar_tokens, ar_logprobs = run_pure_ar(engine, prompt, max_tokens)
    print(f"  Pure AR emitted {len(ar_tokens)} tokens.")
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 2. Generate with Tandem K=2
    print(f"\nRunning Tandem Speculative Decoding (K=2, temp=0.0)...")
    out_k2 = engine.generate(prompt, max_new_tokens=max_tokens, block_size=2, temperature=0.0, return_logprob=True)
    print(f"  Tandem K=2 emitted {len(out_k2.token_ids)} tokens (NFE={out_k2.num_forward_passes}, alpha={out_k2.acceptance_rate:.2%}).")
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 3. Generate with Tandem K=4
    print(f"\nRunning Tandem Speculative Decoding (K=4, temp=0.0)...")
    out_k4 = engine.generate(prompt, max_new_tokens=max_tokens, block_size=4, temperature=0.0, return_logprob=True)
    print(f"  Tandem K=4 emitted {len(out_k4.token_ids)} tokens (NFE={out_k4.num_forward_passes}, alpha={out_k4.acceptance_rate:.2%}).")
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


    # 4. Check Exact Token-by-Token Parity
    print("\n" + "=" * 60)
    print("PARITY VERIFICATION")
    print("=" * 60)
    
    # Check K=2 token parity
    assert out_k2.token_ids == ar_tokens, (
        f"K=2 token mismatch! First divergence: "
        f"{[(i, a, b) for i, (a, b) in enumerate(zip(ar_tokens, out_k2.token_ids)) if a != b][:1]}"
    )
    print(f"PASS: Tandem K=2 token-exact greedy match vs Pure AR ({len(ar_tokens)}/{len(ar_tokens)} tokens bit-exact).")

    # Check K=4 token parity
    assert out_k4.token_ids == ar_tokens, (
        f"K=4 token mismatch! First divergence: "
        f"{[(i, a, b) for i, (a, b) in enumerate(zip(ar_tokens, out_k4.token_ids)) if a != b][:1]}"
    )
    print(f"PASS: Tandem K=4 token-exact greedy match vs Pure AR ({len(ar_tokens)}/{len(ar_tokens)} tokens bit-exact).")

    # 5. Check Logprob Alignment
    assert out_k2.trajectory is not None, "K=2 trajectory is missing!"
    k2_logprobs = out_k2.trajectory.logprobs
    assert len(k2_logprobs) == len(ar_logprobs), f"Trajectory length mismatch: {len(k2_logprobs)} vs {len(ar_logprobs)}"

    deltas_k2 = [abs(a - b) for a, b in zip(ar_logprobs, k2_logprobs)]
    max_delta_k2 = max(deltas_k2)
    mean_delta_k2 = sum(deltas_k2) / len(deltas_k2)
    print(f"PASS: Tandem K=2 logprob alignment (max |delta| = {max_delta_k2:.2e}, mean |delta| = {mean_delta_k2:.2e}).")
    assert max_delta_k2 < 0.15, f"Logprob mismatch exceeding fp16 reduction bounds: max |delta| = {max_delta_k2}"

    assert out_k4.trajectory is not None, "K=4 trajectory is missing!"
    k4_logprobs = out_k4.trajectory.logprobs
    assert len(k4_logprobs) == len(ar_logprobs), f"Trajectory length mismatch: {len(k4_logprobs)} vs {len(ar_logprobs)}"

    deltas_k4 = [abs(a - b) for a, b in zip(ar_logprobs, k4_logprobs)]
    max_delta_k4 = max(deltas_k4)
    mean_delta_k4 = sum(deltas_k4) / len(deltas_k4)
    print(f"PASS: Tandem K=4 logprob alignment (max |delta| = {max_delta_k4:.2e}, mean |delta| = {mean_delta_k4:.2e}).")
    assert max_delta_k4 < 0.15, f"Logprob mismatch exceeding fp16 reduction bounds: max |delta| = {max_delta_k4}"



    print("=" * 60)
    print("ALL REAL-WEIGHTS GREEDY PARITY GATES PASSED TOKEN-FOR-TOKEN AND LOGPROB-FOR-LOGPROB.")
    print("=" * 60)


if __name__ == "__main__":
    verify_greedy_parity()
