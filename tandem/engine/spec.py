from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, List, Set, Union
import torch
import torch.nn.functional as F

from tandem.config import EngineConfig
from tandem.engine.model_runner import TandemModelRunner
from tandem.engine.matcher import match_speculative_tokens
from tandem.engine.sampler import sample_gumbel_max
from tandem.rl.trajectory import TrajectoryCollector, Trajectory


@dataclass
class GenerationOutput:
    text: str
    token_ids: List[int]
    num_generated_tokens: int
    num_forward_passes: int
    wall_time: float
    tok_per_sec: float
    acceptance_rate: float
    trajectory: Optional[Trajectory] = None


def crop_cache(past_key_values, max_length: int) -> None:
    """Crop DynamicCache to max_length across PyTorch / Transformers versions."""
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(max_length)
    else:
        for layer_idx in range(len(past_key_values)):
            past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, :max_length]
            past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, :max_length]
        past_key_values._seen_tokens = max_length


class TandemEngine:
    """High-performance self-speculative engine for Nemotron-Labs-Diffusion on Apple Silicon."""

    def __init__(self, config: Optional[EngineConfig] = None):
        if config is None:
            config = EngineConfig()
        self.config = config
        self.runner = TandemModelRunner(config)

    def generate(
        self,
        prompt: Union[str, List[int], torch.Tensor],
        max_new_tokens: int = 128,
        block_size: Optional[int] = None,
        temperature: Optional[float] = None,
        return_logprob: Optional[bool] = None,
    ) -> GenerationOutput:
        """Run self-speculative generation on the input prompt.
        
        Args:
            prompt: Text prompt string or token ID tensor.
            max_new_tokens: Maximum number of tokens to generate.
            block_size: Speculative draft block size K (defaults to config.block_size).
            temperature: Sampling temperature (0.0 for greedy).
            return_logprob: Whether to record DiffuGRPO RL trajectory logprobs.
        """
        K = block_size if block_size is not None else self.config.block_size
        temp = temperature if temperature is not None else self.config.temperature
        want_logprob = return_logprob if return_logprob is not None else self.config.return_logprob

        device = self.runner.device
        eos_ids = self.runner.eos_token_ids

        # 1. Encode prompt
        if isinstance(prompt, str):
            prompt_ids = self.runner.tokenizer.encode(prompt, return_tensors="pt").to(device)
        elif isinstance(prompt, list):
            prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        else:
            prompt_ids = prompt.to(device)

        prompt_len = prompt_ids.shape[1]
        collector = TrajectoryCollector(prompt_ids[0].tolist()) if want_logprob else None

        t0 = time.perf_counter()
        nfe = 0
        total_proposed_drafts = 0
        total_accepted_drafts = 0

        # 2. Causal prefill on prompt
        logits, past_key_values = self.runner.forward_causal(
            prompt_ids,
            use_cache=True,
        )
        nfe += 1

        last_logit = logits[:, -1, :]
        if temp > 0:
            probs = F.softmax(last_logit / max(temp, 1e-5), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            first_logprob = float(torch.log(probs[0, next_token.item()] + 1e-12).item())
        else:
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)
            probs = F.softmax(last_logit, dim=-1)
            first_logprob = float(torch.log(probs[0, next_token.item()] + 1e-12).item())

        first_token_id = int(next_token.item())
        if collector:
            entropy = float(-(probs * torch.log(probs + 1e-12)).sum(dim=-1).item())
            collector.append_step(token_id=first_token_id, logprob=first_logprob, entropy=entropy)

        generated_ids: List[int] = [first_token_id]

        if first_token_id in eos_ids:
            wall_time = time.perf_counter() - t0
            full_text = self.runner.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return GenerationOutput(
                text=full_text,
                token_ids=generated_ids,
                num_generated_tokens=len(generated_ids),
                num_forward_passes=nfe,
                wall_time=wall_time,
                tok_per_sec=len(generated_ids) / max(wall_time, 1e-6),
                acceptance_rate=0.0,
                trajectory=collector.to_trajectory() if collector else None,
            )

        # 3. Speculative Decode Loop
        block = torch.full((1, K + 1), self.runner.mask_token_id, dtype=torch.long, device=device)

        while len(generated_ids) < max_new_tokens:
            cache_len = past_key_values.get_seq_length()
            block[0, 0] = next_token.item()
            block[0, 1:] = self.runner.mask_token_id

            # Pass 1: Propose K draft tokens via bidirectional diffusion forward
            draft_logits = self.runner.forward_draft(block, past_key_values)
            nfe += 1
            total_proposed_drafts += K

            if temp > 0:
                draft_probs = F.softmax(draft_logits / max(temp, 1e-5), dim=-1)
                draft_tokens = torch.multinomial(
                    draft_probs.view(-1, draft_probs.shape[-1]), num_samples=1
                ).view(1, K + 1)[:, 1:]
            else:
                draft_tokens = draft_logits[:, 1:, :].argmax(dim=-1)

            # Insert proposed draft tokens into block positions 1..K
            block[0, 1:] = draft_tokens[0]

            # Pass 2: Causal verification of [seed, draft_1, ..., draft_K]
            verify_logits, past_key_values = self.runner.forward_causal(
                block,
                past_key_values=past_key_values,
                use_cache=True,
            )
            nfe += 1

            if temp > 0:
                verify_probs = F.softmax(verify_logits / max(temp, 1e-5), dim=-1)
                ar_tokens = torch.multinomial(
                    verify_probs.view(-1, verify_probs.shape[-1]), num_samples=1
                ).view(1, K + 1)
            else:
                ar_tokens = verify_logits.argmax(dim=-1)
                verify_probs = F.softmax(verify_logits, dim=-1) if want_logprob else None

            # Shifted Matching: ar_tokens[0, i] verifies block[0, i+1]
            matches = (ar_tokens[0, :K] == block[0, 1:])
            # Cast bool to int32 before cumprod to prevent MPS runtime error
            cum_matches = matches.to(torch.int32).cumprod(dim=0)
            accepted = int(cum_matches.sum().item())
            total_accepted_drafts += accepted

            # AR also provides one bonus token at index 'accepted'
            accepted_tokens_slice = ar_tokens[0, : accepted + 1].tolist()
            committed_this_round = accepted + 1

            # Advance KV cache by committed tokens and crop out the unverified tail
            crop_cache(past_key_values, cache_len + committed_this_round)

            # Record trajectory if requested
            if collector and verify_probs is not None:
                for idx, tok in enumerate(accepted_tokens_slice):
                    tok_prob = float(verify_probs[0, idx, tok].item())
                    logprob = float(torch.log(torch.tensor(tok_prob) + 1e-12).item())
                    step_entropy = float(
                        -(verify_probs[0, idx] * torch.log(verify_probs[0, idx] + 1e-12)).sum().item()
                    )
                    collector.append_step(token_id=tok, logprob=logprob, entropy=step_entropy)

            generated_ids.extend(accepted_tokens_slice)

            # Invariant assertion: KV cache length must strictly equal
            # prefix prompt tokens + all committed tokens minus the deferred next seed token
            assert past_key_values.get_seq_length() == prompt_len + len(generated_ids) - 1

            # Check if any committed token hit an EOS token

            eos_hit = False
            for tok in accepted_tokens_slice:
                if tok in eos_ids:
                    eos_hit = True
                    break

            if eos_hit:
                break

            # The next seed token is the last committed token
            next_token = ar_tokens[0, accepted : accepted + 1]

        # 4. Truncate at EOS if present
        if collector:
            collector.truncate_at_eos(list(eos_ids))

        wall_time = time.perf_counter() - t0
        full_text = self.runner.tokenizer.decode(generated_ids, skip_special_tokens=True)
        alpha = total_accepted_drafts / max(total_proposed_drafts, 1)

        return GenerationOutput(
            text=full_text,
            token_ids=generated_ids,
            num_generated_tokens=len(generated_ids),
            num_forward_passes=nfe,
            wall_time=wall_time,
            tok_per_sec=len(generated_ids) / max(wall_time, 1e-6),
            acceptance_rate=alpha,
            trajectory=collector.to_trajectory() if collector else None,
        )
