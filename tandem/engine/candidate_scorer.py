from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F

from tandem.engine.spec import TandemEngine
from tandem.engine.umls_trie import UMLSEntityTrie


@dataclass
class CandidateDifferentialScore:
    """Evaluation receipt for a single clinical differential candidate."""

    candidate: str
    token_ids: List[int]
    token_logprobs: List[float]
    total_logprob: float
    mean_logprob: float
    perplexity: float
    cui: Optional[str] = None
    concept_name: Optional[str] = None
    umls_verified: bool = False
    rank: int = 0
    candidate_probability: float = 0.0


@dataclass
class DifferentialScoringResult:
    """Complete ranked clinical differential diagnostic evaluation."""

    vignette: str
    candidates: List[CandidateDifferentialScore]
    top_candidate: CandidateDifferentialScore
    entropy_bits: float
    temperature: float = 1.0

    def summary(self) -> str:
        lines = [
            f"Vignette: {repr(self.vignette[:80])}...",
            (
                f"Top Candidate: '{self.top_candidate.candidate}' "
                f"(mean logp: {self.top_candidate.mean_logprob:.4f}, "
                f"prob: {self.top_candidate.candidate_probability:.2%})"
            ),
            (
                f"Differential Entropy: {self.entropy_bits:.3f} bits across "
                f"{len(self.candidates)} candidates"
            ),
            "Ranked Differential Candidates:",
        ]
        for c in self.candidates:
            umls_badge = f" [UMLS: {c.cui}]" if c.umls_verified else ""
            lines.append(
                f"   #{c.rank}: '{c.candidate}'{umls_badge} | "
                f"Mean logp: {c.mean_logprob:.4f} | Prob: {c.candidate_probability:.2%} | "
                f"Tokens: {len(c.token_ids)}"
            )
        return "\n".join(lines)


class CandidateDifferentialScorer:
    """Evaluates and ranks clinical candidate differentials using causal autoregressive log-likelihoods.

    Key algorithmic properties:
    1. Prefix KV-Cache Reuse: Prefills the clinical vignette once into KV cache, evaluating
       arbitrary candidate differentials via cache rewind/crop (O(1) prompt recomputation).
    2. Length Normalization: Ranks candidates by mean per-token log-likelihood (1/M sum log p)
       to eliminate structural bias against multi-word medical diagnoses.
    3. UMLS Radix Trie Alignment: Verifies candidate strings against UMLS 2026AA concepts,
       resolving Concept Unique Identifiers (CUIs).
    4. Diagnostic Ambiguity Metric: Computes categorical softmax probabilities across the
       candidate set and evaluates Shannon entropy in bits, quantifying diagnostic certainty.
    """

    def __init__(
        self,
        engine: TandemEngine,
        umls_trie: Optional[UMLSEntityTrie] = None,
    ):
        self.engine = engine
        self.runner = engine.runner
        self.umls_trie = umls_trie

    def score_candidates(
        self,
        vignette: str,
        candidates: Sequence[str],
        prefix_template: Optional[str] = None,
        temperature: float = 1.0,
    ) -> DifferentialScoringResult:
        """Score and rank a list of differential candidate diagnoses for a clinical vignette.

        Args:
            vignette: Clinical case vignette or prompt.
            candidates: List of candidate differential diagnosis strings.
            prefix_template: Optional string to append to the vignette before candidate evaluation
                             (e.g. "\\n<answer>" or "\\nDifferential Diagnosis: ").
            temperature: Softmax temperature for candidate probability and entropy calculation.

        Returns:
            DifferentialScoringResult with ranked candidates and diagnostic entropy.
        """
        if not candidates:
            raise ValueError("Candidates list must contain at least one candidate.")

        tokenizer = self.runner.tokenizer
        device = self.runner.device

        # 1. Format prompt prefix
        prompt_str = vignette.rstrip()
        if prefix_template:
            prompt_str += prefix_template

        prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=True)
        P = len(prompt_tokens)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        # 2. Prefill prompt into KV-cache once
        with torch.no_grad():
            prompt_logits, kv_cache = self.runner.forward_causal(
                prompt_tensor, use_cache=True
            )

        first_token_logprobs_dist = F.log_softmax(
            prompt_logits[0, -1, :].float(), dim=-1
        )

        scored_candidates: List[CandidateDifferentialScore] = []

        # 3. Autoregressively score each candidate with cache rewind
        for cand_str in candidates:
            cand_clean = cand_str.strip()
            # Determine subword token splice matching natural context continuation
            full_str = (
                prompt_str
                + (" " if not prompt_str.endswith((" ", "\n", ">")) else "")
                + cand_clean
            )
            full_toks = tokenizer.encode(full_str, add_special_tokens=True)
            cand_toks = full_toks[P:]

            if not cand_toks:
                # Fallback if candidate was whitespace or empty
                cand_toks = tokenizer.encode(" " + cand_clean, add_special_tokens=False)

            M = len(cand_toks)
            assert M > 0, f"Candidate '{cand_str}' produced 0 tokens"

            token_logprobs: List[float] = []

            # First token logprob from prompt's final logits
            t0 = cand_toks[0]
            logp_0 = float(first_token_logprobs_dist[t0].item())
            token_logprobs.append(logp_0)

            # Subsequent tokens evaluated through KV cache
            if M > 1:
                rem_toks = cand_toks[:-1]
                rem_tensor = torch.tensor([rem_toks], dtype=torch.long, device=device)
                with torch.no_grad():
                    rem_logits, _ = self.runner.forward_causal(
                        rem_tensor, past_key_values=kv_cache, use_cache=True
                    )
                # rem_logits[0, j, :] predicts cand_toks[j+1]
                log_probs_rem = F.log_softmax(rem_logits[0].float(), dim=-1)
                for j in range(M - 1):
                    target_tok = cand_toks[j + 1]
                    logp_j = float(log_probs_rem[j, target_tok].item())
                    token_logprobs.append(logp_j)

                # Rewind/crop KV-cache back to prompt length P
                kv_cache.crop(P)

            total_logp = sum(token_logprobs)
            mean_logp = total_logp / M
            perplexity = math.exp(-mean_logp) if mean_logp > -100 else float("inf")

            # Check UMLS trie if provided
            cui = None
            concept_name = None
            umls_verified = False
            if self.umls_trie is not None:
                # 1. Try exact lookup on candidate tokens
                lookup = self.umls_trie.exact_lookup(cand_toks)
                if lookup is None:
                    # Also try looking up raw candidate tokens without leading space
                    raw_c_toks = tokenizer.encode(cand_clean, add_special_tokens=False)
                    lookup = self.umls_trie.exact_lookup(raw_c_toks)
                if lookup is not None:
                    cui, concept_name = lookup
                    umls_verified = True
                elif hasattr(self.umls_trie, "extract_concepts"):
                    # Try concept extraction on candidate string
                    extracted = self.umls_trie.extract_concepts(cand_clean, tokenizer)
                    if extracted:
                        cui = extracted[0][0]
                        concept_name = extracted[0][1]
                        umls_verified = True

            score_obj = CandidateDifferentialScore(
                candidate=cand_str,
                token_ids=cand_toks,
                token_logprobs=token_logprobs,
                total_logprob=total_logp,
                mean_logprob=mean_logp,
                perplexity=perplexity,
                cui=cui,
                concept_name=concept_name,
                umls_verified=umls_verified,
            )
            scored_candidates.append(score_obj)

        # 4. Rank candidates by mean_logprob descending (length-normalized)
        scored_candidates.sort(key=lambda x: x.mean_logprob, reverse=True)
        for rank_idx, cand in enumerate(scored_candidates, 1):
            cand.rank = rank_idx

        # 5. Compute categorical distribution across candidates and Shannon entropy
        temp = max(temperature, 1e-4)
        scores_tensor = torch.tensor(
            [c.mean_logprob / temp for c in scored_candidates], dtype=torch.float32
        )
        probs = F.softmax(scores_tensor, dim=0)

        entropy_bits = 0.0
        for i, c in enumerate(scored_candidates):
            p_val = float(probs[i].item())
            c.candidate_probability = p_val
            if p_val > 1e-12:
                entropy_bits -= p_val * math.log2(p_val)

        return DifferentialScoringResult(
            vignette=vignette,
            candidates=scored_candidates,
            top_candidate=scored_candidates[0],
            entropy_bits=entropy_bits,
            temperature=temperature,
        )
