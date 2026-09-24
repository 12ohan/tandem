# TANDEM ENGINE: ARCHITECTURAL HANDOFF AND STATE OF RECORD

**Repository:** `12ohan/tandem` ([GitHub](https://github.com/12ohan/tandem))  
**Local Path:** `/Users/rohanmaster/Developer/tandem`  
**Target Hardware:** Apple Silicon M2 Pro (macOS Darwin, 16 GB Unified Memory ceiling)  
**Target Model:** NVIDIA `Nemotron-Labs-Diffusion-3B` (`nvidia/Nemotron-Labs-Diffusion-3B`, Ministral-3B base, untied `diffusion_head`, 3,831,659,520 parameters, bfloat16)  
**Date of Record:** 2026-09-23  

---

## 1. Executive Summary & Core Mission

`tandem` is an ultra-fast, mathematically exact self-speculative inference and DiffuGRPO reinforcement learning engine built specifically for NVIDIA's tri-mode diffusion-language models on Apple Silicon. It couples bidirectional diffusion drafting with causal autoregressive verification, and specializes in clinical differential diagnosis reasoning across the Amboss Question Bank (3,317 clinical cases) and the UMLS 2026AA Metathesaurus.

### Guiding Principles
- **Mathematical Exactness over Heuristics:** Self-speculative decoding adheres to Scheme (a) equality-accept with target replacement ($y_{n+1} \sim p(y \mid x, y_{1:n})$). Target distribution is preserved identically.
- **Empirical Operating Noise Floor:** Declared and verified noise threshold for bfloat16 Metal attention tiles and tail logit sampling: $\text{mean } |\Delta| < 0.05$, $\text{max } |\Delta| < 0.15$.
- **Exploit-Proof Deterministic Rewards:** Group Relative Policy Optimization (GRPO) relies strictly on deterministic programmatic rewards (magnitude dominance, proximity attribution, NegEx negation checking) rather than brittle or hackable LLM judges.
- **On-Policy Integrity:** Zero-reward homogeneous groups are resolved via hint-in-prompt conditioning, preserving exact on-policy importance ratios ($r_t = 1.0$) without synthetic completion masking.

---

## 2. Upstream Open-Source Contributions Shipped

Three external pull requests were implemented, verified with hardware receipts, and submitted upstream to unblock the tri-mode ecosystem on non-CUDA/Apple Silicon hardware:

1. **NVIDIA Official Repository:**
   - **PR:** [NVlabs/Nemotron-Labs-Diffusion#3](https://github.com/NVlabs/Nemotron-Labs-Diffusion/pull/3)
   - **Title:** `feat: add dynamic device resolution supporting Apple Silicon (MPS) and CPU`
   - **Scope:** Replaced hardcoded `.cuda()` across chat drivers (`chat_ar.py`, `chat_dlm.py`, `chat_linear_spec.py`) and `evaluate.py` with dynamic accelerator detection (`cuda` -> `mps` -> `cpu`), added `MODEL_ID` environment variable overrides, and corrected upstream doc bugs (HF quickstart `prompt_ids` bug and dead SGLang git branch link).

2. **Apple MLX Upstream:**
   - **PR:** [ml-explore/mlx-lm#1918](https://github.com/ml-explore/mlx-lm/pull/1918)
   - **Title:** `Add Nemotron-Labs-Diffusion architecture support`
   - **Scope:** Implemented `mlx_lm.models.nemotron_labs_diffusion` subclassing `ministral3.Model`. Sanitizes untied `diffusion_head` and `encoder.` prefixes across all 237 checkpoint keys. Verified 96/96 token-exact greedy decode parity against PyTorch MPS reference.

3. **Hugging Face Hub Community:**
   - **PR:** [mlx-community/Nemotron-Labs-Diffusion-3B-4bit#2](https://huggingface.co/mlx-community/Nemotron-Labs-Diffusion-3B-4bit/discussions/2)
   - **Title:** `Add standalone model.py for MLX support`
   - **Scope:** Shipped standalone `model.py` adapter and updated `config.json` enabling immediate zero-install execution in Apple MLX.

---

## 3. Codebase Architecture & File Inventory

The local repository is located at [`/Users/rohanmaster/Developer/tandem`](file:///Users/rohanmaster/Developer/tandem).

```text
tandem/
├── config.py                   # EngineConfig dataclass (device, dtype, temperature, block_size)
├── __init__.py                 # Clean package-level symbol exports
├── engine/
│   ├── model_runner.py         # TandemModelRunner: PyTorch MPS wrapper for NLD-3B
│   ├── kv_cache.py             # DynamicKVCache: O(1) rewind capability for speculation
│   ├── matcher.py              # match_speculative_tokens: Scheme (a) equality matching
│   ├── sampler.py              # Gumbel-max and residual distribution samplers
│   ├── spec.py                 # TandemEngine: Self-speculative generation loop
│   ├── umls_trie.py            # UMLSEntityTrie: Radix trie over UMLS 2026AA (0.58 us proposal)
│   └── candidate_scorer.py     # CandidateDifferentialScorer: KV-cache reuse differential ranking
└── rl/
    ├── dataset.py              # PromptDataset and PromptItem dataclasses
    ├── diffu_grpo.py           # Core GRPO mathematics: group advantages and clipped surrogate loss
    ├── reward.py               # Pluggable deterministic rewards (Format, Regex, Math, Composite)
    ├── amboss.py               # Amboss dataset parser and AmbossDifferentialReward
    ├── trainer.py              # DiffuGRPOTrainer: Rollout collection, autograd step, hint curriculum
    └── trajectory.py           # Trajectory and TrajectoryCollector: Step logprobs and metadata
```

---

## 4. Key Mathematical Decisions & Algorithmic Designs

### A. Speculative Verification Scheme: Scheme (a) Target Replacement
- **Greedy Decode Parity:** Exact match (32/32 tokens bit-exact) against pure autoregressive decoding.
- **Stochastic Sampling:** When $y \sim p(\cdot \mid x, \hat{y}_{<i})$, matching compares $\hat{y}_i == y_i$. On the first mismatch at position $k$, the target sample $y_k$ is accepted and subsequent draft tokens are discarded. The target distribution is strictly preserved with zero drift.

### B. Untied Parameter Accounting
- Base Ministral-3B uses tied embeddings (`tie_word_embeddings: true`, 3,429,006,336 parameters).
- NLD-3B unties the head (`tie_word_embeddings: false`, 3,831,659,520 parameters). The `diffusion_head` contains $3072 \times 131072 = 402,653,184$ parameters.
- Setting `train_head_only=True` in `GRPOTrainerConfig` restricts autograd gradients strictly to these 402M parameters, allowing live RL training within Apple Silicon's 16 GB unified memory envelope.

### C. Reward Function Hardening (AmbossDifferentialReward)
To prevent reinforcement learning reward hacking, the clinical reward function enforces five structural gates:
1. **Magnitude Dominance:** $Gold = 2.0$, $RuleOut = 0.20$ each (max 0.80 across 4 distractors). Total reward bounded in $[0.0, 2.80]$. Any wrong answer is strictly bounded below any right answer ($0.80 < 2.00$). Crucially, partial credit preserves within-group variance $\sigma_R > 0$ on all-wrong groups, preventing policy gradient stalls.
2. **Candidate Set Recall:** Recognizes gold diagnosis in `<answer>` or in `<differential><candidate>...</candidate></differential>` tags.
3. **No Think Block Overcrediting:** Mentioning a hypothesis in `<think>` that is later dropped for a wrong diagnosis scores 0.0. Asymmetric prefix matching (e.g. "acute" matching "acute cholecystitis") is eliminated.
4. **Proximity Window & Regex Negation Guard:** Distractor keywords must appear within $\pm 25$ words of the candidate's mention or inside `<rule_out target="...">`. Negation scope uses word boundaries (e.g. `\bpresent\b`) and expanded affirmative verbs (`shows`, `reveals`, `demonstrates`, `notable`) scanning all keyword occurrences across the context.
5. **Deterministic Held-Out Splitting:** `load_amboss_questions(directory_path, split="all"|"train"|"eval", eval_split_ratio=0.15)` uses deterministic SHA-256 hash on question ID to guarantee zero train/eval leakage.

### D. Semantic Candidate Differential Scorer (KV-Cache Reuse, Contrastive PMI)
- **$O(1)$ Prompt Recomputation:** Given a clinical vignette of length $P$, prefill the KV cache once. For each candidate of length $M$, execute forward pass of length $M-1$, evaluate per-token logprobs, and call `kv_cache.crop(P)`.
- **Prefix Stability Assertion:** Validates that `full_toks[:P] == ctx_tokens` at BPE boundaries; cleanly falls back to continuation tokenization if a boundary merge occurs.
- **Contrastive Pointwise Mutual Information (PMI):** Option to evaluate candidates against a neutral context ($x_{\text{neutral}}$):
  $$\text{PMI}(y; x) = \log p(y \mid x) - \log p(y \mid x_{\text{neutral}})$$
  Subtracting generic term frequency cancels out baseline prior bias, prioritizing diagnoses specifically indicated by findings in the vignette.
- **Diagnostic Entropy:** Computes categorical softmax distribution across candidates and Shannon entropy $H = -\sum p_k \log_2(p_k)$ in bits, directly measuring clinical uncertainty.

### E. Hint-in-Prompt Curriculum & Dead-Group Semantics
- **Answer-Leak Guard:** Multi-source screening across `learning_objective`, `hint`, `gold_why`, and `distractor_buts`. Performs sentence-level filtering and aggressive token-level masking (`[CONDITION]` and `[...]`) ensuring the gold diagnosis is never leaked. Falls back gracefully if fewer than 2 clinical reasoning tokens remain.
- **Split Dead-Group Semantics:**
  - All-failed groups ($R_i < 2.0$, spread $< 10^{-5}$): Triggers screened curriculum hint re-roll.
  - All-correct groups ($R_i \ge 2.0$): Bypasses rescue cleanly without wasting compute re-rolling solved cases.
  - All-truncated rollouts: If rollouts exhaust `max_new_tokens` without EOS, hint rescue is skipped.
- **DAPO Advantage Bypass:** If all group advantages are zero, `compute_rollout_loss` shortcuts immediately without building causal autograd graphs, saving significant GPU FLOPs and memory.

---

## 5. Local Data Map & External Assets

| Asset | Local Path | Description |
|---|---|---|
| Amboss Question Bank | `/Users/rohanmaster/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Resources/data/raw/amboss_qbank/questions` | 3,317 parsed clinical questions with rationales |
| UMLS 2026AA Metathesaurus | `/Users/rohanmaster/Developer/deep-research-pipeline/data/external/umls/2026AA/META/MRCONSO.RRF` | 2.2 GB concept table (SNOMED-CT, MSH, RxNorm) |
| UMLS Semantic Types | `/Users/rohanmaster/Developer/deep-research-pipeline/data/external/umls/2026AA/META/MRSTY.RRF` | 203 MB semantic type mapping |
| Argus Clinical Consensus | `/Users/rohanmaster/Developer/Argus/01_canon` | `MASTER_CONSENSUS.md`, `GRAPH_LAYER_SPEC.md` |
| Upstream NLD Fork | `/Users/rohanmaster/Developer/Nemotron-Labs-Diffusion` | Branch `support-mps-cpu-device-agnostic` |
| Upstream MLX Fork | `/Users/rohanmaster/Developer/mlx-lm` | Branch `add-nemotron-labs-diffusion` |

---

## 6. Verification Receipts & Test Commands

### Fast Unit Test Suite (75 Tests, 8.4s)
```bash
cd /Users/rohanmaster/Developer/tandem
PYTHONPATH=. pytest -m "not integration and not slow" tests/
```
Result: `75 passed, 2 deselected, 1 warning in 8.42s`.

### Real-Weights Amboss GRPO Step (Single-Process MPS)
```bash
PYTHONPATH=. python3 tests/test_amboss_grpo_real_weights.py
```
Receipt: G=4 rollouts on Amboss Case 1 (15.93s, $\alpha = 54.65\%$), backward pass on untied head (6.60s), peak memory 9.39 GB allocated / 9.82 GB driver.

### Teacher-Forced Gold Logprob Sanity Gate
```bash
PYTHONPATH=. python3 tests/test_logprob_sanity.py
```
Receipt: All per-token logprobs finite in $[-20.0, 0.0]$, zero NaNs, mean logprob `-2.1797`.

### Real-Weights Candidate Differential Scorer
```bash
PYTHONPATH=. pytest -s -k "test_candidate_scorer_real_weights_amboss" tests/test_candidate_scorer.py
```
Receipt: Evaluates 5 Amboss options (`all_candidates`) in 22s; ranks gold diagnosis (*Streptococcus pneumoniae*) vs distractors; verified with strict precondition assertions (`item.ground_truth in candidates`).

---

## 7. Roadmap & Next Engineering Frontiers

1. **Two-Tower Generation & Certification Coupling:**
   - Serial Generation Mode: Tower A (Narrative Context) -> UMLS Radix Trie Linker -> Tower B (Diffusion Differential Denoising).
   - Parallel Certification Mode: Tower A (Unstructured Vignette) vs Tower B (Structured KG Subgraph). Divergence trips escalation threshold.
2. **UMLS Persistent Binary Cache:**
   - Serialize pre-tokenized `UMLSEntityTrie` (50,000 concepts, ~88 MB) to a memory-mapped binary cache (`umls_trie_50k.bin`) to eliminate cold-start parsing overhead.
3. **TPU v5e-8 Distributed Scaling Blueprint:**
   - Retain local prototyping on Apple Silicon MPS with 3B model.
   - When scaling to 14B under Kaggle TPU v5e-8 (128 GB distributed HBM), replace custom CUDA diffusion kernels with PyTorch/XLA standard operations under FSDP.
