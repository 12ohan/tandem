from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from tandem.rl.amboss import STOP_WORDS
from tandem.rl.dataset import PromptDataset, PromptItem
from tandem.rl.diffu_grpo import GRPOMetrics, compute_group_advantages, compute_grpo_loss
from tandem.rl.reward import BaseReward
from tandem.rl.trajectory import Trajectory, TrajectoryCollector


@dataclass
class GRPOTrainerConfig:
    """Hyperparameters and runtime settings for DiffuGRPO reinforcement learning."""

    group_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    clip_eps: float = 0.2
    beta_kl: float = 0.04
    max_grad_norm: float = 1.0
    temperature: float = 1.0  # Default fixed to tau=1.0 per Phase 4 contract
    max_new_tokens: int = 128
    block_size: int = 4
    batch_size: int = 1
    grad_accum_steps: int = 1
    normalize_by_std: bool = True  # Flag to disable std-normalization (Dr. GRPO critique)
    divisor_mode: Literal["token_mean", "group_mean"] = "token_mean"
    dual_clip_c: Optional[float] = 3.0  # NeMo RL Dual-clipping parameter (Ye et al., 2019)
    overlong_filtering: bool = False  # Exclude max_tokens completions without EOS from loss
    format_failure_reward: float = 0.0  # Reward assigned when policy drifts to commit mask_token
    train_head_only: bool = False
    curriculum_hint_on_zero: bool = False  # If True, re-rolls zero-variance groups with progressive hint in prompt
    hint_fraction: float = 0.25  # Fraction of gold rationale to inject as progressive prompt hint
    max_hint_retries: int = 1  # Maximum curriculum retry attempts per prompt
    mask_gold_in_hint: bool = True  # Strictly screens and masks gold answer tokens from injected hints
    hint_anneal_steps: Optional[int] = None  # Steps over which hint probability linearly decays to 0.0
    empty_cache_interval: int = 1
    log_interval: int = 1
    save_dir: Optional[str] = None


@dataclass
class GRPORollout:
    """Rollout bundle for a single prompt expanded across G speculative completions."""

    prompt_item: PromptItem
    completions: List[str]
    trajectories: List[Trajectory]
    rewards: torch.Tensor
    advantages: torch.Tensor
    mean_reward: float
    std_reward: float
    acceptance_rate: float


class DiffuGRPOTrainer:
    """Reinforcement learning trainer implementing Group Relative Policy Optimization (GRPO)
    on top of the Tandem self-speculative inference engine.
    """

    def __init__(
        self,
        engine: Any,
        reward_fn: BaseReward,
        config: Optional[GRPOTrainerConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        ref_model: Optional[Any] = None,
    ):
        self.engine = engine
        self.reward_fn = reward_fn
        self.config = config if config is not None else GRPOTrainerConfig()
        self.ref_model = ref_model

        # Setup trainable parameters and optimizer
        model = self.engine.runner.model
        if self.config.train_head_only:
            # Freeze encoder backbone, train only the diffusion output head
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.diffusion_head.parameters():
                param.requires_grad = True
            trainable_params = list(model.diffusion_head.parameters())
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if not trainable_params:
                # If all parameters were frozen, enable autograd on entire model
                for param in model.parameters():
                    param.requires_grad = True
                trainable_params = list(model.parameters())

        self.trainable_params = trainable_params
        self.global_step = 0

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(
                self.trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

    def _extract_prompt_hint(self, prompt_item: PromptItem) -> Optional[str]:
        """Extract a clinical curriculum hint from prompt metadata if available,
        strictly screening and masking any direct mentions, abbreviations, or distinctive
        subwords of the gold diagnosis. Drops answer-bearing sentences and falls back safely.
        """
        meta = prompt_item.metadata
        if not meta:
            return None

        gold = str(prompt_item.ground_truth or "").strip()
        gold_clean = re.sub(r"[^\w\s]", " ", gold).lower().strip()
        gold_words = [w for w in gold_clean.split() if len(w) >= 3 and w not in STOP_WORDS]

        # Generate abbreviations (e.g. "S. pneumoniae" for "Streptococcus pneumoniae")
        abbrevs = []
        parts = gold.split()
        if len(parts) >= 2:
            abbrevs.append(f"{parts[0][0]}. {parts[1]}")
            abbrevs.append(f"{parts[0][0]} {parts[1]}")

        def _screen_text(text: str) -> Optional[str]:
            """Screen and sentence-filter text to drop answer-bearing clauses.
            Returns None if all informative content is dropped.
            """
            if not text:
                return None
            if not self.config.mask_gold_in_hint or not gold:
                return text.strip()

            # 1. Multi-sentence filtering: if multiple sentences present, prefer dropping answer sentences
            raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            if len(raw_sentences) > 1:
                surviving = []
                for sent in raw_sentences:
                    s_lower = sent.lower()
                    if gold.lower() in s_lower or any(ab.lower() in s_lower for ab in abbrevs) or any(re.search(rf"\b{re.escape(gw)}\b", s_lower) for gw in gold_words):
                        continue
                    surviving.append(sent)
                text_to_mask = " ".join(surviving) if surviving else text.strip()
            else:
                text_to_mask = text.strip()

            # 2. Token-level masking across all gold diagnosis mentions and distinctive tokens
            screened = text_to_mask
            screened = re.sub(re.escape(gold), "[CONDITION]", screened, flags=re.IGNORECASE)
            for ab in abbrevs:
                screened = re.sub(re.escape(ab), "[CONDITION]", screened, flags=re.IGNORECASE)
            for gw in gold_words:
                screened = re.sub(rf"\b{re.escape(gw)}\b", "[...]", screened, flags=re.IGNORECASE)

            # Ensure surviving text contains real clinical reasoning tokens
            cleaned_remaining = re.sub(r"\[\.\.\.\]|\[CONDITION\]|[^\w\s]", " ", screened).strip()
            meaningful_words = [w for w in cleaned_remaining.split() if len(w) >= 3 and w not in STOP_WORDS]
            if len(meaningful_words) < 2:
                return None

            return screened

        # Source 1: Learning objective (preferred high-level conceptual pointer)
        if "learning_objective" in meta and meta["learning_objective"]:
            screened_lo = _screen_text(str(meta["learning_objective"]))
            if screened_lo:
                return screened_lo

        # Source 2: Direct explicit hint (screened)
        if "hint" in meta and meta["hint"]:
            screened_hint = _screen_text(str(meta["hint"]))
            if screened_hint:
                return screened_hint

        # Source 3: Gold rationale fraction (strictly screened)
        gold_why = meta.get("gold_why")
        if gold_why and isinstance(gold_why, str) and len(gold_why.strip()) > 0:
            screened_why = _screen_text(gold_why)
            if screened_why:
                words = screened_why.split()
                k = max(1, int(len(words) * self.config.hint_fraction))
                return " ".join(words[:k]) + "..."

        # Source 4: Leading distractor rule-out note (screened)
        distractor_buts = meta.get("distractor_buts")
        if distractor_buts and isinstance(distractor_buts, dict) and len(distractor_buts) > 0:
            first_distractor, rationale = next(iter(distractor_buts.items()))
            raw_note = f"Note: {first_distractor} is ruled out because {rationale}"
            screened_note = _screen_text(raw_note)
            if screened_note:
                return screened_note

        return None

    def rollout_group(
        self, prompt_item: PromptItem, retry_count: int = 0, global_step: int = 0
    ) -> GRPORollout:
        """Generate G speculative completions for a prompt and compute normalized advantages."""
        G = self.config.group_size
        completions: List[str] = []
        trajectories: List[Trajectory] = []
        rewards_list: List[float] = []
        acceptance_rates: List[float] = []

        # Run rollouts under no_grad to conserve Apple Silicon memory
        with torch.no_grad():
            for _ in range(G):
                try:
                    out = self.engine.generate(
                        prompt=prompt_item.prompt,
                        max_new_tokens=self.config.max_new_tokens,
                        block_size=self.config.block_size,
                        temperature=self.config.temperature,
                        return_logprob=True,
                    )
                    completions.append(out.text)
                    acceptance_rates.append(out.acceptance_rate)
                    assert out.trajectory is not None, "Trajectory must be recorded for GRPO rollouts"
                    trajectories.append(out.trajectory)

                    # Compute scalar reward using pluggable evaluator (with token_ids)
                    score = self.reward_fn(
                        prompt=prompt_item.prompt,
                        completion=out.text,
                        token_ids=out.token_ids,
                        ground_truth=prompt_item.ground_truth,
                        **prompt_item.metadata,
                    )
                    rewards_list.append(float(score))

                except RuntimeError as e:
                    # §D: Catch mask_id commitment gracefully during policy drift.
                    # The failed rollout contributes its failure reward to group stats (lowering group mean
                    # and raising valid siblings' advantages), but has zero completion tokens and zero
                    # synthetic logprobs, completely excluding it from the policy gradient loss.
                    if "mask token" in str(e).lower():
                        failed_text = "<failed_mask_token_commitment>"
                        completions.append(failed_text)
                        acceptance_rates.append(0.0)

                        if hasattr(self.engine.runner, "tokenizer"):
                            prompt_toks = self.engine.runner.tokenizer.encode(
                                prompt_item.prompt, add_special_tokens=False
                            )
                        else:
                            prompt_toks = [1]

                        # Empty completion trajectory: T = 0 tokens participate in loss
                        failed_col = TrajectoryCollector(
                            prompt_toks, temperature=self.config.temperature
                        )
                        trajectories.append(failed_col.to_trajectory())

                        # Assign format failure penalty for group advantage baseline
                        rewards_list.append(float(self.config.format_failure_reward))
                    else:
                        raise e

        # Annealing check for progressive curriculum
        should_hint = True
        if self.config.hint_anneal_steps is not None and self.config.hint_anneal_steps > 0:
            prob = max(0.0, 1.0 - global_step / self.config.hint_anneal_steps)
            if random.random() > prob:
                should_hint = False

        # §3 Split dead-group semantics:
        # Check if all rollouts were truncated without reaching EOS
        eos_ids = getattr(self.engine.runner, "eos_token_ids", {2, 131070})
        all_truncated = all(
            len(traj.completion_tokens) >= self.config.max_new_tokens
            and not any(tok in eos_ids for tok in traj.completion_tokens)
            for traj in trajectories
        )

        reward_span = max(rewards_list) - min(rewards_list) if rewards_list else 0.0
        gold_threshold = 2.0  # Gold diagnosis award threshold

        # Rescue condition: all-failed (max_r < 2.0), zero-variance (span < 1e-5), not all-truncated
        # All-correct groups (min_r >= 2.0) are NOT hinted or re-rolled; they bypass rescue cleanly.
        if (
            self.config.curriculum_hint_on_zero
            and should_hint
            and retry_count < self.config.max_hint_retries
            and reward_span < 1e-5
            and max(rewards_list) < gold_threshold
            and not all_truncated
        ):
            hint_text = self._extract_prompt_hint(prompt_item)
            if hint_text:
                augmented_prompt = (
                    f"{prompt_item.prompt.rstrip()}\n\nClinical Guidance: {hint_text}\n"
                )
                augmented_item = PromptItem(
                    prompt=augmented_prompt,
                    ground_truth=prompt_item.ground_truth,
                    metadata={**prompt_item.metadata, "curriculum_hint_applied": True},
                )
                return self.rollout_group(augmented_item, retry_count=retry_count + 1, global_step=global_step)

        device = self.engine.runner.device
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        advantages = compute_group_advantages(
            rewards, normalize_by_std=self.config.normalize_by_std
        )

        mean_r = float(rewards.mean().item())
        std_r = float(rewards.std(unbiased=False).item())
        mean_alpha = float(sum(acceptance_rates) / max(len(acceptance_rates), 1))

        return GRPORollout(
            prompt_item=prompt_item,
            completions=completions,
            trajectories=trajectories,
            rewards=rewards,
            advantages=advantages,
            mean_reward=mean_r,
            std_reward=std_r,
            acceptance_rate=mean_alpha,
        )

    def compute_rollout_loss(self, rollout: GRPORollout) -> Tuple[torch.Tensor, GRPOMetrics]:
        """Compute differentiable GRPO loss for the completions in a group rollout."""
        device = self.engine.runner.device

        # DAPO-style dead group bypass: if all advantages are zero (e.g. all-correct or rescued group still dead),
        # zero gradient update is guaranteed, bypassing expensive forward passes.
        if torch.all(rollout.advantages == 0.0):
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, GRPOMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        G = len(rollout.trajectories)
        device = self.engine.runner.device

        policy_logprobs: List[torch.Tensor] = []
        old_logprobs: List[torch.Tensor] = []
        ref_logprobs: Optional[List[torch.Tensor]] = [] if self.ref_model is not None else None

        for i in range(G):
            traj = rollout.trajectories[i]
            p_tokens = traj.prompt_tokens
            c_tokens = traj.completion_tokens
            P = len(p_tokens)
            T = len(c_tokens)

            if T == 0:
                empty = torch.zeros(0, device=device)
                policy_logprobs.append(empty)
                old_logprobs.append(empty)
                if ref_logprobs is not None:
                    ref_logprobs.append(empty)
                continue

            full_tokens = p_tokens + c_tokens
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)

            # Causal forward pass with autograd active
            logits, _ = self.engine.runner.forward_causal(input_ids, use_cache=False)

            # §A.3 Discipline: Logits predicting completion tokens c_0..c_{T-1}
            # row_i = P - 1 + i, so slice is [P - 1 : P + T - 1]
            comp_logits = logits[0, P - 1 : P + T - 1, :]
            target_ids = torch.tensor(c_tokens, dtype=torch.long, device=device)

            # Use recorded trajectory temperature (§A.1 operationalization)
            temp = traj.temperature
            if temp > 0:
                log_probs = F.log_softmax(comp_logits / max(temp, 1e-5), dim=-1)
            else:
                log_probs = F.log_softmax(comp_logits, dim=-1)

            pi_logp = log_probs[torch.arange(T, device=device), target_ids]
            policy_logprobs.append(pi_logp)

            # Old policy logprob from rollout trajectory
            old_logp = torch.tensor(traj.logprobs, dtype=torch.float32, device=device)
            old_logprobs.append(old_logp)

            # Reference policy logprob if reference model is provided
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_logits, _ = self.ref_model.forward_causal(input_ids, use_cache=False)
                    ref_comp_logits = ref_logits[0, P - 1 : P + T - 1, :]
                    if temp > 0:
                        ref_log_probs = F.log_softmax(ref_comp_logits / max(temp, 1e-5), dim=-1)
                    else:
                        ref_log_probs = F.log_softmax(ref_comp_logits, dim=-1)
                    ref_pi = ref_log_probs[torch.arange(T, device=device), target_ids]
                    ref_logprobs.append(ref_pi)

        # NeMo RL Overlong Filtering: exclude rollouts reaching max length without EOS from gradient
        loss_multipliers = None
        if self.config.overlong_filtering:
            eos_ids = getattr(self.engine.runner, "eos_token_ids", {2, 131070})
            loss_multipliers = []
            for traj in rollout.trajectories:
                c_toks = traj.completion_tokens
                if len(c_toks) >= self.config.max_new_tokens and not any(tok in eos_ids for tok in c_toks):
                    loss_multipliers.append(0.0)
                else:
                    loss_multipliers.append(1.0)

        loss, metrics = compute_grpo_loss(
            policy_logprobs=policy_logprobs,
            old_logprobs=old_logprobs,
            advantages=rollout.advantages,
            ref_logprobs=ref_logprobs,
            clip_eps=self.config.clip_eps,
            beta_kl=self.config.beta_kl,
            divisor_mode=self.config.divisor_mode,
            dual_clip_c=self.config.dual_clip_c,
            loss_multipliers=loss_multipliers,
        )

        return loss, metrics

    def step_rollout(self, rollout: GRPORollout) -> Dict[str, float]:
        """Perform a single RL optimization step directly on a precomputed rollout."""
        self.engine.runner.model.train()
        self.optimizer.zero_grad()

        loss, metrics = self.compute_rollout_loss(rollout)
        loss.backward()

        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.max_grad_norm)

        self.optimizer.step()

        if torch.backends.mps.is_available() and self.config.empty_cache_interval > 0:
            torch.mps.empty_cache()

        return {
            "loss": float(loss.item()),
            "policy_loss": metrics.policy_loss,
            "kl_loss": metrics.kl_loss,
            "mean_reward": rollout.mean_reward,
            "acceptance_rate": rollout.acceptance_rate,
            "clip_fraction": metrics.clip_fraction,
        }

    def step(self, batch: List[PromptItem], global_step: Optional[int] = None) -> Dict[str, float]:
        """Perform a single RL optimization step on a batch of prompts."""
        if not batch:
            return {}

        self.engine.runner.model.train()
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.engine.runner.device)
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        total_reward = 0.0
        total_alpha = 0.0
        total_clip = 0.0
        B = len(batch)
        cur_step = global_step if global_step is not None else self.global_step

        for prompt_item in batch:
            rollout = self.rollout_group(prompt_item, global_step=cur_step)
            loss, metrics = self.compute_rollout_loss(rollout)

            scaled_loss = loss / B
            scaled_loss.backward()

            total_loss = total_loss + scaled_loss.detach()
            total_policy_loss += metrics.policy_loss / B
            total_kl_loss += metrics.kl_loss / B
            total_reward += rollout.mean_reward / B
            total_alpha += rollout.acceptance_rate / B
            total_clip += metrics.clip_fraction / B

        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.max_grad_norm)

        self.optimizer.step()
        self.global_step += 1

        # Free Apple Silicon unified memory buffers between training steps
        if torch.backends.mps.is_available() and self.config.empty_cache_interval > 0:
            torch.mps.empty_cache()

        return {
            "loss": float(total_loss.item()),
            "policy_loss": total_policy_loss,
            "kl_loss": total_kl_loss,
            "mean_reward": total_reward,
            "acceptance_rate": total_alpha,
            "clip_fraction": total_clip,
        }

    def train(
        self,
        dataset: PromptDataset,
        steps: Optional[int] = None,
        epochs: int = 1,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> List[Dict[str, float]]:
        """Train the policy over the dataset using DiffuGRPO."""
        history: List[Dict[str, float]] = []
        global_step = 0
        total_steps = steps if steps is not None else len(dataset) * epochs // self.config.batch_size

        for epoch in range(epochs):
            for batch in dataset.iter_batches(batch_size=self.config.batch_size, shuffle=True):
                if steps is not None and global_step >= steps:
                    return history

                t0 = time.perf_counter()
                metrics = self.step(batch)
                dt = time.perf_counter() - t0

                global_step += 1
                metrics["step"] = global_step
                metrics["epoch"] = epoch + 1
                metrics["step_time"] = dt
                history.append(metrics)

                if global_step % self.config.log_interval == 0:
                    print(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Policy Loss: {metrics['policy_loss']:.4f} | "
                        f"Reward: {metrics['mean_reward']:.3f} | "
                        f"Alpha: {metrics['acceptance_rate']:.1%} | "
                        f"Time: {dt:.2f}s"
                    )

                if callback is not None:
                    callback(global_step, metrics)

                if self.config.save_dir and global_step % max(1, total_steps // 5) == 0:
                    self.save_weights(f"{self.config.save_dir}/step_{global_step}.pt")

        return history

    def save_weights(self, path: Union[str, Path]) -> None:
        """Save trainable parameters to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            name: param
            for name, param in self.engine.runner.model.named_parameters()
            if param.requires_grad
        }
        torch.save(state_dict, path)

    def load_weights(self, path: Union[str, Path]) -> None:
        """Load trainable parameters from disk."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state_dict = torch.load(path, map_location=self.engine.runner.device)
        self.engine.runner.model.load_state_dict(state_dict, strict=False)
