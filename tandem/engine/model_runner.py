from __future__ import annotations

from typing import Optional, Tuple
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache

from tandem.config import EngineConfig



def resolve_eos_token_ids(tokenizer, model_config, explicit_ids=None) -> set[int]:
    """Resolve EOS ids from tokenizer, model config, and explicit overrides."""
    eos_ids: set[int] = set()
    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tok_eos, int):
        eos_ids.add(int(tok_eos))
    cfg_eos = getattr(model_config, "eos_token_id", None)
    if isinstance(cfg_eos, int):
        eos_ids.add(int(cfg_eos))
    elif isinstance(cfg_eos, (list, tuple, set)):
        eos_ids.update(int(x) for x in cfg_eos if x is not None)
    if explicit_ids:
        eos_ids.update(int(x) for x in explicit_ids)
    return eos_ids


class TandemModelRunner:
    """Wraps Nemotron-Labs-Diffusion model for Apple Silicon (MPS/CPU) self-speculative execution."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            config.model_id, trust_remote_code=True
        ).to(self.device).to(self.dtype)

        # Cache mask_token_id and eos_token_id
        self.mask_token_id = getattr(
            self.model.config, "mask_token_id", config.mask_id
        )
        self.eos_token_ids = resolve_eos_token_ids(
            self.tokenizer, self.model.config, config.eos_token_ids
        )

    def set_diffusion_mode(self, enabled: bool) -> None:
        """Toggle bidirectional diffusion attention vs causal autoregressive attention."""
        for layer in self.model.encoder.layers:
            if hasattr(layer.self_attn, "diffusion_lm"):
                layer.self_attn.diffusion_lm = enabled

    def toggle_adapters(self, enabled: bool) -> None:
        """Toggle LoRA adapters if present (active for diffusion draft, inactive for causal verify)."""
        for module in self.model.modules():
            if hasattr(module, "_disable_adapters"):
                module._disable_adapters = not enabled

    def forward_causal(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, DynamicCache]:
        """Execute a causal autoregressive forward pass with KV cache."""
        self.set_diffusion_mode(False)
        self.toggle_adapters(False)

        enc_past = past_key_values if past_key_values is not None else (DynamicCache() if use_cache else None)

        enc_out = self.model.encoder(
            input_ids=input_ids,
            past_key_values=enc_past,
            use_cache=use_cache,
            use_causal_mask=True,
        )
        logits = self.model.diffusion_head(enc_out.last_hidden_state)
        return logits, enc_out.past_key_values

    def forward_draft(
        self,
        block: torch.Tensor,
        past_key_values: DynamicCache,
    ) -> torch.Tensor:
        """Execute a bidirectional diffusion forward pass proposing draft tokens for the block."""
        self.set_diffusion_mode(True)
        self.toggle_adapters(True)

        enc_out = self.model.encoder(
            input_ids=block,
            past_key_values=past_key_values,
            use_cache=False,
        )
        logits = self.model.diffusion_head(enc_out.last_hidden_state)
        return logits
