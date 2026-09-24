from __future__ import annotations

from typing import List, Tuple, Optional
import torch


class DynamicKVCache:
    """Manages Key-Value cache for self-speculative decoding with rewind capability.
    
    Tracks lengths explicitly on CPU to prevent device-synchronizing .item() stalls
    during speculative loops.
    """

    def __init__(
        self,
        num_layers: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.device = torch.device(device)
        self.dtype = dtype

        # key_cache and value_cache per layer: List of torch.Tensor
        # Each tensor has shape [batch_size, num_kv_heads, seq_len, head_dim]
        self.key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.value_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._seq_len: int = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value states to the cache for a given layer.
        
        Args:
            key_states: [batch_size, num_kv_heads, new_tokens, head_dim]
            value_states: [batch_size, num_kv_heads, new_tokens, head_dim]
            layer_idx: index of transformer layer
            
        Returns:
            Tuple of concatenated (key, value) tensors for attention.
        """
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )

        if layer_idx == 0:
            self._seq_len = self.key_cache[0].shape[2]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def rewind(self, num_slots: int) -> int:
        """Roll back the KV cache by num_slots on draft rejection.
        
        Args:
            num_slots: Number of rejected slots to truncate from the tail.
            
        Returns:
            New seq_len after rewind.
        """
        if num_slots <= 0:
            return self._seq_len

        assert num_slots <= self._seq_len, (
            f"Cannot rewind {num_slots} slots from cache with length {self._seq_len}"
        )

        new_len = self._seq_len - num_slots
        for i in range(self.num_layers):
            if self.key_cache[i] is not None:
                self.key_cache[i] = self.key_cache[i][:, :, :new_len, :].contiguous()
                self.value_cache[i] = self.value_cache[i][:, :, :new_len, :].contiguous()

        self._seq_len = new_len
        return self._seq_len

    def reset(self) -> None:
        """Clear all cached tensors."""
        for i in range(self.num_layers):
            self.key_cache[i] = None
            self.value_cache[i] = None
        self._seq_len = 0
