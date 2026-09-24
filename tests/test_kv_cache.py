from __future__ import annotations

import pytest
import torch
from tandem.engine.kv_cache import DynamicKVCache


def test_kv_cache_update_and_rewind():
    cache = DynamicKVCache(num_layers=2, device="cpu", dtype=torch.float32)
    assert cache.seq_len == 0

    # Simulate prompt prefill of 10 tokens
    k0 = torch.randn(1, 4, 10, 64)
    v0 = torch.randn(1, 4, 10, 64)
    cache.update(k0, v0, layer_idx=0)
    cache.update(k0, v0, layer_idx=1)
    assert cache.seq_len == 10

    # Simulate draft pass extending by 4 candidate tokens
    k_draft = torch.randn(1, 4, 4, 64)
    v_draft = torch.randn(1, 4, 4, 64)
    cache.update(k_draft, v_draft, layer_idx=0)
    cache.update(k_draft, v_draft, layer_idx=1)
    assert cache.seq_len == 14

    # Simulate rejection of 2 tokens (accept c=2, reject K-c=2)
    new_len = cache.rewind(num_slots=2)
    assert new_len == 12
    assert cache.seq_len == 12
    assert cache.key_cache[0].shape == (1, 4, 12, 64)
    assert cache.value_cache[1].shape == (1, 4, 12, 64)


def test_kv_cache_reset():
    cache = DynamicKVCache(num_layers=1, device="cpu", dtype=torch.float32)
    k = torch.randn(1, 2, 5, 32)
    v = torch.randn(1, 2, 5, 32)
    cache.update(k, v, layer_idx=0)
    assert cache.seq_len == 5

    cache.reset()
    assert cache.seq_len == 0
    assert cache.key_cache[0] is None
