from __future__ import annotations

import os
import time
import pytest
from typing import List

from tandem.engine.umls_trie import (
    UMLSTrieNode,
    UMLSEntityTrie,
    UMLSSpeculativeDrafter,
)


class MockTokenizer:
    """Fast deterministic tokenizer for unit testing without HF dependencies."""

    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.next_id = 100

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = []
        for word in text.strip().split():
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.inv_vocab[self.next_id] = word
                self.next_id += 1
            tokens.append(self.vocab[word])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return " ".join(self.inv_vocab.get(tid, f"<unk_{tid}>") for tid in token_ids)


def test_node_attributes_and_best_child():
    node = UMLSTrieNode(token_id=42, is_terminal=False)
    assert node.token_id == 42
    assert not node.is_terminal
    assert node.children == {}
    assert node.cui is None
    assert node.concept_name is None
    assert node.frequency == 0
    assert node.best_child is None

    # Test update_best_child
    child1 = UMLSTrieNode(token_id=101, frequency=2)
    child2 = UMLSTrieNode(token_id=102, frequency=5)
    node.children[101] = child1
    node.update_best_child(101, child1)
    assert node.best_child == (101, child1)

    node.children[102] = child2
    node.update_best_child(102, child2)
    assert node.best_child == (102, child2)


def test_insertion_and_exact_continuation():
    trie = UMLSEntityTrie()
    assert len(trie) == 0
    assert trie.num_nodes == 1

    # Insert concept 1: "Hypertension" -> [10, 20, 30]
    trie.insert(
        token_ids=[10, 20, 30],
        concept_name="Hypertension",
        cui="C0020538",
        metadata={"sab": "MSH", "tty": "MH"},
    )
    assert len(trie) == 1
    assert trie.num_unique_terms == 1
    assert trie.num_nodes == 4  # root + 3 tokens

    # Insert concept 2: "Essential Hypertension" -> [10, 20, 30, 40]
    trie.insert(
        token_ids=[10, 20, 30, 40],
        concept_name="Essential Hypertension",
        cui="C0085580",
        metadata={"sab": "SNOMEDCT_US", "tty": "PT"},
    )
    assert len(trie) == 2
    assert trie.num_unique_terms == 2
    assert trie.num_nodes == 5

    # Test exact lookups
    res1 = trie.exact_lookup([10, 20, 30])
    assert res1 == ("C0020538", "Hypertension")

    res2 = trie.exact_lookup([10, 20, 30, 40])
    assert res2 == ("C0085580", "Essential Hypertension")

    # Prefix that is not terminal
    assert trie.exact_lookup([10, 20]) is None
    # Missing sequence
    assert trie.exact_lookup([99, 999]) is None

    # Test find_node
    node = trie.find_node([10, 20, 30])
    assert node is not None
    assert node.is_terminal
    assert node.cui == "C0020538"
    assert "C0020538" in node.metadata["all_cuis"]
    assert node.children[40].cui == "C0085580"


def test_branching_vs_unique_continuation():
    trie = UMLSEntityTrie()

    # Part A: Unique continuation
    # Concept: [1, 2, 3, 4]
    trie.insert([1, 2, 3, 4], "Unique Pathway", "C1000")
    # Deterministic proposal for prefix [1, 2] should follow [3, 4]
    draft_det = trie.propose_draft([1, 2], max_tokens=5, deterministic_only=True)
    assert draft_det == [3, 4]
    draft_top = trie.propose_draft([1, 2], max_tokens=5, deterministic_only=False)
    assert draft_top == [3, 4]

    # Part B: Ambiguous branching continuation
    # Insert competing branches:
    # [10, 20, 30, 40] (1 occurrence)
    # [10, 20, 50, 60] (3 occurrences)
    trie.insert([10, 20, 30, 40], "Minor Branch", "C2001")
    trie.insert([10, 20, 50, 60], "Major Branch", "C2002")
    trie.insert([10, 20, 50, 60], "Major Branch Duplicate", "C2002")
    trie.insert([10, 20, 50, 60], "Major Branch Triplicate", "C2002")

    # At prefix [10, 20], path branches into 30 (freq 1) and 50 (freq 3).
    # Deterministic draft must halt because of branch ambiguity:
    assert trie.propose_draft([10, 20], max_tokens=5, deterministic_only=True) == []

    # Top-branch draft must follow the most frequent branch [50, 60]:
    assert trie.propose_draft([10, 20], max_tokens=5, deterministic_only=False) == [50, 60]

    # Part C: Downstream branching
    # [100, 200, 300, 401] (freq 1)
    # [100, 200, 300, 402] (freq 2)
    trie.insert([100, 200, 300, 401], "Downstream 1", "C3001")
    trie.insert([100, 200, 300, 402], "Downstream 2", "C3002")
    trie.insert([100, 200, 300, 402], "Downstream 2 dup", "C3002")

    # Prefix [100]:
    # Path to 200 and 300 is unique. Branching occurs at 300.
    # Deterministic draft should emit [200, 300] and stop before the branch:
    assert trie.propose_draft([100], max_tokens=5, deterministic_only=True) == [200, 300]
    # Top-branch draft should traverse all the way through the dominant branch:
    assert trie.propose_draft([100], max_tokens=5, deterministic_only=False) == [200, 300, 402]

    # Part D: Terminal node boundary
    # [500, 501] is a complete concept. [500, 501, 502] is a compound extension.
    trie.insert([500, 501], "Base Concept", "C5001")
    trie.insert([500, 501, 502], "Extended Concept", "C5002")

    # For prefix [500, 501]:
    # Deterministic stops because the concept is already complete:
    assert trie.propose_draft([500, 501], max_tokens=5, deterministic_only=True) == []
    # Top-branch continues into the extension:
    assert trie.propose_draft([500, 501], max_tokens=5, deterministic_only=False) == [502]


def test_draft_proposal_generation():
    trie = UMLSEntityTrie()
    trie.insert([10, 20, 30, 40, 50, 60], "Long Concept", "C9001")

    # Test max_tokens clipping
    assert trie.propose_draft([10, 20], max_tokens=2) == [30, 40]
    assert trie.propose_draft([10, 20], max_tokens=3) == [30, 40, 50]
    assert trie.propose_draft([10, 20], max_tokens=10) == [30, 40, 50, 60]

    # Test edge cases
    assert trie.propose_draft([], max_tokens=5) == []
    assert trie.propose_draft([10, 20], max_tokens=0) == []
    assert trie.propose_draft([9999, 8888], max_tokens=5) == []

    # Test longest suffix matching with context history
    # Suppose model has emitted: [901, 902, 903, 10, 20]
    # Where [901, 902, 903] are prior context words outside the trie
    context_prefix = [901, 902, 903, 10, 20]
    draft = trie.propose_draft(context_prefix, max_tokens=4)
    assert draft == [30, 40, 50, 60]

    # Test UMLSSpeculativeDrafter wrapper
    drafter = UMLSSpeculativeDrafter(trie, block_size=3, deterministic_only=False)
    assert drafter.propose([10, 20]) == [30, 40, 50]
    assert drafter.propose([10, 20], max_tokens=2) == [30, 40]


def test_mrconso_slice_ram_and_speed():
    mrconso_path = "/Users/rohanmaster/Developer/deep-research-pipeline/data/external/umls/2026AA/META/MRCONSO.RRF"
    if not os.path.exists(mrconso_path):
        pytest.skip(f"MRCONSO.RRF not found at {mrconso_path}")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("nvidia/Nemotron-Labs-Diffusion-3B", trust_remote_code=True)
    except Exception:
        tok = MockTokenizer()

    # Stream 5,000 concepts from MRCONSO.RRF
    t0_build = time.perf_counter()
    trie = UMLSEntityTrie.build_from_mrconso(
        mrconso_path=mrconso_path,
        tokenizer=tok,
        max_concepts=5000,
        target_sabs=["SNOMEDCT_US", "MSH", "RXNORM"],
    )
    build_time = time.perf_counter() - t0_build
    assert trie.num_concepts == 5000
    assert trie.num_nodes > 5000
    print(f"\n[Profile] Built 5,000 UMLS concepts in {build_time:.3f}s ({5000/build_time:.0f} concepts/sec)")

    # Profile memory consumption
    mem = trie.profile_memory()
    print(f"[Profile] Total Memory: {mem['total_mb']:.2f} MB")
    print(f"[Profile] Memory per Concept: {mem['bytes_per_concept']:.1f} bytes")
    print(f"[Profile] RAM per 10k Concepts: {mem['ram_per_10k_concepts_mb']:.2f} MB")

    # Assertions on memory footprint
    assert mem["total_mb"] < 25.0, f"Memory ({mem['total_mb']:.2f} MB) should be lean (< 25 MB)"
    assert mem["ram_per_10k_concepts_mb"] < 50.0

    # Profile proposal latency
    # Identify a valid prefix from the trie
    first_root_tok = next(iter(trie.root.children.keys()))
    child = trie.root.children[first_root_tok]
    second_tok = next(iter(child.children.keys())) if child.children else 100
    sample_prefix = [first_root_tok, second_tok]

    # Benchmark 2,000 lookups
    num_queries = 2000
    t0_bench = time.perf_counter()
    for _ in range(num_queries):
        _ = trie.propose_draft(sample_prefix, max_tokens=4)
    elapsed = time.perf_counter() - t0_bench
    avg_latency_us = (elapsed / num_queries) * 1e6

    print(f"[Profile] Average Proposal Latency: {avg_latency_us:.3f} microseconds ({num_queries/elapsed:.0f} ops/sec)")
    # Assert sub-microsecond proposal time (with generous threshold of < 3.0 us to avoid CI flakiness)
    assert avg_latency_us < 3.0, f"Latency ({avg_latency_us:.3f} us) must be sub-microsecond or near-zero compute"
